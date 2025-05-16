# sac.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
from __future__ import annotations
import warnings, os, sys
from pathlib import Path

# Headless MuJoCo rendering
os.environ.setdefault("MUJOCO_GL", "egl")

# -------------------------------------------------------------------------
# Make sure ACROAgent is on the path
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ACRO_ROOT = ROOT / "agent-centric-representations"
if str(ACRO_ROOT) not in sys.path:
    sys.path.append(str(ACRO_ROOT))

from acro import ACROAgent
from acro import NoShiftAug

# -------------------------------------------------------------------------
# Third-party imports
# -------------------------------------------------------------------------
import hydra
import numpy as np
import torch, tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
import torch.nn as nn
import torch


# -------------------------------------------------------------------------
# Local helpers
# -------------------------------------------------------------------------
import utils
from sac_acro_utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
    make_acro_optimizer,
)
from numpy_replay_buffer import EfficientReplayBuffer

# -------------------------------------------------------------------------
# ACRO sampler: lets ACROAgent pull batches from our TorchRL buffer
# -------------------------------------------------------------------------
class _TorchRLSampler:
    def __init__(self, torchrl_rb, device):
        self.rb     = torchrl_rb
        self.device = device

    def __next__(self):
        td = self.rb.sample().to("cpu")  # CPU tensordict
        obs      = td["pixels"].numpy()             # [B, C, H, W]
        action   = td["action"].numpy().astype(np.float32)
        reward   = td["next", "reward"].numpy().astype(np.float32)
        discount = (1.0 - td["next", "done"].float()).numpy().astype(np.float32)
        next_obs = td["next", "pixels"].numpy()
        # simple 1-step positive pair
        k_step = np.ones_like(reward, dtype=np.float32)
        obs_k  = next_obs
        return (obs, action, reward, discount, next_obs, k_step, obs_k)

# -------------------------------------------------------------------------
# Wrapper to encode pixels → state before passing to ACROAgent
# -------------------------------------------------------------------------
class PixelToState(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: [E, T, C, H, W]
        if len(pixels.shape) == 5:
            E, T, C, H, W = pixels.shape
        elif len(pixels.shape) == 4:
            E, C, H, W = pixels.shape
            T = 1
        else:
            raise ValueError(f"Invalid pixel shape: {pixels.shape}")

        flat = pixels.view(-1, C, H, W)
        z    = self.encoder(flat)
        return z.view(E, T, z.shape[-1]) if len(pixels.shape) == 5 else z.view(E, z.shape[-1])

def build_acro_agent(cfg, train_env, device):
    obs_shape    = train_env.observation_spec_unbatched["pixels"].shape
    action_spec  = train_env.action_spec_unbatched.to(device)
    action_shape = action_spec.shape
    max_action   = float(action_spec.space.high[0])

    return ACROAgent(
        obs_shape          = obs_shape,
        action_shape       = action_shape,
        device             = device,
        lr                 = cfg.acro.lr,
        feature_dim        = cfg.acro.feature_dim,
        hidden_dim         = getattr(cfg.acro, "hidden_dim", 256),
        critic_target_tau  = getattr(cfg.acro, "critic_target_tau", cfg.optim.target_update_polyak),
        num_expl_steps     = getattr(cfg.acro, "num_seed_frames", cfg.collector.init_random_frames),
        update_every_steps = getattr(cfg.acro, "update_every_steps", 1),
        stddev_schedule    = getattr(cfg.acro, "stddev_schedule", "linear(0.1,0.1,25000)"),
        stddev_clip        = getattr(cfg.acro, "stddev_clip", 0.1),
        use_tb             = False,
        augmentation       = NoShiftAug(),
        use_bc             = False,
        k_embed            = False,
        use_critic_grads   = True,
        max_action         = max_action,
    )

# ========================================================================
#                                  MAIN
# ========================================================================
@hydra.main(version_base="1.1", config_path="", config_name="sac_acro_config")
def main(cfg: DictConfig):
    # -------------------------- device --------------------------
    device_str = cfg.network.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)

    # -------------------------- logger --------------------------
    exp_name = generate_exp_name("SAC-ACRO", cfg.logger.exp_name)
    logger   = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type      = cfg.logger.backend,
            logger_name      = "sac_acro_logging",
            experiment_name  = exp_name,
            wandb_kwargs     = {
                "mode"       : cfg.logger.mode,
                "config"     : dict(cfg),
                "project"    : cfg.logger.project_name,
                "group"      : cfg.logger.group_name,
                "dir"        : "/datastor1/sarthakd/wandb",
            },
        )

    ckpt_dir = Path.cwd() / cfg.checkpoint.save_dir
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # ---------------------- environments -----------------------
    train_env, eval_env = make_environment(cfg, logger=logger)

    # ----------------------- build ACRO ------------------------
    acro        = build_acro_agent(cfg, train_env, device)
    acro_opt    = make_acro_optimizer(cfg, acro)

    # ----------------------- build SAC -------------------------
    model, actor_module = make_sac_agent(cfg, train_env, eval_env, device, acro)
    loss_module, target_net_updater = make_loss_module(cfg, model)
    opt_actor, opt_critic, opt_alpha = make_sac_optimizer(cfg, loss_module)
    sac_opt = group_optimizers(opt_actor, opt_critic, opt_alpha)

    # ---------------- collector & buffers ---------------------
        # ---------------------------------------------------------------------

    encoder_td = TensorDictModule(
        module=PixelToState(acro.encoder),
        in_keys=["pixels"],
        out_keys=["state"],
    )

    # Chain encoder + actor into a single policy module
    policy = TensorDictSequential(encoder_td, actor_module)
    collector     = make_collector(cfg, train_env, policy, compile_mode=cfg.compile.compile_mode)
    
    replay_buffer = make_replay_buffer(
        batch_size    = cfg.optim.batch_size,
        prb           = cfg.replay_buffer.prb,
        buffer_size   = cfg.replay_buffer.size,
        scratch_dir   = cfg.replay_buffer.scratch_dir,
        device        = device,
    )
    acro_sampler  = _TorchRLSampler(replay_buffer, device)  # for ACROAgent.update()

    # ------------------ compile update fn? --------------------
    def sac_update(batch: TensorDict) -> TensorDict:
        batch = batch.to(device)
        td    = loss_module(batch)
        (td["loss_actor"] + td["loss_qvalue"] + td["loss_alpha"]).sum().backward()
        sac_opt.step()
        sac_opt.zero_grad(set_to_none=True)
        target_net_updater.step()
        return td.detach()

    update_f = sac_update
    if cfg.compile.compile:
        mode = cfg.compile.compile_mode or ("reduce-overhead" if not cfg.compile.cudagraphs else "default")
        update_f = compile_with_warmup(sac_update, mode=mode, warmup=1)
    if cfg.compile.cudagraphs:
        warnings.warn("CudaGraphModule is experimental; use with caution.")
        update_f = CudaGraphModule(update_f, in_keys=[], out_keys=[], warmup=5)

    # -------------- training loop setup -----------------------
    init_rand     = cfg.collector.init_random_frames
    frames_per_b  = cfg.collector.frames_per_batch
    num_updates   = int(frames_per_b * cfg.optim.utd_ratio)
    eval_iter     = cfg.logger.eval_iter
    max_ep_steps  = cfg.env.max_episode_steps

    pbar           = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_fr   = 0
    last_sac_loss  = None

    # ====================================================================
    #                             MAIN LOOP
    # ====================================================================
    for td in collector:
        # ---------------- encode + store ----------------------------
        with torch.no_grad():
            td = td.to(device)
            # original pixels shape: [E, T, C, H, W]
            px       = td["pixels"]
            px_next  = td["next", "pixels"]
            E, T, C, H, W = px.shape

            # flatten env & time dims → [E*T, C, H, W]
            flat_px      = px.view(-1, C, H, W)
            flat_px_next = px_next.view(-1, C, H, W)

            # encode
            flat_s       = acro.encoder(flat_px)
            flat_s_next  = acro.encoder(flat_px_next)

            # reshape back to [E, T, repr_dim]
            repr_dim     = flat_s.shape[-1]
            td["state"]       = flat_s.view(E, T, repr_dim)
            td["next", "state"] = flat_s_next.view(E, T, repr_dim)

        replay_buffer.extend(td.reshape(-1))

        # ---------------- bookkeeping -------------------------------
        current_f = td.numel()
        collected_fr += current_f
        pbar.update(current_f)

        # --------------- SAC + ACRO updates -------------------------
        if collected_fr >= init_rand:
            sac_losses = TensorDict(batch_size=[num_updates])
            for i in range(num_updates):
                batch = replay_buffer.sample()
                sac_losses[i] = update_f(batch).select("loss_actor", "loss_qvalue", "loss_alpha")
                if cfg.replay_buffer.prb:
                    replay_buffer.update_priority(batch)
            last_sac_loss = sac_losses.mean()

            # --- online ACRO representation learning ---------------
            if collected_fr < cfg.acro.pretrain_num_frames:
                acro_stats = acro.pretrain(acro_sampler, collected_fr)
            else:
                acro_stats = acro.update(acro_sampler, collected_fr)
        else:
            acro_stats = {}

        # ---------------- logging & evaluation -----------------------
        metrics = {}

        # train rewards
        done_mask = td["next", "done"] if td["next", "done"].any() else td["next", "truncated"]
        if done_mask.any():
            rewards = td["next", "episode_reward"][done_mask]
            lengths = td["next", "step_count"][done_mask].sum() / len(rewards)
            metrics.update({
                "train/reward"         : rewards,
                "train/episode_length" : lengths
            })

        # losses
        if collected_fr >= init_rand:
            metrics.update({
                "train/q_loss"      : last_sac_loss["loss_qvalue"],
                "train/actor_loss"  : last_sac_loss["loss_actor"],
                "train/alpha_loss"  : last_sac_loss["loss_alpha"],
                "train/alpha"       : loss_module.log_alpha.exp().detach(),
            })
            # merge ACRO stats with prefix
            for k, v in acro_stats.items():
                metrics[f"acro/{k}"] = v


        # evaluation
        if abs(collected_fr % eval_iter) < frames_per_b:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad(), timeit("eval"):
                rollout = eval_env.rollout(
                    max_ep_steps,
                    policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                metrics["eval/reward"] = rollout["next", "reward"].sum(-2).mean().item()
        
        if collected_fr >= cfg.checkpoint.interval_frames and collected_fr % cfg.checkpoint.interval_frames < frames_per_b:
            ckpt_path = ckpt_dir / f"ckpt_{collected_fr}.pt"
            torch.save(
                {
                    # SAC networks & optimizer
                    "actor_state":   model[0].state_dict(),
                    "critic_state":  model[1].state_dict(),
                    "sac_opt":       sac_opt.state_dict(),
                    # ACRO components
                    "acro.encoder":         acro.encoder.state_dict(),
                    "acro.decoder":         acro.decoder.state_dict(),
                    "acro.actor":           acro.actor.state_dict(),
                    "acro.predictor":       acro.predictor.state_dict(),
                    "acro.critic":          acro.critic.state_dict(),
                    "acro.critic_target":   acro.critic_target.state_dict(),
                    "acro.k_embedding":     acro.k_embedding.state_dict(),
                    # ACRO optimizer
                    "acro_opt":             acro_opt.state_dict(),
                    # training progress
                    "frame":                collected_fr,
                    "rng_state":            torch.get_rng_state(),
                },
                ckpt_path,
            )

            # optional: also print/log
            print(f"[checkpoint] saved to {ckpt_path}")

        if logger is not None and metrics:
            metrics.update(timeit.todict(prefix="time"))
            metrics["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics, collected_fr)

    # ---------------- shutdown -----------------------
    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
