from __future__ import annotations

import functools, sys
from pathlib import Path

import torch
from tensordict.nn import InteractionType, TensorDictModule, NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import (
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardSum,
    StepCounter,
    ToTensorImage,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ConvNet, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.record import VideoRecorder

ROOT = Path(__file__).resolve().parent
ACRO_ROOT = ROOT / "agent-centric-representations"
if str(ACRO_ROOT) not in sys.path:
    sys.path.append(str(ACRO_ROOT))

from acro import ACROAgent

class CriticNet(nn.Module):
    def __init__(self, emb_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim + act_dim, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # state: [B, emb_dim], action: [B, act_dim]
        x = torch.cat([state, action], dim=-1)   # → [B, emb_dim+act_dim]
        x = self.act(self.fc1(x))
        return self.fc2(x)     

def make_acro_optimizer(cfg, acro: ACROAgent):
    """Optimise ONLY the encoder / predictor / k-embedding (policy parts are
    trained by SAC and ACRO internally)."""
    lr = getattr(cfg.acro, "lr", 1e-4)
    wd = getattr(cfg.acro, "weight_decay", 0.0)
    params = (
        list(acro.encoder.parameters())
        + list(acro.predictor.parameters())
        + list(acro.k_embedding.parameters())
    )
    return optim.Adam(params, lr=lr, weight_decay=wd)

def make_sac_agent(cfg, train_env, eval_env, device, acro: ACROAgent):
    """Actor & critic over ACRO latent `state`."""
    # Determine embedding dimension
    with torch.no_grad():
        fake_px = torch.zeros(1, *train_env.observation_spec_unbatched["pixels"].shape, device=device)
        emb_dim = acro.encoder(fake_px).shape[-1]

    action_spec = train_env.action_spec_unbatched.to(device)
    act_dim     = action_spec.shape[-1]

    actor_backbone = nn.Sequential(
        nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, 2 * act_dim)
    ).to(device)
    actor_backbone = nn.Sequential(actor_backbone, NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-3).to(device))

    actor_tdmod = TensorDictModule(actor_backbone, in_keys=["state"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_tdmod,
        distribution_class=TanhNormal,
        distribution_kwargs={"low": action_spec.space.low, "high": action_spec.space.high, "tanh_loc": False},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    critic_net = CriticNet(emb_dim, act_dim).to(device)
    critic     = ValueOperator(module=critic_net, in_keys=["state", "action"]).to(device)

    # Warm-start parameters by forward pass
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict().to(device)
        td["state"] = acro.encoder(td["pixels"])
        actor(td); critic(td)

    return nn.ModuleList([actor, critic]), actor

def make_loss_module(cfg, model):
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)
    target_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_updater

def make_sac_optimizer(cfg, loss_module):
    actor_params  = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    opt_actor  = optim.Adam(actor_params,  lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.adam_eps)
    opt_critic = optim.Adam(critic_params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.adam_eps)
    opt_alpha  = optim.Adam([loss_module.log_alpha], lr=3e-4)
    return opt_actor, opt_critic, opt_alpha

def log_metrics(logger, metrics, step):
    for k, v in metrics.items():
        logger.log_scalar(k, v, step)

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

# -------------------------------------------------------------------
# Legacy methods from old utils (added for backward compatibility)
# -------------------------------------------------------------------
def env_maker(cfg, device="cpu", from_pixels=True):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=True,
            )
    elif lib == "dm_control":
        return DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=True
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")

def apply_env_transforms(env, max_episode_steps=1000):
    return TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            ToTensorImage(in_keys=["pixels"]),
            GrayScale(in_keys=["pixels"]),
            ObservationNorm(in_keys=["pixels"], loc=0.0, scale=255.0),
            Resize(84, 84, in_keys=["pixels"]),
            StepCounter(max_episode_steps),
            RewardSum(),
        ),
    )

def make_environment(cfg, logger=None):
    """Make environments for training and evaluation."""
    partial_fn = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial_fn),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    partial_fn = functools.partial(env_maker, cfg=cfg, from_pixels=True)
    trsf_clone = train_env.transform.clone()
    if logger and getattr(cfg.logger, "video", False):
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(partial_fn),
            serial_for_single=True,
        ),
        trsf_clone,
    )
    return train_env, eval_env

def make_train_environment(cfg):
    """Make training-only environment."""
    partial_fn = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial_fn),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)
    return apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make synchronous data collector."""
    device = cfg.collector.device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    compile_policy = {"mode": compile_mode} if cfg.compile.compile else None
    cudagraph_policy = {"warmup": 5} if (cfg.compile.compile and cfg.compile.cudagraphs) else None

    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy=compile_policy,
        cudagraph_policy=cudagraph_policy,
        trust_policy=True,
    )
    collector.set_seed(cfg.env.seed)
    return collector

def flatten(td):
    return td.reshape(-1)

def make_collector_async(
    cfg, train_env_make, actor_model_explore, compile_mode, replay_buffer
):
    """Make asynchronous data collector."""
    device = cfg.collector.device or (torch.device("cuda:1") if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else torch.device("cpu"))
    collector = aSyncDataCollector(
        train_env_make,
        actor_model_explore,
        init_random_frames=0,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        env_device=torch.device("cpu"),
        compile_policy={"mode": compile_mode, "warmup": 5} if compile_mode else False,
        cudagraph_policy={"warmup": 20} if getattr(cfg.compile, "cudagraphs", False) else False,
        replay_buffer=replay_buffer,
        extend_buffer=True,
        postproc=flatten,
        no_cuda_sync=True,
        trust_policy=True,
    )
    collector.set_seed(cfg.env.seed)
    collector.start()
    return collector

def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
    shared: bool = False,
):
    storage_cls = (
        functools.partial(LazyMemmapStorage, device="cpu")
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(buffer_size),
            batch_size=batch_size,
            shared=shared,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(buffer_size),
            batch_size=batch_size,
            shared=shared,
        )
    if scratch_dir:
        replay_buffer.append_transform(lambda td: td.to(device))
    return replay_buffer

def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Unknown activation {cfg.network.activation!r}")

def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []
    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params
