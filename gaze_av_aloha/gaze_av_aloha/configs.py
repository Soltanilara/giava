from dataclasses import dataclass, field

@dataclass
class WandBConfig:
    enable: bool = True
    project: str = ""
    entity: str = ""
    job_name: str = ""
    run_id: str = ""

@dataclass 
class EnvConfig:
    env_package: str = "gym_pusht"
    env_name: str = "PushT-v0"
    env_kwargs: dict = field(default_factory=lambda: {})
    eval_n_episodes: int = 50
    eval_n_envs: int = 10
    eval_options: dict = field(default_factory=lambda: {})
    visualization_steps: int = 300

@dataclass
class TaskConfig:
    type: str = "pusht"
    dataset_repo_id: str = "lerobot/pusht"
    dataset_root: str = "" 
    dataset_episodes: list[int] = field(default_factory=lambda: [])
    override_stats: dict = field(default_factory=lambda: {})
    envs: dict[str, EnvConfig] = field(default_factory=lambda: {})
    fps: float = 10
    image_keys: list[str] = field(default_factory=lambda: [])
    state_key: str = ""
    action_key: str = ""
    state_dim: int = 4
    action_dim: int = 2

@dataclass
class TrainConfig:
    steps: int = 100_000
    eval_freq: int = 10_000
    viz_freq: int = 10_000
    log_freq: int = 100
    save_checkpoint: bool = True
    save_freq: int = 5_000
    keep_freq: int = 50_000

    num_workers: int = 4
    batch_size: int = 64

    grad_clip_norm: float = 1.0
    use_amp: bool = False

@dataclass
class PolicyConfig:
    type: str = ""
    
@dataclass
class Config:
    seed: int = 42
    debug: bool = True
    device: str = "cuda"
    resume: bool = False
    checkpoint_path: str = ""
    wandb: WandBConfig = field(default_factory=WandBConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
