import json
from typing import Dict, Tuple
import torch
from pathlib import Path
from pydantic import BaseModel, Field


class LatticeConfig(BaseModel):
    """Lattice system configuration"""

    size: Tuple[int, int, int] = (10, 10, 10)
    lattice_constant: float = 2.85  # Å
    atom_types: Dict[int, float] = Field(
        default_factory=lambda: {0: 0.75, 1: 0.20, 2: 0.05}  # Fe  # Cu  # Vacancy
    )
    periodic_boundary: bool = True
    T: float = 800.0  # Temperature (K)


class ModelConfig(BaseModel):
    """Neural network architecture configuration"""

    # Vacancy Embedding Network
    vacancy_neighbor_radius: float = 20.0
    vacancy_max_num_neighbors: int = 32
    vacancy_node_feature_dim: int = 6
    vacancy_hidden_dim1: int = 64
    vacancy_hidden_dim2: int = 128
    vacancy_embedding_dim: int = 64
    vacancy_attention_heads: int = 4
    vacancy_attention_dropout: float = 0.1
    vacancy_gnn_layers: int = 3
    periodic_boundary: bool = True

    # Actor-Critic Network
    state_dim: int = vacancy_embedding_dim
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128


class TrainingConfig(BaseModel):
    """Training hyperparameters"""

    # Basic training settings
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    save_freq: int = 10
    eval_freq: int = 10

    # PPO specific parameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    max_grad_norm: float = 0.5

    # Experience collection
    num_steps: int = 2048
    batch_size: int = 64

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class LoggingConfig(BaseModel):
    """Logging and visualization settings"""

    # Base directories
    run_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    log_dir: Path = Path("logs")

    # Plotting settings
    plot_every_n_steps: int = 1000
    save_video: bool = False
    video_fps: int = 30

    # Logging settings
    log_level: str = "INFO"
    wandb_project: str = "kmc-rl"
    use_wandb: bool = False

    def setup_dirs(self, timestamp: str):
        """Setup directory structure for a training run"""
        run_path = self.run_dir / timestamp
        for path in [
            run_path,
            run_path / "checkpoints",
            run_path / "plots",
            run_path / "logs",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return run_path


class RunnerConfig(BaseModel):
    """Configuration for the main runner, including distributed settings"""
    device: str = "cpu"
    comm_backend: str = "mpi4py"
    scheduler_type: str = "static_queue"
    model_type: str = "SGDNTC_Model"

    leader_finalize_mpi: bool = True
    leader_tick_interval: float = 1.0  # Leader主循环的时间间隔，单位为秒
    
    worker_idle_sleep_sec: float = 0.5  # Worker在没有任务可领取时的睡眠时间，单位为秒
    worker_claim_size: int = 1  # Worker每次领取的任务数量
    worker_join_timeout_sec: float = 50.0  # Leader等待worker退出的最大时长，<=0表示无限等待
    worker_join_poll_interval_sec: float = 1.0  # Leader轮询worker退出状态的间隔，单位为秒
    worker_dump_stacks_on_timeout: bool = True  # join超时时向存活worker发送SIGUSR1打印栈
    worker_pending_log_interval_sec: float = 30.0  # Leader定期打印未完成任务的日志间隔，单位为秒
    


class Config(BaseModel):
    """Main configuration class combining all sub-configs"""

    # Sub-configurations
    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)

    def to_device(self, tensor):
        """Helper method to move tensors to configured device"""
        return tensor.to(self.training.device)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary"""
        return cls.model_validate(config_dict)

    def save(self, path: Path):
        """Save config to file"""
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=4)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from file"""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


CONFIG = Config()
