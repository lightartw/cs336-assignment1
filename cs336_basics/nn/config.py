from pydantic import BaseModel
from typing import Optional

class ModelConfig(BaseModel):
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_head: int
    d_ff: int
    theta: float
    device: str = "cpu"
    dtype: str = "float32"

class OptimizerConfig(BaseModel):
    learning_rate: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    min_lr: float # schedule 里的 amin

class TrainingConfig(BaseModel):
    batch_size: int
    max_iters: int
    warmup_iters: int     # schedule 里的 tw
    cosine_cycle_iters: int # schedule 里的 tc
    max_l2_norm: float    # gradient_clipping 需要的阈值

    eval_interval: int
    log_interval: int
    out_dir: str
    train_data_path: str
    val_data_path: Optional[str] = None # 默认不开启validate
    wandb_project: str

class Config(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read()) 