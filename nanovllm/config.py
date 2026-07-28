import os
from dataclasses import dataclass
from transformers import PretrainedConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 4
    load_format: str = "auto"
    fp8_all_reduce: bool = True
    fp8_all_reduce_min_bytes: int = 3 * 1024 * 1024
    record_collective_stats: bool = False
    shm_name: str = ""
    hf_config: PretrainedConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.tensor_parallel_size == 4
        assert self.load_format in ["auto", "dummy"]
        self.hf_config = PretrainedConfig.from_pretrained(self.model)
        assert self.hf_config.model_type == "minimax_m2"
        assert self.hf_config.hidden_size == 3072
        assert self.hf_config.intermediate_size == 1536
        assert self.hf_config.head_dim == 128
        assert self.hf_config.num_attention_heads == 48
        assert self.hf_config.num_key_value_heads == 8
        assert self.hf_config.num_hidden_layers == 62
        assert self.hf_config.num_local_experts == 256
        assert self.hf_config.num_experts_per_tok == 8
        assert self.hf_config.rotary_dim == 64
        assert self.hf_config.vocab_size == 200064
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
