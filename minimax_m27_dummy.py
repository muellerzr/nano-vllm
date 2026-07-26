import json
import os
import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.models.minimax_m2 import MiniMaxM2ForCausalLM
from nanovllm.utils.loader import initialize_dummy_weights


def main():
    path = os.environ.get("MINIMAX_M27_CONFIG", "config.json")
    with open(path) as f:
        config = PretrainedConfig.from_dict(json.load(f))
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cuda")
    model = MiniMaxM2ForCausalLM(config)
    initialize_dummy_weights(model)
    parameters = sum(x.numel() for x in model.parameters())
    hidden_states = torch.zeros(1, config.hidden_size)
    output = model.model.layers[0].block_sparse_moe(hidden_states)
    torch.cuda.synchronize()
    print({"parameters_per_rank": parameters, "output": list(output.shape)})


if __name__ == "__main__":
    main()
