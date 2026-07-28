import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def initialize_dummy_weights(model: nn.Module, seed: int = 1234):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.dtype == torch.float4_e2m1fn_x2:
            param.data.view(torch.uint8).zero_()
        elif name.endswith("weight_scale_inv") or name.endswith("weight_global_scale"):
            param.data.fill_(1)
        elif param.dtype == torch.float8_e4m3fn:
            param.data.copy_(
                torch.randn(
                    param.shape,
                    device=param.device,
                    dtype=torch.bfloat16,
                ).to(param.dtype)
            )
        elif param.ndim == 1 and name.endswith(".weight"):
            param.data.fill_(1)
        else:
            param.data.normal_(
                0,
                1 / max(1, param.shape[-1]) ** 0.5,
            )
