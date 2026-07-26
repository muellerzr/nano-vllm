import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.layers.linear import FP8ColumnParallelLinear


def quantize(weight):
    blocks = weight.view(weight.size(0) // 128, 128, weight.size(1) // 128, 128)
    scale = blocks.abs().amax((1, 3)) / torch.finfo(torch.float8_e4m3fn).max
    weight = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return weight.view_as(blocks).reshape(blocks.size(0) * 128, blocks.size(2) * 128), scale


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    torch.set_default_device("cuda")
    layer = FP8ColumnParallelLinear(512, 512)
    dense_weight = torch.randn(512, 512, dtype=torch.bfloat16)
    weight, scale = quantize(dense_weight)
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.weight_scale_inv, scale)
    x = torch.randn(16, 512, dtype=torch.bfloat16)
    y = layer(x)
    reference = F.linear(x, dense_weight.chunk(dist.get_world_size(), 0)[dist.get_rank()])
    torch.cuda.synchronize()
    error = (y - reference).float().abs().mean() / reference.float().abs().mean()
    assert y.shape == (16, 512 // dist.get_world_size())
    assert torch.isfinite(y).all()
    assert error < 0.05
    print({"rank": dist.get_rank(), "shape": list(y.shape), "error": error.item()})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
