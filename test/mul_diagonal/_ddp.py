import sys

sys.path.append("./")

import os
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from diagonal import diagonal_mul


def _create_model():
    class Model(torch.nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.linear_1 = torch.nn.Linear(10, 10)
            self.linear_2 = torch.nn.Linear(10, 1)

        def forward(self, x):
            x = F.relu(self.linear_1(x))
            x = diagonal_mul(x, 1.0)
            return F.sigmoid(self.linear_2(x))

    return Model()


def _setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _cleanup():
    dist.destroy_process_group()


def _run(fn, world_size):
    mp.spawn(fn=fn, args=(world_size,), nprocs=world_size, join=True)


def _ddp_training(rank, world_size):
    _setup(rank, world_size)
    model = _create_model().to(f"cuda:{rank}")
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters())
    x = torch.randn((10, 10), dtype=torch.float32, device=f"cuda:{rank}")
    one = torch.tensor(
        [0.5 for _ in range(10)], dtype=torch.float32, device=f"cuda:{rank}"
    ).unsqueeze(dim=-1)

    for _ in range(2):
        output = ddp_model(x)
        loss = loss_fn(one, output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    _cleanup()


if __name__ == "__main__":
    sleep(0.5)
    world_size = torch.cuda.device_count()
    if world_size < 2:
        exit(0)
    world_size = 2
    try:
        _run(_ddp_training, world_size)
    except BaseException as e:
        _cleanup()
        print(e)
        exit(1)
