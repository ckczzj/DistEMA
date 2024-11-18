# Introduction
This is a simple distributed EMA implementation to save GPU VRAM.

I flatten all the EMA parameters and distribute them equally across all ranks, with each rank only updating the EMA parameters assigned to it.

I now use a separate distributed CPU group to gather the EMA parameters from all devices, as using an NCCL group for this task would consume GPU VRAM.
Perhaps you can save them separately, similar to how optimizer states are handled in ZeRO-1 and ZeRO-2 of DeepSpeed.

# Usage

```python3
import torch
from torch.nn.parallel import DistributedDataParallel

from .dist_ema import DistEMA

torch.distributed.init_process_group(backend='nccl')
gloo_group = torch.distributed.new_group(backend="gloo")

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ["RANK"])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = build_model()
# maybe load pretrained model or resume from previous ckpt
# state_dict = torch.load(ckpt_path, map_location="cpu")
# model.load_state_dict(state_dict["model"])
model = DistributedDataParallel(model.cuda())

# EMA model must be built after the main model
# If your model is fp16 or bf16, I also recommend to use fp32 EMA model with a large decay.
dist_ema = DistEMA(
    model=model,
    world_size=world_size,
    rank=rank,
    dtype="fp32",
    decay=0.999,
    warmup=False,
)

# build optimizer and dataloader
optimizer = build_optimizer(model)
dataloader = build_dataloader()

# maybe resume from previous ckpt
# dist_ema.load_state_dict(state_dict["ema"]["model"])
# dist_ema.set_decay_steps(state_dict["ema"]["config"]["decay_steps"])

for batch in dataloader:
    loss = model(batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dist_ema.update(model)
    

    if save_ckpt:
        dist_ema_state_dict = dist_ema.state_dict(cpu_group=gloo_group)
        if rank == 0:
            torch.save({
                "model": model.state_dict(),
                "ema": {
                    "model": dist_ema_state_dict,
                    "config": dist_ema.config,
                },
            }, save_path)

```