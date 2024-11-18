from collections import OrderedDict

import torch

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class DistEMA(object):
    def __init__(
        self,
        model,
        world_size,
        rank,
        dtype="fp32",
        decay=0.999,
        warmup=False,
        power=None,
        min_decay=0.0,
        inv_gamma=1.0,
        update_after_step=0,
    ):
        self.model = model

        # distributed
        self.world_size = world_size
        self.rank = rank

        # dtype
        self.dtype = dtype
        self.torch_dtype = PRECISION_TO_TYPE[dtype]

        # decay
        self.decay = decay

        # warmup
        self.warmup = warmup
        self.power = power
        self.min_decay = min_decay
        self.inv_gamma = inv_gamma
        self.update_after_step = update_after_step
        self.decay_steps = 0

        self.build_ema_model()

    @torch.no_grad()
    def build_ema_model(self):
        num_params = sum([param.numel() for param in list(self.model.parameters())])

        num_params_per_rank = (num_params + self.world_size - 1) // self.world_size  # ceil
        start_index = self.rank * num_params_per_rank
        end_index = min(start_index + num_params_per_rank, num_params) - 1  # closed interval

        self.name_list = []
        self.index_list = []
        self.ema_param_list = []

        cur_param_start_index = 0
        for name, param in self.model.named_parameters():
            numel = param.numel()

            cur_param_end_index = cur_param_start_index + numel - 1  # closed interval

            max_start_index = max(start_index, cur_param_start_index)
            min_end_index = min(end_index, cur_param_end_index)
            if max_start_index <= min_end_index:
                self.name_list.append(name)
                index = (max_start_index - cur_param_start_index, min_end_index - cur_param_start_index)
                self.index_list.append(index)
                # Tensor.view will not create new tensor
                self.ema_param_list.append(param.view(-1)[index[0] : index[1] + 1].clone().detach().to(dtype=self.torch_dtype))

            cur_param_start_index += numel

    def set_decay_steps(self, decay_steps):
        self.decay_steps = decay_steps

    def get_decay(self):
        if self.warmup:
            step = max(0, self.decay_steps - self.update_after_step - 1)
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
            return max(self.min_decay, min(value, self.decay))
        else:
            return self.decay

    @torch.no_grad()
    def update(self, model, decay=None):
        """
        Step the EMA model towards the current model.
        """
        if decay is None:
            decay = self.get_decay()

        state_dict = OrderedDict(model.named_parameters())
        for ema_param, name, index in zip(self.ema_param_list, self.name_list, self.index_list):
            param = state_dict[name].view(-1)
            ema_param.mul_(decay).add_(param[index[0] : index[1] + 1], alpha=1.0 - decay)

        self.decay_steps += 1

    # use cpu group to save VRAM
    def state_dict(self, cpu_group):
        gather_data = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gather_data, torch.cat(self.ema_param_list, dim=0).cpu(), group=cpu_group)
        flat_params = torch.cat(gather_data, dim=0)
        # return flat_params
        state_dict = OrderedDict()
        offset = 0
        for name, param_shape in [[name, param.shape] for name, param in self.model.named_parameters()]:
            numel = param_shape.numel()
            state_dict[name] = flat_params[offset : offset + numel].view(param_shape)
            offset += numel
        return state_dict

    def load_state_dict(self, state_dict):
        for ema_param, name, index in zip(self.ema_param_list, self.name_list, self.index_list):
            load_ema_param = state_dict[name].to(device=ema_param.device, dtype=self.torch_dtype)
            ema_param.copy_(load_ema_param[index[0] : index[1] + 1])
 
    @property
    def config(self):
        return {
            "dtype": self.dtype,
            "decay": self.decay,
            "warmup": self.warmup,
            "power": self.power,
            "min_decay": self.min_decay,
            "inv_gamma": self.inv_gamma,
            "update_after_step": self.update_after_step,
            "decay_steps": self.decay_steps,
        }
