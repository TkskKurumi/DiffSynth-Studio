import math
import torch
from typing import Tuple, List

# class CosineWarmupRestartLR(torch.optim.lr_scheduler.LambdaLR):
#     """
#     Learning rate scheduler with linear warmup + cosine annealing + periodic restarts.

#     During warmup (step <= warmup_steps):
#         lr = lr_max * step / warmup_steps

#     After warmup:
#         lr = lr_max * (cos(pi * ((step - warmup_steps) % restart_steps) / restart_steps) + 1) / 2
#     """

#     def __init__(self, optimizer, warmup_steps: int, restart_steps: int, edecay_steps: int, decay_min=0.5, last_epoch: int = -1):
#         self.warmup_steps = max(int(warmup_steps), 1)
#         self.restart_steps = max(int(restart_steps), 1)
#         self.edecay_steps = max(int(edecay_steps), 1)
#         self.decay_min = decay_min

#         def lr_lambda(current_step):
#             if current_step <= self.warmup_steps:
#                 return current_step / self.warmup_steps
#             else:
#                 tmp = current_step - self.warmup_steps
#                 edecay = math.exp(-tmp/self.edecay_steps)
#                 tmp = tmp % self.restart_steps
#                 tmp = tmp / self.restart_steps * math.pi
#                 tmp = (math.cos(tmp) + 1) / 2
#                 tmp = tmp*edecay
#                 return self.decay_min + tmp * (1-self.decay_min)

#         super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class WarmupCosineRestart(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, 
                 optimizer, 
                 warmup_steps: int, 
                 restart_step_first: int, 
                 restart_step_multiplier: float = 2.0, 
                 restart_annealing: float = 1.0,  # 修正为 1.0 保持 float 类型一致性
                 min_annealing: float = 0.1, 
                 last_epoch: int = -1, 
                 high_deadzone: float = 0.0, 
                 low_deadzone: float = 0.0):
        
        # 【防御性检查】：确保死区之和严格小于 1，避免除以零
        assert restart_step_multiplier >= 1.0, "restart_step_multiplier must be >= 1.0"
        assert high_deadzone >= 0.0 and low_deadzone >= 0.0, "Deadzones must be non-negative"
        assert (high_deadzone + low_deadzone) < 1.0, "high_deadzone + low_deadzone must be strictly less than 1.0"

        self.warmup_steps = warmup_steps
        self.restart_step_first = restart_step_first
        self.restart_step_multiplier = restart_step_multiplier
        self.restart_annealing = restart_annealing
        self.min_annealing = min_annealing
        self.high_deadzone = high_deadzone
        self.low_deadzone = low_deadzone

        def lr_lambda(current_step):
            # 1. Warmup 阶段：线性增加
            if current_step < self.warmup_steps:
                return current_step / self.warmup_steps
            
            # 2. 寻找当前所在的 Restart 周期
            mx = 1.0
            mn = self.min_annealing
            period_st = self.warmup_steps
            period_len = self.restart_step_first
            
            # 【性能优化】：如果 multiplier == 1，直接用除法算周期，避免 O(N) 的 while 循环
            if self.restart_step_multiplier == 1.0:
                periods_passed = (current_step - self.warmup_steps) // self.restart_step_first
                period_st = self.warmup_steps + periods_passed * self.restart_step_first
                period_ed = period_st + self.restart_step_first
                # 计算 mx 衰减：mx_new = mn + (mx_initial - mn) * (annealing ^ periods)
                mx = mn + (1.0 - mn) * (self.restart_annealing ** periods_passed)
            else:
                # 正常情况：O(log N) 复杂度寻找周期
                period_ed = period_st + period_len
                while current_step > period_ed:
                    period_st = period_ed
                    period_len *= self.restart_step_multiplier
                    period_ed = period_st + period_len
                    # 每次 restart，最大学习率向最小学习率衰减
                    mx = mn + (mx - mn) * self.restart_annealing

            # 3. 计算当前周期内的原始进度 (0 到 1)
            raw_progress = (current_step - period_st) / (period_ed - period_st)
            
            # 4. 应用 Deadzone 逻辑 (映射到 0 到 1)
            if raw_progress < self.high_deadzone:
                in_period_progress = 0.0
            elif raw_progress > (1.0 - self.low_deadzone):
                in_period_progress = 1.0
            else:
                # 线性映射到 [0, 1] 区间
                in_period_progress = (raw_progress - self.high_deadzone) / (1.0 - self.high_deadzone - self.low_deadzone)
            
            # 5. cosine1to0 从 1 平滑降到 0
            cosine1to0 = (math.cos(in_period_progress * math.pi) + 1.0) / 2.0
            
            # 6. 在 mx 和 mn 之间进行插值
            ret = mn + (mx - mn) * cosine1to0
            
            return ret

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(
    optimizer,
    scheduler_type: str = "constant",
    scheduler_kwargs: dict = None,
):
    """
    Factory function to create a learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        scheduler_type: One of "constant" or "cosine-warmup-restart".
        scheduler_kwargs: Dict of kwargs for the scheduler, e.g., {"warmup_steps": 0, "restart_steps": 1000, "edecay_steps": 10000}.

    Returns:
        A torch.optim.lr_scheduler instance.
    """
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    
    if scheduler_type == "cosine-warmup-restart":
        return WarmupCosineRestart(
            optimizer,
            **scheduler_kwargs
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown lr scheduler type: {scheduler_type!r}. Supported: constant, cosine-warmup-restart.")
