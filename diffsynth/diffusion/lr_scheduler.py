import math
import torch
from typing import Tuple, List

class PiecewiseLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 pieces: List[Tuple[float, float]],
                 warmup_steps: int,
                 restart_step_first: int,
                 warmup_interp: str = "linear",
                 interp: str = "linear",
                 last_epoch: int = -1,
                 restart_step_multiplier: float = 1.0):
        
        # 1. 定义插值函数
        def interp_linear(st: float, ed: float, x: float) -> float:
            return st + (ed - st) * x
        
        def interp_cosine(st: float, ed: float, x: float) -> float:
            # x in [0, 1], weight 从 1.0 衰减到 0.0
            x_weight = (math.cos(x * math.pi) + 1.0) / 2.0
            return interp_linear(st, ed, 1-x_weight)
        
        def interp_exp(st: float, ed: float, x: float) -> float:
            # 指数插值要求 st 和 ed 严格大于 0
            if st <= 0 or ed <= 0:
                return interp_linear(st, ed, x)
            logs, loge = math.log(st), math.log(ed)
            return math.exp(interp_linear(logs, loge, x))

        interp_fn_map = {
            "linear": interp_linear,
            "cosine": interp_cosine,
            "exp": interp_exp
        }

        # 2. 参数校验
        if not pieces:
            raise ValueError("pieces 不能为空")
        if warmup_interp not in interp_fn_map:
            raise ValueError(f"不支持的 warmup_interp: {warmup_interp}")
        if interp not in interp_fn_map:
            raise ValueError(f"不支持的 interp: {interp}")

        # 按时间点 t (元组的第一个元素) 排序
        self.pieces = sorted(pieces, key=lambda x: x[0])
        self.warmup_steps = warmup_steps
        self.restart_step_first = restart_step_first
        self.restart_step_multiplier = restart_step_multiplier
        self.interp = interp
        self.warmup_interp = warmup_interp

        # 3. 定义学习率计算逻辑
        def lr_lambda(current_step: int) -> float:
            # --- 阶段 1: Warmup ---
            if current_step < self.warmup_steps:
                progress = current_step / self.warmup_steps
                # 修复：明确传入起点 0.0 和终点 1.0
                return interp_fn_map[self.warmup_interp](0.0, 1.0, progress)

            # --- 阶段 2: 周期重启 (Restart) ---
            current_step_after_warmup = current_step - self.warmup_steps
            
            if self.restart_step_multiplier == 1.0:
                periods_passed = current_step_after_warmup // self.restart_step_first
                period_st = self.warmup_steps + periods_passed * self.restart_step_first
                period_len = self.restart_step_first
                period_ed = period_st + period_len
            else:
                # 修复：正确初始化 period_st 和 period_len
                period_st = self.warmup_steps
                period_len = float(self.restart_step_first)
                period_ed = period_st + period_len
                
                # O(log N) 复杂度寻找当前 step 所在的周期
                while current_step >= period_ed:
                    period_st = period_ed
                    period_len *= self.restart_step_multiplier
                    period_ed = period_st + period_len
            
            # 计算当前周期内的相对进度 [0, 1]
            period_t = (current_step - period_st) / (period_ed - period_st)
            # 防止浮点数精度问题导致越界
            period_t = max(0.0, min(1.0, period_t))

            # --- 阶段 3: 分段插值 (Piecewise) ---
            # 边界情况：在第一个点之前或最后一个点之后
            if period_t <= self.pieces[0][0]:
                return self.pieces[0][1]
            if period_t >= self.pieces[-1][0]:
                return self.pieces[-1][1]

            # 修复：正确遍历相邻的区间段
            for i in range(len(self.pieces) - 1):
                t0, v0 = self.pieces[i]
                t1, v1 = self.pieces[i+1]
                
                if t0 <= period_t <= t1:
                    if t1 == t0:  # 防止除零错误
                        return v0
                    piece_t = (period_t - t0) / (t1 - t0)
                    return interp_fn_map[self.interp](v0, v1, piece_t)
            
            # 兜底返回 (理论上不会执行到这里)
            
            return self.pieces[-1][1]

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

        

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
    elif scheduler_type == "piece":
            return PiecewiseLR(
                optimizer,
                **scheduler_kwargs
            )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown lr scheduler type: {scheduler_type!r}. Supported: constant, cosine-warmup-restart.")
