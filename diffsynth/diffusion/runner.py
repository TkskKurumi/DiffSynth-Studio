import os, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .lr_scheduler import get_scheduler
from diffsynth.core import OffloadTrainingManager


class LossMonitor:
    def __init__(self, accelerator, optimizer=None, show_loss=False, show_smooth_loss=False, dump_loss_file=None, smooth_steps=500):
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.show_loss = show_loss
        self.show_smooth_loss = show_smooth_loss
        self.smooth_steps = max(int(smooth_steps), 1)
        self.is_main = accelerator.is_main_process
        self.optimizer_step = 0
        self.accum_loss_sum = 0.0
        self.accum_count = 0
        self.ema_loss = 0.0
        self.ema_steps = 0
        self.dump_file = None
        if dump_loss_file is not None and self.is_main:
            dump_dir = os.path.dirname(os.path.abspath(dump_loss_file))
            os.makedirs(dump_dir, exist_ok=True)
            self.dump_file = open(dump_loss_file, "w", buffering=1)

    def close(self):
        if self.dump_file is not None:
            self.dump_file.close()
            self.dump_file = None

    def _local_batch_size(self, data):
        if isinstance(data, dict):
            for value in data.values():
                if torch.is_tensor(value):
                    return int(value.shape[0]) if value.ndim > 0 else 1
        if torch.is_tensor(data):
            return int(data.shape[0]) if data.ndim > 0 else 1
        if isinstance(data, (list, tuple)):
            return len(data)
        return 1

    def _global_loss(self, loss, data):
        if loss is None:
            return None
        loss = loss.detach().float()
        if loss.device.type != self.accelerator.device.type:
            loss = loss.to(self.accelerator.device)
        batch_size = torch.tensor(float(self._local_batch_size(data)), device=loss.device)
        if self.accelerator.num_processes > 1:
            gathered_loss = self.accelerator.gather(loss)
            gathered_batch_size = self.accelerator.gather(batch_size)
            loss = (gathered_loss * gathered_batch_size).sum() / gathered_batch_size.sum().clamp_min(1.0)
        return float(loss.detach().cpu())

    def _update_smooth_loss(self, loss):
        beta = 1.0 - 1.0 / self.smooth_steps
        self.ema_loss = beta * self.ema_loss + (1.0 - beta) * loss
        self.ema_steps += 1
        bias_correction = 1.0 - beta ** self.ema_steps
        if bias_correction <= 0:
            bias_correction = 1.0
        return self.ema_loss / bias_correction

    def _get_current_lr(self):
        if self.optimizer is None:
            return None
        if len(self.optimizer.param_groups) == 0:
            return None
        return self.optimizer.param_groups[0]["lr"]

    def update(self, loss, data, progress_bar=None):
        # All processes must execute _global_loss() because gather() is a DDP sync point.
        global_loss = self._global_loss(loss, data)
        if global_loss is None:
            return

        self.accum_loss_sum += global_loss
        self.accum_count += 1
        postfix = {}
        if self.show_loss:
            postfix["loss"] = f"{global_loss:.6g}"

        current_lr = self._get_current_lr()
        if current_lr is not None:
            postfix["lr"] = f"{current_lr:.6g}"

        sync_gradients = getattr(self.accelerator, "sync_gradients", True)
        if sync_gradients:
            effective_loss = self.accum_loss_sum / self.accum_count
            self.accum_loss_sum = 0.0
            self.accum_count = 0
            self.optimizer_step += 1
            if self.show_smooth_loss:
                postfix["smooth_loss"] = f"{self._update_smooth_loss(effective_loss):.6g}"
            if self.dump_file is not None:
                lr_str = f",{current_lr:.10g}" if current_lr is not None else ""
                self.dump_file.write(f"{self.optimizer_step},{effective_loss:.10g}{lr_str}\n")

        if postfix and progress_bar is not None and self.is_main:
            progress_bar.set_postfix(postfix)


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    optimizer_kwargs: str = None,
    lr_scheduler: str = "constant",
    scheduler_kwargs: str = None,
    show_loss: bool = False,
    show_smooth_loss: bool = False,
    show_lr: bool = False,
    dump_loss_file: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
        optimizer_kwargs = args.optimizer_kwargs
        lr_scheduler = args.lr_scheduler
        scheduler_kwargs = args.scheduler_kwargs
        show_loss = args.show_loss
        show_smooth_loss = args.show_smooth_loss
        show_lr = args.show_lr
        dump_loss_file = args.dump_loss_file

    optimizer_class = get_optimizer_class(customized_optimizer)
    
    # Build optimizer kwargs
    opt_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if optimizer_kwargs is not None:
        import ast
        opt_kwargs.update(ast.literal_eval(optimizer_kwargs))
    
    optimizer = optimizer_class(model.trainable_modules(), **opt_kwargs)
    
    # Build scheduler kwargs
    sched_kwargs = {}
    if scheduler_kwargs is not None:
        import ast
        sched_kwargs = ast.literal_eval(scheduler_kwargs)
    
    scheduler = get_scheduler(optimizer, scheduler_type=lr_scheduler, scheduler_kwargs=sched_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    loss_monitor = None
    if show_loss or show_smooth_loss or show_lr or dump_loss_file is not None:
        loss_monitor = LossMonitor(
            accelerator,
            optimizer=optimizer,
            show_loss=show_loss,
            show_smooth_loss=show_smooth_loss,
            dump_loss_file=dump_loss_file,
        )
    try:
        for epoch_id in range(num_epochs):
            progress_bar = tqdm(dataloader, dynamic_ncols=True)
            for data in progress_bar:
                with accelerator.accumulate(model):
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if loss_monitor is not None:
                        loss_monitor.update(loss, data, progress_bar=progress_bar)
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

        model_logger.on_training_end(accelerator, model, save_steps)
    finally:
        if loss_monitor is not None:
            loss_monitor.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
