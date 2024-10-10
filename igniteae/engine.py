from enum import Enum, auto
from typing import Any, Callable, Dict, NamedTuple, Type, Sequence

import torch
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler

import ignite.distributed as idist
from ignite.engine import DeterministicEngine, Engine
from ignite.metrics import Metric
from ignite.utils import convert_tensor


class Device(Enum):
    CPU = auto()
    GPU = auto()
    TPU = auto()
    MPS = auto()


class AMPMode(Enum):
    AMP = auto()
    APEX = auto()


class TrainingMode(Enum):
    NonDeterministic = auto()
    Deterministic = auto()


class StepType(NamedTuple):
    device: Device
    amp: tuple[AMPMode | None, GradScaler | None]


def parse_torch_device(device: str) -> Device:
    match device:
        case "cpu":
            return Device.CPU
        case "cuda":
            return Device.GPU
        case "xla":
            return Device.TPU
        case "mps":
            return Device.MPS
        case _:
            raise ValueError("panic!")


def parse_device(device: torch.device | str | int | None) -> Device:
    match device:
        case torch.device():
            return parse_torch_device(device.type)
        case str():
            return parse_torch_device(torch.device(device).type)
        case int():
            return Device.GPU
        case None:
            return Device.CPU
        case _:
            raise ValueError("panic!")


def parse_amp_mode(mode: str | None) -> AMPMode | None:
    match mode:
        case "amp":
            return AMPMode.AMP
        case "apex":
            return AMPMode.APEX
        case None:
            return None
        case _:
            raise ValueError("panic!")


def prepare_grad_scaler(scaler: bool | GradScaler) -> GradScaler | None:
    match scaler:
        case GradScaler():
            return scaler
        case True:
            return GradScaler(enabled=True)
        case False:
            return None
        case _:
            raise RuntimeError("panic!")


def self_supervised_training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    loss_fn: Callable[[nn.Module, Any, Any, Engine], Tensor] = lambda loss, model_output, batch, engine: loss(model_output, batch),
    output_transform: Callable[[Any, Any, Any], Any] = lambda model_output, batch, loss_output: loss_output.item(),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler | None = None,
):
    # Based on ignite's original supervised_training_step:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L44-L124
    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient_accumulation_steps must be strictly positive. " "No gradient accumulation if the value set to one (default).")

    def update(engine: Engine, batch: list[Tensor]) -> Any:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        model_output = model_fn(model, batch)
        model_output = model_transform(model_output)
        loss_output = loss_fn(loss, model_output, batch, engine)

        if gradient_accumulation_steps > 1:
            loss_output = loss_output / gradient_accumulation_steps
        loss_output.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()

        return output_transform(model_output, batch, loss_output * gradient_accumulation_steps)

    return update


def self_supervised_training_step_tpu(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    loss_fn: Callable[[nn.Module, Any, Any, Engine], Tensor] = lambda loss, model_output, batch, engine: loss(model_output, batch),
    output_transform: Callable[[Any, Any, Any], Any] = lambda model_output, batch, loss_output: loss_output.item(),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler | None = None,
) -> Callable:
    # Based on ignite's original supervised_training_step_tpu:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L310-L392
    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError:
        raise ModuleNotFoundError("torch_xla cannot be imported, please install PyTorch XLA.")

    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient_accumulation_steps must be strictly positive. " "No gradient accumulation if the value set to one (default).")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        model_output = model_fn(model, batch)
        model_output = model_transform(model_output)
        loss_output = loss_fn(loss, model_output, batch, engine)

        if gradient_accumulation_steps > 1:
            loss_output = loss_output / gradient_accumulation_steps
        loss_output.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            xm.optimizer_step(optimizer, barrier=True)

        return output_transform(model_output, batch, loss_output * gradient_accumulation_steps)

    return update


def self_supervised_training_step_apex(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    loss_fn: Callable[[nn.Module, Any, Any, Engine], Tensor] = lambda loss, model_output, batch, engine: loss(model_output, batch),
    output_transform: Callable[[Any, Any, Any], Any] = lambda model_output, batch, loss_output: loss_output.item(),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler | None = None,
) -> Callable:
    # Based on ignite's original supervised_training_step_apex:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L223-L307
    try:
        from apex import amp as apex_amp
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install apex from https://github.com/nvidia/apex to use amp_mode='apex'.")

    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient_accumulation_steps must be strictly positive. " "No gradient accumulation if the value set to one (default).")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        model_output = model_fn(model, batch)
        model_output = model_transform(model_output)
        loss_output = loss_fn(loss, model_output, batch, engine)

        if gradient_accumulation_steps > 1:
            loss_output = loss_output / gradient_accumulation_steps

        with apex_amp.scale_loss(loss_output, optimizer) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return output_transform(model_output, batch, loss_output * gradient_accumulation_steps)

    return update


def self_supervised_training_step_amp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    loss_fn: Callable[[nn.Module, Any, Any, Engine], Tensor] = lambda loss, model_output, batch, engine: loss(model_output, batch),
    output_transform: Callable[[Any, Any, Any], Any] = lambda model_output, batch, loss_output: loss_output.item(),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler | None = None,
) -> Callable:
    # Based on ignite's original supervised_training_step_amp:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L127-L220
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient_accumulation_steps must be strictly positive. " "No gradient accumulation if the value set to one (default).")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        model.train()
        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)

        with autocast(enabled=True):
            model_output = model_fn(model, batch)
            model_output = model_transform(model_output)
            loss_output = loss_fn(loss, model_output, batch, engine)
            if gradient_accumulation_steps > 1:
                loss_output = loss_output / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss_output).backward()
            if engine.state.iteration % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss_output.backward()
            if engine.state.iteration % gradient_accumulation_steps == 0:
                optimizer.step()

        return output_transform(model_output, batch, loss_output)

    return update


def training_step(step: StepType):
    match step:
        case StepType(Device.CPU | Device.GPU | Device.MPS, (None, None)):
            return self_supervised_training_step
        case StepType(Device.TPU, (None, None)) if idist.has_xla_support:
            return self_supervised_training_step_tpu
        case StepType(Device.GPU, (AMPMode.APEX, None)):
            return self_supervised_training_step_apex
        case StepType(Device.GPU, (AMPMode.AMP, scaler)):
            return self_supervised_training_step_amp
        case _:
            raise RuntimeError("panic!")


def select_trainer(mode: TrainingMode) -> Type[Engine] | Type[DeterministicEngine]:
    match mode:
        case TrainingMode.NonDeterministic:
            return Engine
        case TrainingMode.Deterministic:
            return DeterministicEngine
        case _:
            raise RuntimeError("panic!")


def create_self_supervised_trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    loss_fn: Callable[[nn.Module, Any, Any, Engine], Tensor] = lambda loss, model_output, batch, engine: loss(model_output, batch),
    output_transform: Callable[[Any, Any, Any], Any] = lambda model_output, batch, loss_output: loss_output.item(),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler | None = None,
    training_mode: TrainingMode = TrainingMode.NonDeterministic,
    amp_mode: AMPMode | None = None,
) -> Engine:
    return select_trainer(training_mode)(
        training_step(StepType(parse_device(device), (amp_mode, scaler)))(
            model,
            optimizer,
            loss,
            prepare_batch,
            model_fn,
            model_transform,
            loss_fn,
            output_transform,
            device,
            non_blocking,
            gradient_accumulation_steps,
            scaler,
        )
    )


def self_supervised_evaluation_step(
    model: torch.nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    output_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda model_output, batch: (model_output, batch),
    device: str | int | torch.device | None = None,
    non_blocking: bool = False,
) -> Callable:
    # Based on ignite's original supervised_evaluation_step:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L624-L679
    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            model_output = model_fn(model, batch)
            model_output = model_transform(model_output)
            return output_transform(model_output, batch)

    return evaluate_step


def self_supervised_evaluation_step_amp(
    model: torch.nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    output_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda model_output, batch: (model_output, batch),
    device: str | int | torch.device | None = None,
    non_blocking: bool = False,
) -> Callable:
    # Based on ignite's original supervised_evaluation_step_amp:
    # https://github.com/pytorch/ignite/blob/302c707f79dbef50a3920baf11b76eba5ee4200e/ignite/engine/__init__.py#L682-L742
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Any | tuple[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            with autocast(enabled=True):
                model_output = model_fn(model, batch)
                model_output = model_transform(model_output)
            return output_transform(model_output, batch)

    return evaluate_step


def evaluation_step(step: StepType):
    match step:
        case StepType(Device.CPU | Device.GPU | Device.TPU | Device.MPS, (None, None)):
            return self_supervised_evaluation_step
        case StepType(Device.GPU, (AMPMode.APEX, None)):
            return self_supervised_evaluation_step
        case StepType(Device.GPU, (AMPMode.AMP, None)):
            return self_supervised_evaluation_step_amp
        case _:
            raise RuntimeError("panic!")


def create_self_supervised_evaluator(
    model: torch.nn.Module,
    prepare_batch: Callable[[Any, str | int | torch.device | None, bool], Any] = lambda batch, device, non_blocking: convert_tensor(*batch, device, non_blocking),
    model_fn: Callable[[nn.Module, Any], Any] = lambda model, batch: model(batch),
    model_transform: Callable[[Any], Any] = lambda model_output: model_output,
    output_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda model_output, batch: (model_output, batch),
    device: torch.device | str | int | None = None,
    non_blocking: bool = False,
    amp_mode: AMPMode | None = None,
    metrics: Dict[str, Metric] | None = None,
) -> Engine:
    metrics = metrics or {}

    evaluator = Engine(
        evaluation_step(StepType(parse_device(device), (amp_mode, None)))(
            model,
            prepare_batch,
            model_fn,
            model_transform,
            output_transform,
            device,
            non_blocking,
        )
    )

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
