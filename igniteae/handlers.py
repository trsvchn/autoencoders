import os

from torch.utils.data import DataLoader

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger, WeightsHistHandler
from ignite.metrics import RunningAverage

from .engine import create_self_supervised_evaluator


def setup_evaluators(eval_model, evaluators):
    out_evaluators = {}
    for name_eval, config in evaluators.items():
        match config:
            case {
                "prepare_batch": prepare_batch,
                "model_fn": model_fn,
                "model_transform": model_transform,
                "output_transform": output_transform,
                "device": device,
                "non_blocking": non_blocking,
                "amp_mode": amp_mode,
                "metrics": metrics,
            }:
                out_evaluators[name_eval] = create_self_supervised_evaluator(
                    model=eval_model,
                    prepare_batch=prepare_batch,
                    model_fn=model_fn,
                    model_transform=model_transform,
                    output_transform=output_transform,
                    device=device,
                    non_blocking=non_blocking,
                    amp_mode=amp_mode,
                    metrics=metrics,
                )
    return out_evaluators


def attach_evaluators(trainer, evaluator_objects, data, **kwargs):
    match kwargs:
        case {"evaluators": evaluators}:
            for eval_name, eval_config in evaluators.items():
                eval_data = data[eval_config["data"]]
                evaluator = evaluator_objects[eval_name]
                trainer.add_event_handler(
                    Events.EPOCH_COMPLETED(every=eval_config["every_n_epochs"]),
                    (lambda evaluator=evaluator, eval_data=eval_data: lambda: evaluator.run(eval_data))(),
                    # NOTE: maybe this is better?
                    # (lambda evaluator: lambda: evaluator.run(eval_data))(evaluator),
                )


def setup_handlers(trainer, evaluators, **kwargs):
    match kwargs:
        case {"run_name": name}:
            trainer.add_event_handler(Events.STARTED, lambda: print(f"Training `{name}`..."))
            trainer.add_event_handler(Events.COMPLETED, lambda: print(f"Training `{name}`...DONE!"))


def prepare_print_trainer_logs(engine):
    def print_trainer_logs():
        line_kwargs = {"iter": engine.state.iteration}
        line = "[Running] - ITER {iter}"
        for k, v in engine.state.metrics.items():
            if k.endswith("_loss"):
                k = k.removeprefix("running_")
                line_kwargs[k] = v
                line += f" - {k}: {v:.4f}"
        print(line.format(**line_kwargs))

    return print_trainer_logs


def prepare_print_eval_logs(trainer, evaluator, name):
    def print_eval_logs():
        line_kwargs = {"name": name, "epoch": trainer.state.epoch}
        line = "[{name}] - EPOCH {epoch}"
        for k, v in evaluator.state.metrics.items():
            line_kwargs[k] = v
            line += f" - {k}: {v:.4f}"
        print(line.format(**line_kwargs))

    return print_eval_logs


def attach_print_logs(trainer, evaluators, **kwargs):
    match kwargs:
        case {"print_logs_every_n_iters": every}:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=every), prepare_print_trainer_logs(trainer))
            for name, evaluator in evaluators.items():
                evaluator.add_event_handler(Events.COMPLETED, prepare_print_eval_logs(trainer, evaluator, name))


def attach_running_losses(engine, **kwargs):
    match kwargs:
        case {"running_losses": losses} if losses:
            for running_metric in losses:
                RunningAverage(
                    output_transform=(lambda running_metric=running_metric: lambda x: x[running_metric])()
                    # output_transform=(lambda running_metric=running_metric: lambda x: x)()
                ).attach(engine, f"running_{running_metric}")


def setup_data(data, **kwargs):
    dataloaders = {}
    match kwargs:
        case {"trainer": trainer, "evaluators": evaluators}:
            dataloaders["trainer"] = DataLoader(
                data["train"],
                batch_size=trainer["batch_size"],
                shuffle=True,
                num_workers=trainer["num_workers"],
                pin_memory=trainer["pin_memory"],
            )
            for evaluator in evaluators.values():
                dataloaders[evaluator["data"]] = DataLoader(
                    data[evaluator["data"]],
                    batch_size=evaluator["batch_size"],
                    shuffle=False,
                    num_workers=evaluator["num_workers"],
                    pin_memory=evaluator["pin_memory"],
                )
            return dataloaders
        case {"trainer": trainer}:
            dataloaders["trainer"] = DataLoader(
                data["train"],
                batch_size=trainer["batch_size"],
                shuffle=True,
                num_workers=trainer["num_workers"],
                pin_memory=trainer["pin_memory"],
            )
            return dataloaders
        case _:
            raise RuntimeError("panic!")


def setup_model_checkpoints(trainer, evaluators, to_save, /, **kwargs):
    checkpoints = {}
    match kwargs:
        case {
            "run_name": run_name,
            "model_checkpoints": checkpoints_config,
        }:
            for checkpoint_name, checkpoint_config in checkpoints_config.items():
                model_ckpt = ModelCheckpoint(
                    filename_prefix="_".join(["best", run_name]),
                    global_step_transform=global_step_from_engine(trainer),
                    **checkpoint_config["config"],
                )
                evaluators[checkpoint_config["evaluator"]].add_event_handler(
                    Events.EPOCH_COMPLETED(every=checkpoint_config["save_every_n_epochs"]),
                    model_ckpt,
                    to_save,
                )
                checkpoints[checkpoint_name] = model_ckpt
        case _:
            raise RuntimeError("panic!")
    return checkpoints


def setup_early_stopping(trainer, evaluators, **kwargs):
    match kwargs:
        case {
            "early_stopping": early_stopping_config,
        } if early_stopping_config:
            evaluators[early_stopping_config["evaluator"]].add_event_handler(
                Events.COMPLETED,
                EarlyStopping(
                    trainer=trainer,
                    **early_stopping_config["config"],
                ),
            )
        # case _:
        #     raise RuntimeError("panic!")
        # Otherwise do nothing, for now.


def setup_tensorboard_logger(model, trainer, evaluators, optimizer, /, **kwargs):
    engines = evaluators | {"trainer": trainer}
    match kwargs:
        case {
            "log_dir": log_dir,
            "run_name": run_name,
            "tensorboard_logger": tensorboard_config,
        }:
            # TensorBoard Logger
            logger = TensorboardLogger(
                logdir=os.path.join(log_dir, run_name)
            )
            for handler, handler_config in tensorboard_config.items():
                engine = engines[handler_config["engine"]]
                match handler_config["handler"]:
                    case "output_handler":
                        if engine is trainer:
                            logger.attach_output_handler(engine=engine, **handler_config["config"])
                        else:
                            logger.attach_output_handler(engine=engine, global_step_transform=global_step_from_engine(trainer), **handler_config["config"])
                    case "opt_params_handler":
                        logger.attach_opt_params_handler(engine=engine, optimizer=optimizer, **handler_config["config"])
                    case "weights_hist_handler":
                        logger.attach(
                            engine=engine,
                            event_name=handler_config["config"]["event_name"],
                            log_handler=WeightsHistHandler(
                                model,
                                whitelist=handler_config["config"]["whitelist"],
                            ),
                        )
        case _:
            raise RuntimeError("panic!")
    return logger
