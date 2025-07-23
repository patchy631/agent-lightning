import json
import logging
from typing import Optional


def configure_logger(level: int = logging.INFO, name: str = "agentlightning") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()  # clear existing handlers

    # log to stdout
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging
    return logger


class LightningLogger:
    """Agent-lightning logger that supports tracing events and metrics throughout the training."""

    def log_event(self, event: str, data: dict):
        """
        Log an event with its associated data when something happens.
        """

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        """
        Log a metric with its value and an optional step.
        """

    def log_message(self, level: int, message: str):
        """
        Log a message at a specific logging level.
        """


class ConsoleLogger(LightningLogger):
    """A simple logger that logs messages to the console using Python's logging module."""

    def __init__(self, level: int = logging.INFO):
        self.logger = configure_logger(level, name="agentlightning.logging.console")
        self.default_level = level

    def log_event(self, event: str, data: dict):
        data_str = str(data)
        if len(data_str) > 512:
            data_str = f"{data_str[:512]}... (truncated)"
        message = f"Event: {event}, Data: {data_str}"
        self.log_message(self.default_level, message)

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        step_str = f" at step {step}" if step is not None else ""
        message = f"Metric: {metric} = {value}{step_str}"
        self.log_message(self.default_level, message)

    def log_message(self, level: int, message: str):
        if level >= self.default_level:
            self.logger.log(level, message)
        # else skip logging if below default level


class WandbLogger(LightningLogger):

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        *,
        flush_every_n_events: int = 1000,
    ):
        import wandb

        wandb.init(project=project, entity=entity, name=name, config=config)

        self.event_table: Optional[wandb.Table] = None
        self.flush_every_n_events = flush_every_n_events

    def log_event(self, event: str, data: dict):
        import wandb

        if self.event_table is None:
            self.event_table = wandb.Table(columns=["event", "data"])

        try:
            data_str = json.dumps(data)  # Ensure data is JSON serializable
        except (TypeError, ValueError):
            data_str = str(data)
        self.event_table.add_data(event, data_str)

        if len(self.event_table.data) % self.flush_every_n_events == 0:
            wandb.log({"events": self.event_table})

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        import wandb

        if step is not None:
            wandb.log({metric: value}, step=step)
        else:
            wandb.log({metric: value})

    def log_message(self, level: int, message: str):
        pass  # Wandb handles logging internally, so we don't need to implement this
