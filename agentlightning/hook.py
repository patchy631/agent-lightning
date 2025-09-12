from __future__ import annotations

from typing import TYPE_CHECKING

from .types import Task, Rollout

if TYPE_CHECKING:
    from .runner import AgentRunner
    from .tracer import BaseTracer


class Hook:
    """Base class for defining hooks in the agent runner's lifecycle."""

    def on_rollout_start(self, task: Task, runner: AgentRunner, tracer: BaseTracer) -> None:
        """Hook called immediately before a rollout begins.

        Args:
            task: The :class:`Task` object that will be processed.
            runner: The :class:`AgentRunner` managing the rollout.
            tracer: The tracer instance associated with the runner.

        Subclasses can override this method to implement custom logic such as
        logging, metric collection, or resource setup. By default, this is a
        no-op.
        """

    def on_rollout_end(self, task: Task, rollout: Rollout, runner: AgentRunner, tracer: BaseTracer) -> None:
        """Hook called after a rollout completes.

        Args:
            task: The :class:`Task` object that was processed.
            rollout: The resulting :class:`Rollout` object.
            runner: The :class:`AgentRunner` managing the rollout.
            tracer: The tracer instance associated with the runner.

        Subclasses can override this method for cleanup or additional
        logging. By default, this is a no-op.
        """
