"""Utilities for running uvicorn servers in background threads or processes."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import uvicorn

logger = logging.getLogger(__name__)


@dataclass
class UvicornServerHandle:
    """Handle for a running uvicorn server."""

    server: Optional[uvicorn.Server]
    thread: Optional[threading.Thread] = None
    process: Optional[multiprocessing.Process] = None

    def is_running(self) -> bool:
        """Return True if the underlying server/process is alive."""

        if self.server is not None:
            return bool(self.server.started and not self.server.should_exit)
        if self.process is not None:
            return self.process.is_alive()
        return False

    def wait_until_started(self, timeout: float = 20.0, poll_interval: float = 0.01) -> bool:
        """Block until the server reports started or timeout occurs."""

        if self.server is None:
            raise RuntimeError("Cannot wait for start on a process-based server handle.")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.server.started:
                return True
            if self.server.should_exit:
                return False
            time.sleep(poll_interval)
        return bool(self.server.started)

    async def wait_until_started_async(self, timeout: float = 20.0, poll_interval: float = 0.01) -> bool:
        """Async variant of :meth:`wait_until_started`."""

        if self.server is None:
            raise RuntimeError("Cannot wait for start on a process-based server handle.")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.server.started:
                return True
            if self.server.should_exit:
                return False
            await asyncio.sleep(poll_interval)
        return bool(self.server.started)

    def stop(self, timeout: float = 10.0, force: bool = False) -> bool:
        """Attempt to stop the running server."""

        success = True
        if self.server is not None:
            if self.server.started and not self.server.should_exit:
                self.server.should_exit = True
            if self.thread is not None:
                self.thread.join(timeout=timeout)
                if self.thread.is_alive():
                    logger.error("uvicorn server thread is still alive after %.1f seconds", timeout)
                    success = False
                    if force:
                        logger.warning("Force flag has no effect for threads; manual intervention required.")
                self.thread = None
            self.server = None
        elif self.process is not None:
            if self.process.is_alive():
                if force:
                    logger.warning("Forcefully terminating uvicorn process.")
                    self.process.kill()
                else:
                    self.process.terminate()
                self.process.join(timeout=timeout)
                if self.process.is_alive():
                    logger.error("uvicorn server process is still alive after %.1f seconds", timeout)
                    success = False
            self.process = None
        else:
            success = False
        return success


def create_uvicorn_server(
    app: Any,
    host: str,
    port: int,
    *,
    log_level: str = "info",
    config_factory: Callable[..., uvicorn.Config] | None = None,
    **config_kwargs: Any,
) -> uvicorn.Server:
    """Create a uvicorn server for the given ASGI app."""

    factory = config_factory or uvicorn.Config
    config = factory(app, host=host, port=port, log_level=log_level, **config_kwargs)
    return uvicorn.Server(config)


def start_uvicorn_in_thread(
    server: uvicorn.Server,
    *,
    daemon: bool = True,
) -> UvicornServerHandle:
    """Start a uvicorn server inside a background thread."""

    def run() -> None:
        asyncio.run(server.serve())

    thread = threading.Thread(target=run, daemon=daemon)
    thread.start()
    return UvicornServerHandle(server=server, thread=thread)


def start_uvicorn_in_process(
    config: uvicorn.Config,
    *,
    daemon: bool = True,
) -> UvicornServerHandle:
    """Start a uvicorn server inside a separate process."""

    def target() -> None:
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    process = multiprocessing.Process(target=target, daemon=daemon)
    process.start()
    return UvicornServerHandle(server=None, process=process)
