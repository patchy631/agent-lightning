# Copyright (c) Microsoft. All rights reserved.

import asyncio
import importlib
import inspect
import logging
import multiprocessing
import signal
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, cast

from agentlightning.adapter import TraceTripletAdapter
from agentlightning.algorithm.base import BaseAlgorithm
from agentlightning.client import AgentLightningClient
from agentlightning.execution.base import ExecutionStrategy
from agentlightning.execution.shared_memory import SharedMemoryExecutionStrategy
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.runner import AgentRunner, AgentRunnerV2, BaseRunner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import Dataset, Hook, ParallelWorkerBase

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class Trainer(ParallelWorkerBase):
    """Orchestrates the distributed execution of agent rollouts.

    The Trainer is responsible for launching one or more worker processes
    that run the agent's execution loop. It manages multiprocessing,
    handles graceful shutdown, and serves as the main entry point for
    running a client-side agent fleet.

    Attributes:
        algorithm: An instance of `BaseAlgorithm` to use for training.
        store: An instance of `LightningStore` to use for storing tasks and traces.
        runner: An instance of `BaseRunner` to use for running the agent.
        dev: If True, rollouts are run against the dev endpoint provided in `fit`.
        n_runners: Number of agent runners to run in parallel.
        max_rollouts: Maximum number of rollouts to process per runner. If None,
                      workers run until no more rollouts are available.
        strategy: An instance of `ExecutionStrategy` to use for spawning the algorithm and runners.
        tracer: A tracer instance, or a string pointing to the class full name or a dictionary with a 'type' key
                that specifies the class full name and other initialization parameters.
                If None, a default `AgentOpsTracer` will be created with the current settings.
        adapter: An instance of `TraceTripletAdapter` to export data consumble by algorithms from traces.
        llm_proxy: An instance of `LLMProxy` to use for intercepting the LLM calls.
                   If not provided, algorithm will create one on its own.
        n_workers: Number of agent workers to run in parallel. Deprecated in favor of `n_runners`.
        max_tasks: Maximum number of tasks to process per runner. Deprecated in favor of `max_rollouts`.
        daemon: Whether worker processes should be daemons. Daemon processes
                are terminated automatically when the main process exits. Deprecated.
                Only have effect with `fit_v0`.
        triplet_exporter: An instance of `TraceTripletAdapter` to export triplets from traces,
                          or a dictionary with the initialization parameters for the exporter.
                          Deprecated. Use `adapter` instead.
    """

    def __init__(
        self,
        *,
        dev: bool = False,
        n_runners: Optional[int] = None,
        max_rollouts: Optional[int] = None,
        tracer: Union[BaseTracer, str, Dict[str, Any], None] = None,
        adapter: Union[TraceTripletAdapter, Dict[str, Any], None] = None,
        store: Union[LightningStore, str, Dict[str, Any], None] = None,
        runner: Union[
            BaseRunner[Any],
            type[BaseRunner[Any]],
            Callable[[], BaseRunner[Any]],
            str,
            Dict[str, Any],
            None,
        ] = None,
        strategy: Union[ExecutionStrategy, str, Dict[str, Any], None] = None,
        algorithm: Union[BaseAlgorithm, str, Dict[str, Any], None] = None,
        llm_proxy: Union[LLMProxy, Dict[str, Any], None] = None,
        n_workers: Optional[int] = None,
        max_tasks: Optional[int] = None,
        daemon: bool = True,
        triplet_exporter: Union[TraceTripletAdapter, Dict[str, Any], None] = None,
    ):
        super().__init__()
        self.dev = dev
        self.daemon = daemon
        self._client: AgentLightningClient | None = None  # Will be initialized in fit or fit_v0

        if n_workers is not None:
            warnings.warn(
                "`n_workers` is deprecated. Please use `n_runners`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if n_runners is None:
            n_runners = n_workers if n_workers is not None else 1
        else:
            if n_workers is not None and n_workers != n_runners:
                warnings.warn(
                    "`n_workers` is ignored when `n_runners` is provided.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self.n_runners = n_runners
        self.n_workers = n_runners  # Backwards compatibility for fit_v0

        if max_tasks is not None:
            warnings.warn(
                "`max_tasks` is deprecated. Please use `max_rollouts`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if max_rollouts is None:
            max_rollouts = max_tasks
        elif max_tasks is not None and max_tasks != max_rollouts:
            warnings.warn(
                "`max_tasks` is ignored when `max_rollouts` is provided.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.max_rollouts = max_rollouts
        self.max_tasks = max_tasks if max_tasks is not None else max_rollouts

        self._tracer_spec: Union[BaseTracer, str, Dict[str, Any], None] = tracer
        self._tracer_factory: Callable[[], BaseTracer] = self._build_tracer_factory(tracer)
        self.tracer = self._tracer_factory()

        if adapter is not None and triplet_exporter is not None:
            warnings.warn(
                "`triplet_exporter` is deprecated and ignored because `adapter` is provided.",
                DeprecationWarning,
                stacklevel=2,
            )

        adapter_spec = adapter if adapter is not None else triplet_exporter
        self.adapter = self._make_adapter(adapter_spec)
        self.triplet_exporter = self.adapter  # Backwards compatibility

        self.algorithm = self._make_algorithm(algorithm)

        # The active store for the current execution context
        self.store = self._make_store(store)
        self._runner_spec = runner

        self.strategy = self._make_strategy(strategy, n_runners=self.n_runners)
        if hasattr(self.strategy, "n_runners"):
            strategy_runners = getattr(self.strategy, "n_runners")
            if isinstance(strategy_runners, int) and strategy_runners > 0:
                self.n_runners = strategy_runners
                self.n_workers = strategy_runners

        self.llm_proxy = self._make_llm_proxy(llm_proxy, store=self.store)

        if not self.daemon:
            logger.warning(
                "daemon=False. Worker processes are non-daemonic. "
                "The worker processes will NOT be terminated when the main process exits. "
                "The cleanup must be handled manually."
            )

    def _make_tracer(self, tracer: Union[BaseTracer, str, Dict[str, Any], None]) -> BaseTracer:
        """Creates a tracer instance based on the provided configuration."""
        if isinstance(tracer, BaseTracer):
            return tracer
        if isinstance(tracer, str):
            module_name, class_name = tracer.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tracer_cls = getattr(module, class_name)
            return tracer_cls()
        if isinstance(tracer, dict):
            tracer_type = tracer.get("type")
            if tracer_type is None:
                raise ValueError("tracer dict must have a 'type' key with the class full name")
            module_name, class_name = tracer_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tracer_cls = getattr(module, class_name)
            # Remove 'type' key and pass remaining keys as kwargs
            tracer_kwargs = {k: v for k, v in tracer.items() if k != "type"}
            return tracer_cls(**tracer_kwargs)
        if tracer is None:
            return AgentOpsTracer(agentops_managed=True, instrument_managed=True, daemon=self.daemon)
        raise ValueError(f"Invalid tracer type: {type(tracer)}. Expected BaseTracer, str, dict, or None.")

    def _build_tracer_factory(self, tracer: Union[BaseTracer, str, Dict[str, Any], None]) -> Callable[[], BaseTracer]:
        if isinstance(tracer, BaseTracer):
            return lambda: tracer

        def _factory() -> BaseTracer:
            return self._make_tracer(tracer)

        return _factory

    def _make_algorithm(self, algorithm: Union[BaseAlgorithm, str, Dict[str, Any], None]) -> Optional[BaseAlgorithm]:
        """Creates an algorithm instance based on the provided configuration."""
        if isinstance(algorithm, BaseAlgorithm):
            return algorithm
        if isinstance(algorithm, str):
            module_name, class_name = algorithm.rsplit(".", 1)
            module = importlib.import_module(module_name)
            algorithm_cls = getattr(module, class_name)
            return algorithm_cls()
        if isinstance(algorithm, dict):
            algorithm_type = algorithm.get("type")
            if algorithm_type is None:
                raise ValueError("algorithm dict must have a 'type' key with the class full name")
            module_name, class_name = algorithm_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            algorithm_cls = getattr(module, class_name)
            # Remove 'type' key and pass remaining keys as kwargs
            algorithm_kwargs = {k: v for k, v in algorithm.items() if k != "type"}
            return algorithm_cls(**algorithm_kwargs)
        if algorithm is None:
            return None
        raise ValueError(f"Invalid algorithm type: {type(algorithm)}. Expected BaseAlgorithm, str, dict, or None.")

    def _make_adapter(self, adapter: Union[TraceTripletAdapter, Dict[str, Any], None]) -> TraceTripletAdapter:
        if isinstance(adapter, TraceTripletAdapter):
            return adapter
        if isinstance(adapter, dict):
            adapter_conf = dict(adapter)
            adapter_type = adapter_conf.pop("type", None)
            if adapter_type is None:
                return TraceTripletAdapter(**adapter_conf)
            adapter_cls = self._load_class(adapter_type)
            adapter_instance = self._instantiate_component(adapter_cls, adapter_conf)
            if not isinstance(adapter_instance, TraceTripletAdapter):
                raise TypeError(
                    f"Adapter '{adapter_type}' does not inherit from TraceTripletAdapter (got {type(adapter_instance)})."
                )
            return adapter_instance
        if adapter is None:
            return TraceTripletAdapter()
        raise ValueError(f"Invalid adapter type: {type(adapter)}. Expected TraceTripletAdapter, dict, or None.")

    def _make_store(self, store: Union[LightningStore, str, Dict[str, Any], None]) -> LightningStore:
        if isinstance(store, LightningStore):
            return store
        if isinstance(store, str):
            store_cls = self._load_class(store)
            store_instance = self._instantiate_component(store_cls)
        elif isinstance(store, dict):
            store_conf = dict(store)
            store_type = store_conf.pop("type", None)
            if store_type is None:
                raise ValueError("store dict must have a 'type' key with the class full name")
            store_cls = self._load_class(store_type)
            store_instance = self._instantiate_component(store_cls, store_conf)
        elif store is None:
            store_instance = InMemoryLightningStore()
        else:
            raise ValueError(f"Invalid store type: {type(store)}. Expected LightningStore, str, dict, or None.")

        if not isinstance(store_instance, LightningStore):
            raise TypeError(f"Store factory returned {type(store_instance)}, which is not a LightningStore subclass.")
        return store_instance

    def _make_strategy(
        self,
        strategy: Union[ExecutionStrategy, str, Dict[str, Any], None],
        *,
        n_runners: int,
    ) -> ExecutionStrategy:
        if isinstance(strategy, ExecutionStrategy):
            return strategy
        if isinstance(strategy, str):
            strategy_cls = self._load_class(strategy)
            strategy_instance = self._instantiate_component(
                strategy_cls,
                optional_defaults={"n_runners": lambda: n_runners},
            )
        elif isinstance(strategy, dict):
            strategy_conf = dict(strategy)
            strategy_type = strategy_conf.pop("type", None)
            if strategy_type is None:
                raise ValueError("strategy dict must have a 'type' key with the class full name")
            strategy_cls = self._load_class(strategy_type)
            strategy_instance = self._instantiate_component(
                strategy_cls,
                strategy_conf,
                {"n_runners": lambda: n_runners},
            )
        elif strategy is None:
            strategy_instance = SharedMemoryExecutionStrategy(n_runners=n_runners)
        else:
            raise ValueError(
                f"Invalid strategy type: {type(strategy)}. Expected ExecutionStrategy, str, dict, or None."
            )

        if not isinstance(strategy_instance, ExecutionStrategy):
            raise TypeError(
                f"Strategy factory returned {type(strategy_instance)}, which is not an ExecutionStrategy subclass."
            )
        return strategy_instance

    def _make_llm_proxy(
        self,
        llm_proxy: Union[LLMProxy, Dict[str, Any], str, None],
        *,
        store: LightningStore,
    ) -> Optional[LLMProxy]:
        if isinstance(llm_proxy, LLMProxy):
            return llm_proxy
        if isinstance(llm_proxy, dict):
            proxy_conf = dict(llm_proxy)
            proxy_type = proxy_conf.pop("type", None)
            if proxy_type is None:
                raise ValueError("llm_proxy dict must have a 'type' key with the class full name")
            proxy_cls = self._load_class(proxy_type)
            proxy_conf.setdefault("store", store)
            proxy_instance = self._instantiate_component(proxy_cls, proxy_conf)
            if not isinstance(proxy_instance, LLMProxy):
                raise TypeError(
                    f"llm_proxy factory returned {type(proxy_instance)}, which is not an LLMProxy subclass."
                )
            return proxy_instance
        if isinstance(llm_proxy, str):
            proxy_cls = self._load_class(llm_proxy)
            proxy_instance = self._instantiate_component(
                proxy_cls,
                optional_defaults={"store": lambda: store},
            )
            if not isinstance(proxy_instance, LLMProxy):
                raise TypeError(
                    f"llm_proxy factory returned {type(proxy_instance)}, which is not an LLMProxy subclass."
                )
            return proxy_instance
        if llm_proxy is None:
            return None
        raise ValueError(f"Invalid llm_proxy type: {type(llm_proxy)}. Expected LLMProxy, dict, str, or None.")

    def _create_runner_instance(self) -> BaseRunner[Any]:
        spec = self._runner_spec
        optional_defaults: Dict[str, Callable[[], Any]] = {"tracer": self._tracer_factory}
        if self.max_rollouts is not None:
            optional_defaults["max_rollouts"] = lambda: self.max_rollouts

        if spec is None:
            return AgentRunnerV2(
                tracer=self._tracer_factory(),
                max_rollouts=self.max_rollouts,
            )
        if isinstance(spec, BaseRunner):
            if self.n_runners > 1:
                logger.warning(
                    "A single runner instance was provided; it will be shared across %d workers.",
                    self.n_runners,
                )
            return cast(BaseRunner[Any], spec)
        if isinstance(spec, type) and issubclass(spec, BaseRunner):
            return self._instantiate_component(cast(type[BaseRunner[Any]], spec), optional_defaults=optional_defaults)
        if callable(spec) and not isinstance(spec, type):  # type: ignore
            runner_instance = spec()  # type: ignore
            if not isinstance(runner_instance, BaseRunner):  # type: ignore
                raise TypeError("Runner factory callable must return an instance of BaseRunner.")
            return runner_instance
        if isinstance(spec, str):
            runner_cls = self._load_class(spec)
            return self._instantiate_component(runner_cls, optional_defaults=optional_defaults)
        if isinstance(spec, dict):
            runner_conf = dict(spec)
            runner_type = runner_conf.pop("type", None)
            if runner_type is None:
                raise ValueError("runner dict must have a 'type' key with the class full name")
            runner_cls = self._load_class(runner_type)
            return self._instantiate_component(runner_cls, runner_conf, optional_defaults)
        raise ValueError(f"Invalid runner type: {type(spec)}. Expected BaseRunner, callable, str, dict, or None.")

    @staticmethod
    def _load_class(path: str) -> type[Any]:
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _instantiate_component(
        self,
        cls: type[Any],
        provided_kwargs: Optional[Dict[str, Any]] = None,
        optional_defaults: Optional[Dict[str, Callable[[], Any] | Any]] = None,
    ) -> Any:
        kwargs = dict(provided_kwargs or {})
        if optional_defaults:
            signature = inspect.signature(cls.__init__)
            for name, value in optional_defaults.items():
                if name in kwargs:
                    continue
                if name in signature.parameters:
                    kwargs[name] = value() if callable(value) else value
        return cls(**kwargs)

    def _normalize_hooks(self, hooks: Optional[Union[Hook, Sequence[Hook]]]) -> Sequence[Hook]:
        if hooks is None:
            return ()
        if isinstance(hooks, Hook):
            return (hooks,)
        return tuple(hooks)

    def fit(
        self,
        agent: LitAgent[T_co],
        *,
        train_dataset: Optional[Dataset[Any]] = None,
        validation_dataset: Optional[Dataset[Any]] = None,
        dev_dataset: Optional[Dataset[Any]] = None,
        hooks: Optional[Union[Hook, Sequence[Hook]]] = None,
    ) -> None:
        """Run the training loop using the configured strategy, store, and runner."""
        agent.set_trainer(self)
        hook_sequence = self._normalize_hooks(hooks)

        algorithm = self.algorithm
        if algorithm is not None:
            algorithm.set_trainer(self)
            algorithm.set_llm_proxy(self.llm_proxy)

        async def algorithm_bundle(store: LightningStore, event: Any) -> None:
            if self.llm_proxy is not None:
                self.llm_proxy.set_store(store)

            if algorithm is None:
                while not event.is_set():
                    await asyncio.sleep(0.1)
                return
            try:
                if inspect.iscoroutinefunction(algorithm.run):
                    await algorithm.run(
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        dev_dataset=dev_dataset,
                    )
                else:
                    # This will block the event loop to maximize the debugging experience
                    # It's the responsibility of the execution strategy to enable async execution
                    algorithm.run(
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        dev_dataset=dev_dataset,
                    )
            except Exception:
                logger.exception("Algorithm bundle encountered an error.")
                raise

        async def runner_bundle(store: LightningStore, worker_id: int, event: Any) -> None:
            runner_instance: BaseRunner[Any] | None = None
            runner_initialized = False
            worker_initialized = False
            try:
                runner_instance = self._create_runner_instance()
                runner_instance.init(agent=agent, hooks=hook_sequence)
                runner_initialized = True
                runner_instance.init_worker(worker_id, store)
                worker_initialized = True
                await runner_instance.iter(event=event)
            except Exception:
                logger.exception("Runner bundle encountered an error (worker_id=%s).", worker_id)
                raise
            finally:
                if runner_instance is not None:
                    if worker_initialized:
                        try:
                            runner_instance.teardown_worker(worker_id)
                        except Exception:
                            logger.exception("Error during runner worker teardown (worker_id=%s).", worker_id)
                    if runner_initialized:
                        try:
                            runner_instance.teardown()
                        except Exception:
                            logger.exception("Error during runner teardown (worker_id=%s).", worker_id)

        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    def _extract_client_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[AgentLightningClient]:
        """Extract client from data if it's a string URL or AgentLightningClient."""
        if isinstance(data, str):
            if not data.startswith("http://") and not data.startswith("https://"):
                raise ValueError("String data must be a valid URL starting with http:// or https://")
            return AgentLightningClient(endpoint=data)
        elif isinstance(data, AgentLightningClient):
            return data
        return None

    def _extract_dataset_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[Dataset[Any]]:
        """Extract dataset from data if it's a Dataset."""
        if isinstance(data, str) or isinstance(data, AgentLightningClient):
            return None
        return data

    def _determine_backend(
        self,
        train_data: Union[str, AgentLightningClient, Dataset[Any]],
        dev_data: Union[str, AgentLightningClient, Dataset[Any], None] = None,
    ) -> Union[str, AgentLightningClient]:
        """Determine which backend to use for initialization."""
        if self.dev:
            if dev_data is None:
                raise ValueError("dev_data must be provided when dev=True.")
            client = self._extract_client_from_data(dev_data)
            if client is None:
                raise ValueError("dev_data must be a string URL or AgentLightningClient when dev=True.")
            return client
        else:
            client = self._extract_client_from_data(train_data)
            if client is None and self.algorithm is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            elif client is None and self.algorithm is not None:
                # Algorithm will be responsible for creating the client
                client = self.algorithm.get_client()
                logger.info(f"Algorithm created client: {client}")
                return client
            if client is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            return client

    def init(self, backend: Union[str, AgentLightningClient]) -> None:
        logger.info(f"Initializing Trainer...")

        self._init_client(backend)

        self.tracer.init()

        logger.info(f"Trainer main initialization complete.")

    def teardown(self) -> None:
        logger.info(f"Cleaning up Trainer...")
        self.tracer.teardown()

        self._client = None
        logger.info(f"Trainer main cleanup complete.")

    def client(self) -> AgentLightningClient:
        """Returns the AgentLightningClient instance."""
        if self._client is None:
            raise RuntimeError("AgentLightningClient has not been initialized. Call `init` first.")
        return self._client

    def _init_client(self, backend: Union[str, AgentLightningClient]) -> AgentLightningClient:
        if self._client is None:
            if isinstance(backend, AgentLightningClient):
                logger.info("Using provided AgentLightningClient instance.")
                self._client = backend
            else:
                logger.info(f"Initializing AgentLightningClient with endpoint: {backend}")
                if not isinstance(backend, str):  # type: ignore
                    raise ValueError("backend must be a string URL or an AgentLightningClient instance.")
                if not backend.startswith("http://") and not backend.startswith("https://"):
                    raise ValueError("backend must be a valid URL starting with http:// or https://")
                # Initialize the client with the provided backend URL
                self._client = AgentLightningClient(endpoint=backend)
        else:
            logger.warning("AgentLightningClient already initialized. Returning existing instance.")
        return self._client

    def _worker_main_loop(self, agent: LitAgent[Any], worker_id: int, is_async: bool):
        """The main function for each worker process.

        This function initializes the client and the loop, then starts the
        execution. It also configures process-specific settings like the
        process title and signal handling.

        Args:
            agent: The `LitAgent` instance to run.
            worker_id: The unique ID for this worker.
            is_async: A boolean indicating if the async loop should be run.
        """
        if self.n_workers > 1:
            import setproctitle

            # Ignore Ctrl+C in worker processes; the main process handles it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            setproctitle.setproctitle(multiprocessing.current_process().name)

        # Now we are in child processes, so we can safely set up the environment.
        agent.set_trainer(self)
        # TODO: this should be set elsewhere
        if agent.trained_agents:
            self.triplet_exporter.agent_match = agent.trained_agents
        self._initialize_worker_env(worker_id)

        mode = "Async" if is_async else "Sync"
        logger.info(f"[Worker {worker_id}] {mode} worker process started.")

        num_processed = 0

        try:
            client = self.client()
            loop = AgentRunner(
                agent=agent,
                client=client,
                tracer=self.tracer,
                triplet_exporter=self.triplet_exporter,
                max_tasks=self.max_tasks,
                worker_id=worker_id,
            )
            loop.init_worker(worker_id)  # type: ignore
            if is_async:
                num_processed = asyncio.run(loop.iter_async())
            else:
                num_processed = loop.iter()
        except Exception:
            logger.exception(f"[Worker {worker_id}] Unhandled exception in worker loop.")
        finally:
            self._teardown_worker_env(worker_id)

        return num_processed

    def _initialize_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up trainer environment...")  # worker_id included in process name
        self.tracer.init_worker(worker_id)

    def _teardown_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Cleaning up trainer environment...")
        self.tracer.teardown_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Environment cleanup complete.")

    @staticmethod
    def kill_orphaned_processes() -> None:
        """
        Kill any orphaned processes that may have been left behind by previous runs.
        This is useful for cleaning up after crashes or unexpected exits.
        """
        import psutil

        for proc in psutil.process_iter():  # type: ignore
            # check whether the process name matches
            if proc.name().startswith("AgentLightning-"):
                proc.kill()

    def _terminate_processes(self, processes: List[multiprocessing.Process]) -> None:
        if self.n_workers > 1 and len(processes) > 0:
            for i, p in enumerate(processes):
                if p.is_alive():
                    logger.info(f"Terminating worker {i} (name: {p.name}, PID: {p.pid})...")
                    p.terminate()
                else:
                    logger.info(f"Worker {i} (name: {p.name}, PID: {p.pid}) is not alive or has already terminated.")
            for i, p in enumerate(processes):
                if p.is_alive():
                    p.join(timeout=10)  # Give some time to terminate
                if p.is_alive():  # If still alive, kill
                    logger.warning(
                        f"Worker {i} (name: {p.name}, PID: {p.pid}) did not terminate gracefully, killing..."
                    )
                    p.kill()
                    p.join(timeout=10)  # Ensure it's reaped

    def fit_v0(
        self,
        agent: LitAgent[T_co],
        train_data: Union[str, AgentLightningClient, Dataset[T_co]],
        *,
        val_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_backend: Union[str, AgentLightningClient, None] = None,
    ):
        """Train the agent using the provided data.

        Each data argument can be a string URL connecting to a agent-lightning server,
        or an AgentLightningClient instance connecting to a server (or mock server), or a dataset.
        If no algorithm is provided when instantiating the trainer, the data must be
        provided to connecting a server. Otherwise, dataset is also allowed and will be
        passed to the algorithm.

        If the algorithm is instantiated and there is no URL/client provided,
        the algorithm will be responsible for creating a client that will connect to itself.
        It can also create a mock client if the algorithm does not require a server.
        """

        if dev_backend is not None:
            warnings.warn("dev_backend is deprecated. Use dev_data instead.")
            if dev_data is not None:
                raise ValueError("dev_data and dev_backend cannot be provided at the same time.")
            dev_data = dev_backend

        # Extract datasets for algorithm if available
        train_dataset = self._extract_dataset_from_data(train_data)
        val_dataset = self._extract_dataset_from_data(val_data) if val_data else None
        dev_dataset = self._extract_dataset_from_data(dev_data) if dev_data else None

        # Initialize the algorithm with trainer if provided
        if self.algorithm is not None:
            self.algorithm.set_trainer(self)
            # DO NOT RUN TRAINING HERE. Need to spawn the worker first.

        # Determine the backend to use for client-server mode
        backend = self._determine_backend(train_data, dev_data)

        if self.dev:
            logger.warning(f"Running in dev mode. Using dev backend: {backend}")
        else:
            logger.debug(f"Running in non-dev mode. Using backend: {backend}")

        self.init(backend)

        processes: List[multiprocessing.Process] = []

        # Determine if the agent is asynchronous

        mode = "asynchronous" if agent.is_async else "synchronous"

        try:
            if self.n_workers == 1:
                logger.info(f"Running with n_workers=1 ({mode} in main process).")

                # Warn if algorithm is set with single worker mode
                if self.algorithm is not None:
                    logger.warning(
                        "Algorithm is set but using single worker mode. Algorithm will never get the chance to run."
                    )
                    # Ideally the single worker should be run in a separate thread or process.

                num_tasks = self._worker_main_loop(agent, 0, agent.is_async)
                logger.info(f"Single worker mode finished. Tasks processed: {num_tasks}")

                # If algorithm is provided and we have datasets, run algorithm after worker completes
                if self.algorithm is not None and train_dataset is not None:
                    logger.info("Running algorithm training after worker completion.")
                    self.algorithm.run(
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset,
                        dev_dataset=dev_dataset,
                    )
            else:
                logger.info(f"Running with n_workers={self.n_workers} ({mode} multiprocessing).")
                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,
                        args=(agent, i, agent.is_async),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    logger.info(f"Starting worker process {i} (name: {process_name})...")
                    p.start()

                if self.daemon:
                    # If algorithm is provided and we have datasets, pass them to the algorithm
                    if self.algorithm is not None:
                        logger.info("All workers have been spawned. Running algorithm training with provided datasets.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            validation_dataset=val_dataset,
                            dev_dataset=dev_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

                    for i, p in enumerate(processes):
                        p.join()  # Wait for the process to complete
                        logger.info(
                            f"Worker process {i} (name: {p.name}, PID: {p.pid}) joined with exit code {p.exitcode}."
                        )
                        if p.exitcode != 0:
                            logger.warning(
                                f"Worker process {i} (name: {p.name}, PID: {p.pid}) exited with non-zero code: {p.exitcode}."
                            )

                    logger.info(f"All {self.n_workers} worker processes have completed.")
                else:
                    logger.info("All worker processes started. Main process will not wait.")

                    # A hack to stop the main process from waiting for child processes to finish.
                    time.sleep(1)  # Give workers time to start
                    import multiprocessing.process as multiprocessing_process

                    multiprocessing_process._children.clear()  # type: ignore

                    if self.algorithm is not None:
                        logger.info("Main process continues to run algorithm.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            validation_dataset=val_dataset,
                            dev_dataset=dev_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Killing the workers.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        except Exception:
            logger.exception(f"Unhandled exception in fit method.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        finally:
            if self.daemon:
                self.teardown()
            else:
                logger.info("Main process exiting. Please use Trainer.kill_orphaned_processes() for cleanup.")
