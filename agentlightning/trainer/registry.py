# Copyright (c) Microsoft. All rights reserved.

ExecutionStrategyRegistry = {
    "shm": "agentlightning.execution.shared_memory.SharedMemoryExecutionStrategy",
    # "ipc": "agentlightning.execution.inter_process.InterProcessExecutionStrategy",
    "cs": "agentlightning.execution.client_server.ClientServerExecutionStrategy",
}
