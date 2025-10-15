import asyncio


async def a():
    print("a")
    b()
    print("finish")


def b():
    print("b")
    loop = asyncio.get_running_loop()
    fut = asyncio.run_coroutine_threadsafe(c(), loop)
    fut.result(timeout=5.0)


async def c():
    print("c")
    await asyncio.sleep(0.1)


asyncio.run(a())
