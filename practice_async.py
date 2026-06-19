import asyncio
import random

async def fake_request(i : int) -> str:
    delay = random.uniform(0.1,1.0)
    await asyncio.sleep(delay)
    return f"request-{i} done after {delay} seconds"

async def main():
    tasks  = []
    for i in range(10):
        task  = asyncio.create_task(fake_request(i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
    