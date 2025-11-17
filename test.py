import asyncio
import time


# Simulate reading an image (async for I/O-bound task)
async def read_image(image_id):
    print("reading")
    await asyncio.sleep(3)  # Simulate network latency
    return f"Image-{image_id}"


# Simulate processing an image (CPU-bound task)
async def process_image(image):
    print("processing")
    await asyncio.sleep(1)  # Simulate processing time
    print(f"Processed {image}")


# Producer: Reads images asynchronously in parallel and puts them in the queue
async def producer(queue, num_images, max_read_tasks):
    semaphore = asyncio.Semaphore(max_read_tasks)

    async def read_and_enqueue(image_id):
        async with semaphore:  # Limit parallelism to `max_read_tasks`
            image = await read_image(image_id)
            await queue.put(image)

    read_tasks = [read_and_enqueue(image_id) for image_id in range(num_images)]
    await asyncio.gather(*read_tasks)

    # Signal that all tasks are done
    for _ in range(queue.maxsize):
        await queue.put(None)


# Consumer: Processes images from the queue
async def consumer(queue):
    while True:
        image = await queue.get()
        if image is None:  # Sentinel to stop the consumer
            break
        await process_image(image)


async def main():
    num_images = 10
    buffer_capacity = 4  # Maximum number of images in the queue at a time
    max_read_tasks = 4  # Maximum number of concurrent read tasks

    # Create a bounded queue
    queue = asyncio.Queue(maxsize=buffer_capacity)

    # Start measuring time
    start_time = time.time()

    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(producer(queue, num_images, max_read_tasks))
    consumer_task = asyncio.create_task(consumer(queue))

    await asyncio.gather(producer_task, consumer_task)

    # Measure total time
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
