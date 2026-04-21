"""Async utilities for parallel execution."""
import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

from fake_news_detector.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_in_thread(func: Callable[..., T], *args: Any) -> T:
    """Run blocking function in thread pool.

    Args:
        func: Blocking function to run
        *args: Arguments to pass to function

    Returns:
        Result of function
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


async def gather_with_timeout(
    *tasks: Coroutine[Any, Any, T],
    timeout: float = 30.0,
) -> list[T | None]:
    """Gather tasks with timeout.

    Args:
        *tasks: Coroutines to gather
        timeout: Timeout in seconds

    Returns:
        List of results or None if timeout
    """
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        return list(results)
    except asyncio.TimeoutError:
        logger.error(f"Gather timeout after {timeout}s")
        return [None] * len(tasks)


async def parallel_analysis(
    article: str,
    baseline_func: Callable[[str], dict[str, Any]],
    claim_func: Callable[[str], list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run baseline and claim extraction in parallel.

    Args:
        article: Input article
        baseline_func: Baseline prediction function
        claim_func: Claim extraction function

    Returns:
        Tuple of (baseline_result, claims)
    """
    baseline_task = asyncio.create_task(
        run_in_thread(baseline_func, article)
    )
    claim_task = asyncio.create_task(
        run_in_thread(claim_func, article)
    )

    baseline_result, claims = await gather_with_timeout(
        baseline_task, claim_task, timeout=60.0
    )

    if baseline_result is None:
        baseline_result = {"label": "UNKNOWN", "confidence": 0.0}
    if claims is None:
        claims = []

    return baseline_result, claims


def create_task_with_retry(
    coro: Coroutine[Any, Any, T],
    max_retries: int = 3,
    delay: float = 1.0,
) -> asyncio.Task[T]:
    """Create task with automatic retry.

    Args:
        coro: Coroutine to execute
        max_retries: Maximum retry attempts
        delay: Delay between retries

    Returns:
        Task object
    """
    async def retry_wrapper() -> T:
        last_error = None
        for attempt in range(max_retries):
            try:
                return await coro
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (attempt + 1))
        raise last_error if last_error else Exception("All retries failed")

    return asyncio.create_task(retry_wrapper())


class AsyncBatch:
    """Process items in batches with concurrency limit."""

    def __init__(self, max_concurrent: int = 5) -> None:
        """Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process(
        self,
        items: list[Any],
        func: Callable[[Any], Coroutine[Any, Any, T]],
    ) -> list[T]:
        """Process items with concurrency limit.

        Args:
            items: Items to process
            func: Async function to apply

        Returns:
            List of results
        """
        async def limited_process(item: Any) -> T:
            async with self.semaphore:
                return await func(item)

        tasks = [limited_process(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)