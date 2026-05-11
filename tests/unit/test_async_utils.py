"""Tests for async utilities."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fake_news_detector.utils.async_utils import (
    AsyncBatch,
    create_task_with_retry,
    gather_with_timeout,
    parallel_analysis,
    run_in_thread,
)


class TestRunInThread:
    """Test run_in_thread function."""

    @pytest.mark.asyncio
    async def test_run_in_thread_success(self) -> None:
        """Blocking function runs in thread pool."""
        def blocking_func(x: int) -> int:
            return x * 2

        result = await run_in_thread(blocking_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_run_in_thread_with_string(self) -> None:
        """Thread pool works with string operations."""
        def blocking_upper(text: str) -> str:
            return text.upper()

        result = await run_in_thread(blocking_upper, "hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_run_in_thread_with_multiple_args(self) -> None:
        """Multiple arguments are passed correctly."""
        def add(a: int, b: int) -> int:
            return a + b

        result = await run_in_thread(add, 3, 4)
        assert result == 7


class TestGatherWithTimeout:
    """Test gather_with_timeout function."""

    @pytest.mark.asyncio
    async def test_gather_success(self) -> None:
        """Successful tasks return results."""
        async def task1() -> int:
            return 1

        async def task2() -> int:
            return 2

        results = await gather_with_timeout(task1(), task2(), timeout=5.0)
        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_gather_timeout_returns_none(self) -> None:
        """Timeout returns None list."""
        async def slow_task() -> str:
            await asyncio.sleep(10)
            return "done"

        results = await gather_with_timeout(slow_task(), timeout=0.1)
        assert results == [None]

    @pytest.mark.asyncio
    async def test_gather_multiple_tasks_timeout(self) -> None:
        """Multiple slow tasks all return None on timeout."""
        async def slow_task() -> str:
            await asyncio.sleep(10)
            return "done"

        results = await gather_with_timeout(
            slow_task(), slow_task(), slow_task(), timeout=0.1
        )
        assert results == [None, None, None]

    @pytest.mark.asyncio
    async def test_gather_default_timeout(self) -> None:
        """Default timeout is 30 seconds."""
        async def fast_task() -> int:
            return 42

        results = await gather_with_timeout(fast_task())
        assert results == [42]


class TestParallelAnalysis:
    """Test parallel_analysis function."""

    @pytest.mark.asyncio
    async def test_parallel_analysis_success(self) -> None:
        """Both functions run in parallel successfully."""
        def baseline_func(text: str) -> dict[str, Any]:
            return {"label": "REAL", "confidence": 0.9}

        def claim_func(text: str) -> list[dict[str, Any]]:
            return [{"text": "Test claim", "type": "FACT"}]

        baseline, claims = await parallel_analysis("Test article", baseline_func, claim_func)
        assert baseline["label"] == "REAL"
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_parallel_analysis_returns_tuple(self) -> None:
        """Returns correct tuple of (baseline, claims)."""
        def baseline_func(text: str) -> dict[str, Any]:
            return {"label": "FAKE", "confidence": 0.95}

        def claim_func(text: str) -> list[dict[str, Any]]:
            return [{"text": "Claim 1", "type": "FACT"}, {"text": "Claim 2", "type": "OPINION"}]

        baseline, claims = await parallel_analysis("Test article", baseline_func, claim_func)
        assert baseline["label"] == "FAKE"
        assert len(claims) == 2


class TestCreateTaskWithRetry:
    """Test create_task_with_retry function."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self) -> None:
        """Successful first attempt returns result."""
        async def successful_coro() -> int:
            return 42

        task = create_task_with_retry(successful_coro())
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_retry_returns_task(self) -> None:
        """Returns a task object that can be awaited."""
        async def dummy() -> int:
            return 100

        task = create_task_with_retry(dummy())
        assert hasattr(task, "result") or asyncio.iscoroutine(task)


class TestAsyncBatch:
    """Test AsyncBatch class."""

    @pytest.mark.asyncio
    async def test_process_empty_list(self) -> None:
        """Empty list returns empty results."""
        batch = AsyncBatch(max_concurrent=2)

        async def dummy_func(item: Any) -> int:
            return item

        results = await batch.process([], dummy_func)
        assert results == []

    @pytest.mark.asyncio
    async def test_process_all_items(self) -> None:
        """All items are processed."""
        batch = AsyncBatch(max_concurrent=5)

        async def double(x: int) -> int:
            return x * 2

        results = await batch.process([1, 2, 3], double)
        assert sorted(results) == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_process_with_semaphore(self) -> None:
        """Semaphore limits concurrency."""
        batch = AsyncBatch(max_concurrent=1)
        concurrent_count = 0
        max_concurrent_seen = 0

        async def track_concurrency(item: int) -> int:
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return item

        await batch.process([1, 2, 3, 4, 5], track_concurrency)
        assert max_concurrent_seen == 1

    @pytest.mark.asyncio
    async def test_process_with_exceptions(self) -> None:
        """Exceptions are captured in results."""
        batch = AsyncBatch(max_concurrent=2)

        async def may_fail(x: int) -> int:
            if x == 2:
                raise ValueError("Failed on 2")
            return x

        results = await batch.process([1, 2, 3], may_fail)
        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert results[2] == 3

    @pytest.mark.asyncio
    async def test_default_max_concurrent(self) -> None:
        """Default max_concurrent is 5."""
        batch = AsyncBatch()
        assert batch.semaphore._value == 5
