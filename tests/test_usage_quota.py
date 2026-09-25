"""Tests du quota quotidien d'analyses."""

from __future__ import annotations

import threading
from datetime import date

from usage_quota import DailyQuota


class FakeClock:
    """Horloge manipulable à la main."""

    def __init__(self, day: date) -> None:
        self.day = day

    def __call__(self) -> date:
        return self.day


def test_quota_allows_requests_up_to_the_limit():
    quota = DailyQuota(limit=2, today=FakeClock(date(2026, 9, 26)))

    assert [quota.try_consume() for _ in range(3)] == [True, True, False]


def test_quota_resets_on_a_new_day():
    clock = FakeClock(date(2026, 9, 26))
    quota = DailyQuota(limit=1, today=clock)
    assert quota.try_consume() is True
    assert quota.try_consume() is False

    clock.day = date(2026, 9, 27)

    assert quota.try_consume() is True


def test_zero_limit_means_unlimited():
    quota = DailyQuota(limit=0)

    assert all(quota.try_consume() for _ in range(1000))


def test_quota_is_thread_safe():
    quota = DailyQuota(limit=10, today=FakeClock(date(2026, 9, 26)))
    results: list[bool] = []
    results_lock = threading.Lock()

    def consume() -> None:
        allowed = quota.try_consume()
        with results_lock:
            results.append(allowed)

    threads = [threading.Thread(target=consume) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 10
