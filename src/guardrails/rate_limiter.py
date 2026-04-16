"""Rate-limiter helper for the defense-in-depth pipeline."""
from collections import defaultdict, deque
from dataclasses import dataclass
import time
from typing import Deque, Dict, Optional, Tuple


@dataclass
class RateLimitDecision:
    """Represents the outcome of a rate-limit check for a single user."""

    allowed: bool
    retry_after: float
    remaining: int


class SlidingWindowRateLimiter:
    """Track per-user requests to enforce a sliding-window limit."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        """Initialize state so the guardrail can reject bursts while still allowing normal traffic."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: Dict[str, Deque[float]] = defaultdict(deque)

    def _cleanup(self, timestamps: Deque[float], now: float) -> None:
        """Drop timestamps outside the current window to keep memory bounded."""
        cutoff = now - self.window_seconds
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

    def check(self, user_id: Optional[str] = None) -> RateLimitDecision:
        """Return whether the next request for a user should be allowed and how long to wait if blocked."""
        user = user_id or "anonymous"
        now = time.time()
        window = self.user_windows[user]
        self._cleanup(window, now)

        if len(window) < self.max_requests:
            window.append(now)
            return RateLimitDecision(allowed=True, retry_after=0.0, remaining=self.max_requests - len(window))

        oldest = window[0]
        retry_after = (oldest + self.window_seconds) - now
        return RateLimitDecision(allowed=False, retry_after=max(retry_after, 0.0), remaining=0)

    def reset(self, user_id: Optional[str] = None) -> None:
        """Clear a user's history, useful for unit tests or when suspending enforcement."""
        user = user_id or "anonymous"
        self.user_windows.pop(user, None)
