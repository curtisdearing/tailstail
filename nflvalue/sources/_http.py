"""JSON-over-HTTP with explicit timeouts, bounded retry, and typed failure.

Standard library only, deliberately: this runs in CI and on a laptop, and a
transitive dependency on ``requests`` is not worth the surface area.

Retry policy is narrow on purpose. Timeouts, connection resets, 5xx, and 429
are transient and retried with exponential backoff plus deterministic jitter.
A 4xx is us asking wrong -- retrying a 401 eleven times just burns API credits
and delays the error. ``Retry-After`` is honoured when the server sends it,
because Discord and The Odds API both do.

Nothing here swallows: on exhaustion it raises ``SourceFetchError`` carrying
every attempt, so the failure report names the endpoint and each cause.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

from ..failures import Attempt, SourceRejected, SourceTimeout, SourceUnavailable

DEFAULT_TIMEOUT = 15.0
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF = 0.75      # seconds; doubles each retry
MAX_BACKOFF = 20.0
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


# --- credential redaction -------------------------------------------------
# A URL is not safe to put in an exception. Three things hide credentials in
# one: userinfo (``https://user:pass@host``), query parameters whose *name*
# marks the value secret, and webhook paths -- a Discord webhook URL is a
# bearer credential in its entirety, so its path segments are replaced rather
# than trusted to look harmless. Every URL that reaches a raised error, a log
# line, or a CI transcript goes through :func:`redact_url` first.

_SENSITIVE_PARAMS = frozenset({
    "apikey", "api_key", "key", "token", "access_token", "accesstoken",
    "auth", "auth_token", "authorization", "secret", "client_secret",
    "signature", "sig", "password", "passwd", "pwd", "swid", "espn_s2",
    "session", "sessionid", "sid", "code", "credential",
})
_SECRET_PATH_ROOTS = ("webhooks", "webhook", "hooks")
REDACTED = "<redacted>"


def redact_url(url: str) -> str:
    """Return ``url`` with anything credential-bearing replaced."""
    try:
        parts = urllib.parse.urlsplit(str(url))
    except ValueError:
        return REDACTED
    if not parts.scheme and not parts.netloc:
        return str(url)

    netloc = parts.netloc
    if "@" in netloc:                       # strip userinfo entirely
        netloc = REDACTED + "@" + netloc.rsplit("@", 1)[1]

    segments = parts.path.split("/")
    for index, segment in enumerate(segments):
        if segment.lower() in _SECRET_PATH_ROOTS:
            # everything after the hook root is the credential
            segments[index + 1:] = [REDACTED for _ in segments[index + 1:]]
            break
    path = "/".join(segments)

    query = parts.query
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        if pairs:
            query = urllib.parse.urlencode(
                [(k, REDACTED if k.lower() in _SENSITIVE_PARAMS else v) for k, v in pairs]
            )
    return urllib.parse.urlunsplit((parts.scheme, netloc, path, query, parts.fragment))


def _sleep_for(attempt_number: int, retry_after: Optional[float]) -> float:
    """Exponential backoff, with the server's ``Retry-After`` taking priority.

    Jitter is derived from the attempt number rather than drawn randomly, so a
    retry schedule is reproducible in a test and in a log.
    """
    if retry_after is not None:
        return min(float(retry_after), MAX_BACKOFF)
    base = DEFAULT_BACKOFF * (2 ** (attempt_number - 1))
    return min(base + 0.1 * (attempt_number % 3), MAX_BACKOFF)


def _retry_after_seconds(headers) -> Optional[float]:
    raw = headers.get("Retry-After") if headers else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None      # HTTP-date form; fall back to exponential backoff


def request_bytes(url: str, *, params: Optional[Dict] = None,
                  data: Optional[bytes] = None,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: float = DEFAULT_TIMEOUT,
                  attempts: int = DEFAULT_ATTEMPTS,
                  source: str = "http",
                  sleep: Callable[[float], None] = time.sleep) -> bytes:
    """Fetch ``url``, retrying transient failures. Raises on exhaustion."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    request_headers = {"User-Agent": "nfl-value/1.0"}
    request_headers.update(headers or {})

    log: List[Attempt] = []
    last_kind = SourceUnavailable
    for number in range(1, max(1, attempts) + 1):
        req = urllib.request.Request(url, data=data, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            retry_after = _retry_after_seconds(exc.headers)
            log.append(Attempt(number, f"HTTP {exc.code} {exc.reason}",
                               status=exc.code, retry_after=retry_after))
            if exc.code not in RETRY_STATUSES:
                # 401/403/404 will not fix themselves; fail now, keep the credit.
                raise SourceRejected(source, redact_url(url), log) from exc
            last_kind = SourceUnavailable
        except TimeoutError:
            log.append(Attempt(number, f"timeout after {timeout}s"))
            last_kind = SourceTimeout
            retry_after = None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            is_timeout = isinstance(reason, TimeoutError) or "timed out" in str(reason).lower()
            log.append(Attempt(number, f"{type(reason).__name__}: {reason}"))
            last_kind = SourceTimeout if is_timeout else SourceUnavailable
            retry_after = None
        except OSError as exc:
            log.append(Attempt(number, f"{type(exc).__name__}: {exc}"))
            last_kind = SourceUnavailable
            retry_after = None
        else:  # pragma: no cover - unreachable, kept for clarity
            retry_after = None

        if number < attempts:
            sleep(_sleep_for(number, log[-1].retry_after))

    raise last_kind(source, redact_url(url), log)


def get_json(url: str, params: Optional[Dict] = None,
             timeout: float = DEFAULT_TIMEOUT, *,
             attempts: int = DEFAULT_ATTEMPTS,
             headers: Optional[Dict[str, str]] = None,
             source: str = "http",
             sleep: Callable[[float], None] = time.sleep):
    """Fetch and parse JSON. A malformed body is a source failure, not a crash."""
    raw = request_bytes(url, params=params, timeout=timeout, attempts=attempts,
                        headers=headers, source=source, sleep=sleep)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceUnavailable(
            source, redact_url(url), [Attempt(1, f"unparseable body: {type(exc).__name__}: {exc}")],
            detail=f"first 120 bytes: {raw[:120]!r}") from exc
