"""
Atomic file writes that survive Windows.

`os.replace` is atomic on POSIX and *usually* atomic on Windows. It fails with
PermissionError / WinError 5 when anything else holds a handle on the
destination for even a moment - antivirus scanning the file you just wrote,
the search indexer, or a cloud sync client (OneDrive, Dropbox) uploading it.

None of those are bugs in your code, and all of them are transient. The fix is
to retry with backoff rather than to abandon atomicity, because the reason we
write this way is that a half-written risk state file is worse than no file.

If your project lives under Documents, it may well be inside a OneDrive-synced
folder, which makes this dramatically more likely. Pointing the state directory
somewhere unsynced (`--state-dir C:\\trading-state`) removes the problem
entirely rather than working around it.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_RETRIES = 8
DEFAULT_DELAY = 0.02


class AtomicWriteError(OSError):
    """Raised when a file could not be replaced after repeated attempts."""


def _safe_unlink(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def atomic_write_text(
    path: Path,
    text: str,
    retries: int = DEFAULT_RETRIES,
    delay: float = DEFAULT_DELAY,
) -> None:
    """
    Write `text` to `path` atomically, retrying transient replace failures.

    Writes to a temp file in the same directory (so the replace is same-volume
    and therefore atomic), fsyncs it, then replaces the destination.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(retries):
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            _safe_unlink(tmp)
            raise

        try:
            os.replace(tmp, path)
            if attempt:
                log.info("Wrote %s after %d retries.", path.name, attempt)
            return
        except PermissionError as exc:
            # Windows: something else has the destination open. Back off.
            last_error = exc
            _safe_unlink(tmp)
            time.sleep(delay * (2**attempt))
        except OSError as exc:
            last_error = exc
            _safe_unlink(tmp)
            time.sleep(delay * (2**attempt))

    raise AtomicWriteError(
        f"Could not replace {path} after {retries} attempts: {last_error}. "
        f"Something is holding the file open - antivirus, the search indexer, "
        f"or a cloud sync client. If this directory is inside OneDrive, move the "
        f"state directory somewhere unsynced."
    )
