# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the XPU Qwen4Exp PLE tables: pinned slabs, file reads, row cache.

Every row of the n-gram table has one owner among the TP ranks. A rank holds its
rows either in pinned slabs (``PinnedRowSlabs``) or behind a bounded row cache
over the table's files (``RowCache``, ``RowReader``, ``RowServer``,
``RowLookahead``). This module only needs numpy and numba, so it says nothing
about devices; ``ple_table.py`` connects it to the model.
"""

from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import NamedTuple, cast

import numpy as np

logger = logging.getLogger(__name__)

PAGE = mmap.PAGESIZE
# A row of up to one page crosses at most one page boundary.
READ_SLOT = 2 * PAGE
# Retry backoff for a failed or short read, seconds (then raise).
RETRY_BACKOFF_S = (0.001, 0.010, 0.100)

# The caching host allocator rounds a pinned request up to a power of two, and
# XPU refuses large pinned allocations without a debug key: 16 GiB allocated on
# torch 2.15 at the time of writing, 17 did not.
MAX_SLAB_BYTES = 1 << 33
TAIL_BYTES = 1 << 20
MAX_SLABS = 8


class RowExtent(NamedTuple):
    """A run of consecutive table rows in one file."""

    path: str
    offset: int
    num_rows: int


def merge_extents(extents: Sequence[RowExtent], row_bytes: int) -> list[RowExtent]:
    """Join extents that are adjacent in one file."""
    merged: list[RowExtent] = []
    for extent in extents:
        if merged:
            last = merged[-1]
            if (
                last.path == extent.path
                and last.offset + last.num_rows * row_bytes == extent.offset
            ):
                merged[-1] = last._replace(num_rows=last.num_rows + extent.num_rows)
                continue
        merged.append(extent)
    return merged


def safetensors_tensor_location(path: str, name: str) -> tuple[int, list[int], str]:
    """(absolute byte offset, shape, dtype) of one tensor of a .safetensors file."""
    return _tensor_location(safetensors_header(path), name)


def safetensors_header(path: str) -> dict:
    """The header of a .safetensors file, with the data start under ``""``."""
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    header[""] = 8 + header_len
    return header


def _tensor_location(header: dict, name: str) -> tuple[int, list[int], str]:
    info = header[name]
    begin = int(info["data_offsets"][0])
    return header[""] + begin, [int(v) for v in info["shape"]], str(info["dtype"])


# --------------------------------------------------------------------------
# Compiled loops
# --------------------------------------------------------------------------


@cache
def _compiled():
    """Compile the loops that run without the GIL, once and on first use."""
    import numba

    libc = ctypes.CDLL(None, use_errno=True)
    pread = libc.pread
    # Addresses are integers here: numba computes them inside the loop.
    pread.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int64)
    pread.restype = ctypes.c_int64
    clock_gettime = libc.clock_gettime
    clock_gettime.argtypes = (ctypes.c_int, ctypes.c_size_t)
    clock_gettime.restype = ctypes.c_int
    clock_monotonic = 1

    @numba.njit(nogil=True)
    def pread_batch(fds, offsets, lengths, base, slot, lo, hi, results, latencies_ns):
        now = np.empty(2, dtype=np.int64)
        now_addr = now.ctypes.data
        for i in range(lo, hi):
            clock_gettime(clock_monotonic, now_addr)
            start = now[0] * 1_000_000_000 + now[1]
            results[i] = pread(fds[i], base + i * slot, lengths[i], offsets[i])
            clock_gettime(clock_monotonic, now_addr)
            latencies_ns[i] = now[0] * 1_000_000_000 + now[1] - start

    @numba.njit(nogil=True)
    def clock_alloc(n, hand, ref, epoch, min_epoch, step, out):
        capacity = ref.shape[0]
        got = 0
        scanned = 0
        limit = 2 * capacity + n
        while got < n and scanned < limit:
            slot = hand
            hand = hand + 1 if hand + 1 < capacity else 0
            scanned += 1
            if epoch[slot] >= min_epoch:
                continue
            if ref[slot]:
                ref[slot] = 0
                continue
            out[got] = slot
            epoch[slot] = step
            ref[slot] = 1
            got += 1
        return got, hand

    return pread_batch, clock_alloc


# --------------------------------------------------------------------------
# Whole ranges and pinned slabs
# --------------------------------------------------------------------------


def _open_for_reads(path: str, direct: bool) -> tuple[int, bool]:
    """Open with O_DIRECT where the filesystem takes it, else through the page cache."""
    if direct:
        try:
            return os.open(path, os.O_RDONLY | os.O_DIRECT), True
        except OSError:
            logger.warning("%s does not take O_DIRECT; reads use the page cache", path)
    return os.open(path, os.O_RDONLY), False


def read_file_range(
    path: str,
    byte_start: int,
    out: np.ndarray,
    *,
    direct: bool = True,
    threads: int = 4,
    chunk_bytes: int = 64 << 20,
    preadv: Callable = os.preadv,
) -> bool:
    """Fill ``out`` (C-contiguous uint8) with file bytes from ``byte_start``.

    Sequential page-aligned chunks go through one aligned bounce per worker; the
    workers take the chunks round-robin, so the reads in flight stay adjacent in
    the file. Returns whether O_DIRECT was used.
    """
    if out.dtype != np.uint8 or not out.flags.c_contiguous:
        raise ValueError("PLE table read: out must be C-contiguous uint8")
    total = int(out.nbytes)
    if total == 0:
        return direct
    flat = out.reshape(-1)
    chunk = max(PAGE, int(chunk_bytes) // PAGE * PAGE)
    begin = int(byte_start)
    end = begin + total
    starts = list(range(begin & ~(PAGE - 1), end, chunk))
    fd, direct = _open_for_reads(path, direct)
    size = os.fstat(fd).st_size
    if end > size:
        os.close(fd)
        raise ValueError(f"{path}: range [{begin}, {end}) is past the file ({size} B)")
    workers = max(1, min(int(threads), len(starts)))

    def read_chunk(view: memoryview, pos: int, need: int) -> None:
        length = -(-need // PAGE) * PAGE  # whole pages (O_DIRECT); EOF may cut it
        got = 0
        attempt = 0
        last_error = ""
        while got < need:
            try:
                count = preadv(fd, [view[got:length]], pos + got)
                if count > 0:
                    next_got = got + count
                    if direct and next_got < need:
                        next_got &= ~(PAGE - 1)  # O_DIRECT resumes on a page boundary
                    if next_got > got:
                        got = next_got
                        continue
                last_error = f"short read ({count} B, {got} < {need} B)"
            except OSError as exc:
                last_error = f"errno {exc.errno} ({exc.strerror})"
            if attempt >= len(RETRY_BACKOFF_S):
                raise RuntimeError(
                    f"PLE table read failed at {pos + got}: {last_error} ({path})"
                )
            time.sleep(RETRY_BACKOFF_S[attempt])
            attempt += 1

    def work(worker: int) -> None:
        bounce = mmap.mmap(-1, chunk)
        view = memoryview(bounce)
        source = np.frombuffer(bounce, dtype=np.uint8)
        for pos in starts[worker::workers]:
            lo = max(begin - pos, 0)
            hi = min(chunk, end - pos)
            read_chunk(view, pos, hi)
            flat[pos + lo - begin : pos + hi - begin] = source[lo:hi]

    try:
        if workers == 1:
            work(0)
        else:
            with ThreadPoolExecutor(workers, thread_name_prefix="ple-load") as pool:
                for future in [pool.submit(work, w) for w in range(workers)]:
                    future.result()
    finally:
        os.close(fd)
    return direct


def plan_slabs(
    rows: int,
    row_bytes: int,
    max_bytes: int = MAX_SLAB_BYTES,
    tail_bytes: int = TAIL_BYTES,
    max_slabs: int = MAX_SLABS,
) -> list[int]:
    """Rows per slab, such that the pinned footprint is about the table's own size.

    Each slab is the largest power of two of bytes up to ``max_bytes`` that holds
    whole rows, so the allocator's rounding adds less than a row. What remains
    below ``tail_bytes``, or at the last allowed slab, is one slab of its own.
    """
    if rows < 0 or row_bytes <= 0 or max_bytes < row_bytes or max_slabs < 1:
        raise ValueError(f"plan_slabs: {rows=} {row_bytes=} {max_bytes=} {max_slabs=}")
    out: list[int] = []
    left = int(rows)
    while left > 0:
        nbytes = left * row_bytes
        if nbytes <= tail_bytes or len(out) == max_slabs - 1:
            if nbytes > max_bytes:
                raise ValueError(
                    f"{rows} rows of {row_bytes} B do not fit {max_slabs} slabs "
                    f"of {max_bytes} B"
                )
            out.append(left)
            break
        take = min(left, (1 << (min(nbytes, max_bytes).bit_length() - 1)) // row_bytes)
        out.append(take)
        left -= take
    return out


class PinnedRowSlabs:
    """A rank's rows [start, end) of the table, resident in host memory, in slabs.

    ``alloc(rows, row_bytes)`` makes one slab as a uint8 array of that shape; the
    table passes pinned storage, tests pass plain numpy.
    """

    def __init__(
        self,
        start: int,
        end: int,
        row_bytes: int,
        *,
        alloc: Callable[[int, int], np.ndarray] | None = None,
        max_bytes: int = MAX_SLAB_BYTES,
        tail_bytes: int = TAIL_BYTES,
    ) -> None:
        if not 0 <= start <= end:
            raise ValueError(f"PLE slabs: bad row range [{start}, {end})")
        self.start = int(start)
        self.end = int(end)
        self.row_bytes = int(row_bytes)
        self.slab_rows = plan_slabs(end - start, row_bytes, max_bytes, tail_bytes)
        alloc = alloc or (lambda r, b: np.empty((r, b), dtype=np.uint8))
        self.slabs = [alloc(rows, self.row_bytes) for rows in self.slab_rows]
        # Local row starts of each slab, then the end.
        self.bounds = np.concatenate(
            [[0], np.cumsum(self.slab_rows, dtype=np.int64)]
        ).astype(np.int64)

    @property
    def num_rows(self) -> int:
        return self.end - self.start

    @property
    def nbytes(self) -> int:
        return self.num_rows * self.row_bytes

    def load(
        self,
        extents: Sequence[RowExtent],
        *,
        direct: bool = True,
        threads: int = 4,
        chunk_bytes: int = 64 << 20,
    ) -> tuple[float, bool]:
        """Read the rows from the table's files. Returns (seconds, used O_DIRECT)."""
        started = time.perf_counter()
        used_direct = True
        first_rows = np.cumsum([0] + [e.num_rows for e in extents], dtype=np.int64)
        if self.end > int(first_rows[-1]):
            raise ValueError(
                f"PLE table holds {int(first_rows[-1])} rows, rank needs {self.end}"
            )
        for index, slab in enumerate(self.slabs):
            slab_start = self.start + int(self.bounds[index])
            slab_end = self.start + int(self.bounds[index + 1])
            for extent, first in zip(extents, first_rows.tolist()):
                lo = max(slab_start, first)
                hi = min(slab_end, first + extent.num_rows)
                if lo >= hi:
                    continue
                used_direct &= read_file_range(
                    extent.path,
                    extent.offset + (lo - first) * self.row_bytes,
                    slab[lo - slab_start : hi - slab_start],
                    direct=direct,
                    threads=threads,
                    chunk_bytes=chunk_bytes,
                )
        return time.perf_counter() - started, used_direct

    def rows(self, global_rows: np.ndarray) -> np.ndarray:
        """uint8 [n, row_bytes] copies of owned rows, by global row id."""
        ids = np.asarray(global_rows, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < self.start or ids.max() >= self.end):
            raise IndexError(f"PLE rows outside [{self.start}, {self.end})")
        local = ids - self.start
        which = np.searchsorted(self.bounds, local, side="right") - 1
        out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        for slab in np.unique(which).tolist():
            selected = which == slab
            out[selected] = self.slabs[slab][local[selected] - self.bounds[slab]]
        return out

    def check_scales(self, scale_offset: int, step: int = 1 << 22) -> None:
        """Every row's float32 scale at ``scale_offset`` is finite and not negative."""
        for index, slab in enumerate(self.slabs):
            for start in range(0, slab.shape[0], step):
                part = slab[start : start + step]
                scales = np.ndarray(
                    (part.shape[0],),
                    dtype="<f4",
                    buffer=part,
                    offset=scale_offset,
                    strides=(self.row_bytes,),
                )
                bad = ~(np.isfinite(scales) & (scales >= 0))
                if bad.any():
                    row = self.start + int(self.bounds[index]) + start
                    row += int(np.nonzero(bad)[0][0])
                    raise ValueError(f"PLE row {row} has a bad scale")


# --------------------------------------------------------------------------
# Row reads
# --------------------------------------------------------------------------


class RowReader:
    """Read single table rows with O_DIRECT into an aligned bounce, in parallel.

    Row reads are small and random, so the page cache and its read-around do not
    help them: 4 KiB each, or 8 KiB when a row crosses a page. A batch splits over
    ``threads`` workers, each of which runs a compiled ``pread`` loop without the
    GIL. A failed or short read is retried three times, then raises; nothing is
    ever zero-filled. One reader is used by one thread at a time.
    """

    def __init__(
        self,
        extents: Sequence[RowExtent],
        row_bytes: int,
        *,
        threads: int = 16,
        max_batch_rows: int = 8192,
        direct: bool = True,
    ) -> None:
        if row_bytes > PAGE:
            raise ValueError(f"PLE row of {row_bytes} B is larger than a page")
        self.row_bytes = int(row_bytes)
        self.threads = max(1, int(threads))
        self.max_batch_rows = max(1, int(max_batch_rows))
        self.direct = direct
        self._fds: dict[str, int] = {}
        self._extent_fds = np.empty(len(extents), dtype=np.int32)
        self._extent_offsets = np.array([e.offset for e in extents], dtype=np.int64)
        self._first_rows = np.cumsum(
            [0] + [e.num_rows for e in extents], dtype=np.int64
        )
        self.num_rows = int(self._first_rows[-1])
        for index, extent in enumerate(extents):
            if extent.path not in self._fds:
                fd, self.direct = _open_for_reads(extent.path, self.direct)
                self._fds[extent.path] = fd
            fd = self._fds[extent.path]
            self._extent_fds[index] = fd
            end = extent.offset + extent.num_rows * self.row_bytes
            size = os.fstat(fd).st_size
            if end > size:
                self.close()
                raise ValueError(
                    f"{extent.path}: {extent.num_rows} rows from byte {extent.offset} "
                    f"end at {end}, past the file ({size} B)"
                )
        self._bounce = mmap.mmap(-1, self.max_batch_rows * READ_SLOT)
        self._bounce_view = memoryview(self._bounce)
        self._bounce_np = np.frombuffer(self._bounce, dtype=np.uint8).reshape(
            self.max_batch_rows, READ_SLOT
        )
        self._bounce_addr = self._bounce_np.ctypes.data
        self._results = np.zeros(self.max_batch_rows, dtype=np.int64)
        self._latencies_ns = np.zeros(self.max_batch_rows, dtype=np.int64)
        self._row_offsets = np.arange(self.row_bytes, dtype=np.int64)
        self._pool = (
            ThreadPoolExecutor(self.threads, thread_name_prefix="ple-read")
            if self.threads > 1
            else None
        )
        self._pread_batch: Callable[..., None] = _compiled()[0]
        # Numba compiles on the first call: have it here, not in a step.
        self._pread_batch(
            self._extent_fds,
            self._extent_offsets,
            self._extent_offsets,
            self._bounce_addr,
            READ_SLOT,
            0,
            0,
            self._results,
            self._latencies_ns,
        )
        self._preadv = os.preadv  # tests swap this to inject faults
        self.retries = 0

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._pool = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def _retry(self, slot: int, offset: int, length: int, need: int) -> None:
        view = self._bounce_view[slot * READ_SLOT : slot * READ_SLOT + length]
        last_error = ""
        for attempt in range(len(RETRY_BACKOFF_S) + 1):
            try:
                got = self._preadv(int(self._read_fds[slot]), [view], offset)
                if got >= need:
                    return
                last_error = f"short read {got} < {need} B"
            except OSError as exc:
                last_error = f"errno {exc.errno} ({exc.strerror})"
            if attempt < len(RETRY_BACKOFF_S):
                time.sleep(RETRY_BACKOFF_S[attempt])
        raise RuntimeError(
            f"PLE row read failed after {len(RETRY_BACKOFF_S)} retries at offset "
            f"{offset}, length {length}: {last_error}"
        )

    def read_rows(
        self,
        rows: np.ndarray,
        out: np.ndarray | None = None,
        latencies: list | None = None,
    ) -> np.ndarray:
        """Bytes of ``rows`` (global row ids) as uint8 [n, row_bytes]."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        count = rows.shape[0]
        if out is None:
            out = np.empty((count, self.row_bytes), dtype=np.uint8)
        if count == 0:
            return out
        if rows.min() < 0 or rows.max() >= self.num_rows:
            raise IndexError(
                f"PLE row out of range [0, {self.num_rows}): "
                f"{int(rows.min())}..{int(rows.max())}"
            )
        for lo in range(0, count, self.max_batch_rows):
            hi = min(count, lo + self.max_batch_rows)
            self._read_batch(rows[lo:hi], out[lo:hi], latencies)
        return out

    def _read_batch(
        self, rows: np.ndarray, out: np.ndarray, latencies: list | None
    ) -> None:
        count = rows.shape[0]
        extent = np.searchsorted(self._first_rows, rows, side="right") - 1
        byte = (
            self._extent_offsets[extent]
            + (rows - self._first_rows[extent]) * self.row_bytes
        )
        page = byte & ~np.int64(PAGE - 1)
        inner = byte - page
        lengths = np.where(inner + self.row_bytes > PAGE, READ_SLOT, PAGE).astype(
            np.int64
        )
        needs = inner + self.row_bytes
        self._read_fds = self._extent_fds[extent]
        args = (
            self._read_fds,
            page,
            lengths,
            self._bounce_addr,
            READ_SLOT,
        )
        tail = (self._results, self._latencies_ns)
        workers = min(self.threads, count)
        pool = self._pool
        if workers <= 1 or pool is None:
            self._pread_batch(*args, 0, count, *tail)
        else:
            bounds = np.linspace(0, count, workers + 1).astype(np.int64).tolist()
            futures = [
                pool.submit(self._pread_batch, *args, bounds[w], bounds[w + 1], *tail)
                for w in range(workers)
            ]
            for future in futures:
                future.result()
        for slot in np.nonzero(self._results[:count] < needs)[0].tolist():
            self.retries += 1
            self._retry(slot, int(page[slot]), int(lengths[slot]), int(needs[slot]))
        if latencies is not None:
            latencies.append(self._latencies_ns[:count] * 1e-9)
        columns = inner[:, None] + self._row_offsets[None, :]
        out[:] = np.take_along_axis(self._bounce_np[:count], columns, axis=1)


# --------------------------------------------------------------------------
# Row cache
# --------------------------------------------------------------------------

_EPOCH_FREE = np.iinfo(np.int64).min // 2


class RowCache:
    """A rank's row cache: uint8 [capacity, row_bytes] slots and the host maps.

    ``row2slot`` is a direct int32 map over the rank's own rows (-1 = absent).
    Eviction is CLOCK with epoch protection: a slot touched in this step or the
    previous one is never evicted. A step that needs more slots than that leaves
    raises before any map changes; the table refuses a capacity below two full
    steps at boot. Speculative rows are not protected until a lookup touches
    them, so lookahead never uses up those two steps.
    """

    def __init__(self, num_rows: int, slab: np.ndarray) -> None:
        capacity, row_bytes = slab.shape
        if capacity <= 0 or capacity > np.iinfo(np.int32).max:
            raise ValueError(f"PLE row cache capacity {capacity} is out of range")
        self.capacity = int(capacity)
        self.row_bytes = int(row_bytes)
        self.slab = slab
        self.row2slot = np.full(int(num_rows), -1, dtype=np.int32)
        self.slot2row = np.full(self.capacity, -1, dtype=np.int64)
        self.ref = np.zeros(self.capacity, dtype=np.uint8)
        self.epoch = np.full(self.capacity, _EPOCH_FREE, dtype=np.int64)
        self.hand = 0
        self.step = 0
        self._alloc_out = np.empty(1024, dtype=np.int64)
        self._clock_alloc = _compiled()[1]
        # Numba compiles on the first call: have it here, not in a step.
        self._clock_alloc(0, 0, self.ref, self.epoch, -1, 0, self._alloc_out)

    def begin_step(self) -> None:
        self.step += 1

    def lookup(self, local_rows: np.ndarray) -> np.ndarray:
        """int64 slots of ``local_rows`` (-1 = miss); marks the hits used."""
        slots = self.row2slot[local_rows].astype(np.int64)
        used = slots[slots >= 0]
        if used.size:
            self.ref[used] = 1
            self.epoch[used] = self.step
        return slots

    def _allocate(self, count: int) -> np.ndarray:
        if count > self._alloc_out.shape[0]:
            self._alloc_out = np.empty(count, dtype=np.int64)
        out = self._alloc_out
        got, hand = self._clock_alloc(
            count, self.hand, self.ref, self.epoch, self.step - 1, self.step, out
        )
        self.hand = int(hand)
        if got < count:
            # The partial pick was evictable, so "free" is its honest epoch.
            self.epoch[out[:got]] = _EPOCH_FREE
            self.ref[out[:got]] = 0
            raise RuntimeError(
                f"PLE row cache: step needs {count} new slots but only {got} are "
                f"evictable (capacity {self.capacity}); raise disk_row_cache_gib"
            )
        return out[:count].copy()

    def install(
        self, local_rows: np.ndarray, data: np.ndarray, speculative: bool = False
    ) -> None:
        """Place unique missing ``local_rows`` into freshly allocated slots.

        ``speculative`` rows (a lookahead guess no lookup has asked for) stay
        evictable until a lookup touches them, so they never take the room the
        two-step protection keeps for rows a step actually needs.
        """
        slots = self._allocate(local_rows.shape[0])
        evicted = self.slot2row[slots]
        evicted = evicted[evicted >= 0]
        if evicted.size:
            self.row2slot[evicted] = -1
        self.slab[slots] = data
        self.slot2row[slots] = local_rows
        self.row2slot[local_rows] = slots.astype(np.int32)
        if speculative:
            self.epoch[slots] = _EPOCH_FREE
            self.ref[slots] = 0


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


class RowStats:
    """Counters over an interval; one log line per interval, per rank."""

    def __init__(self, rank: int, interval_s: float) -> None:
        self.rank = rank
        self.interval_s = float(interval_s)
        self.enabled = self.interval_s > 0
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.started = now
        self.steps = self.lookups = self.hits = self.misses = 0
        self.reads = self.from_lookahead = 0
        self.step_ms: list[float] = []
        self.wait_ms: list[float] = []
        self.read_latencies: list[np.ndarray] = []

    def record(
        self,
        lookups: int,
        misses: int,
        reads: int,
        step_ms: float,
        from_lookahead: int = 0,
        wait_ms: float = 0.0,
    ) -> None:
        """``misses`` at step start; ``reads`` of them went to the file in the step;
        ``from_lookahead`` arrived ready; ``wait_ms`` was spent ending a job."""
        if not self.enabled:
            return
        self.steps += 1
        self.lookups += lookups
        self.hits += lookups - misses
        self.misses += misses
        self.reads += reads
        self.from_lookahead += from_lookahead
        self.step_ms.append(step_ms)
        if wait_ms > 0.0:
            self.wait_ms.append(wait_ms)
        now = time.monotonic()
        if now - self.started >= self.interval_s:
            logger.info("%s", self.line(now))
            self._reset(now)

    def line(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        latencies = (
            np.concatenate(self.read_latencies) * 1e3
            if self.read_latencies
            else np.zeros(0)
        )
        step_ms = np.asarray(self.step_ms) if self.step_ms else np.zeros(1)

        def percentile(values: np.ndarray, q: float) -> float:
            return float(np.percentile(values, q)) if values.size else 0.0

        rate = self.hits / self.lookups if self.lookups else 0.0
        return (
            f"PLE row cache rank {self.rank}: {self.steps} steps in "
            f"{now - self.started:.0f} s, hit rate {rate:.3f} "
            f"({self.hits}/{self.lookups}), misses/step "
            f"{self.misses / max(self.steps, 1):.1f}, reads {self.reads}, "
            f"read p50 {percentile(latencies, 50):.3f} ms "
            f"p99 {percentile(latencies, 99):.3f} ms, "
            f"host step p50 {percentile(step_ms, 50):.3f} ms "
            f"p99 {percentile(step_ms, 99):.3f} ms "
            f"max {float(step_ms.max()):.3f} ms, "
            f"from lookahead {self.from_lookahead}, lookahead waits "
            f"{len(self.wait_ms)} (max {max(self.wait_ms, default=0.0):.3f} ms)"
        )


# --------------------------------------------------------------------------
# Per-rank server
# --------------------------------------------------------------------------


def owned_heads(
    sizes: np.ndarray, offsets: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Heads whose global rows intersect [start, end)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    hit = (offsets < end) & (offsets + sizes > start)
    return np.nonzero(hit)[0].astype(np.int64)


def host_ngram_ids(
    tokens: np.ndarray,
    query_start_loc: np.ndarray,
    ngram_context: np.ndarray,
    *,
    multipliers: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    eos_token_id: int,
    heads_per_ngram: int,
    heads: np.ndarray,
) -> np.ndarray:
    """N-gram ids [T, len(heads)] on the host, bit-exact with ``ple_ngram_ids``.

    ``tokens`` [T] are laid out by ``query_start_loc`` [R + 1], and ``ngram_context``
    [R, ngram_size - 1] holds the tokens before each request's first token. A
    shifted token before the start of the token's EOS segment becomes EOS.
    """
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    qsl = np.asarray(query_start_loc, dtype=np.int64)
    multipliers = np.asarray(multipliers, dtype=np.int64)
    ngram_size = int(multipliers.shape[0])
    ctx_len = ngram_size - 1
    num_tokens = tokens.shape[0]
    num_reqs = qsl.shape[0] - 1
    if num_tokens == 0 or num_reqs <= 0:
        return np.empty((num_tokens, heads.shape[0]), dtype=np.int64)
    lengths = np.diff(qsl)
    context = np.asarray(ngram_context, dtype=np.int64)[:num_reqs]
    # Flat layout: each request's context tokens, then its tokens.
    req_start = qsl[:-1] + ctx_len * np.arange(num_reqs, dtype=np.int64)
    total = num_tokens + ctx_len * num_reqs
    sequence = np.empty(total, dtype=np.int64)
    req_of_token = np.repeat(np.arange(num_reqs, dtype=np.int64), lengths)
    token_at = np.arange(num_tokens, dtype=np.int64) + ctx_len * (req_of_token + 1)
    sequence[token_at] = tokens
    context_at = req_start[:, None] + np.arange(ctx_len, dtype=np.int64)
    sequence[context_at.reshape(-1)] = context.reshape(-1)
    # Last EOS strictly before each position, or just before the request.
    index = np.arange(total, dtype=np.int64)
    inclusive = np.maximum.accumulate(np.where(sequence == eos_token_id, index, -1))
    previous = np.empty(total, dtype=np.int64)
    previous[0] = -1
    previous[1:] = inclusive[:-1]
    previous = np.maximum(previous, np.repeat(req_start - 1, lengths + ctx_len))
    in_segment = (index - previous - 1)[token_at]
    terms = [sequence[token_at] * multipliers[0]]
    for shift in range(1, ngram_size):
        shifted = np.where(
            in_segment >= shift,
            sequence[np.maximum(token_at - shift, 0)],
            np.int64(eos_token_id),
        )
        terms.append(shifted * multipliers[shift])
    out = np.empty((num_tokens, heads.shape[0]), dtype=np.int64)
    mixed_by_order: dict[int, np.ndarray] = {}
    for column, head in enumerate(heads.tolist()):
        order = head // heads_per_ngram + 2
        mixed = mixed_by_order.get(order)
        if mixed is None:
            mixed = terms[0]
            for term in terms[1:order]:
                mixed = np.bitwise_xor(mixed, term)
            mixed_by_order[order] = mixed
        out[:, column] = np.remainder(mixed, sizes[head]) + offsets[head]
    return out


class RowServer:
    """One rank's step: own row ids -> cache slots, reading what is missing."""

    def __init__(
        self,
        *,
        start: int,
        end: int,
        multipliers: np.ndarray,
        sizes: np.ndarray,
        offsets: np.ndarray,
        eos_token_id: int,
        heads_per_ngram: int,
        cache: RowCache,
        reader: RowReader,
        stats: RowStats,
    ) -> None:
        self.start = int(start)
        self.end = int(end)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.eos_token_id = int(eos_token_id)
        self.heads_per_ngram = int(heads_per_ngram)
        self.own_heads = owned_heads(self.sizes, self.offsets, self.start, self.end)
        self.cache = cache
        self.reader = reader
        self.stats = stats

    def hash_own(self, tokens, query_start_loc, context) -> np.ndarray:
        return host_ngram_ids(
            tokens,
            query_start_loc,
            context,
            multipliers=self.multipliers,
            sizes=self.sizes,
            offsets=self.offsets,
            eos_token_id=self.eos_token_id,
            heads_per_ngram=self.heads_per_ngram,
            heads=self.own_heads,
        )

    def resolve(
        self,
        own: np.ndarray,
        prefetched: tuple[np.ndarray, np.ndarray] | None = None,
        wait_ms: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slots of ``own`` [T, len(own_heads)] global row ids, as (slots, mask).

        Order, per step: the lookup marks every hit with this step's epoch, so
        nothing below can evict it. Misses come from ``prefetched`` (sorted unique
        local rows and their bytes) where present and from the file otherwise, and
        are installed together. Prefetched rows this step did not need go in last,
        best effort, so they only displace rows older than the previous step, and
        stay evictable until a later lookup asks for them.
        """
        started = time.perf_counter()
        self.cache.begin_step()
        own = np.asarray(own, dtype=np.int64)
        mask = (own >= self.start) & (own < self.end)
        local = own[mask] - self.start
        slots = self.cache.lookup(local)
        missing = slots < 0
        lookups, misses = int(local.shape[0]), int(missing.sum())
        reads = from_lookahead = 0
        if prefetched is None or not prefetched[0].size:
            prefetched = None
        else:
            pre_rows, pre_data = prefetched
            pre_used = np.zeros(pre_rows.shape[0], dtype=bool)
        if misses:
            miss_rows = np.unique(local[missing])
            data = np.empty((miss_rows.shape[0], self.cache.row_bytes), np.uint8)
            found = np.zeros(miss_rows.shape[0], dtype=bool)
            if prefetched is not None:
                position = np.minimum(
                    np.searchsorted(pre_rows, miss_rows), pre_rows.shape[0] - 1
                )
                found = pre_rows[position] == miss_rows
                data[found] = pre_data[position[found]]
                pre_used[position[found]] = True
                from_lookahead = int(found.sum())
            need = miss_rows[~found]
            if need.size:
                latencies: list | None = [] if self.stats.enabled else None
                data[~found] = self.reader.read_rows(
                    need + self.start, latencies=latencies
                )
                reads = int(need.shape[0])
                if latencies:
                    self.stats.read_latencies.extend(latencies)
            self.cache.install(miss_rows, data)
            slots = self.cache.row2slot[local].astype(np.int64)
        out = np.zeros(own.shape, dtype=np.int64)
        out[mask] = slots
        if prefetched is not None:
            self._install_unused(pre_rows[~pre_used], pre_data[~pre_used])
        self.stats.record(
            lookups,
            misses,
            reads,
            (time.perf_counter() - started) * 1e3,
            from_lookahead,
            wait_ms,
        )
        return out, mask

    def _install_unused(self, rows: np.ndarray, data: np.ndarray) -> None:
        absent = self.cache.row2slot[rows] < 0
        if not absent.any():
            return
        try:
            self.cache.install(rows[absent], data[absent], speculative=True)
        except RuntimeError:
            logger.debug(
                "PLE row cache had no room for %d prefetched rows", absent.sum()
            )


# --------------------------------------------------------------------------
# Prefill lookahead
# --------------------------------------------------------------------------


def plan_next_chunks(
    idx_mapping: np.ndarray,
    num_computed: np.ndarray,
    num_scheduled: np.ndarray,
    prefill_len: np.ndarray,
    budget: int,
) -> list[tuple[int, int, int]]:
    """Predict the next step's prefill chunks as [(req_idx, start, end)].

    Every request still prefilling after this step continues where this step ends,
    within a token budget shared in batch order; requests that decode next step
    take one token each. A wrong guess only costs reads, since the next step
    resolves its real ids either way.
    """
    num = int(idx_mapping.shape[0])
    start = num_computed[:num].astype(np.int64) + num_scheduled[:num].astype(np.int64)
    end = prefill_len[:num].astype(np.int64)
    prefilling = start < end
    left = int(budget) - int((~prefilling).sum())
    plan: list[tuple[int, int, int]] = []
    for i in np.nonzero(prefilling)[0].tolist():
        if left <= 0:
            break
        chunk_end = min(int(end[i]), int(start[i]) + left)
        plan.append((int(idx_mapping[i]), int(start[i]), chunk_end))
        left -= chunk_end - int(start[i])
    return plan


def current_chunk_keys(
    idx_mapping: np.ndarray,
    num_computed: np.ndarray,
    num_scheduled: np.ndarray,
    prefill_len: np.ndarray,
) -> frozenset:
    """Keys (req_idx, start) of this step's prefill chunks, to match a past plan."""
    keys = set()
    for i in range(int(idx_mapping.shape[0])):
        start = int(num_computed[i])
        if start < int(prefill_len[i]) and int(num_scheduled[i]) > 0:
            keys.add((int(idx_mapping[i]), start))
    return frozenset(keys)


class _LookaheadJob:
    def __init__(self, keys, tokens, qsl, context) -> None:
        self.keys = keys
        self.tokens = tokens
        self.qsl = qsl
        self.context = context
        self.cancel = False
        self.started = False
        self.done = False
        self.rows: list[np.ndarray] = []
        self.data: list[np.ndarray] = []
        self.error: BaseException | None = None


class RowLookahead:
    """Reads the rows of the next prefill chunk ahead of its step, one per rank.

    The step thread is the only writer of the cache: it calls ``take`` at the start
    of a step and ``submit`` after staging, so at most one job exists and none spans
    a step. The worker hashes the job's tokens, reads the row map without a lock to
    skip cached rows (a stale answer costs one extra read, or one step-time read of
    a row evicted since; never wrong bytes), and reads the rest with its own reader
    into the job's arrays, in sub-batches so a cancel takes effect quickly. Errors
    are logged and dropped: the step reads what is missing, and raises there if the
    disk is really failing.
    """

    def __init__(
        self,
        server: RowServer,
        reader: RowReader,
        *,
        sub_batch: int = 256,
        max_rows: int | None = None,
    ) -> None:
        self.server = server
        self.reader = reader
        self.sub_batch = max(1, min(int(sub_batch), reader.max_batch_rows))
        self.max_rows = max_rows
        self._cv = threading.Condition()
        self._job: _LookaheadJob | None = None
        self._stop = False
        self.submitted = 0
        self.cancelled = 0
        self.rows_read = 0
        self.errors = 0
        self._thread = threading.Thread(
            target=self._main, name="ple-lookahead", daemon=True
        )
        self._thread.start()

    def submit(self, keys, tokens, query_start_loc, context) -> None:
        """Start reading the rows of a predicted chunk."""
        job = _LookaheadJob(
            keys,
            np.array(tokens, dtype=np.int64),
            np.array(query_start_loc, dtype=np.int64),
            np.array(context, dtype=np.int64),
        )
        with self._cv:
            if self._job is not None:
                raise RuntimeError("PLE lookahead: submit before take")
            self._job = job
            self.submitted += 1
            self._cv.notify_all()

    def take(
        self, current_keys: frozenset
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, float]:
        """End the pending job; return ((rows, data) or None, wait in ms).

        A job still running with none of its chunks in this step is cancelled and
        keeps its finished sub-batches. Otherwise the step waits for it: what
        remains is no more than the step would read itself.
        """
        with self._cv:
            job = self._job
            if job is None:
                return None, 0.0
            started = time.perf_counter()
            waited = not job.done
            if waited and not (job.keys & current_keys):
                job.cancel = True
            while not job.done:
                self._cv.wait()
            self._job = None
        wait_ms = (time.perf_counter() - started) * 1e3 if waited else 0.0
        if job.cancel:
            self.cancelled += 1
        if job.error is not None:
            self.errors += 1
            logger.warning("PLE lookahead job failed: %s", job.error)
        if not job.rows:
            return None, wait_ms
        return (np.concatenate(job.rows), np.concatenate(job.data)), wait_ms

    def close(self) -> None:
        with self._cv:
            self._stop = True
            if self._job is not None:
                self._job.cancel = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)
        self.reader.close()

    def _main(self) -> None:
        while True:
            with self._cv:
                while not self._stop and (self._job is None or self._job.started):
                    self._cv.wait()
                if self._stop:
                    return
                job = cast(_LookaheadJob, self._job)
                job.started = True
            try:
                self._work(job)
            except Exception as exc:  # handed to the step thread
                job.error = exc
            with self._cv:
                job.done = True
                self._cv.notify_all()

    def _work(self, job: _LookaheadJob) -> None:
        server = self.server
        if job.cancel or job.tokens.shape[0] == 0:
            return
        own = server.hash_own(job.tokens, job.qsl, job.context)
        mask = (own >= server.start) & (own < server.end)
        local = np.unique(own[mask] - server.start)
        local = local[server.cache.row2slot[local] < 0]
        if self.max_rows is not None:
            local = local[: self.max_rows]
        for lo in range(0, local.shape[0], self.sub_batch):
            if job.cancel:
                return
            part = local[lo : lo + self.sub_batch]
            data = self.reader.read_rows(part + server.start)
            job.rows.append(part)
            job.data.append(data)
            self.rows_read += int(part.shape[0])
