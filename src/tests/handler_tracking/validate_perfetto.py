#!/usr/bin/env python3
#
# validate_perfetto.py
# ~~~~~~~~~~~~~~~~~~~~
#
# Usage: python validate_perfetto.py <trace.pftrace>
#
# Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

import struct
import sys

failures = 0


def check(cond, msg):
    global failures
    print(("PASS" if cond else "FAIL") + ": " + msg)
    if not cond:
        failures += 1


# ---- Structural checks via trace_processor -------------------------------


def structural_checks(path):
    from perfetto.trace_processor import TraceProcessor

    tp = TraceProcessor(trace=path)

    def q(sql):
        return list(tp.query(sql))

    bad = q(
        "SELECT name, value FROM stats "
        "WHERE severity IN ('error', 'data_loss') AND value > 0"
    )
    check(
        len(bad) == 0,
        "no error/data_loss stats " + str([(r.name, r.value) for r in bad]),
    )

    nslice = q("SELECT count(*) AS c FROM slice")[0].c
    check(nslice > 0, "slice table non-empty (%d slices)" % nslice)

    unfinished = q("SELECT count(*) AS c FROM slice WHERE dur < 0")[0].c
    check(unfinished == 0, "no unfinished slices (%d)" % unfinished)

    threads = [
        r
        for r in q("SELECT tid FROM thread WHERE tid IS NOT NULL")
        if r.tid and r.tid > 0
    ]
    check(len(threads) >= 1, "at least one thread with a real tid")

    procs = q(
        "SELECT pid, name FROM process "
        "WHERE pid IS NOT NULL AND pid > 0 AND name IS NOT NULL"
    )
    check(len(procs) >= 1, "at least one process with pid and name")

    flows = q("SELECT slice_out, slice_in FROM flow")
    resolved = 0
    for f in flows:
        a = q("SELECT count(*) AS c FROM slice WHERE id = %d" % f.slice_out)[0].c
        b = q("SELECT count(*) AS c FROM slice WHERE id = %d" % f.slice_in)[0].c
        if a == 1 and b == 1:
            resolved += 1
    check(len(flows) >= 1, "at least one flow (%d)" % len(flows))
    check(
        resolved == len(flows),
        "every flow resolves to two slices (%d/%d)" % (resolved, len(flows)),
    )


# ---- Location slice nesting via a wire-format parse ----------------------


def read_varint(b, i):
    r = 0
    s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            break
        s += 7
    return r, i


def fields(b):
    i = 0
    out = []
    while i < len(b):
        key, i = read_varint(b, i)
        fn = key >> 3
        wt = key & 7
        if wt == 0:
            v, i = read_varint(b, i)
            out.append((fn, wt, v))
        elif wt == 1:
            v = struct.unpack("<Q", b[i : i + 8])[0]
            i += 8
            out.append((fn, wt, v))
        elif wt == 2:
            ln, i = read_varint(b, i)
            out.append((fn, wt, b[i : i + ln]))
            i += ln
        elif wt == 5:
            v = struct.unpack("<I", b[i : i + 4])[0]
            i += 4
            out.append((fn, wt, v))
        else:
            raise ValueError("bad wire type %d" % wt)
    return out


def slice_nesting_check(path):
    data = open(path, "rb").read()
    i = 0

    # Collect track events as (timestamp, type, name) per thread track. A
    # thread's timestamps are strictly increasing. Sorting per track
    # reconstructs its exact begin/end order regardless of file/buffer order.
    events = {}
    while i < len(data):
        key, i = read_varint(data, i)  # Trace.packet = 1, length-delimited
        ln, i = read_varint(data, i)
        pkt = data[i : i + ln]
        i += ln
        ts = None
        te = None
        for fn, wt, val in fields(pkt):
            if fn == 8 and wt == 0:  # TracePacket.timestamp
                ts = val
            elif fn == 11 and wt == 2:  # TracePacket.track_event
                te = val
        if te is None:
            continue
        track = None
        etype = None
        name = None
        for f2, w2, v2 in fields(te):
            if f2 == 11 and w2 == 0:  # TrackEvent.track_uuid
                track = v2
            elif f2 == 9 and w2 == 0:  # TrackEvent.type (1=begin, 2=end)
                etype = v2
            elif f2 == 23 and w2 == 2:  # TrackEvent.name
                name = v2.decode("utf-8", "replace")
        if track is None or etype not in (1, 2):
            continue
        events.setdefault(track, []).append((ts, etype, name))

    # Reconstruct each track's nested slices via a LIFO stack, recording the
    # (name, begin, end) interval of every slice.
    intervals = []
    for track, evs in events.items():
        evs.sort(key=lambda e: e[0])
        stack = []
        for ts, etype, name in evs:
            if etype == 1:  # slice_begin
                stack.append((name, ts))
            elif stack:  # slice_end
                nm, begin = stack.pop()
                intervals.append((track, nm, begin, ts))

    # The inner async_write slice must open after and close before an outer
    # async_double_write slice on the same track.
    found = False
    for t1, n1, b1, e1 in intervals:
        if n1 != "async_double_write":
            continue
        for t2, n2, b2, e2 in intervals:
            if n2 == "async_write" and t2 == t1 and b1 < b2 and e2 < e1:
                found = True
    check(
        found,
        "async_write location slice nested within async_double_write "
        "(opens after, closes before, same thread track)",
    )


def main():
    path = sys.argv[1]
    try:
        structural_checks(path)
    except ImportError:
        print(
            "SKIP: perfetto trace_processor not installed; "
            "running the dependency-free checks only"
        )
    slice_nesting_check(path)
    print("RESULT:", "ALL PASS" if failures == 0 else ("%d FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
