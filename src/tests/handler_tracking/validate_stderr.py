#!/usr/bin/env python3
#
# validate_stderr.py
# ~~~~~~~~~~~~~~~~~~
#
# Usage: python validate_stderr.py <stderr-log>
#
# Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

import sys


def field(line):
    parts = line.split("|")
    return parts[2] if len(parts) > 2 else ""


def main():
    if len(sys.argv) != 2:
        print("usage: validate_stderr.py <stderr-log>")
        return 2

    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    asio = [l for l in lines if l.startswith("@asio|")]
    creations = [l for l in asio if "*" in field(l)]
    begins = [l for l in asio if field(l).startswith(">")]
    ends = [l for l in asio if field(l).startswith("<")]

    print(
        "@asio records: %d  (creations=%d begins=%d ends=%d)"
        % (len(asio), len(creations), len(begins), len(ends))
    )

    ok = True
    if not asio:
        print("FAIL: no @asio| tracking records on stderr")
        ok = False
    if not creations:
        print("FAIL: no operation creations recorded")
        ok = False
    if not begins:
        print("FAIL: no handler invocation begins recorded")
        ok = False
    if not ends:
        print("FAIL: no handler invocation ends recorded")
        ok = False
    # Every begun invocation must also end.
    if begins and ends and len(begins) != len(ends):
        print(
            "FAIL: %d invocation begins but %d ends (unbalanced)"
            % (len(begins), len(ends))
        )
        ok = False

    print("RESULT:", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
