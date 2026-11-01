#!/usr/bin/env python3
#
# check_links.py
# ~~~~~~~~~~~~~~
#
# Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# Checks the internal links in the generated HTML documentation.
#
# Usage: check_links.py <html-directory>

import os
import re
import sys
from urllib.parse import unquote


def find_pages(root):
    pages = []
    for directory, subdirectories, files in os.walk(root):
        for name in files:
            if name.endswith(".html"):
                pages.append(os.path.normpath(os.path.join(directory, name)))
    return pages


def main():
    root = sys.argv[1]

    anchors = {}
    links = []
    for page in find_pages(root):
        with open(page, errors="replace") as f:
            text = f.read()
        anchors[page] = set(re.findall(r'(?:id|name)="([^"]+)"', text))
        for href in re.findall(r'href="([^"]*)"', text):
            links.append((page, href))

    if not anchors:
        sys.exit("no HTML pages found")

    broken = 0
    for page, href in links:
        # Skip empty links and links to other sites.
        if not href or re.match(r"^[a-z]+:", href):
            continue

        path, separator, anchor = href.partition("#")
        if path:
            target = os.path.join(os.path.dirname(page), unquote(path))
            target = os.path.normpath(target)
        else:
            target = page

        if not os.path.exists(target):
            ok = False
        elif anchor and target in anchors:
            ok = anchor in anchors[target]
        else:
            ok = True

        if not ok:
            print("%s: broken link: %s" % (os.path.relpath(page, root), href))
            broken += 1

    print("%d internal links checked, %d broken" % (len(links), broken))
    if broken:
        sys.exit(1)


if __name__ == "__main__":
    main()
