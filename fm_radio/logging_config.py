#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) [2025] FortuneOfLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Logging configuration for the FM receiver system."""

from __future__ import annotations

import logging


def setup_logging(log_level: int = logging.INFO, log_file: str | None = None) -> None:
    """Setup logging configuration for the FM receiver system.

    The file is written as UTF-8 whatever the console is.  Left to
    Python it takes the platform encoding, which on this machine is
    cp932: a log full of station names, areas and site names, and
    ``This looks like 関東`` unreadable in every editor that opens
    the file as UTF-8.  The console handler keeps the platform
    encoding, because that is what the console can display.

    Args:
        log_level: Logging level (default: INFO).
        log_file: Optional log file path. If None, logs to console only.
    """
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )


# Initialize logger
logger: logging.Logger = logging.getLogger('fm_receiver')
