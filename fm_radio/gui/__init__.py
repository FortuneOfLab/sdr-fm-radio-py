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
"""Qt front end for the receiver.

Optional: the receiver runs without Qt installed, and nothing outside this
package imports PySide6.  ``python -m fm_radio --gui`` reaches it through
:func:`run`, which says what to install if it is not there.

The window reads state through ``controller.get_status()`` and changes it
through the controller's facade.  It never touches the processing thread's
objects, and the processing thread never waits on it.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger('fm_receiver.gui')

#: What to tell someone who asked for the GUI without the dependency.
MISSING_QT_MESSAGE = (
    "The GUI needs PySide6, which is not installed.\n"
    "    pip install PySide6\n"
    "The receiver itself does not need it; run without --gui to use the "
    "command line."
)


def run(controller) -> int:
    """Show the window and run the Qt event loop until it is closed.

    Args:
        controller: A started :class:`~fm_radio.controller.FMReceiverController`.

    Returns:
        The exit code for the process: 0 normally, 1 if PySide6 is missing.
    """
    try:
        from fm_radio.gui.main_window import run_window
    except ImportError as exc:
        logger.error("Could not start the GUI: %s", exc)
        print(MISSING_QT_MESSAGE, file=sys.stderr)
        return 1
    return run_window(controller)
