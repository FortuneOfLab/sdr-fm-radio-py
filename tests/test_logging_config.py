"""What setup_logging writes, and in what encoding.

The receiver logs station names, areas and transmitter sites, all of
which are Japanese.  Left to Python the log file takes the platform
encoding - cp932 on the machine this runs on - and every one of those
lines is unreadable in an editor that opens the file as UTF-8, which
is how a log gets read.
"""

from __future__ import annotations

import contextlib
import logging

from fm_radio.logging_config import setup_logging

# A line the receiver really writes; see band_scan.where_this_is.
A_REAL_LINE = "This looks like 関東: 東京 explains 4 of 5"


@contextlib.contextmanager
def a_root_logger_of_our_own():
    """Let setup_logging configure the root logger, then put it back.

    basicConfig does nothing when the root already has handlers, and
    under pytest it always does.
    """
    root = logging.getLogger()
    was_handlers, was_level = root.handlers[:], root.level
    root.handlers = []
    try:
        yield root
    finally:
        for handler in root.handlers:
            # Windows will not let the file be deleted while it is
            # open, and tmp_path is deleted after the test.
            handler.close()
        root.handlers = was_handlers
        root.level = was_level


def test_the_log_file_is_written_as_utf8(tmp_path):
    """Written and read back as UTF-8, not as whatever the console is.

    This only catches the defect where the platform encoding is not
    UTF-8 to begin with - on the Linux CI job the bytes are the same
    either way.  On Windows, taking the encoding off the handler
    fails it.
    """
    path = tmp_path / "run.log"
    with a_root_logger_of_our_own():
        setup_logging(log_file=str(path))
        logging.getLogger("fm_receiver.band_scan").info(A_REAL_LINE)
        for handler in logging.getLogger().handlers:
            handler.flush()
        written = path.read_bytes()

    assert A_REAL_LINE.encode("utf-8") in written
    assert A_REAL_LINE in written.decode("utf-8")


def test_the_file_handler_says_utf8(tmp_path):
    """The encoding is on the handler, for the platforms where it matters."""
    path = tmp_path / "run.log"
    with a_root_logger_of_our_own():
        setup_logging(log_file=str(path))
        files = [h for h in logging.getLogger().handlers
                 if isinstance(h, logging.FileHandler)]
        assert len(files) == 1
        assert (files[0].encoding or "").lower().replace("-", "") == "utf8"


def test_the_console_handler_is_still_there(tmp_path):
    """A file to read later does not replace the lines on the screen."""
    path = tmp_path / "run.log"
    with a_root_logger_of_our_own():
        setup_logging(log_file=str(path))
        handlers = logging.getLogger().handlers
        streams = [h for h in handlers
                   if isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)]
        assert len(streams) == 1


def test_without_a_file_nothing_is_written(tmp_path):
    with a_root_logger_of_our_own():
        setup_logging()
        assert not any(isinstance(h, logging.FileHandler)
                       for h in logging.getLogger().handlers)
