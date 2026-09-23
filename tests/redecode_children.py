"""Stand-in children for fm_radio.redecode.Job, in tests.

A spawned child imports its target by module and name, so the targets
live here, in a module that imports nothing but the standard library:
the child has to get as far as calling one before a test can say
anything about how it ended.
"""

import os
import threading
from multiprocessing.connection import Client


def wait_forever(send, address, window_s):
    """Tell *address* it is running, then never answer the job.

    *address* is a ``multiprocessing.connection.Listener``'s, passed
    where the job passes the WAV's path: the test waits on it to know
    the child is running, and the job's own pipe carries nothing.
    """
    with Client(address) as told:
        told.send(os.getpid())
    threading.Event().wait()


def answer_when_told(send, address, window_s):
    """Tell *address* it is running, and answer the job once told to."""
    with Client(address) as told:
        told.send(os.getpid())
        told.recv()
    send.send(("failed", "told to"))


def die_without_answering(send, wav_path, window_s):
    """End the process the way a crash would: no answer, no clean-up."""
    os._exit(3)
