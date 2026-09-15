"""The entry point's startup and shutdown paths.

What matters here is that a receiver which got part way up is always taken
back down: start_background() starts the SDR thread before the processing
thread, so a failure between the two leaves a thread running that only
cleanup() stops.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

import fm_radio.__main__ as entry


class FakeController:
    """Records the order of the lifecycle calls made against it."""

    def __init__(self, *, fail_at: str | None = None, **kwargs) -> None:
        self.kwargs = kwargs
        self.fail_at = fail_at
        self.calls: list[str] = []
        # A real Event, not None: the threads the receiver starts watch this,
        # and a stand-in that cannot record it being set hides whether
        # anything ever asked them to stop.
        self.quit_event = threading.Event()

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")

    def start_background(self) -> None:
        self._record("start_background")

    def start(self) -> None:
        self._record("start")

    def cleanup(self) -> None:
        self._record("cleanup")


@pytest.fixture
def entry_point(monkeypatch):
    """Run main() with a stand-in controller and a stand-in GUI."""
    built: list[FakeController] = []
    gui_calls: list[object] = []

    def _run(argv, *, fail_at=None, gui_exit=0, gui_error=None):
        def build(**kwargs):
            controller = FakeController(fail_at=fail_at, **kwargs)
            built.append(controller)
            return controller

        def run_gui(controller):
            gui_calls.append(controller)
            if gui_error is not None:
                raise gui_error
            return gui_exit

        monkeypatch.setattr(entry, "FMReceiverController", build)
        monkeypatch.setattr(sys, "argv", ["fm_receiver.py"] + argv)
        from fm_radio import gui
        monkeypatch.setattr(gui, "run", run_gui)

        # The GUI path exits with the window's code; the command line path
        # returns normally, so neither is required to raise.
        try:
            entry.main()
            code = None
        except SystemExit as exc:
            code = exc.code
        return built[-1], gui_calls, code

    return _run


def test_the_gui_path_starts_runs_and_cleans_up(entry_point):
    controller, gui_calls, code = entry_point(["--gui"], gui_exit=0)

    assert controller.calls == ["start_background", "cleanup"]
    assert gui_calls == [controller]
    assert code == 0


def test_the_gui_exit_code_reaches_the_process(entry_point):
    _, _, code = entry_point(["--gui"], gui_exit=1)
    assert code == 1


def test_a_failure_starting_the_threads_still_cleans_up(entry_point):
    """The SDR thread may already be running when the next start fails."""
    controller, gui_calls, code = entry_point(
        ["--gui"], fail_at="start_background")

    assert controller.calls == ["start_background", "cleanup"]
    assert gui_calls == [], "the window opened over a receiver that never started"
    assert code == 1


def test_a_window_that_raises_still_cleans_up(entry_point):
    controller, _, code = entry_point(
        ["--gui"], gui_error=RuntimeError("window exploded"))

    assert controller.calls == ["start_background", "cleanup"]
    assert code == 1


def test_without_gui_the_controller_runs_its_own_loop(entry_point):
    controller, gui_calls, code = entry_point([])

    # start() owns the CLI loop and its own cleanup.
    assert controller.calls == ["start"]
    assert gui_calls == []
    assert code is None, "the command line path should return, not exit"


def test_the_stations_path_reaches_the_controller(entry_point, tmp_path):
    controller, _, _ = entry_point(["--gui", "--stations", str(tmp_path / "s.toml")])
    assert controller.kwargs["stations_path"] == str(tmp_path / "s.toml")


def test_light_mode_reaches_the_controller(entry_point):
    controller, _, _ = entry_point(["--gui", "--light"])
    assert controller.kwargs["light"] is True


# ----------------------------------------------------------------------
# The threads have to be told to stop, not just have their things closed
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"gui_exit": 0},                                    # the window closed
    {"gui_exit": 1},                                    # PySide6 missing
    {"gui_error": RuntimeError("window exploded")},     # it never opened
    {"fail_at": "start_background"},                    # it never started
])
def test_every_gui_path_reaches_cleanup(entry_point, kwargs):
    """What main() owns: cleanup happens however the GUI path ends.

    Whether cleanup then stops the threads is the receiver's own contract,
    which a stand-in cannot demonstrate - see the test below, which uses the
    real one.
    """
    controller, _, _ = entry_point(["--gui"], **kwargs)
    assert "cleanup" in controller.calls


def test_the_real_receiver_stops_when_the_window_never_opens(monkeypatch):
    """The stand-in above cannot show this: it has no threads to leave behind.

    gui.run() returning 1 is what happens when PySide6 is missing, and it is
    the path that used to close the audio output from under a processing
    thread that was still running.
    """
    from fm_radio import gui
    from fm_radio.controller import FMReceiverController

    monkeypatch.setattr(gui, "run", lambda controller: 1)
    monkeypatch.setattr(sys, "argv",
                        ["fm_receiver.py", "--gui", "--stations", "/none.toml"])

    before = set(threading.enumerate())
    with pytest.raises(SystemExit) as exit_info:
        entry.main()
    assert exit_info.value.code == 1

    # cleanup() waits for them, so they are gone by the time it returns.
    started = [t for t in set(threading.enumerate()) - before if t.is_alive()]
    assert not started, f"still running: {[t.name for t in started]}"


def test_cleanup_can_be_called_twice(no_user_config):
    """The GUI path's finally may run after the window already closed."""
    from fm_radio.controller import FMReceiverController

    controller = FMReceiverController(light=True,
                                      stations_path=str(no_user_config))
    controller.start_background()
    controller.cleanup()
    controller.cleanup()                # must not raise
    assert controller.quit_event.is_set()


def test_cleanup_on_a_receiver_that_never_started(no_user_config):
    from fm_radio.controller import FMReceiverController

    controller = FMReceiverController(light=True,
                                      stations_path=str(no_user_config))
    controller.cleanup()                # no threads at all
    assert controller.quit_event.is_set()
