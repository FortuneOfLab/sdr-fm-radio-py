"""The entry point's startup and shutdown paths.

What matters here is that a receiver which got part way up is always taken
back down: start_background() starts the SDR thread before the processing
thread, so a failure between the two leaves a thread running that only
cleanup() stops.
"""

from __future__ import annotations

import sys

import pytest

import fm_radio.__main__ as entry


class FakeController:
    """Records the order of the lifecycle calls made against it."""

    def __init__(self, *, fail_at: str | None = None, **kwargs) -> None:
        self.kwargs = kwargs
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.quit_event = None

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
