from __future__ import annotations

from types import SimpleNamespace

from alpha_evolve.dashboard.api.subprocess_runtime import pump_text_subprocess_output


class _DummyProc:
    def __init__(self, lines: list[str], exit_code: int = 0) -> None:
        self.stdout = iter(lines)
        self._exit_code = exit_code

    def wait(self, timeout=None):  # noqa: ANN001
        return self._exit_code

    def poll(self):
        return self._exit_code


def test_pump_text_subprocess_output_streams_lines_and_returns_exit_code() -> None:
    proc = _DummyProc(["hello\n", "world\n"], exit_code=3)
    seen: list[str] = []

    code = pump_text_subprocess_output(proc=proc, handle_line=seen.append)

    assert code == 3
    assert seen == ["hello\n", "world\n"]


def test_pump_text_subprocess_output_reports_missing_stdout() -> None:
    proc = SimpleNamespace(stdout=None, wait=lambda timeout=None: 1, poll=lambda: 1)
    errors: list[str] = []

    code = pump_text_subprocess_output(
        proc=proc,
        handle_line=lambda _line: None,
        handle_error=lambda exc: errors.append(str(exc)),
    )

    assert code == 1
    assert errors
