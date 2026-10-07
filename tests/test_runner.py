"""Harmless local process tests for bounded I/O and Linux lifecycle policy."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.capabilities import registry
from app.core.database import async_session_factory
from app.core.errors import ToolError
from app.models.asset import Asset
from app.models.enums import JobStatus
from app.schemas.job import JobCreate
from app.services import jobs, media_metadata
from app.services.storage import LocalStorageService
from app.tools import ffmpeg, imagemagick, process_policy, runner, waveform


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def test_stdout_below_and_at_cap(monkeypatch) -> None:
    monkeypatch.setattr(runner, "MAX_STDOUT_BYTES", 1000)
    assert await runner.run_stdout(python("print('hello', end='')")) == "hello"
    assert await runner.run_stdout(python("print('x' * 1000, end='')")) == "x" * 1000


async def test_stdout_overflow_is_failure_and_ignored_stdout_is_discarded(monkeypatch) -> None:
    monkeypatch.setattr(runner, "MAX_STDOUT_BYTES", 1000)
    with pytest.raises(ToolError, match="output exceeded its limit"):
        await runner.run_stdout(python("print('x' * 1001, end='')"))
    assert await runner.run(python("print('x' * 1001, end='')")) == ""


@pytest.mark.parametrize("newline", [False, True])
async def test_stderr_tail_is_byte_bounded_and_public_error_is_fixed(newline, caplog) -> None:
    data = "INTERNAL-/storage/private-https://internal.invalid/" * 3000
    if newline:
        data = data.replace("/", "\n")
    # Construct the large diagnostic in the child, not in a huge argv element.
    part = data[:len(data) // 3000]
    code = f"import sys;sys.stderr.write({part!r} * 3000);sys.stderr.write('TAIL')"
    tail = await runner.run(python(code))
    assert len(tail.encode()) == runner.MAX_STDERR_BYTES
    assert tail.endswith("TAIL")
    with pytest.raises(ToolError) as error:
        await runner.run(python(code + ";sys.exit(1)"))
    assert error.value.message == "Media processing failed."
    assert "INTERNAL" not in str(error.value)
    assert data[:20] not in caplog.text  # no diagnostics/argv at ordinary log levels


async def test_explicit_environment_stdin_cwd_and_session(monkeypatch) -> None:
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "must-not-reach-media")
    body = await runner.run_stdout(python(
        "import os,sys,json;print(json.dumps(dict(env=dict(os.environ),"
        "cwd=os.getcwd(),pid=os.getpid(),sid=os.getsid(0),stdin=sys.stdin.read())))"
    ))
    result = json.loads(body)
    env = result["env"]
    assert "FAKE_PROVIDER_API_KEY" not in env
    assert env["PATH"] == process_policy.TOOL_PATH
    assert env["LANG"] == env["LC_ALL"] == "C.UTF-8"
    assert env["HOME"] == env["TMPDIR"] == result["cwd"]
    assert result["cwd"] != os.getcwd()
    assert Path(result["cwd"]).name.startswith("media-process-")
    assert not Path(result["cwd"]).exists()  # private workspace cleaned
    assert result["stdin"] == ""
    assert result["pid"] == result["sid"]


async def wait_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not path.is_file():
            await asyncio.sleep(0.01)


def stopped(pid: int) -> bool:
    try:
        # A killed orphan may await PID 1's reap in a minimal container. A
        # zombie cannot run; cooperating-parent cases below assert full reap.
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("force", [False, True])
async def test_timeout_cancellation_kill_entire_process_group(tmp_path, cancel, force) -> None:
    ready = tmp_path / "pids"
    child_code = "import time;time.sleep(30)"
    if force:
        child_code = "import signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);" + child_code
    code = (
        "import os,sys,subprocess,signal,time;from pathlib import Path;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        # Cooperating parent waits/reaps child after group TERM, proving no orphan.
        "signal.signal(signal.SIGTERM,lambda *_:(child.wait(),sys.exit(0)));"
    )
    if force:
        code += "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
    code += f"Path({str(ready)!r}).write_text(f'{{os.getpid()}} {{child.pid}}');time.sleep(30)"
    task = asyncio.create_task(runner.run(python(code), timeout=1 if not cancel else 10))
    await wait_file(ready)
    parent, child = map(int, ready.read_text().split())
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ToolError) as error:
        await task
    if not cancel:
        assert error.value.message == "Media processing timed out."
    assert not Path(f"/proc/{parent}").exists()
    async with asyncio.timeout(2):
        while not stopped(child):
            await asyncio.sleep(0.01)
    if not force:
        assert not Path(f"/proc/{child}").exists()


@pytest.mark.parametrize("kind", ["metadata", "timeline", "waveform"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_direct_launchers_timeout_and_cancel_reap(
    tmp_path, monkeypatch, kind, cancel,
) -> None:
    ready = tmp_path / "pid"
    executable = "ffmpeg" if kind == "waveform" else "ffprobe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / executable
    fake.write_text(
        f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
        f"Path({str(ready)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    # The isolated validation container's /tmp is noexec; run the fixture via
    # Python while preserving every production launch-policy keyword argument.
    spawn = asyncio.create_subprocess_exec
    async def fixture_spawn(*args, **kwargs):
        assert args[0] == executable
        proc = await spawn(sys.executable, str(fake), **kwargs)
        # Start the short operation timeout only once the fixture is running,
        # independent of host/container load and Python interpreter startup.
        await wait_file(ready)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fixture_spawn)
    monkeypatch.setattr(media_metadata, "PROBE_TIMEOUT_SECONDS", 0.3 if not cancel else 10)
    monkeypatch.setattr(waveform, "WAVEFORM_TIMEOUT_SECONDS", 0.3 if not cancel else 10)
    media = tmp_path / "input.wav"
    media.write_bytes(b"fixture")
    if kind == "waveform":
        call = waveform.generate_waveform(media)
        error = waveform.WaveformGenerationError
    elif kind == "timeline":
        call = media_metadata.probe_video_timeline(media)
        error = media_metadata.MetadataProbeError
    else:
        call = media_metadata._run_probe(["ffprobe", str(media)])
        error = media_metadata.MetadataProbeError
    task = asyncio.create_task(call)
    await wait_file(ready)
    pid = int(ready.read_text())
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else error) as caught:
        await task
    if not cancel:
        if kind == "waveform":
            assert str(caught.value) == "Decoder timeout"
        else:
            assert caught.value.reason == "timeout"
    assert not Path(f"/proc/{pid}").exists()


async def test_relative_storage_root_is_anchored(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    storage = LocalStorageService("relative-storage")
    assert storage.get_path("test.wav") == tmp_path / "relative-storage/test.wav"


async def test_stdout_overflow_reaps_a_still_running_child(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "MAX_STDOUT_BYTES", 1000)
    ready = tmp_path / "pid"
    code = (
        "import os,time;from pathlib import Path;"
        f"Path({str(ready)!r}).write_text(str(os.getpid()));"
        "print('x'*2000,flush=True);time.sleep(30)"
    )
    with pytest.raises(ToolError, match="output exceeded its limit"):
        await runner.run_stdout(python(code))
    assert not Path(f"/proc/{ready.read_text()}").exists()


async def test_repeated_cancellation_cannot_interrupt_group_cleanup(tmp_path, monkeypatch) -> None:
    ready = tmp_path / "pid"
    entered = asyncio.Event()
    terminate = process_policy.terminate_process_group
    async def observed_termination(proc):
        entered.set()
        await terminate(proc)

    monkeypatch.setattr(process_policy, "terminate_process_group", observed_termination)
    task = asyncio.create_task(runner.run(python(
        "import os,time,signal;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )))
    await wait_file(ready)
    task.cancel()
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not Path(f"/proc/{ready.read_text()}").exists()


async def test_cancellation_during_spawn_does_not_lose_child(monkeypatch) -> None:
    spawn = asyncio.create_subprocess_exec
    entered, release = asyncio.Event(), asyncio.Event()
    processes = []
    async def delayed_spawn(*args, **kwargs):
        proc = await spawn(*args, **kwargs)
        processes.append(proc)
        entered.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(runner.run(python("import time;time.sleep(30)")))
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None
    assert not Path(f"/proc/{processes[0].pid}").exists()


async def test_delayed_spawn_cancellation_is_bounded_and_cleanup_remains_owned(monkeypatch) -> None:
    spawn = asyncio.create_subprocess_exec
    entered, release = asyncio.Event(), asyncio.Event()
    processes = []
    async def delayed_spawn(*args, **kwargs):
        proc = await spawn(*args, **kwargs)
        processes.append(proc)
        entered.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    monkeypatch.setattr(process_policy, "REAP_TIMEOUT_SECONDS", 0.1)
    task = asyncio.create_task(runner.run(python("import time;time.sleep(30)")))
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 2)
    finally:
        release.set()
    async with asyncio.timeout(3):
        while processes[0].returncode is None or process_policy._pending_cleanup:
            await asyncio.sleep(0.01)
    assert not Path(f"/proc/{processes[0].pid}").exists()


async def test_exited_leader_does_not_leave_live_descendant(tmp_path) -> None:
    ready = tmp_path / "child"
    code = (
        "import sys,subprocess;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        f"Path({str(ready)!r}).write_text(str(child.pid))"
    )
    await runner.run(python(code))
    assert stopped(int(ready.read_text()))


async def test_missing_tool_failure_is_fixed() -> None:
    with pytest.raises(ToolError) as error:
        await runner.run(["/nonexistent-private-location/media-tool"])
    assert error.value.message == "Media processing tool unavailable."


async def test_real_ffmpeg_ffprobe_and_imagemagick(tmp_path) -> None:
    audio = tmp_path / "tone.wav"
    await runner.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                      "sine=frequency=440:duration=0.1", str(audio)])
    assert await ffmpeg.probe_duration(str(audio)) == pytest.approx(0.1)
    assert (await waveform.generate_waveform(audio)).bucket_count > 0
    image = tmp_path / "card.png"
    thumb = tmp_path / "thumb.jpg"
    await imagemagick.render_text_card(str(image), "Safe fonts", width=80, height=80)
    await imagemagick.thumbnail(str(image), str(thumb))
    assert thumb.is_file()
    policy = await runner.run_stdout(["convert", "-list", "policy"])
    assert "Resource" in policy and "rights: None" in policy and "@*" in policy


async def test_tool_failure_job_error_is_sanitized_and_publishes_nothing(
    make_project, monkeypatch,
) -> None:
    class FailingHandler:
        async def run(self, ctx):
            await runner.run(python(
                "import sys;sys.stderr.write('/private/storage/ SECRET ' * 2000);sys.exit(1)"
            ))

    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", FailingHandler())
    project = uuid.UUID(await make_project())
    async with async_session_factory() as session:
        job = await jobs.create_job(session, JobCreate(
            capability_id="media.titlecard", project_id=project, input={"text": "Test"},
        ))
        await jobs.execute_job(session, job.id)
        assert job.status == JobStatus.failed
        assert job.error == "Media processing failed."
        assert job.output is None
        assert not list(await session.scalars(select(Asset)))
        await jobs.execute_job(session, job.id)  # terminal redelivery no-op
        assert job.status == JobStatus.failed
