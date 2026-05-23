"""Command-line interface.

Workflow:
    heed init my_phrase --phrase "hey andre"
    heed record my_phrase --kind positive --count 8   # uses mic if available
    heed record my_phrase --kind negative --count 8
    heed train my_phrase
    heed test  my_phrase some_audio.wav
    heed listen my_phrase                              # needs sounddevice
    heed eval  my_phrase --positive-dir ... --negative-dir ...
    heed smoke                                         # synthetic E2E test

If your microphone is not accessible (e.g., WSL2 without PortAudio), drop
WAV files into <name>/positive/ and <name>/negative/ and skip `record`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch

from . import SAMPLE_RATE
from .audio import load_wav, log_mel, peak_normalize, prepare_clip, save_wav
from .eval import evaluate_dirs, fmt_report
from .infer import WakeWordDetector, scan_file
from .trainer import TrainerConfig, train_wake_word


# ----------------------- utilities ----------------------------------------


def _project_path(name: str) -> Path:
    return Path(name).expanduser().resolve()


def _config_path(project: Path) -> Path:
    return project / "config.json"


def _model_path(project: Path) -> Path:
    return project / "model.pt"


def _read_config(project: Path) -> dict:
    cfg_path = _config_path(project)
    if not cfg_path.exists():
        raise click.ClickException(
            f"no project at {project}. Run `heed init {project.name}` first."
        )
    return json.loads(cfg_path.read_text())


def _try_import_sounddevice():
    try:
        import sounddevice as sd  # noqa: WPS433
        return sd
    except Exception as exc:  # noqa: BLE001
        return None


# ----------------------- commands -----------------------------------------


@click.group()
@click.version_option(package_name="heed-wakeword", prog_name="heed")
def cli() -> None:
    """Tiny, custom, on-device wake-word detector."""


@cli.command()
@click.argument("name")
@click.option("--phrase", required=True, help='Wake phrase, e.g. "hey andre".')
def init(name: str, phrase: str) -> None:
    """Create a new wake-word project."""
    project = _project_path(name)
    if project.exists() and any(project.iterdir()):
        raise click.ClickException(
            f"{project} already exists and is not empty - pick another name."
        )
    project.mkdir(parents=True, exist_ok=True)
    (project / "positive").mkdir(exist_ok=True)
    (project / "negative").mkdir(exist_ok=True)
    config = {
        "phrase": phrase,
        "created": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }
    _config_path(project).write_text(json.dumps(config, indent=2))
    click.secho(f"created project at {project}", fg="green")
    click.echo(f"  phrase: {phrase}")
    click.echo(f"  next:   heed record {name} --kind positive --count 8")
    click.echo(f"          (or drop .wav files into {project / 'positive'})")


@cli.command()
@click.argument("name")
@click.option(
    "--kind",
    type=click.Choice(["positive", "negative"]),
    required=True,
    help='"positive" = the wake word; "negative" = distractor phrases.',
)
@click.option("--count", default=8, help="How many clips to record.")
@click.option(
    "--duration",
    default=1.5,
    type=float,
    help="Recording length in seconds per clip.",
)
def record(name: str, kind: str, count: int, duration: float) -> None:
    """Record clips via the microphone (requires sounddevice + working mic)."""
    project = _project_path(name)
    cfg = _read_config(project)
    sd = _try_import_sounddevice()
    target_dir = project / kind
    target_dir.mkdir(exist_ok=True)

    if sd is None:
        click.secho(
            "sounddevice is not installed or PortAudio is missing.",
            fg="yellow",
        )
        click.echo(
            f"Record {count} clip(s) of "
            f"{'the wake word ' + repr(cfg['phrase']) if kind == 'positive' else 'distractor phrases'}"
            f" on any device and copy them as .wav (16 kHz mono is ideal)"
            f" into:\n  {target_dir}\nThen run `heed train {name}`."
        )
        return

    existing = sorted(target_dir.glob("*.wav"))
    start_idx = len(existing) + 1
    prompt_phrase = (
        f'"{cfg["phrase"]}"' if kind == "positive" else "any short sentence"
    )
    click.secho(
        f"Recording {count} {kind} clip(s). Say {prompt_phrase} after each "
        f"prompt. Press Ctrl-C to stop early.",
        fg="cyan",
    )
    for i in range(count):
        click.echo(f"\n  [{start_idx + i}/{start_idx + count - 1}] Get ready…")
        for c in (3, 2, 1):
            click.echo(f"    starting in {c}…", nl=False)
            time.sleep(1)
            click.echo("\r" + " " * 30 + "\r", nl=False)
        click.secho("    RECORDING", fg="red")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", blocking=True)
        click.echo("    saved.")
        path = target_dir / f"{kind}_{start_idx + i:03d}.wav"
        save_wav(path, audio.squeeze())

    click.secho(
        f"\nWrote {count} clip(s) to {target_dir}", fg="green"
    )


@cli.command()
@click.argument("name")
@click.option("--epochs", default=35, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--aug-pos", default=40, type=int,
              help="Augmented copies per real positive (incl. the original).")
@click.option("--aug-neg", default=25, type=int,
              help="Augmented copies per real negative.")
@click.option("--target-fpr", default=0.01, type=float,
              help="Calibrate threshold for <= this FPR on validation negatives.")
@click.option("--tts-pos", default=0, type=int,
              help="Number of multi-speaker Piper TTS positives to synthesize. "
                   "Requires piper-tts + downloaded voice. 0 disables.")
@click.option("--tts-neg", "tts_neg", multiple=True,
              help="A distractor phrase to synthesize TTS negatives for. "
                   "Can be passed multiple times. Off by default.")
@click.option("--tts-neg-count", default=20, type=int,
              help="Number of TTS negatives per --tts-neg phrase.")
@click.option("--auto-distractors/--no-auto-distractors", default=True,
              help="When --tts-pos > 0, also synthesize a built-in pool of "
                   "common confusable phrases as hard negatives. On by default.")
@click.option("--kokoro-pos", default=0, type=int,
              help="Number of Kokoro TTS positives (second acoustic family). "
                   "Adds cross-TTS-family generalization. Requires "
                   "kokoro-onnx + downloaded model. Recommended ~150-300 "
                   "alongside --tts-pos. 0 disables.")
@click.option("--force-regenerate-tts", is_flag=True, default=False,
              help="Ignore cached TTS samples and re-synthesize everything. "
                   "Cache auto-invalidates on phrase/voice/count/seed changes, "
                   "so this is only needed if you want fresh randomness with "
                   "all other settings unchanged.")
@click.option("--model-size", default="small",
              type=click.Choice(["small", "medium", "large"]),
              help="Model capacity. small (~10K params) is the default; "
                   "medium (~27K) and large (~60K) give more discriminative "
                   "capacity at the cost of disk/compute (still tiny).")
def train(name: str, epochs: int, batch_size: int, aug_pos: int,
          aug_neg: int, target_fpr: float, tts_pos: int,
          tts_neg: tuple[str, ...], tts_neg_count: int,
          auto_distractors: bool, kokoro_pos: int,
          force_regenerate_tts: bool, model_size: str) -> None:
    """Train a tiny wake-word model from recorded clips."""
    project = _project_path(name)
    cfg_json = _read_config(project)
    pos_dir = project / "positive"
    neg_dir = project / "negative"
    if not any(pos_dir.glob("*.wav")):
        raise click.ClickException(f"no .wav files in {pos_dir}")
    if not any(neg_dir.glob("*.wav")):
        raise click.ClickException(f"no .wav files in {neg_dir}")

    if tts_pos > 0 or tts_neg:
        from .tts import is_voice_available
        if not is_voice_available():
            raise click.ClickException(
                "TTS requested but voice model is not downloaded. "
                "Run `heed download-tts` first."
            )
    if kokoro_pos > 0:
        from .tts_kokoro import (is_kokoro_importable,
                                 is_voice_available as kokoro_avail)
        if not is_kokoro_importable():
            raise click.ClickException(
                "--kokoro-pos requested but kokoro-onnx is not installed.\n"
                "  → pip install kokoro-onnx"
            )
        if not kokoro_avail():
            raise click.ClickException(
                "kokoro voices not downloaded.\n"
                "  → heed download-kokoro"
            )

    distractor_phrases = list(tts_neg)
    wake_phrase_lower = cfg_json.get("phrase", "").strip().lower()
    if tts_pos > 0 and auto_distractors:
        from .tts import DEFAULT_NEAR_DISTRACTOR_PHRASES, phonetic_neighbor_distractors
        for d in DEFAULT_NEAR_DISTRACTOR_PHRASES:
            if d.lower() == wake_phrase_lower:
                continue   # never include the wake phrase itself
            if d not in distractor_phrases:
                distractor_phrases.append(d)
        # Wake-phrase-specific phonetic neighbors - fixes "hey *" over-firing
        for n in phonetic_neighbor_distractors(wake_phrase_lower, max_neighbors=60):
            if n != wake_phrase_lower and n not in distractor_phrases:
                distractor_phrases.append(n)
        click.echo(
            f"auto-distractors on → {len(distractor_phrases)} "
            f"distractor phrases × {tts_neg_count} samples each = "
            f"{len(distractor_phrases) * tts_neg_count} TTS hard negatives"
        )

    trainer_cfg = TrainerConfig(
        phrase=cfg_json.get("phrase", ""),
        epochs=epochs,
        batch_size=batch_size,
        aug_positives_per_real=aug_pos,
        aug_negatives_per_real=aug_neg,
        threshold_target_fpr=target_fpr,
        tts_positives=tts_pos,
        tts_negative_phrases=distractor_phrases,
        tts_negatives_per_phrase=tts_neg_count,
        kokoro_positives=kokoro_pos,
        force_regenerate_tts=force_regenerate_tts,
        model_size=model_size,
    )
    artifact = train_wake_word(
        positive_dir=pos_dir,
        negative_dir=neg_dir,
        output_path=_model_path(project),
        phrase=cfg_json.get("phrase", ""),
        cfg=trainer_cfg,
        log_fn=click.echo,
    )
    click.secho(
        f"\nDone. model = {_model_path(project)}\n"
        f"  threshold = {artifact.threshold:.3f}\n"
        f"  val accuracy = {artifact.val_accuracy:.3f}\n"
        f"  TPR @ FPR≤{target_fpr:.2f} = {artifact.val_tpr_at_target_fpr:.2f}\n"
        f"  parameters = {artifact.n_params}\n"
        f"  trained in {artifact.seconds:.1f}s",
        fg="green",
    )


@cli.command()
@click.argument("name")
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--threshold", type=float, default=None,
              help="Override the calibrated threshold for this run.")
@click.option("--hop", type=float, default=0.1,
              help="Hop length in seconds for sliding-window scan.")
def test(name: str, audio_path: str, threshold: float | None, hop: float) -> None:
    """Score an audio file with a trained model."""
    project = _project_path(name)
    model_path = _model_path(project)
    if not model_path.exists():
        raise click.ClickException(
            f"no model at {model_path}. Run `heed train {name}` first."
        )
    audio = load_wav(audio_path)
    duration = audio.numel() / SAMPLE_RATE
    click.echo(f"scanning {audio_path}  ({duration:.2f}s)")
    results = scan_file(model_path, audio, hop_seconds=hop,
                       threshold_override=threshold)
    triggers = [r for r in results if r["triggered"]]
    peak = max((r["prob"] for r in results), default=0.0)
    peak_ema = max((r["ema"] for r in results), default=0.0)
    click.echo(f"  peak prob = {peak:.3f}    peak EMA = {peak_ema:.3f}")
    if triggers:
        click.secho(f"  {len(triggers)} trigger(s) detected", fg="green")
        for t in triggers:
            click.echo(f"    at t={t['time']:.2f}s  prob={t['prob']:.3f}  ema={t['ema']:.3f}")
    else:
        click.secho("  no triggers", fg="yellow")


@cli.command()
@click.argument("name")
@click.option("--threshold", type=float, default=None)
def listen(name: str, threshold: float | None) -> None:
    """Stream audio from the microphone and print triggers in real time."""
    project = _project_path(name)
    model_path = _model_path(project)
    if not model_path.exists():
        raise click.ClickException(
            f"no model at {model_path}. Run `heed train {name}` first."
        )
    sd = _try_import_sounddevice()
    if sd is None:
        raise click.ClickException(
            "sounddevice is not installed or PortAudio is missing - "
            "use `heed test <name> <file.wav>` instead."
        )

    detector = WakeWordDetector(model_path, threshold_override=threshold)
    chunk = int(0.1 * SAMPLE_RATE)
    click.secho(
        f"listening for {detector.phrase!r}. threshold={detector.threshold:.3f}. "
        f"Ctrl-C to stop.",
        fg="cyan",
    )
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk) as stream:
            t0 = time.monotonic()
            while True:
                data, _ = stream.read(chunk)
                res = detector.step(data[:, 0])
                t = time.monotonic() - t0
                if res["triggered"]:
                    click.secho(
                        f"  [{t:6.2f}s] TRIGGER  prob={res['prob']:.3f} ema={res['ema']:.3f}",
                        fg="green", bold=True,
                    )
                elif not res["gated"] and res["prob"] > 0.3:
                    click.echo(
                        f"  [{t:6.2f}s]   prob={res['prob']:.3f} ema={res['ema']:.3f}"
                    )
    except KeyboardInterrupt:
        click.echo("\nstopped.")


@cli.command()
@click.argument("name")
@click.option("--positive-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--negative-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--threshold", type=float, default=None)
def eval(name: str, positive_dir: str | None, negative_dir: str | None,
         threshold: float | None) -> None:
    """Evaluate a model on labelled .wav directories."""
    project = _project_path(name)
    model_path = _model_path(project)
    if not model_path.exists():
        raise click.ClickException(
            f"no model at {model_path}. Run `heed train {name}` first."
        )
    if not positive_dir and not negative_dir:
        raise click.ClickException(
            "supply --positive-dir and/or --negative-dir for evaluation."
        )
    report = evaluate_dirs(model_path, positive_dir, negative_dir,
                           threshold_override=threshold)
    click.echo(fmt_report(report))


# ----------------------- synthetic smoke test ------------------------------


def _synthetic_clip(
    *,
    fundamental: float,
    formants: list[tuple[float, float]],
    duration: float = 1.0,
    onset: float = 0.25,
    offset_silence: float = 0.25,
    seed: int = 0,
) -> torch.Tensor:
    """Create a fake wake-word: a harmonic stack with formant emphasis.

    More speech-like than a pure tone - has fundamental + harmonics, formant
    peaks roughly where vowels sit, plus amplitude envelope and noise floor.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    audio = np.zeros(n, dtype=np.float32)
    # harmonic stack
    for k in range(1, 16):
        gain = 0.5 / k
        # emphasize harmonics inside formant peaks
        f = fundamental * k
        boost = 0.0
        for fpeak, bw in formants:
            boost += np.exp(-((f - fpeak) ** 2) / (2 * bw**2))
        gain *= 1.0 + 2.0 * boost
        audio += gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # amplitude envelope (attack + sustain + release)
    env = np.ones(n)
    burst_start = int(onset * SAMPLE_RATE)
    burst_end = n - int(offset_silence * SAMPLE_RATE)
    env[:burst_start] = np.linspace(0, 1, burst_start)
    env[burst_end:] = np.linspace(1, 0, n - burst_end)
    audio = audio * env
    # noise floor
    audio = audio + 0.005 * rng.standard_normal(n).astype(np.float32)
    # normalize
    audio = audio / (np.abs(audio).max() + 1e-6) * 0.8
    return torch.from_numpy(audio.astype(np.float32))


@cli.command()
def doctor() -> None:
    """Diagnose the install. Reports Python, torch/CUDA, piper-tts, voice files."""
    import sys
    click.echo(f"python:        {sys.executable}")
    click.echo(f"sys.prefix:    {sys.prefix}")
    click.echo(f"python ver:    {sys.version.split()[0]}")

    # torch
    try:
        import torch
        click.echo(f"torch:         {torch.__version__}")
        click.echo(f"  cuda:         {torch.cuda.is_available()}"
                   + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception as exc:  # noqa: BLE001
        click.secho(f"torch:         FAIL - {exc}", fg="red")

    # soundfile
    try:
        import soundfile
        click.echo(f"soundfile:     {soundfile.__version__}")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"soundfile:     FAIL - {exc}", fg="red")

    # scipy
    try:
        import scipy
        click.echo(f"scipy:         {scipy.__version__}")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"scipy:         FAIL - {exc}", fg="red")

    # flask
    try:
        import flask  # noqa: F401
        from importlib.metadata import version as _v
        click.echo(f"flask:         {_v('flask')}")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"flask:         not installed ({exc})", fg="yellow")

    # sounddevice
    try:
        import sounddevice  # noqa: F401
        click.echo("sounddevice:   ok")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"sounddevice:   not installed / PortAudio missing ({exc})",
                    fg="yellow")

    # onnxruntime - checked BEFORE piper because piper depends on it
    click.echo()
    click.secho("onnxruntime (piper + kokoro share this runtime):", bold=True)
    ort_packages = _installed_onnxruntime_packages(sys.executable)
    if ort_packages:
        for pkg, ver in ort_packages.items():
            click.echo(f"  {pkg}:  {ver}")
        if "onnxruntime" in ort_packages and "onnxruntime-gpu" in ort_packages:
            click.secho(
                "  ⚠ both onnxruntime AND onnxruntime-gpu are installed.\n"
                "    They conflict on Windows (same DLL name, different builds).\n"
                "    Uninstall both then install ONE:\n"
                f"      {sys.executable} -m pip uninstall onnxruntime onnxruntime-gpu\n"
                f"      {sys.executable} -m pip install onnxruntime-gpu",
                fg="red")
    else:
        click.secho("  (no onnxruntime* packages found via pip)", fg="yellow")
    try:
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        click.echo(f"  import:      ok - version {ort.__version__}")
        click.echo(f"  providers:   {providers}")
        # Explicit GPU-acceleration verdict - this is the single most common
        # reason TTS is 6-10x slower than it should be on a CUDA-capable box.
        has_cuda_provider = any("CUDA" in p for p in providers)
        try:
            import torch  # noqa: WPS433
            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda = False
        if torch_cuda and has_cuda_provider:
            click.secho("  GPU accel:   ✓ onnxruntime can use CUDA - "
                        "Piper/Kokoro will run on GPU", fg="green")
        elif torch_cuda and not has_cuda_provider:
            click.secho(
                "  GPU accel:   ✗ torch sees CUDA but onnxruntime is CPU-only.\n"
                "    This is THE common reason TTS synthesis is slow on a 4090.\n"
                "    Fix (in this exact order):\n"
                f"      {sys.executable} -m pip uninstall -y onnxruntime onnxruntime-gpu\n"
                f"      {sys.executable} -m pip install onnxruntime-gpu\n"
                "    Then re-install piper-tts WITHOUT pulling CPU runtime back in:\n"
                f"      {sys.executable} -m pip install --upgrade --no-deps piper-tts\n"
                "    Verify by re-running `heed doctor` - providers should\n"
                "    include 'CUDAExecutionProvider'.",
                fg="yellow")
        elif not torch_cuda:
            click.echo("  GPU accel:   (no CUDA-capable torch detected - CPU mode)")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"  import:      FAIL - {type(exc).__name__}: {exc}", fg="red")
        if "DLL load failed" in str(exc):
            click.secho(
                "  → Windows DLL load failure. Try, in order:\n"
                "    1. Install the Microsoft Visual C++ Redistributable (x64):\n"
                "       https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "       (this is by far the most common fix on miniconda Win10/11)\n"
                "    2. Then re-install onnxruntime cleanly:\n"
                f"       {sys.executable} -m pip uninstall onnxruntime onnxruntime-gpu\n"
                f"       {sys.executable} -m pip install --no-cache-dir onnxruntime\n"
                "    3. If still failing, try an older known-good version:\n"
                f"       {sys.executable} -m pip install --no-cache-dir onnxruntime==1.18.1",
                fg="yellow")

    # piper-tts itself
    click.echo()
    click.secho("piper-tts (TTS augmentation):", bold=True)
    try:
        import piper  # noqa: F401
        from piper.voice import PiperVoice  # noqa: F401
        from piper.config import SynthesisConfig  # noqa: F401
        click.secho("  import:      ok", fg="green")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"  import:      FAIL - {type(exc).__name__}: {exc}", fg="red")
        msg = str(exc)
        if "onnxruntime" in msg:
            click.echo("  → piper failed because onnxruntime is unusable - see the onnxruntime section above.")
        else:
            click.echo(f"  → try:  {sys.executable} -m pip install --force-reinstall piper-tts")

    # voice files
    from .tts import voice_paths, is_voice_available
    onnx_path, _ = voice_paths()
    if is_voice_available():
        size_mb = onnx_path.stat().st_size / 1e6
        click.secho(f"  voice file:  {onnx_path} ({size_mb:.1f} MB)", fg="green")
    else:
        click.secho(f"  voice file:  NOT FOUND at {onnx_path}", fg="yellow")
        click.echo("  → run `heed download-tts`")

    # kokoro-onnx - second TTS engine, used for cross-TTS generalization test
    click.echo()
    click.secho("kokoro-onnx (second TTS family - cross-TTS test):", bold=True)
    try:
        from .tts_kokoro import (voice_paths as k_paths,
                                 is_voice_available as k_avail,
                                 is_kokoro_importable)
        if is_kokoro_importable():
            click.secho("  import:      ok", fg="green")
            k_onnx, k_voices = k_paths()
            if k_avail():
                k_size_mb = k_onnx.stat().st_size / 1e6
                click.secho(f"  model file:  {k_onnx} ({k_size_mb:.1f} MB)",
                            fg="green")
            else:
                click.secho(f"  model file:  NOT FOUND at {k_onnx}",
                            fg="yellow")
                click.echo("  → run `heed download-kokoro`")
        else:
            click.secho("  import:      not installed", fg="yellow")
            click.echo(f"  → {sys.executable} -m pip install kokoro-onnx")
    except Exception as exc:  # noqa: BLE001
        click.secho(f"  module:      FAIL - {type(exc).__name__}: {exc}",
                    fg="red")


def _installed_onnxruntime_packages(python_exe: str) -> dict[str, str]:
    """Use pip to detect every onnxruntime* package installed in this interpreter."""
    import subprocess
    found: dict[str, str] = {}
    try:
        out = subprocess.run(
            [python_exe, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=20,
        )
        for line in out.stdout.splitlines():
            if "==" not in line:
                continue
            name, ver = line.split("==", 1)
            if name.lower().startswith("onnxruntime"):
                found[name.lower()] = ver.strip()
    except Exception:
        pass
    return found


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 to expose).")
@click.option("--port", default=7777, type=int, help="Bind port.")
@click.option("--workspace", default=None,
              type=click.Path(file_okay=False),
              help="Directory holding projects. Defaults to the current dir.")
def ui(host: str, port: int, workspace: str | None) -> None:
    """Launch the local browser UI for recording, training, and live-testing."""
    try:
        from .web import run_server
    except ImportError as exc:
        raise click.ClickException(
            f"web UI needs flask. Install it with `pip install flask`. ({exc})"
        )
    from pathlib import Path
    run_server(host=host, port=port,
               workspace=Path(workspace).resolve() if workspace else None)


@cli.command(name="download-tts")
@click.option("--voice", default="en_US-libritts_r-medium",
              help="Voice model name (Piper format).")
def download_tts(voice: str) -> None:
    """Download a Piper multi-speaker voice model for TTS augmentation."""
    try:
        from .tts import download_voice, voice_paths, is_voice_available
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    if is_voice_available(voice):
        onnx, _ = voice_paths(voice)
        click.secho(f"voice {voice!r} already present at {onnx.parent}",
                    fg="green")
        return
    click.echo(f"downloading voice {voice!r}…")
    download_voice(voice, progress_fn=click.echo)
    click.secho(f"\nVoice ready. You can now pass --tts-pos N to `train`.",
                fg="green")


@cli.command(name="download-kokoro")
def download_kokoro() -> None:
    """Download the Kokoro ONNX model + voices file (~340 MB total).

    Kokoro is a second multi-speaker TTS engine used for cross-TTS
    generalization tests and (optionally) additional training augmentation.
    Different acoustic family than Piper, so a model that triggers on
    Kokoro voices it has never seen is meaningfully more "real-human-like"
    than one that only passes the same-family Piper held-out test.
    """
    try:
        from .tts_kokoro import (download_voice, voice_paths, is_voice_available,
                                 is_kokoro_importable, DEFAULT_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"kokoro module failed to load: {exc}")
    if not is_kokoro_importable():
        raise click.ClickException(
            "kokoro-onnx is not installed in this environment.\n"
            "  → pip install kokoro-onnx\n"
            "Then re-run `heed download-kokoro`."
        )
    if is_voice_available():
        onnx, voices = voice_paths()
        click.secho(f"kokoro voices already present at {onnx.parent}",
                    fg="green")
        return
    click.echo(f"downloading kokoro model {DEFAULT_MODEL_NAME!r}…")
    download_voice(progress_fn=click.echo)
    click.secho(f"\nKokoro ready. You can now run `heed cross-tts-test "
                f"<project>` to evaluate cross-TTS generalization.",
                fg="green")


@cli.command(name="export")
@click.argument("name")
@click.option("--output", default=None,
              help="Output directory (default: <project>/export/).")
@click.option("--int8/--no-int8", default=True,
              help="Also produce an INT8-quantized variant (~25% the size, "
                   "lossy but typically indistinguishable on our test set).")
@click.option("--tflite/--no-tflite", default=True,
              help="Also produce a TFLite/LiteRT variant if `litert-torch` is "
                   "installed. TFLite is the preferred format for Android "
                   "NNAPI and iOS Core ML delegate paths to NPU acceleration. "
                   "Gracefully skipped if the converter isn't available.")
@click.option("--opset", default=17, type=int,
              help="ONNX opset version to target (default 17).")
def export_cmd(name: str, output: str | None, int8: bool, tflite: bool,
               opset: int) -> None:
    """Export a trained model to ONNX (and optionally TFLite) for deployment.

    Produces wake.onnx (+ wake.int8.onnx, + wake.tflite if litert-torch is
    installed) + wake.json metadata in the output directory. Models expect
    log-mel features as input (B, n_mels, T). See the generated
    export/README.md for deployment code in Python, Android, and iOS, and
    a description of the preprocessing chain to reproduce in JS / Swift / etc.
    """
    project = _project_path(name)
    model_path = _model_path(project)
    if not model_path.exists():
        raise click.ClickException(
            f"no model at {model_path}. Run `heed train {name}` first."
        )
    out_dir = Path(output) if output else (project / "export")

    from .export import export_to_onnx, export_to_tflite
    try:
        result = export_to_onnx(model_path, out_dir, int8=int8, opset=opset,
                                log_fn=click.echo)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    if tflite:
        try:
            export_to_tflite(model_path, out_dir, log_fn=click.echo)
        except ImportError as exc:
            click.echo(
                f"  [skip] TFLite export not available: {exc}\n"
                f"         Install with: pip install litert-torch"
            )
        except RuntimeError as exc:
            click.echo(f"  [warn] TFLite export failed: {exc}")

    click.echo()
    click.secho(f"✓ exported to {out_dir}", fg="green", bold=True)


@cli.command(name="cross-tts-test")
@click.argument("name")
def cross_tts_test(name: str) -> None:
    """Run the trained model against Kokoro voices it has never seen.

    The existing cross-speaker test uses 3 held-out **Piper** voices. The
    model has seen ~900 OTHER Piper voices, so it can pass that test while
    still failing on real humans whose acoustics differ from Piper's TTS
    family. This test re-runs the same idea with Kokoro voices - a
    different neural family entirely - which is much closer to the real
    "does this generalize to humans?" question.
    """
    project = _project_path(name)
    model_path = _model_path(project)
    if not model_path.exists():
        raise click.ClickException(
            f"no model at {model_path}. Run `heed train {name}` first."
        )
    cfg_json = _read_config(project)
    phrase = cfg_json.get("phrase", "").strip()
    if not phrase:
        raise click.ClickException("project has no phrase set.")

    try:
        from .tts_kokoro import (HELDOUT_VOICE_IDS, is_kokoro_importable,
                                 is_voice_available, synthesize_from_voices)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"kokoro module failed to load: {exc}")
    if not is_kokoro_importable():
        raise click.ClickException(
            "kokoro-onnx is not installed.  → pip install kokoro-onnx"
        )
    if not is_voice_available():
        raise click.ClickException(
            "kokoro voices not downloaded.  → heed download-kokoro"
        )

    from .trainer import load_model
    model, payload = load_model(model_path)
    threshold = float(payload["threshold"])

    false_trigger_phrases = ["hey siri", "hey google", "good morning",
                             "see you later"]

    def _score(audio: torch.Tensor) -> float:
        prepped = prepare_clip(audio)
        mel = log_mel(prepped)
        with torch.no_grad():
            return float(torch.sigmoid(model(mel)).item())

    click.echo(f"synthesizing held-out Kokoro voices "
               f"({', '.join(HELDOUT_VOICE_IDS)})…")
    wake_clips = synthesize_from_voices(phrase, HELDOUT_VOICE_IDS)

    pos_dir = project / "kokoro_eval"
    pos_dir.mkdir(exist_ok=True)
    for old in pos_dir.glob("*.wav"):
        old.unlink()

    click.secho(f"\nphrase = {phrase!r}   threshold = {threshold:.3f}",
                bold=True)
    click.secho("\npositives (wake phrase, should TRIGGER):", fg="cyan")
    n_pos_fire = 0
    for vid, clip in zip(HELDOUT_VOICE_IDS, wake_clips):
        score = _score(clip)
        save_wav(pos_dir / f"pos_{vid}.wav", clip)
        fire = score > threshold
        if fire:
            n_pos_fire += 1
        mark = "✓" if fire else "✗"
        color = "green" if fire else "red"
        click.secho(f"  {mark} {vid:14s}  score={score:.3f}", fg=color)

    click.secho("\nnegatives (distractors, should NOT trigger):", fg="cyan")
    n_neg_fire = 0
    n_neg_total = 0
    for distractor in false_trigger_phrases:
        clips = synthesize_from_voices(distractor, HELDOUT_VOICE_IDS)
        for vid, clip in zip(HELDOUT_VOICE_IDS, clips):
            score = _score(clip)
            safe = "".join(c if c.isalnum() else "_" for c in distractor.lower())[:30]
            save_wav(pos_dir / f"neg_{safe}_{vid}.wav", clip)
            fire = score > threshold
            n_neg_total += 1
            if fire:
                n_neg_fire += 1
            mark = "✗" if fire else "✓"
            color = "red" if fire else "green"
            click.secho(f"  {mark} {vid:14s} / {distractor!r:25s}  "
                        f"score={score:.3f}", fg=color)

    if n_pos_fire >= 2 and n_neg_fire <= 2:
        verdict, vcolor = "GENERALIZES (cross-TTS)", "green"
    elif n_pos_fire == 0:
        verdict, vcolor = "TTS-FAMILY-LOCKED", "red"
    else:
        verdict, vcolor = "BORDERLINE", "yellow"

    click.echo()
    click.secho(f"verdict: {verdict}", fg=vcolor, bold=True)
    click.echo(f"  positives triggered: {n_pos_fire} / {len(HELDOUT_VOICE_IDS)}")
    click.echo(f"  false triggers:      {n_neg_fire} / {n_neg_total}")
    click.echo(f"  clips saved to:      {pos_dir}")


@cli.command()
@click.option("--out", default="/tmp/heed_smoke",
              type=click.Path(), help="Workdir for synthetic data.")
@click.option("--keep/--no-keep", default=False,
              help="Keep the synthetic project directory after the test.")
def smoke(out: str, keep: bool) -> None:
    """Run a synthetic end-to-end test (no microphone needed).

    Generates two distinguishable synthetic 'phrases' (different fundamental
    + formant patterns), trains a model on a few of each, and evaluates on
    held-out variations. Exits non-zero if the model fails to learn.
    """
    workdir = Path(out)
    if workdir.exists():
        click.echo(f"clearing {workdir}")
        import shutil
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    project = workdir / "synthetic"
    project.mkdir()
    (project / "positive").mkdir()
    (project / "negative").mkdir()
    _config_path(project).write_text(json.dumps(
        {"phrase": "synthetic wake", "created": "smoke"}, indent=2))

    pos_formants = [(700, 90), (1200, 100), (2600, 120)]  # /a/-like vowel
    neg_formants = [(300, 80), (870, 90), (2200, 100)]    # /u/-like vowel

    click.echo("generating synthetic clips…")
    n_per = 10
    for i in range(n_per):
        clip = _synthetic_clip(
            fundamental=120 + i * 8,
            formants=pos_formants,
            onset=0.20 + 0.04 * (i % 3),
            offset_silence=0.25 + 0.03 * (i % 2),
            seed=100 + i,
        )
        save_wav(project / "positive" / f"pos_{i:03d}.wav", clip)
    for i in range(n_per):
        clip = _synthetic_clip(
            fundamental=180 + i * 7,
            formants=neg_formants,
            onset=0.18 + 0.03 * (i % 4),
            offset_silence=0.22 + 0.05 * (i % 3),
            seed=200 + i,
        )
        save_wav(project / "negative" / f"neg_{i:03d}.wav", clip)

    click.echo("training…")
    # Deterministic smoke: CPU + single-thread so the verdict reproduces run to
    # run. On tiny synthetic data, GPU/cuDNN nondeterminism otherwise swings the
    # held-out AUC anywhere from ~0.78 to 1.0. The trainer seeds itself, so CPU
    # single-thread pins the result. CI runs on CPU regardless.
    import torch
    torch.set_num_threads(1)
    cfg = TrainerConfig(epochs=25, aug_positives_per_real=20,
                        aug_negatives_per_real=15, val_split=0.2,
                        device="cpu")
    artifact = train_wake_word(
        positive_dir=project / "positive",
        negative_dir=project / "negative",
        output_path=project / "model.pt",
        phrase="synthetic wake",
        cfg=cfg,
        log_fn=click.echo,
    )

    # held-out eval: brand new synthetic clips
    held_pos = workdir / "held_pos"
    held_neg = workdir / "held_neg"
    held_pos.mkdir()
    held_neg.mkdir()
    for i in range(8):
        save_wav(held_pos / f"hp_{i}.wav",
                 _synthetic_clip(fundamental=110 + i * 9,
                                 formants=pos_formants, seed=500 + i))
        save_wav(held_neg / f"hn_{i}.wav",
                 _synthetic_clip(fundamental=190 + i * 6,
                                 formants=neg_formants, seed=600 + i))

    click.echo("\nheld-out evaluation:")
    report = evaluate_dirs(project / "model.pt", held_pos, held_neg)
    click.echo(fmt_report(report))

    # Judge on AUC (rank-based separability), not TPR/FPR at the calibrated
    # threshold. On only 8 synthetic held-out clips the threshold calibration is
    # noisy and FPR swings run-to-run, but AUC is stable (~0.95-0.98 on a healthy
    # pipeline, ~0.5 if broken) - so this is a reliable CI signal, not a flaky one.
    ok = report.auc_approx >= 0.85
    if ok:
        click.secho(
            f"smoke test PASSED ✓  (AUC {report.auc_approx:.3f})",
            fg="green", bold=True,
        )
    else:
        click.secho(
            f"smoke test FAILED - AUC {report.auc_approx:.3f} < 0.85 "
            f"(tpr={report.tpr:.2f}, fpr={report.fpr:.2f})",
            fg="red", bold=True,
        )

    if not keep:
        import shutil
        shutil.rmtree(workdir)
        click.echo(f"removed {workdir}")
    else:
        click.echo(f"kept {workdir}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cli()
