"""Local browser UI for recording, training, and live-testing wake words.

Launches a small Flask app at 127.0.0.1:7777 (configurable). The browser
captures audio via the Web Audio API and POSTs raw float32 PCM to the
server, which resamples to 16 kHz and saves WAV files into the standard
project layout used by the CLI (positive/, negative/, model.pt).
"""

from __future__ import annotations

import io
import json
import math
import secrets
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from . import SAMPLE_RATE
from .audio import load_wav, log_mel, prepare_clip, save_wav
from .infer import WakeWordDetector
from .trainer import TrainerConfig, load_model, train_wake_word
from .tts import DEFAULT_NEAR_DISTRACTOR_PHRASES, is_piper_importable, is_voice_available


# ----- workspace + project helpers ------------------------------------------


def _workspace(app: Flask) -> Path:
    return Path(app.config["TINYWW_WORKSPACE"])


def _project_dir(app: Flask, name: str) -> Path:
    safe = name.strip().replace("/", "_").replace("\\", "_")
    return _workspace(app) / safe


def _project_config(project: Path) -> dict:
    cfg_path = project / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _list_projects(app: Flask) -> list[dict]:
    ws = _workspace(app)
    projects = []
    for p in sorted(ws.iterdir()) if ws.exists() else []:
        if not p.is_dir():
            continue
        cfg_path = p / "config.json"
        if not cfg_path.exists():
            continue
        cfg = _project_config(p)
        n_pos = len(list((p / "positive").glob("*.wav"))) if (p / "positive").exists() else 0
        n_neg = len(list((p / "negative").glob("*.wav"))) if (p / "negative").exists() else 0
        has_model = (p / "model.pt").exists()
        projects.append({
            "name": p.name,
            "phrase": cfg.get("phrase", ""),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "has_model": has_model,
        })
    return projects


# ----- training jobs (background threads) -----------------------------------


_train_state: dict[str, dict] = {}
_train_lock = threading.Lock()


def _train_status(name: str) -> dict:
    with _train_lock:
        return _train_state.get(name, {"state": "idle", "logs": []})


def _train_in_background(app, name: str, cfg: TrainerConfig) -> None:
    project = _project_dir(app, name)

    def log(msg: str) -> None:
        line = str(msg)
        with _train_lock:
            _train_state.setdefault(name, {"state": "running", "logs": []})
            _train_state[name]["logs"].append(line)
        print(f"[train:{name}] {line}", flush=True)

    def runner():
        try:
            with _train_lock:
                _train_state[name] = {"state": "running", "logs": [], "started": time.time()}
            artifact = train_wake_word(
                project / "positive",
                project / "negative",
                project / "model.pt",
                phrase=cfg.phrase,
                cfg=cfg,
                log_fn=log,
            )
            with _train_lock:
                _train_state[name]["state"] = "done"
                _train_state[name]["artifact"] = asdict(artifact)
                _train_state[name]["elapsed"] = time.time() - _train_state[name]["started"]
        except Exception as exc:  # noqa: BLE001
            import traceback
            log("ERROR ----")
            for line in str(exc).splitlines():
                log("  " + line)
            log("traceback ----")
            for line in traceback.format_exc().splitlines():
                log("  " + line)
            with _train_lock:
                _train_state[name]["state"] = "error"
                _train_state[name]["error"] = str(exc)

    threading.Thread(target=runner, daemon=True).start()


# ----- live-listen sessions (one detector per session) ----------------------


_live_sessions: dict[str, dict] = {}
_live_lock = threading.Lock()


def _gc_live_sessions() -> None:
    now = time.monotonic()
    with _live_lock:
        dead = [sid for sid, s in _live_sessions.items()
                if now - s["last_seen"] > 30.0]
        for sid in dead:
            _live_sessions.pop(sid, None)


# ----- audio handling -------------------------------------------------------


def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    g = math.gcd(src_sr, SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = src_sr // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


# ----- Flask app ------------------------------------------------------------


def create_app(workspace: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["TINYWW_WORKSPACE"] = str(workspace or Path.cwd())

    # ---- HTML + assets ----
    @app.route("/")
    def index() -> Response:
        return Response(_INDEX_HTML, mimetype="text/html", headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })

    @app.route("/favicon.ico")
    def favicon() -> Response:
        # Tiny 1x1 transparent ICO so the browser stops 404-ing on us.
        ico = bytes.fromhex(
            "00000100010001010000010020003000000016000000280000000100000002"
            "00000001000000010000200000000000040000000000000000000000000000"
            "00000000000000000000000000000000000000"
        )
        return Response(ico, mimetype="image/x-icon",
                        headers={"Cache-Control": "public, max-age=86400"})

    # ---- project CRUD ----
    @app.route("/api/projects", methods=["GET"])
    def api_list():
        piper_ok = is_piper_importable()
        voice_ok = is_voice_available()
        return jsonify({"projects": _list_projects(app),
                        "workspace": str(_workspace(app)),
                        "cuda": torch.cuda.is_available(),
                        "tts_available": piper_ok and voice_ok,
                        "piper_importable": piper_ok,
                        "voice_downloaded": voice_ok,
                        "near_distractors": DEFAULT_NEAR_DISTRACTOR_PHRASES})

    @app.route("/api/projects", methods=["POST"])
    def api_create():
        data = request.get_json(force=True)
        name = (data.get("name") or "").strip()
        phrase = (data.get("phrase") or "").strip()
        if not name or not phrase:
            return jsonify({"error": "name and phrase are required"}), 400
        if "/" in name or "\\" in name:
            return jsonify({"error": "name cannot contain slashes"}), 400
        project = _project_dir(app, name)
        if project.exists():
            return jsonify({"error": f"project {name!r} already exists"}), 400
        project.mkdir(parents=True)
        (project / "positive").mkdir()
        (project / "negative").mkdir()
        (project / "config.json").write_text(json.dumps({
            "phrase": phrase,
            "created": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
        }, indent=2))
        return jsonify({"ok": True, "name": name})

    @app.route("/api/projects/<name>", methods=["GET"])
    def api_get(name: str):
        project = _project_dir(app, name)
        if not project.exists():
            return jsonify({"error": "not found"}), 404
        cfg = _project_config(project)
        positives = sorted([p.name for p in (project / "positive").glob("*.wav")])
        negatives = sorted([p.name for p in (project / "negative").glob("*.wav")])
        info = {"phrase": cfg.get("phrase", ""),
                "positives": positives,
                "negatives": negatives,
                "has_model": (project / "model.pt").exists()}
        # If model exists, include metadata
        model_path = project / "model.pt"
        if model_path.exists():
            sidecar = model_path.with_suffix(".json")
            if sidecar.exists():
                info["model_info"] = json.loads(sidecar.read_text())
        return jsonify(info)

    @app.route("/api/projects/<name>/clip/<kind>/<filename>", methods=["GET"])
    def api_get_clip(name: str, kind: str, filename: str):
        if kind not in ("positive", "negative"):
            return jsonify({"error": "bad kind"}), 400
        project = _project_dir(app, name)
        clip_dir = project / kind
        return send_from_directory(clip_dir, filename, mimetype="audio/wav")

    @app.route("/api/projects/<name>/tts_cache", methods=["GET"])
    def api_list_tts_cache(name: str):
        """List TTS samples saved during the last training, for inspection."""
        project = _project_dir(app, name)
        cache = project / "tts_cache"
        if not cache.exists():
            return jsonify({"positives": [], "negatives": []})
        positives = sorted([
            p.name for p in (cache / "positive").glob("*.wav")
        ]) if (cache / "positive").exists() else []
        neg_root = cache / "negative"
        negatives: dict[str, list[str]] = {}
        if neg_root.exists():
            for phrase_dir in sorted(neg_root.iterdir()):
                if not phrase_dir.is_dir():
                    continue
                negatives[phrase_dir.name] = sorted(p.name for p in phrase_dir.glob("*.wav"))
        return jsonify({"positives": positives, "negatives": negatives})

    @app.route("/api/projects/<name>/tts_clip/<path:filepath>", methods=["GET"])
    def api_get_tts_clip(name: str, filepath: str):
        """Serve a TTS sample. `filepath` is e.g.
        'positive/tts_pos_0001.wav' or 'negative/hey_there/tts_neg_0002.wav'."""
        if ".." in filepath or filepath.startswith("/"):
            return jsonify({"error": "bad path"}), 400
        parts = filepath.split("/")
        if not parts or parts[0] not in ("positive", "negative"):
            return jsonify({"error": "bad subkind"}), 400
        project = _project_dir(app, name)
        full = project / "tts_cache" / filepath
        if not full.exists():
            return jsonify({"error": "not found"}), 404
        return send_from_directory(full.parent, full.name, mimetype="audio/wav")

    @app.route("/api/projects/<name>/clip", methods=["DELETE"])
    def api_delete_clip(name: str):
        data = request.get_json(force=True)
        kind = data.get("kind")
        filename = data.get("filename")
        if kind not in ("positive", "negative") or not filename:
            return jsonify({"error": "kind and filename required"}), 400
        project = _project_dir(app, name)
        path = project / kind / filename
        if path.exists():
            path.unlink()
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    # ---- recording (raw float32 audio from browser) ----
    @app.route("/api/projects/<name>/record", methods=["POST"])
    def api_record(name: str):
        kind = request.args.get("kind")
        sr = int(request.args.get("sr", "0"))
        if kind not in ("positive", "negative"):
            return jsonify({"error": "kind must be positive|negative"}), 400
        if sr <= 0:
            return jsonify({"error": "sr query param required"}), 400
        project = _project_dir(app, name)
        if not project.exists():
            return jsonify({"error": "project not found"}), 404
        audio_bytes = request.get_data()
        if len(audio_bytes) < 64:
            return jsonify({"error": "too little audio"}), 400
        # Interpret as float32
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        audio = _resample_to_16k(audio, sr)
        # next index
        existing = sorted((project / kind).glob("*.wav"))
        idx = len(existing) + 1
        path = project / kind / f"{kind}_{idx:03d}.wav"
        save_wav(path, audio)
        return jsonify({"ok": True, "filename": path.name})

    @app.route("/api/projects/<name>/upload", methods=["POST"])
    def api_upload(name: str):
        kind = request.args.get("kind")
        if kind not in ("positive", "negative"):
            return jsonify({"error": "kind must be positive|negative"}), 400
        project = _project_dir(app, name)
        if not project.exists():
            return jsonify({"error": "project not found"}), 404
        files = request.files.getlist("file")
        saved = []
        for f in files:
            if not f.filename.lower().endswith(".wav"):
                continue
            data, src_sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if src_sr != SAMPLE_RATE:
                data = _resample_to_16k(data, src_sr)
            existing = sorted((project / kind).glob("*.wav"))
            idx = len(existing) + 1
            path = project / kind / f"{kind}_{idx:03d}.wav"
            save_wav(path, data)
            saved.append(path.name)
        return jsonify({"ok": True, "saved": saved})

    # ---- training ----
    @app.route("/api/projects/<name>/train", methods=["POST"])
    def api_train(name: str):
        data = request.get_json(force=True) or {}
        project = _project_dir(app, name)
        if not project.exists():
            return jsonify({"error": "project not found"}), 404
        cfg_json = _project_config(project)
        with _train_lock:
            cur = _train_state.get(name, {}).get("state")
            if cur == "running":
                return jsonify({"error": "training already running"}), 409

        cfg = TrainerConfig(
            phrase=cfg_json.get("phrase", ""),
            epochs=int(data.get("epochs", 100)),
            aug_positives_per_real=int(data.get("aug_pos", 40)),
            aug_negatives_per_real=int(data.get("aug_neg", 25)),
            tts_positives=int(data.get("tts_pos", 0)),
            tts_negatives_per_phrase=int(data.get("tts_neg_count", 20)),
            kokoro_positives=int(data.get("kokoro_pos", 200)),
            threshold_target_fpr=float(data.get("target_fpr", 0.01)),
            device=data.get("device", "auto"),
            partial_negatives=bool(data.get("partial_negatives", True)),
            specaugment=bool(data.get("specaugment", True)),
            use_rir=bool(data.get("use_rir", True)),
            use_parametric_noise=bool(data.get("use_parametric_noise", True)),
            spectral_matching=bool(data.get("spectral_matching", True)),
            loss_function=str(data.get("loss_function", "focal")),
            end_aligned_variants=bool(data.get("end_aligned_variants", True)),
            force_regenerate_tts=bool(data.get("force_regenerate_tts", False)),
            model_size=str(data.get("model_size", "medium")),
        )

        if cfg.tts_positives > 0:
            if not is_piper_importable():
                import sys
                return jsonify({
                    "error": (
                        f"TTS requested but piper-tts is not importable from "
                        f"this Python ({sys.executable}). "
                        f"Run `{sys.executable} -m pip install piper-tts` "
                        f"or `heed doctor` to diagnose."
                    )
                }), 400
            if not is_voice_available():
                return jsonify({
                    "error": "TTS requested but voice not downloaded. "
                             "Run `heed download-tts` from a terminal."
                }), 400

        if cfg.kokoro_positives > 0:
            from .tts_kokoro import (is_kokoro_importable as k_importable,
                                     is_voice_available as k_avail)
            if not k_importable():
                import sys
                return jsonify({
                    "error": (
                        f"Kokoro positives requested but kokoro-onnx is not "
                        f"importable from this Python ({sys.executable}). "
                        f"Run `{sys.executable} -m pip install kokoro-onnx`."
                    )
                }), 400
            if not k_avail():
                return jsonify({
                    "error": "Kokoro positives requested but voices not "
                             "downloaded. Run `heed download-kokoro`."
                }), 400

        wake_lower = cfg.phrase.strip().lower()
        if cfg.tts_positives > 0 and bool(data.get("auto_distractors", True)):
            from .tts import phonetic_neighbor_distractors
            extras: list[str] = list(data.get("extra_distractors") or [])
            for d in DEFAULT_NEAR_DISTRACTOR_PHRASES:
                if d.lower() != wake_lower and d not in extras:
                    extras.append(d)
            # Wake-phrase-specific phonetic neighbors: the most important
            # hard negatives - they prevent the "hey *" over-firing bug.
            # Bumped cap 30→50 to surface more phonetically-diverse "hey X"
            # rhymes/near-rhymes (e.g. "hey dog", "hey rock", "hey john",
            # "hey hot" for wake phrase "hey doc").
            for n in phonetic_neighbor_distractors(wake_lower, max_neighbors=60):
                if n != wake_lower and n not in extras:
                    extras.append(n)
            cfg.tts_negative_phrases = extras
        else:
            cfg.tts_negative_phrases = data.get("extra_distractors") or []

        _train_in_background(app, name, cfg)
        return jsonify({"ok": True})

    @app.route("/api/projects/<name>/train_status", methods=["GET"])
    def api_train_status(name: str):
        return jsonify(_train_status(name))

    # ---- live listen (per-session detector) ----
    @app.route("/api/projects/<name>/ambient", methods=["POST"])
    def api_record_ambient(name: str):
        """Save a recording of room/mic ambient (no speech) - used as a noise
        pool when augmenting TTS samples, so synthetic audio sounds more like
        the user's actual recording environment."""
        project = _project_dir(app, name)
        if not project.exists():
            return jsonify({"error": "project not found"}), 404
        sr = int(request.args.get("sr", "0"))
        if sr <= 0:
            return jsonify({"error": "sr query param required"}), 400
        audio = np.frombuffer(request.get_data(), dtype=np.float32)
        if audio.size < 64:
            return jsonify({"error": "too little audio"}), 400
        audio = _resample_to_16k(audio, sr)
        save_wav(project / "ambient.wav", audio)
        return jsonify({"ok": True,
                        "duration_s": round(len(audio) / SAMPLE_RATE, 2)})

    @app.route("/api/projects/<name>/ambient", methods=["GET"])
    def api_get_ambient_status(name: str):
        project = _project_dir(app, name)
        p = project / "ambient.wav"
        if not p.exists():
            return jsonify({"present": False})
        return jsonify({"present": True,
                        "duration_s": round(p.stat().st_size / 2 / SAMPLE_RATE, 2)})

    @app.route("/api/projects/<name>/ambient/audio", methods=["GET"])
    def api_get_ambient_audio(name: str):
        project = _project_dir(app, name)
        return send_from_directory(project, "ambient.wav", mimetype="audio/wav")

    @app.route("/api/projects/<name>/suggested_neighbors", methods=["GET"])
    def api_suggested_neighbors(name: str):
        """Return phonetic-neighbor phrases for the user to record in their
        own voice - the single biggest lever for fixing 'hey * triggers'."""
        from .tts import phonetic_neighbor_distractors
        project = _project_dir(app, name)
        cfg = _project_config(project)
        phrase = cfg.get("phrase", "").strip()
        suggestions = phonetic_neighbor_distractors(phrase, max_neighbors=20)
        # Detect which ones already have a matching recording. Filename
        # convention: neg_neighbor_<safe_phrase>_NN.wav
        recorded: dict[str, list[str]] = {}
        if (project / "negative").exists():
            for p in (project / "negative").glob("neg_neighbor_*.wav"):
                # extract phrase between "neg_neighbor_" and last "_NN"
                stem = p.stem  # "neg_neighbor_hey_siri_001"
                without_prefix = stem[len("neg_neighbor_"):]
                # split off trailing "_NNN"
                rsplit = without_prefix.rsplit("_", 1)
                phrase_key = rsplit[0] if len(rsplit) > 1 and rsplit[1].isdigit() else without_prefix
                recorded.setdefault(phrase_key, []).append(p.name)
        return jsonify({"suggestions": suggestions, "recorded": recorded})

    @app.route("/api/projects/<name>/record_neighbor", methods=["POST"])
    def api_record_neighbor(name: str):
        """Save a user-voice recording of a specific phonetic neighbor as
        a hard negative."""
        project = _project_dir(app, name)
        neighbor = request.args.get("phrase", "").strip()
        sr = int(request.args.get("sr", "0"))
        if not neighbor or sr <= 0:
            return jsonify({"error": "phrase and sr required"}), 400
        audio = np.frombuffer(request.get_data(), dtype=np.float32)
        if audio.size < 64:
            return jsonify({"error": "too little audio"}), 400
        audio = _resample_to_16k(audio, sr)
        neg_dir = project / "negative"
        neg_dir.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in neighbor.lower())[:40]
        existing = list(neg_dir.glob(f"neg_neighbor_{safe}_*.wav"))
        idx = len(existing) + 1
        path = neg_dir / f"neg_neighbor_{safe}_{idx:03d}.wav"
        save_wav(path, audio)
        return jsonify({"ok": True, "filename": path.name, "phrase": neighbor})

    @app.route("/api/projects/<name>/hard_negative_status", methods=["GET"])
    def api_hard_negative_status(name: str):
        """Return per-category counts of user-recorded hard negatives so the
        UI can show "✓ recorded (N)" badges. Categories match the prompts
        in the UI's hard-negative library section."""
        project = _project_dir(app, name)
        neg_dir = project / "negative"
        counts: dict[str, int] = {}
        if neg_dir.exists():
            for p in neg_dir.glob("neg_hardneg_*.wav"):
                # filename format: neg_hardneg_<category>_NNN.wav
                stem = p.stem[len("neg_hardneg_"):]
                # strip trailing _NNN
                parts = stem.rsplit("_", 1)
                cat = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                counts[cat] = counts.get(cat, 0) + 1
        return jsonify({"counts": counts})

    @app.route("/api/projects/<name>/record_hard_negative", methods=["POST"])
    def api_record_hard_negative(name: str):
        """Save a user-voice recording of a non-phonetic hard negative.
        Categories: breathing, mouth_sounds, typing, mumbling, loud_breath,
        etc. Saved into negative/ with descriptive filename so the user (and
        we) can audit which categories are represented in training."""
        project = _project_dir(app, name)
        category = request.args.get("category", "").strip().lower()
        sr = int(request.args.get("sr", "0"))
        if not category or sr <= 0:
            return jsonify({"error": "category and sr required"}), 400
        # Sanitize category to a safe filename slug
        safe_cat = "".join(c if c.isalnum() else "_" for c in category)[:40]
        if not safe_cat:
            return jsonify({"error": "invalid category"}), 400
        audio = np.frombuffer(request.get_data(), dtype=np.float32)
        if audio.size < 64:
            return jsonify({"error": "too little audio"}), 400
        audio = _resample_to_16k(audio, sr)
        neg_dir = project / "negative"
        neg_dir.mkdir(exist_ok=True)
        existing = list(neg_dir.glob(f"neg_hardneg_{safe_cat}_*.wav"))
        idx = len(existing) + 1
        path = neg_dir / f"neg_hardneg_{safe_cat}_{idx:03d}.wav"
        save_wav(path, audio)
        return jsonify({"ok": True, "filename": path.name, "category": safe_cat})

    @app.route("/api/projects/<name>/holdout_test", methods=["POST"])
    def api_holdout_test(name: str):
        """Run the trained model against synthesized utterances from 3 voices
        the model NEVER saw during training. The objective generalization
        check - if it triggers on user but not on these unseen voices, the
        model is speaker-locked. If it triggers on both → it generalizes."""
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400
        cfg_json = _project_config(project)
        phrase = cfg_json.get("phrase", "").strip()
        if not phrase:
            return jsonify({"error": "project has no phrase"}), 400
        try:
            from .tts import (HELDOUT_SPEAKER_IDS, is_piper_importable,
                              is_voice_available, synthesize_from_speakers)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"TTS unavailable: {exc}"}), 400
        if not is_piper_importable() or not is_voice_available():
            return jsonify({
                "error": "TTS voice not importable or not downloaded. "
                         "Run `heed doctor` for diagnostics."
            }), 400

        # Negative phrases that exercise the "hey *" failure mode + general
        # short distractors - enough to spot speaker-locked behavior fast.
        false_trigger_phrases = ["hey siri", "hey google", "good morning",
                                 "see you later"]

        from .trainer import load_model
        model, payload = load_model(model_path)
        threshold = float(payload["threshold"])

        def _score(audio: torch.Tensor) -> float:
            prepped = prepare_clip(audio)
            mel = log_mel(prepped)
            with torch.no_grad():
                return float(torch.sigmoid(model(mel)).item())

        # Positives: held-out speakers saying the wake phrase
        wake_clips = synthesize_from_speakers(phrase, HELDOUT_SPEAKER_IDS)
        positives = [
            {"speaker_id": int(sid), "phrase": phrase, "score": _score(clip),
             "expected": "trigger"}
            for sid, clip in zip(HELDOUT_SPEAKER_IDS, wake_clips)
        ]

        # Negatives: same held-out speakers saying common false-trigger phrases
        negatives = []
        for distractor in false_trigger_phrases:
            for sid, clip in zip(HELDOUT_SPEAKER_IDS,
                                 synthesize_from_speakers(distractor, HELDOUT_SPEAKER_IDS)):
                negatives.append({
                    "speaker_id": int(sid),
                    "phrase": distractor,
                    "score": _score(clip),
                    "expected": "no trigger",
                })

        # Persist these clips so the UI can let the user listen
        eval_dir = project / "holdout_eval"
        eval_dir.mkdir(exist_ok=True)
        for old in eval_dir.glob("*.wav"):
            old.unlink()
        for sid, clip in zip(HELDOUT_SPEAKER_IDS, wake_clips):
            save_wav(eval_dir / f"pos_spk{sid:03d}.wav", clip)
        idx = 0
        for distractor in false_trigger_phrases:
            for sid, clip in zip(HELDOUT_SPEAKER_IDS,
                                 synthesize_from_speakers(distractor, HELDOUT_SPEAKER_IDS)):
                safe = "".join(c if c.isalnum() else "_" for c in distractor.lower())[:30]
                save_wav(eval_dir / f"neg_{safe}_spk{sid:03d}.wav", clip)
                negatives[idx]["filename"] = f"neg_{safe}_spk{sid:03d}.wav"
                idx += 1
        for i, sid in enumerate(HELDOUT_SPEAKER_IDS):
            positives[i]["filename"] = f"pos_spk{sid:03d}.wav"

        n_pos_fire = sum(1 for p in positives if p["score"] > threshold)
        n_neg_fire = sum(1 for n in negatives if n["score"] > threshold)
        verdict = "generalizes" if (n_pos_fire >= 2 and n_neg_fire <= 2) else \
                  "speaker-locked" if n_pos_fire == 0 else "borderline"
        return jsonify({
            "ok": True,
            "phrase": phrase,
            "threshold": threshold,
            "positives": positives,
            "negatives": negatives,
            "n_pos_fire": n_pos_fire,
            "n_neg_fire": n_neg_fire,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "verdict": verdict,
        })

    @app.route("/api/projects/<name>/holdout_clip/<filename>", methods=["GET"])
    def api_get_holdout_clip(name: str, filename: str):
        project = _project_dir(app, name)
        return send_from_directory(project / "holdout_eval", filename,
                                   mimetype="audio/wav")

    @app.route("/api/projects/<name>/cross_tts_test", methods=["POST"])
    def api_cross_tts_test(name: str):
        """Cross-TTS-family generalization test.

        The /holdout_test endpoint above uses 3 held-out *Piper* voices, but
        the model trained on 900 other Piper voices that share the same
        vocoder and acoustic model. Passing that test means "generalizes
        across Piper speakers" - strictly weaker than "generalizes to real
        humans." This endpoint repeats the holdout idea with **Kokoro**
        voices, a different neural family entirely. Passing this is much
        closer evidence of human-level generalization.
        """
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400
        cfg_json = _project_config(project)
        phrase = cfg_json.get("phrase", "").strip()
        if not phrase:
            return jsonify({"error": "project has no phrase"}), 400
        try:
            from .tts_kokoro import (HELDOUT_VOICE_IDS, is_kokoro_importable,
                                     is_voice_available, synthesize_from_voices)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"kokoro module failed: {exc}"}), 400
        if not is_kokoro_importable():
            return jsonify({
                "error": "kokoro-onnx is not installed.  → pip install kokoro-onnx"
            }), 400
        if not is_voice_available():
            return jsonify({
                "error": "kokoro voices not downloaded.  → heed download-kokoro"
            }), 400

        false_trigger_phrases = ["hey siri", "hey google", "good morning",
                                 "see you later"]

        from .trainer import load_model
        model, payload = load_model(model_path)
        threshold = float(payload["threshold"])

        def _score(audio: torch.Tensor) -> float:
            prepped = prepare_clip(audio)
            mel = log_mel(prepped)
            with torch.no_grad():
                return float(torch.sigmoid(model(mel)).item())

        wake_clips = synthesize_from_voices(phrase, HELDOUT_VOICE_IDS)
        positives = [
            {"voice_id": vid, "phrase": phrase, "score": _score(clip),
             "expected": "trigger"}
            for vid, clip in zip(HELDOUT_VOICE_IDS, wake_clips)
        ]

        negatives = []
        for distractor in false_trigger_phrases:
            for vid, clip in zip(HELDOUT_VOICE_IDS,
                                 synthesize_from_voices(distractor, HELDOUT_VOICE_IDS)):
                negatives.append({
                    "voice_id": vid,
                    "phrase": distractor,
                    "score": _score(clip),
                    "expected": "no trigger",
                })

        eval_dir = project / "kokoro_eval"
        eval_dir.mkdir(exist_ok=True)
        for old in eval_dir.glob("*.wav"):
            old.unlink()
        for vid, clip in zip(HELDOUT_VOICE_IDS, wake_clips):
            save_wav(eval_dir / f"pos_{vid}.wav", clip)
        idx = 0
        for distractor in false_trigger_phrases:
            for vid, clip in zip(HELDOUT_VOICE_IDS,
                                 synthesize_from_voices(distractor, HELDOUT_VOICE_IDS)):
                safe = "".join(c if c.isalnum() else "_" for c in distractor.lower())[:30]
                fname = f"neg_{safe}_{vid}.wav"
                save_wav(eval_dir / fname, clip)
                negatives[idx]["filename"] = fname
                idx += 1
        for i, vid in enumerate(HELDOUT_VOICE_IDS):
            positives[i]["filename"] = f"pos_{vid}.wav"

        n_pos_fire = sum(1 for p in positives if p["score"] > threshold)
        n_neg_fire = sum(1 for n in negatives if n["score"] > threshold)
        if n_pos_fire >= 2 and n_neg_fire <= 2:
            verdict = "generalizes"
        elif n_pos_fire == 0:
            verdict = "tts-family-locked"
        else:
            verdict = "borderline"
        return jsonify({
            "ok": True,
            "phrase": phrase,
            "threshold": threshold,
            "positives": positives,
            "negatives": negatives,
            "n_pos_fire": n_pos_fire,
            "n_neg_fire": n_neg_fire,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "verdict": verdict,
        })

    @app.route("/api/projects/<name>/cross_tts_clip/<filename>", methods=["GET"])
    def api_get_cross_tts_clip(name: str, filename: str):
        project = _project_dir(app, name)
        return send_from_directory(project / "kokoro_eval", filename,
                                   mimetype="audio/wav")

    @app.route("/api/projects/<name>/self_test", methods=["POST"])
    def api_self_test(name: str):
        """Score the trained model on the user's OWN positive/negative
        recordings. This is the most directly relevant metric - val accuracy
        is dominated by TTS samples (hundreds of voices), so a model can
        score 97 % val while failing on the trainer's voice. Self-test
        tells you the answer directly: does the model trigger on you?
        """
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400

        from .trainer import load_model
        model, payload = load_model(model_path)
        threshold = float(payload["threshold"])

        def _score(wav_path):
            audio = load_wav(wav_path)
            clip = prepare_clip(audio)
            mel = log_mel(clip)
            with torch.no_grad():
                return float(torch.sigmoid(model(mel)).item())

        pos_dir = project / "positive"
        neg_dir = project / "negative"
        positives = []
        for p in sorted(pos_dir.glob("*.wav")):
            s = _score(p)
            positives.append({"filename": p.name, "score": s,
                              "expected": "trigger",
                              "ok": s > threshold})
        negatives = []
        for p in sorted(neg_dir.glob("*.wav")):
            s = _score(p)
            negatives.append({"filename": p.name, "score": s,
                              "expected": "no trigger",
                              "ok": s <= threshold})

        n_pos_correct = sum(1 for p in positives if p["ok"])
        n_neg_correct = sum(1 for n in negatives if n["ok"])
        tpr = n_pos_correct / max(1, len(positives))
        fpr = (len(negatives) - n_neg_correct) / max(1, len(negatives))

        # User-specific verdict - the only metric that matches the user's
        # subjective experience of "does this work on my voice."
        if tpr >= 0.9 and fpr <= 0.1:
            verdict = "user-voice-ready"
        elif tpr >= 0.7:
            verdict = "mostly works on your voice"
        elif tpr >= 0.4:
            verdict = "borderline on your voice"
        else:
            verdict = "model does NOT recognize your voice"

        return jsonify({
            "ok": True,
            "threshold": threshold,
            "positives": positives,
            "negatives": negatives,
            "tpr": tpr,
            "fpr": fpr,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "verdict": verdict,
        })

    # ---- model slots ----

    @app.route("/api/projects/<name>/models", methods=["GET"])
    def api_list_models(name: str):
        """List saved model slots under <project>/models/.

        Each slot is a (name.pt, name.json) pair saved by the trainer.
        Returns metadata (size, params, training time, user-voice TPR, etc.)
        so the UI can display a useful summary.

        Also reports which slot is currently "active" - i.e. matches the
        project-root model.pt by file content. The active model is what
        every existing code path uses (Test, Listen, Export, Self-test).
        """
        project = _project_dir(app, name)
        models_dir = project / "models"
        active_path = project / "model.pt"
        active_bytes = active_path.read_bytes() if active_path.exists() else None

        slots = []
        if models_dir.exists():
            for pt in sorted(models_dir.glob("*.pt")):
                sidecar = pt.with_suffix(".json")
                meta = {}
                if sidecar.exists():
                    try:
                        meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                size_bytes = pt.stat().st_size
                mtime = int(pt.stat().st_mtime)
                is_active = (
                    active_bytes is not None
                    and pt.read_bytes() == active_bytes
                )
                slots.append({
                    "name": pt.stem,
                    "size_bytes": size_bytes,
                    "mtime": mtime,
                    "is_active": is_active,
                    "phrase": meta.get("phrase", ""),
                    "n_params": meta.get("n_params"),
                    "threshold": meta.get("threshold"),
                    "seconds": meta.get("seconds"),
                    "user_voice_tpr": meta.get("user_voice_tpr"),
                    "user_voice_fpr": meta.get("user_voice_fpr"),
                    "n_user_positives_eval": meta.get("n_user_positives_eval"),
                    "n_user_negatives_eval": meta.get("n_user_negatives_eval"),
                    "tts_positives_used": meta.get("tts_positives_used"),
                    "kokoro_positives_used": meta.get("kokoro_positives_used"),
                })
        return jsonify({"slots": slots,
                        "active_exists": active_path.exists()})

    @app.route("/api/projects/<name>/models/<slot>/activate", methods=["POST"])
    def api_activate_model(name: str, slot: str):
        """Make `slot` the active model by copying models/<slot>.pt → model.pt
        (and the sidecar). All existing code paths use model.pt at project
        root so this single copy switches what gets tested / listened to /
        exported."""
        if "/" in slot or "\\" in slot or ".." in slot or not slot:
            return jsonify({"error": "bad slot"}), 400
        project = _project_dir(app, name)
        slot_pt = project / "models" / f"{slot}.pt"
        slot_json = project / "models" / f"{slot}.json"
        if not slot_pt.exists():
            return jsonify({"error": f"slot {slot!r} not found"}), 404
        import shutil
        shutil.copy2(slot_pt, project / "model.pt")
        if slot_json.exists():
            shutil.copy2(slot_json, project / "model.json")
        return jsonify({"ok": True, "active_slot": slot})

    @app.route("/api/projects/<name>/models/<slot>", methods=["DELETE"])
    def api_delete_model(name: str, slot: str):
        """Delete a slot (just removes the slot files; does not touch the
        active model.pt)."""
        if "/" in slot or "\\" in slot or ".." in slot or not slot:
            return jsonify({"error": "bad slot"}), 400
        project = _project_dir(app, name)
        slot_pt = project / "models" / f"{slot}.pt"
        slot_json = project / "models" / f"{slot}.json"
        deleted = []
        for p in (slot_pt, slot_json):
            if p.exists():
                p.unlink()
                deleted.append(p.name)
        if not deleted:
            return jsonify({"error": f"slot {slot!r} not found"}), 404
        return jsonify({"ok": True, "deleted": deleted})

    @app.route("/api/projects/<name>/export", methods=["POST"])
    def api_export(name: str):
        """Trigger ONNX export. Produces wake.onnx + wake.int8.onnx +
        wake.json + export/README.md under <project>/export/."""
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model - train first"}), 400
        data = request.get_json(silent=True) or {}
        int8 = bool(data.get("int8", True))

        from .export import export_to_onnx
        try:
            result = export_to_onnx(model_path, project / "export",
                                    int8=int8, log_fn=lambda _msg: None)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:  # noqa: BLE001
            return jsonify({
                "error": f"export failed: {type(exc).__name__}: {exc}"
            }), 500

        return jsonify({
            "ok": True,
            "n_params": result.n_params,
            "onnx_size_bytes": result.onnx_size_bytes,
            "int8_size_bytes": result.int8_size_bytes,
            "max_abs_error_fp32": result.max_abs_error_fp32,
            "max_abs_error_int8": result.max_abs_error_int8,
        })

    @app.route("/api/projects/<name>/send-to-mobile", methods=["POST"])
    def api_send_to_mobile(name: str):
        """Export the project and copy the model files into the bundled React
        Native demo's assets/, replacing its current active model (wake.onnx /
        wake.int8.onnx / wake.tflite / wake.json). Reload Metro on the phone
        afterwards to pick them up."""
        import shutil
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model - train first"}), 400
        data = request.get_json(silent=True) or {}
        try:
            slot = int(data.get("slot", 0))
        except (TypeError, ValueError):
            slot = 0
        if not (0 <= slot <= 4):
            return jsonify({"error": "slot must be 0-4"}), 400

        # RN demo assets dir, relative to the installed package (code/heed/..).
        assets_dir = (Path(__file__).resolve().parent.parent
                      / "examples" / "inference_react_native" / "assets")
        if not assets_dir.is_dir():
            return jsonify({
                "error": f"React Native demo not found ({assets_dir}). "
                         f"This needs the repo's examples/ directory."
            }), 400

        export_dir = project / "export"
        from .export import export_to_onnx
        try:
            export_to_onnx(model_path, export_dir, int8=True, log_fn=lambda _m: None)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"ONNX export failed: {type(exc).__name__}: {exc}"}), 500

        # TFLite is best-effort (needs litert-torch). If skipped, the demo's
        # TFLite chip keeps its previous model until a CLI `heed export`.
        skipped = []
        try:
            from .export import export_to_tflite
            export_to_tflite(model_path, export_dir, log_fn=lambda _m: None)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"wake.tflite ({type(exc).__name__})")

        # Copy into the chosen slot (slotN.*) so the demo can hold several words.
        sent = []
        slot_map = {
            "wake.onnx": f"slot{slot}.onnx",
            "wake.int8.onnx": f"slot{slot}.int8.onnx",
            "wake.tflite": f"slot{slot}.tflite",
            "wake.json": f"slot{slot}.json",
        }
        for src_name, dst_name in slot_map.items():
            src = export_dir / src_name
            if src.exists():
                shutil.copy2(src, assets_dir / dst_name)
                sent.append(dst_name)

        try:
            phrase = json.loads(
                (export_dir / "wake.json").read_text(encoding="utf-8")
            ).get("phrase", "")
        except Exception:  # noqa: BLE001
            phrase = ""
        return jsonify({"ok": True, "phrase": phrase, "slot": slot, "sent": sent,
                        "skipped": skipped, "assets_dir": str(assets_dir)})

    @app.route("/api/projects/<name>/export/info", methods=["GET"])
    def api_export_info(name: str):
        """List exported files (if any) with sizes + mtime."""
        project = _project_dir(app, name)
        export_dir = project / "export"
        if not export_dir.exists():
            return jsonify({"exists": False, "files": []})
        files = []
        for p in sorted(export_dir.iterdir()):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                })
        return jsonify({"exists": True, "files": files})

    @app.route("/api/projects/<name>/export/download/<filename>", methods=["GET"])
    def api_export_download(name: str, filename: str):
        """Serve a single exported file as a download."""
        if "/" in filename or "\\" in filename or ".." in filename:
            return jsonify({"error": "bad filename"}), 400
        project = _project_dir(app, name)
        export_dir = project / "export"
        if not (export_dir / filename).exists():
            return jsonify({"error": "not found"}), 404
        return send_from_directory(
            export_dir, filename,
            as_attachment=True,
        )

    @app.route("/api/projects/<name>/export/verify", methods=["POST"])
    def api_export_verify(name: str):
        """Score user positives/negatives through PyTorch, ONNX fp32, and
        ONNX int8 (if present). Returns side-by-side scores so the user can
        confirm the exported models behave the same as the original."""
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        export_dir = project / "export"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400
        if not (export_dir / "wake.onnx").exists():
            return jsonify({"error": "no export - run export first"}), 400

        try:
            import onnxruntime as ort
        except Exception as exc:
            return jsonify({
                "error": f"onnxruntime not usable in this Python: "
                         f"{type(exc).__name__}: {exc}"
            }), 500

        from .trainer import load_model
        model, payload = load_model(model_path)
        threshold = float(payload["threshold"])

        sess_fp32 = ort.InferenceSession(
            str(export_dir / "wake.onnx"),
            providers=["CPUExecutionProvider"],
        )
        sess_int8 = None
        if (export_dir / "wake.int8.onnx").exists():
            sess_int8 = ort.InferenceSession(
                str(export_dir / "wake.int8.onnx"),
                providers=["CPUExecutionProvider"],
            )

        import numpy as _np

        def _scores(audio):
            clip = prepare_clip(audio)
            mel = log_mel(clip).numpy()
            with torch.no_grad():
                pt = 1.0 / (1.0 + float(_np.exp(-model(torch.from_numpy(mel)).item())))
            fp = sess_fp32.run(None, {"mel": mel})[0]
            fp_p = 1.0 / (1.0 + float(_np.exp(-float(fp.flatten()[0]))))
            int8_p = None
            if sess_int8 is not None:
                i8 = sess_int8.run(None, {"mel": mel})[0]
                int8_p = 1.0 / (1.0 + float(_np.exp(-float(i8.flatten()[0]))))
            return pt, fp_p, int8_p

        def _scan(d, expected_trigger):
            out = []
            for p in sorted(d.glob("*.wav")):
                try:
                    audio = load_wav(p)
                    pt, fp, i8 = _scores(audio)
                except Exception:
                    continue
                out.append({
                    "filename": p.name,
                    "pytorch": round(pt, 4),
                    "onnx_fp32": round(fp, 4),
                    "onnx_int8": round(i8, 4) if i8 is not None else None,
                    "expected_trigger": expected_trigger,
                    "pt_trigger": pt > threshold,
                    "fp32_trigger": fp > threshold,
                    "int8_trigger": (i8 > threshold) if i8 is not None else None,
                })
            return out

        positives = _scan(project / "positive", True)
        negatives = _scan(project / "negative", False)

        # Aggregate divergence stats
        def _abs_diff(a, b):
            if a is None or b is None:
                return None
            return abs(a - b)

        all_rows = positives + negatives
        fp32_diffs = [
            _abs_diff(r["pytorch"], r["onnx_fp32"]) for r in all_rows
        ]
        int8_diffs = [
            _abs_diff(r["pytorch"], r["onnx_int8"]) for r in all_rows
        ]
        fp32_diffs = [d for d in fp32_diffs if d is not None]
        int8_diffs = [d for d in int8_diffs if d is not None]
        # Trigger agreement
        n_disagree_fp32 = sum(
            1 for r in all_rows if r["pt_trigger"] != r["fp32_trigger"]
        )
        n_disagree_int8 = sum(
            1 for r in all_rows
            if r["int8_trigger"] is not None and r["pt_trigger"] != r["int8_trigger"]
        )

        return jsonify({
            "ok": True,
            "threshold": threshold,
            "has_int8": sess_int8 is not None,
            "positives": positives,
            "negatives": negatives,
            "summary": {
                "fp32_max_abs_diff": max(fp32_diffs) if fp32_diffs else 0.0,
                "fp32_mean_abs_diff": (
                    sum(fp32_diffs) / len(fp32_diffs) if fp32_diffs else 0.0
                ),
                "int8_max_abs_diff": max(int8_diffs) if int8_diffs else None,
                "int8_mean_abs_diff": (
                    sum(int8_diffs) / len(int8_diffs) if int8_diffs else None
                ),
                "fp32_trigger_disagreements": n_disagree_fp32,
                "int8_trigger_disagreements": n_disagree_int8,
                "n_total": len(all_rows),
            },
        })

    @app.route("/api/projects/<name>/test_take", methods=["POST"])
    def api_test_take(name: str):
        """One-shot diagnostic: receive raw float32 audio, save it under
        test_takes/, run the trained model on the trimmed-and-centered clip
        directly (no streaming, no gate), return the probability + the
        saved filename so the UI can play it back."""
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400
        sr = int(request.args.get("sr", "0"))
        if sr <= 0:
            return jsonify({"error": "sr query param required"}), 400
        audio_bytes = request.get_data()
        if len(audio_bytes) < 64:
            return jsonify({"error": "too little audio"}), 400

        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        audio_np = _resample_to_16k(audio_np, sr)

        # save the raw take for playback / inspection
        takes_dir = project / "test_takes"
        takes_dir.mkdir(exist_ok=True)
        existing = sorted(takes_dir.glob("take_*.wav"))
        idx = len(existing) + 1
        take_path = takes_dir / f"take_{idx:03d}.wav"
        save_wav(take_path, audio_np)

        # run the model on the prepared clip (same path as training does)
        clip = prepare_clip(torch.from_numpy(audio_np))
        from .trainer import load_model
        model, payload = load_model(model_path)
        mel = log_mel(clip)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(mel)).item())

        # gate diagnostic for context
        from .gate import EnergyGate
        gate = EnergyGate()
        _, gate_diag = gate(clip)

        threshold = float(payload["threshold"])
        return jsonify({
            "ok": True,
            "filename": take_path.name,
            "prob": prob,
            "would_trigger": bool(prob > threshold),
            "threshold": threshold,
            "rms_dbfs": gate_diag.get("rms_dbfs"),
            "band_frac": gate_diag.get("band_frac"),
            "rumble_frac": gate_diag.get("rumble_frac"),
            "hiss_frac": gate_diag.get("hiss_frac"),
            "gate_pass": gate_diag.get("reason") == "pass",
        })

    @app.route("/api/projects/<name>/test_take/<filename>", methods=["GET"])
    def api_get_take(name: str, filename: str):
        project = _project_dir(app, name)
        return send_from_directory(project / "test_takes", filename,
                                   mimetype="audio/wav")

    @app.route("/api/projects/<name>/live_start", methods=["POST"])
    def api_live_start(name: str):
        project = _project_dir(app, name)
        model_path = project / "model.pt"
        if not model_path.exists():
            return jsonify({"error": "no trained model"}), 400
        data = request.get_json(force=True) or {}
        threshold = data.get("threshold")
        backend = data.get("backend", "pytorch")
        # Map backend → ONNX file path (only used for ONNX backends)
        onnx_path = None
        if backend == "onnx_fp32":
            onnx_path = project / "export" / "wake.onnx"
        elif backend == "onnx_int8":
            onnx_path = project / "export" / "wake.int8.onnx"
        elif backend != "pytorch":
            return jsonify({"error": f"unknown backend: {backend!r}"}), 400
        if onnx_path is not None and not onnx_path.exists():
            return jsonify({
                "error": f"backend {backend!r} needs {onnx_path} - "
                         f"run Export first."
            }), 400
        try:
            det = WakeWordDetector(
                model_path,
                threshold_override=threshold,
                backend=backend,
                onnx_path=onnx_path,
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        session = secrets.token_hex(8)
        with _live_lock:
            _live_sessions[session] = {
                "detector": det,
                "src_sr": int(data.get("sr", 0)) or None,
                "last_seen": time.monotonic(),
                "started": time.monotonic(),
                "buffer": np.zeros(0, dtype=np.float32),
            }
        return jsonify({"ok": True, "session": session,
                        "threshold": det.threshold, "phrase": det.phrase,
                        "backend": backend})

    @app.route("/api/projects/<name>/live_chunk", methods=["POST"])
    def api_live_chunk(name: str):
        session = request.args.get("session")
        sr = int(request.args.get("sr", "0"))
        if not session:
            return jsonify({"error": "session required"}), 400
        _gc_live_sessions()
        with _live_lock:
            state = _live_sessions.get(session)
        if state is None:
            return jsonify({"error": "session not found / expired"}), 404
        state["last_seen"] = time.monotonic()
        chunk_bytes = request.get_data()
        chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
        if sr and sr != SAMPLE_RATE:
            chunk = _resample_to_16k(chunk, sr)
        # Detector wants ~100ms; we may receive 50-200ms; either is fine
        res = state["detector"].step(torch.from_numpy(chunk.copy()))
        # Append to per-session ring buffer for diagnostics
        diag = res.get("diag", {}) or {}
        log_entry = {
            "t": round(time.monotonic() - state["started"], 2),
            "prob": round(res["prob"], 3),
            "ema": round(res["ema"], 3),
            "triggered": res["triggered"],
            "gated": res["gated"],
            "rms_dbfs": round(diag.get("rms_dbfs", -120.0), 1),
            "band": round(diag.get("band_frac", 0.0) * 100, 0),
            "rumble": round(diag.get("rumble_frac", 0.0) * 100, 0),
            "hiss": round(diag.get("hiss_frac", 0.0) * 100, 0),
            "reason": diag.get("reason", ""),
        }
        log = state.setdefault("log", [])
        log.append(log_entry)
        # Keep last ~600 frames (~60 s at 100ms/frame)
        if len(log) > 600:
            del log[: len(log) - 600]
        return jsonify({
            "prob": res["prob"],
            "ema": res["ema"],
            "triggered": res["triggered"],
            "gated": res["gated"],
            "diag": diag,
        })

    @app.route("/api/projects/<name>/live_log", methods=["GET"])
    def api_live_log(name: str):
        """Return the recent per-frame log for a live session. Used by the
        UI to display a scrollable diagnostic table.

        Optional query: ?since=<t> to filter entries newer than t seconds.
        """
        session = request.args.get("session")
        since = float(request.args.get("since", "0") or 0)
        if not session:
            return jsonify({"error": "session required"}), 400
        with _live_lock:
            state = _live_sessions.get(session)
        if state is None:
            return jsonify({"error": "session not found / expired"}), 404
        log = state.get("log", [])
        if since > 0:
            log = [e for e in log if e["t"] > since]
        threshold = float(state["detector"].threshold)
        return jsonify({"ok": True, "threshold": threshold, "entries": log})

    @app.route("/api/projects/<name>/live_stop", methods=["POST"])
    def api_live_stop(name: str):
        session = request.args.get("session")
        with _live_lock:
            _live_sessions.pop(session, None)
        return jsonify({"ok": True})

    return app


def run_server(host: str = "127.0.0.1", port: int = 7777,
               workspace: Path | None = None) -> None:
    import sys
    app = create_app(workspace)
    print(f"heed UI on http://{host}:{port}")
    print(f"workspace: {app.config['TINYWW_WORKSPACE']}")
    print(f"python:    {sys.executable}")
    piper_ok = is_piper_importable()
    voice_ok = is_voice_available()
    print(f"piper:     {'importable' if piper_ok else 'NOT IMPORTABLE - TTS disabled'}")
    print(f"voice:     {'downloaded' if voice_ok else 'not downloaded - run `heed download-tts`'}")
    if not piper_ok:
        print("  → run `heed doctor` to see why piper-tts is unusable")
    app.run(host=host, port=port, debug=False, threaded=True)


# ----- embedded single-page HTML -------------------------------------------

_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>heed</title>
<style>
  :root {
    --bg: #15171c; --fg: #ececec; --muted: #8b8e95; --line: #2a2d35;
    --accent: #6ee7d3; --danger: #ef4444; --ok: #22c55e;
  }
  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; }
  body {
    background: var(--bg); color: var(--fg);
    font: 15px/1.55 -apple-system,Segoe UI,Inter,system-ui,sans-serif;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* layout */
  header {
    padding: 16px 24px;
    display: flex; justify-content: space-between; align-items: center;
  }
  header .title { font-weight: 600; letter-spacing: -.01em; font-size: 17px; }
  header .meta { color: var(--muted); font-size: 13px; }
  header .meta b { color: var(--fg); font-weight: 500; }
  main { max-width: 560px; margin: 0 auto; padding: 12px 24px 80px; }

  /* sections */
  section { padding: 28px 0; border-top: 1px solid var(--line); }
  section:first-of-type { border-top: 0; padding-top: 8px; }
  section h2 {
    margin: 0 0 4px; font-size: 14px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: .08em;
  }
  section .hint { color: var(--muted); font-size: 13px; margin-bottom: 16px; }

  /* progress dots */
  .dots { display: flex; gap: 6px; margin: 12px 0; }
  .dot {
    width: 10px; height: 10px; border-radius: 999px;
    background: var(--line);
  }
  .dot.filled { background: var(--accent); }

  /* buttons */
  button {
    background: var(--accent); color: #0c1014; border: 0;
    border-radius: 8px; padding: 12px 18px; font-weight: 600;
    cursor: pointer; font: inherit; font-weight: 600;
    transition: transform .08s, opacity .15s;
  }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: .4; cursor: not-allowed; }
  button.ghost {
    background: transparent; color: var(--fg);
    border: 1px solid var(--line);
  }
  button.danger { background: var(--danger); color: #fff; }

  /* big call-to-action */
  .cta {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; font-size: 16px;
  }
  .cta .rec-dot {
    width: 12px; height: 12px; border-radius: 999px; background: var(--danger);
  }

  /* prob bar */
  .prob-wrap { background: #0d1015; border-radius: 8px; padding: 14px 16px; }
  .prob-row { display: flex; justify-content: space-between; align-items: baseline; }
  .prob-row .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
  .prob-row .val { font: 14px ui-monospace,Consolas,monospace; }
  .prob-bar {
    height: 8px; background: var(--line); border-radius: 99px;
    overflow: hidden; margin-top: 10px;
  }
  .prob-fill {
    height: 100%; background: linear-gradient(90deg, #3b82f6, var(--accent));
    width: 0%; transition: width 80ms linear;
  }
  .flash { animation: flash .6s ease-out; }
  @keyframes flash { 0% { box-shadow: 0 0 0 4px var(--ok) inset; } }

  /* recording overlay */
  .overlay {
    position: fixed; inset: 0; background: rgba(15,17,22,.92);
    display: flex; align-items: center; justify-content: center;
    z-index: 50; backdrop-filter: blur(6px);
  }
  .overlay-inner {
    text-align: center;
  }
  .overlay .count {
    font: 700 96px/1 -apple-system,Segoe UI,Inter,sans-serif;
    color: var(--fg);
  }
  .overlay .recording-label {
    color: var(--danger); font-weight: 700; font-size: 22px;
    letter-spacing: .15em; text-transform: uppercase;
    animation: pulse 1s ease-in-out infinite;
  }
  .overlay .saving-label { color: var(--muted); font-size: 16px; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }

  /* settings drawer */
  .drawer {
    position: fixed; top: 0; right: 0; height: 100%; width: 360px;
    background: #1a1d24; border-left: 1px solid var(--line);
    padding: 24px; transform: translateX(100%); transition: transform .25s;
    z-index: 40; overflow-y: auto;
  }
  .drawer.open { transform: translateX(0); }
  .drawer h3 { margin: 0 0 16px; font-size: 14px; color: var(--muted);
                text-transform: uppercase; letter-spacing: .08em; }
  .drawer .field { margin-bottom: 18px; }
  .drawer .field label { display: block; margin-bottom: 6px;
                          font-size: 13px; color: var(--muted); }
  .drawer .field input[type="number"], .drawer .field input[type="text"] {
    width: 100%; padding: 8px 10px; background: #0d1015;
    color: var(--fg); border: 1px solid var(--line); border-radius: 6px;
    font: inherit;
  }
  .drawer .field label.toggle {
    display: flex; align-items: center; gap: 8px; cursor: pointer;
    color: var(--fg); margin: 0;
  }
  .drawer .close { position: absolute; top: 18px; right: 18px;
                   color: var(--muted); cursor: pointer; font-size: 22px;
                   background: none; padding: 4px 10px; }

  /* misc */
  .muted { color: var(--muted); }
  .hidden { display: none !important; }
  details { margin-top: 14px; font-size: 13px; }
  details summary { color: var(--muted); cursor: pointer; padding: 4px 0; }
  details summary:hover { color: var(--fg); }
  .clip-list {
    margin-top: 10px; max-height: 200px; overflow-y: auto;
    border: 1px solid var(--line); border-radius: 6px;
  }
  .clip-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; border-bottom: 1px solid var(--line);
    font: 12px ui-monospace,Consolas,monospace; gap: 8px;
  }
  .clip-row:last-child { border-bottom: 0; }
  .clip-row audio { height: 24px; flex: 1; }
  .clip-row button { padding: 2px 8px; font-size: 12px; }
  .log {
    margin-top: 12px; background: #0d1015; padding: 12px;
    border-radius: 6px; max-height: 220px; overflow-y: auto;
    font: 11px/1.5 ui-monospace,Consolas,monospace;
    color: #c9cdd2; white-space: pre-wrap;
  }

  /* phrase + project selector */
  .project-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 0 4px;
  }
  .phrase-label { color: var(--muted); font-size: 13px; }
  .phrase { font-weight: 600; }
  select, .drawer select {
    background: #0d1015; color: var(--fg); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 8px; font: inherit;
    max-width: 100%; box-sizing: border-box;
  }
  /* Drawer selects fill the panel width and never push it off-screen, even
     with long option text (the native dropdown still shows full labels). */
  .drawer select { width: 100%; }
</style>
</head>
<body>

<header>
  <span class="title">heed</span>
  <div class="meta">
    <span id="env-line"></span>
    <button class="ghost" id="open-settings" style="padding: 6px 10px; margin-left: 8px;">⚙</button>
  </div>
</header>

<main>

  <!-- Project bar (always at top) -->
  <div class="project-bar" id="project-bar">
    <span class="phrase-label">Wake phrase</span>
    <select id="project-select"></select>
    <button class="ghost" id="new-project" style="padding: 6px 12px; font-size: 13px;">+ new</button>
  </div>

  <!-- New project form (hidden until needed) -->
  <section id="new-project-section" class="hidden">
    <h2>Create project</h2>
    <p class="hint">Pick a name (any string) and a phrase you want to trigger on.</p>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
      <input id="new-name" placeholder="name" style="flex: 1; min-width: 120px; padding: 10px; background: #0d1015; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; font: inherit;">
      <input id="new-phrase" placeholder='phrase, e.g. "hey andre"' style="flex: 2; min-width: 180px; padding: 10px; background: #0d1015; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; font: inherit;">
      <button id="new-create">Create</button>
    </div>
  </section>

  <!-- Step 1 -->
  <section id="step-pos">
    <h2>1 - Record the wake word</h2>
    <p class="hint" id="hint-pos">Say it once per click. <b>3 seconds per take</b> - speak naturally and don't rush.</p>
    <div class="dots" id="dots-pos"></div>
    <button class="cta" id="rec-pos"><span class="rec-dot"></span>Record</button>
    <details>
      <summary>Recorded clips</summary>
      <div id="list-pos"></div>
      <div style="margin-top: 8px;">
        <input type="file" id="upload-pos" accept=".wav" multiple class="hidden">
        <a href="#" id="upload-pos-link">Or drop .wav files here →</a>
      </div>
    </details>
  </section>

  <!-- Step 2 -->
  <section id="step-neg">
    <h2>2 - Record any other sentence</h2>
    <p class="hint">Just talk - different short sentences each time. These are "what NOT to fire on".</p>
    <div class="dots" id="dots-neg"></div>
    <button class="cta" id="rec-neg"><span class="rec-dot"></span>Record</button>
    <details>
      <summary>Recorded clips</summary>
      <div id="list-neg"></div>
      <div style="margin-top: 8px;">
        <input type="file" id="upload-neg" accept=".wav" multiple class="hidden">
        <a href="#" id="upload-neg-link">Or drop .wav files here →</a>
      </div>
    </details>
  </section>

  <!-- Step 2b - Phonetic neighbors (only meaningful if there's a phrase) -->
  <section id="step-neighbors" class="hidden">
    <h2>2b - RECORD PHONETIC NEIGHBORS  <span style="color:var(--accent); font-weight:600; text-transform:none; letter-spacing:0;">highly recommended</span></h2>
    <p class="hint">
      The biggest lever for fixing "hey X / Y andre triggers it too". Say
      each of these <b>once</b>, in your normal voice. Takes ~30 s total
      and improves discrimination way more than another TTS run would.
    </p>
    <div id="neighbors-list"></div>
  </section>

  <!-- Step 2c - Hard-negative library (the things that false-trigger) -->
  <section id="step-hardneg" class="hidden">
    <h2>2c - RECORD CRITICAL HARD NEGATIVES  <span style="color:var(--accent); font-weight:600; text-transform:none; letter-spacing:0;">critical</span></h2>
    <p class="hint">
      <b>This is what fixes "breathing / typing / 'hey' alone triggers the model".</b>
      The TTS distractors cover other speakers' "hey X" phrases - they don't
      cover YOU breathing, YOU clicking, YOU saying "hey" to yourself, etc.
      Each item below takes 5-10 seconds. Combined: ~60 s of recording, and
      it's the biggest single fix for the streaming false-triggers we just
      saw in the log.
    </p>
    <div id="hardneg-list" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"></div>
  </section>

  <!-- Step 2d - Room ambient -->
  <section id="step-ambient">
    <h2>2d - capture your room ambient  <span style="color:var(--muted); font-weight:400; text-transform:none; letter-spacing:0;">recommended</span></h2>
    <p class="hint">
      Stay silent for <b>5 seconds</b> while your mic captures just the
      room. This noise is mixed into the synthetic TTS samples during
      training so they sound recorded by your mic - bridges the
      synth-real domain gap.
    </p>
    <div style="display:flex; gap:12px; align-items:center;">
      <button class="cta" id="rec-ambient"><span class="rec-dot"></span>Record 5 s of silence</button>
      <span class="muted" id="ambient-status" style="font-size: 13px;"></span>
    </div>
  </section>

  <!-- Step 3 -->
  <section id="step-train">
    <h2>3 - Train</h2>
    <p class="hint" id="train-hint">Mixes your recordings with hundreds of synthetic speakers and trains a small model. ~15-30 seconds.</p>
    <div id="train-plan" style="margin-bottom: 12px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; font-size: 13px;"></div>
    <details style="margin-bottom: 14px;">
      <summary style="color: var(--muted); font-size: 13px; cursor: pointer;">Tips for higher quality ↓</summary>
      <div id="quality-tips" style="margin-top: 10px; padding: 10px 12px; background: #0d1015; border-radius: 6px; font-size: 13px; line-height: 1.6;"></div>
    </details>
    <button class="cta" id="train-btn">Train model</button>
    <div id="train-status" class="muted" style="margin-top: 10px; font-size: 13px;"></div>
    <details id="train-log-wrap" class="hidden">
      <summary>Training log</summary>
      <pre class="log" id="train-log"></pre>
    </details>
  </section>

  <!-- TTS cache inspector (between steps 3 and 4, only when present) -->
  <section id="step-tts-cache" class="hidden">
    <h2>TTS samples used in last training</h2>
    <p class="hint" id="tts-cache-hint">Listen to these to verify pronunciation and quality. If a voice mangles your phrase, add explicit distractors for the bad-sounding word so the model doesn't lean on those samples.</p>
    <details>
      <summary id="tts-pos-summary">Positives - synthetic "<span id="tts-cache-phrase"></span>"</summary>
      <div id="tts-pos-list" class="clip-list" style="margin-top:10px;"></div>
    </details>
    <details style="margin-top:10px;">
      <summary id="tts-neg-summary">Hard-negative phrases</summary>
      <div id="tts-neg-list" style="margin-top:10px;"></div>
    </details>
  </section>

  <!-- Step 3.5 - Saved model slots -->
  <section id="step-models" class="hidden">
    <h2>3.5 - Saved models  <span style="color:var(--muted); font-weight:400; text-transform:none; letter-spacing:0;">slots</span></h2>
    <p class="hint">
      Every training also saves a copy under <code>models/&lt;size&gt;.pt</code>.
      Train at multiple sizes (small / medium / large) and switch between them
      without losing the others. The "active" model - the one used by Test,
      Listen, and Export - is whichever you last activated.
    </p>
    <div id="models-list" style="margin-top: 12px;"></div>
  </section>

  <!-- Step 4 -->
  <section id="step-listen" class="hidden">
    <h2>4 - Test</h2>
    <p class="hint" id="listen-hint">
      Quick diagnostic: click "Test once" and say the phrase, see the score.
      Then "Listen" for continuous mode.
    </p>

    <!-- One-shot diagnostic -->
    <div style="margin-bottom: 18px;">
      <button class="cta" id="test-once-btn"><span class="rec-dot"></span>Test once</button>
      <span class="muted" id="test-once-status" style="margin-left: 10px; font-size: 13px;"></span>
      <div id="test-once-result" class="hidden" style="margin-top: 12px; padding: 12px; border: 1px solid var(--line); border-radius: 6px;">
        <div class="prob-row"><span class="label">probability</span><span class="val" id="t1-prob">-</span></div>
        <div class="prob-bar" style="margin-top: 8px;"><div class="prob-fill" id="t1-fill"></div></div>
        <div class="prob-row" style="margin-top: 12px;"><span class="label">decision</span><span class="val" id="t1-decision">-</span></div>
        <div class="prob-row" style="margin-top: 4px;"><span class="label">rms</span><span class="val" id="t1-rms">-</span></div>
        <div class="prob-row" style="margin-top: 4px;"><span class="label">voice band (100-7000 Hz)</span><span class="val" id="t1-band">-</span></div>
        <div class="prob-row" style="margin-top: 4px;"><span class="label">rumble (&lt;100 Hz)</span><span class="val" id="t1-rumble">-</span></div>
        <div class="prob-row" style="margin-top: 4px;"><span class="label">hiss (&gt;7000 Hz)</span><span class="val" id="t1-hiss">-</span></div>
        <div id="t1-spectrum-hint" class="muted hidden" style="font-size: 12px; margin-top: 8px; padding: 8px; background: #0d1015; border-radius: 4px;"></div>
        <audio id="t1-audio" controls style="width: 100%; margin-top: 10px;"></audio>
      </div>
    </div>

    <!-- Cross-speaker (held-out) test -->
    <div style="margin-bottom: 18px;">
      <button id="holdout-btn">Cross-speaker test (3 unseen voices)</button>
      <span class="muted" id="holdout-status" style="margin-left: 10px; font-size: 13px;"></span>
      <div id="holdout-result" class="hidden" style="margin-top: 12px;"></div>
    </div>

    <!-- Cross-TTS (Kokoro held-out) test -->
    <div style="margin-bottom: 18px;">
      <button id="crosstts-btn">Cross-TTS test (Kokoro voices · different family)</button>
      <span class="muted" id="crosstts-status" style="margin-left: 10px; font-size: 13px;"></span>
      <div id="crosstts-result" class="hidden" style="margin-top: 12px;"></div>
      <div class="muted" style="font-size: 12px; margin-top: 6px; max-width: 720px;">
        The cross-speaker test above uses held-out Piper voices - same neural
        family as the training data. This test uses Kokoro voices instead - a
        different acoustic family - so passing it is much closer evidence of
        real-human generalization. Requires <code>pip install kokoro-onnx</code>
        + <code>heed download-kokoro</code>.
      </div>
    </div>

    <!-- Self-test on user's own training data -->
    <div style="margin-bottom: 18px;">
      <button id="selftest-btn">Self-test (model vs YOUR recordings)</button>
      <span class="muted" id="selftest-status" style="margin-left: 10px; font-size: 13px;"></span>
      <div id="selftest-result" class="hidden" style="margin-top: 12px;"></div>
      <div class="muted" style="font-size: 12px; margin-top: 6px; max-width: 720px;">
        Scores the trained model on the WAV files in <code>positive/</code> and
        <code>negative/</code>. This is the most directly relevant metric for
        "does this trigger when I say it" - val accuracy and the cross-speaker
        tests are TTS-dominated and can both pass while the model fails on
        your actual voice. If self-test TPR is high but Test Once fails,
        your test-time pronunciation/distance/volume differs from your training
        recordings - re-record positives the way you'll actually use the wake
        word (natural pace, normal distance from mic).
      </div>
    </div>

    <!-- Continuous live mode -->
    <div class="prob-wrap" id="prob-wrap">
      <div class="prob-row">
        <span class="label">probability</span>
        <span class="val" id="prob-val">0.00</span>
      </div>
      <div class="prob-bar"><div class="prob-fill" id="prob-fill"></div></div>
      <div class="prob-row" style="margin-top: 10px;">
        <span class="label">rms</span><span class="val" id="prob-rms">-</span>
      </div>
      <div class="prob-row" style="margin-top: 4px;">
        <span class="label">voice band</span><span class="val" id="prob-band">-</span>
      </div>
      <div class="prob-row" style="margin-top: 4px;">
        <span class="label">gate</span><span class="val" id="prob-gate">-</span>
      </div>
      <div class="prob-row" style="margin-top: 10px;">
        <span class="label">triggers</span>
        <span class="val" id="trigger-count">0</span>
      </div>
    </div>
    <div style="display: flex; gap: 8px; margin-top: 14px; align-items: center; flex-wrap: wrap;">
      <button id="listen-start">▶ Listen</button>
      <button id="listen-stop" class="ghost hidden">■ Stop</button>
      <label style="font-size: 13px; color: var(--muted); margin-left: 6px;">
        backend:
        <select id="listen-backend" style="padding: 4px 8px; background: #0d1015; color: var(--fg); border: 1px solid var(--line); border-radius: 4px; font: inherit;">
          <option value="pytorch" selected>PyTorch (active model.pt)</option>
          <option value="onnx_fp32">ONNX fp32 (export/wake.onnx)</option>
          <option value="onnx_int8">ONNX int8 (export/wake.int8.onnx)</option>
        </select>
      </label>
      <button id="listen-log-copy" class="ghost hidden" style="margin-left: auto;">Copy log</button>
    </div>
    <div id="listen-log-wrap" class="hidden" style="margin-top: 14px; padding: 10px 12px; background: #0d1015; border-radius: 6px; max-height: 280px; overflow: auto;">
      <div style="display: grid; grid-template-columns: 60px 70px 70px 90px 60px 60px 60px 70px 1fr; gap: 6px; font-size: 11px; font-family: ui-monospace, Consolas, monospace; color: var(--muted); padding-bottom: 4px; border-bottom: 1px solid var(--line);">
        <div>t (s)</div><div>prob</div><div>ema</div><div>rms dBFS</div><div>voice%</div><div>rumble%</div><div>hiss%</div><div>trig</div><div>gate</div>
      </div>
      <div id="listen-log-rows" style="font-family: ui-monospace, Consolas, monospace; font-size: 11px;"></div>
    </div>
  </section>

  <!-- Step 5 - Export to ONNX -->
  <section id="step-export" class="hidden">
    <h2>5 - Export to ONNX  <span style="color:var(--muted); font-weight:400; text-transform:none; letter-spacing:0;">deployment</span></h2>
    <p class="hint">
      Export your trained model to ONNX format for deployment on phones,
      browsers, embedded devices, or any platform with an
      <a href="https://onnxruntime.ai" target="_blank">ONNX Runtime</a>.
      Float32 is bit-exact identical to PyTorch; INT8 is ~3× smaller and
      typically indistinguishable in accuracy at the calibrated threshold.
    </p>
    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px;">
      <button id="export-btn" class="cta">Export ONNX (fp32 + int8)</button>
      <button id="export-verify-btn" class="ghost">Verify vs PyTorch</button>
      <button id="send-mobile-btn" class="ghost">Send to mobile demo</button>
      <select id="send-slot" title="which mobile slot to overwrite">
        <option value="0">to slot 1</option>
        <option value="1">to slot 2</option>
        <option value="2">to slot 3</option>
        <option value="3">to slot 4</option>
        <option value="4" selected>to slot 5 (custom)</option>
      </select>
      <span class="muted" id="export-status" style="font-size: 13px;"></span>
    </div>

    <!-- Exported files list -->
    <div id="export-files" class="hidden" style="margin-bottom: 16px; padding: 12px; border: 1px solid var(--line); border-radius: 6px;">
      <div style="font-size: 13px; color: var(--muted); margin-bottom: 8px;">
        Files in <code>&lt;project&gt;/export/</code> - click any to download.
      </div>
      <div id="export-files-list"></div>
    </div>

    <!-- Verify results -->
    <div id="export-verify-result" class="hidden" style="margin-bottom: 16px;"></div>

    <details style="margin-top: 14px;">
      <summary style="color: var(--muted); font-size: 13px; cursor: pointer;">Deployment notes ↓</summary>
      <div class="muted" style="font-size: 13px; line-height: 1.6; margin-top: 10px; padding: 10px 12px; background: #0d1015; border-radius: 6px;">
        <p style="margin: 0 0 8px;"><b>What's in the export:</b></p>
        <ul style="margin: 0 0 8px; padding-left: 20px;">
          <li><code>wake.onnx</code> - fp32 model, input: log-mel features <code>(B, 40, 101)</code>, output: raw logit. Apply sigmoid in your runtime.</li>
          <li><code>wake.int8.onnx</code> - INT8 quantized variant, same I/O. Use for size-constrained deployment.</li>
          <li><code>wake.json</code> - metadata: threshold, sample rate, preprocessing chain spec.</li>
          <li><code>README.md</code> - deployment code examples and full parameter reference.</li>
        </ul>
        <p style="margin: 8px 0 4px;"><b>Browser deployment:</b> use <code>onnxruntime-web</code> (WASM + SIMD). The model loads via <code>fetch()</code> + <code>InferenceSession</code>.</p>
        <p style="margin: 4px 0;"><b>Mobile:</b> <code>onnxruntime-objc</code> (iOS) or <code>onnxruntime-android</code> (Android). ~3 MB runtime + your &lt;250 KB model.</p>
        <p style="margin: 4px 0 0;"><b>Preprocessing</b> (must be reproduced in your target language): high-pass filter at 100 Hz + 50/60 Hz notches → peak normalize → log-mel → CMN. Full parameters in <code>wake.json[preprocessing]</code>.</p>
      </div>
    </details>
  </section>

</main>

<!-- Settings drawer -->
<aside class="drawer" id="drawer">
  <button class="close" id="close-settings">×</button>
  <h3>Settings</h3>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-tts">
      Use TTS augmentation (multi-speaker)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;" id="tts-status"></div>
  </div>
  <div class="field">
    <label>Piper positives (synthetic voices, family #1)</label>
    <input type="number" id="set-tts-count" value="400" min="0" max="2000">
  </div>
  <div class="field">
    <label>Kokoro positives (synthetic voices, family #2)</label>
    <input type="number" id="set-kokoro-count" value="200" min="0" max="1000">
    <div class="muted" style="font-size: 12px; margin-top: 4px;" id="kokoro-status">
      Mixing Kokoro positives with Piper ones forces the model to learn
      phrase-content features that survive a TTS-family change - the most
      effective single guard against the "passes cross-speaker test but
      fails on real humans" failure mode. Recommended: 150-300. Requires
      kokoro-onnx + downloaded voices.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-force-regen-tts">
      Force regenerate TTS samples (ignore cache)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      By default, TTS samples are cached per project (under
      <code>tts_cache/</code>) and reused whenever phrase / voice / count /
      seed are unchanged. Check this only when you want fresh randomness
      with the same settings - e.g. to verify the cache isn't degrading
      results. Cache auto-invalidates on real param changes.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-autodist" checked>
      Auto hard-negatives (common confusable phrases)
    </label>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-partials" checked>
      Partial-utterance negatives (split halves of the phrase)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Adds first-half / second-half clips of each positive (and TTS positive)
      as negatives - forces the model to require the FULL phrase, not just
      "hey" or "andre" alone.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-specaug" checked>
      SpecAugment (frequency + time masking)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Random freq/time masks on the mel spectrogram during training. Cheap, proven, on by default.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-rir" checked>
      Real RIR convolution (parametric room reverb)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Replaces comb-filter reverb with parametric image-source RIR convolution. ~20% accuracy gain in KWS literature.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-noisepool" checked>
      Parametric noise pool (white / pink / brown / hum / fan / babble)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      30 procedurally-generated noise samples across real-world classes, mixed in alongside your room ambient.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-specmatch" checked>
      Spectral envelope matching (lightweight voice conversion)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Applies an EQ to TTS samples that pulls their power spectrum toward your mic's profile. Bridges synth-real gap in &lt;1 ms per clip, no extra dependencies.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-focal" checked>
      Focal loss (better for class imbalance)
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Down-weights easy examples so gradient budget concentrates on hard ones. Default on.
    </div>
  </div>
  <div class="field">
    <label class="toggle">
      <input type="checkbox" id="set-endalign" checked>
      End-aligned positive variants
    </label>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Adds copies of each positive shifted to the right edge of the window. Teaches the model to fire on phrase completion regardless of position in the sliding inference buffer.
    </div>
  </div>
  <div class="field">
    <label>Model size</label>
    <select id="set-model-size" style="padding: 8px; background: #0d1015; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; font: inherit;">
      <option value="small">small · ~10K params · 41 KB</option>
      <option value="medium" selected>medium · ~27K params · 105 KB</option>
      <option value="large">large · ~60K params · 235 KB</option>
    </select>
    <div class="muted" style="font-size: 12px; margin-top: 4px;">
      Bigger model = more capacity to discriminate phonetically-close phrases.
      Try medium first if "hey doc" vs. "hey john" still confuses; large for
      the hardest cases. All sizes still tiny and fast.
    </div>
  </div>
  <div class="field">
    <label>Epochs</label>
    <input type="number" id="set-epochs" value="100" min="5" max="100">
  </div>
  <div class="field">
    <label>Target false-positive rate at calibration</label>
    <input type="number" id="set-fpr" value="0.01" min="0.001" max="0.1" step="0.005">
  </div>
  <div class="field">
    <label>Listen threshold (override; blank = auto)</label>
    <input type="number" id="set-thr" placeholder="auto" min="0.0" max="1.0" step="0.01">
  </div>
  <h3 style="margin-top: 28px;">Environment</h3>
  <div class="muted" id="env-detail" style="font-size: 13px;"></div>
</aside>

<script>
const RECORD_SECONDS = 3.0;          // user has 3 s per take; silence is auto-trimmed server-side
const COUNTDOWN_MS_PER_NUMBER = 700; // 3-2-1, ~2.1 s total before recording starts
const RECOMMENDED_CLIPS = 8;
const $ = sel => document.querySelector(sel);

let env = {};
let currentProject = null;
let currentInfo = null;
let liveSession = null;
let liveStop = null;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let err; try { err = (await r.json()).error; } catch {}
    throw new Error(err || `${r.status} ${r.statusText}`);
  }
  if ((r.headers.get("content-type") || "").includes("application/json")) return r.json();
  return r;
}

// ---------- audio capture ----------
// AudioWorklet replaces the deprecated ScriptProcessorNode. The processor
// itself is just a passthrough that posts the raw PCM frames to the main
// thread, where we accumulate and chunk them.
const _workletCode = `
  class PCMRecorderProcessor extends AudioWorkletProcessor {
    process(inputs) {
      const input = inputs[0];
      if (input && input[0] && input[0].length > 0) {
        // Copy because the underlying buffer is reused
        this.port.postMessage(new Float32Array(input[0]));
      }
      return true;
    }
  }
  registerProcessor('pcm-recorder', PCMRecorderProcessor);
`;
const _workletURL = URL.createObjectURL(
  new Blob([_workletCode], { type: 'application/javascript' })
);

async function _openMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.audioWorklet.addModule(_workletURL);
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'pcm-recorder');
  // Keep the audio graph "alive" without playing audio back to the user
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(node); node.connect(sink); sink.connect(ctx.destination);
  const close = async () => {
    try { src.disconnect(); node.disconnect(); sink.disconnect(); } catch {}
    stream.getTracks().forEach(t => t.stop());
    try { await ctx.close(); } catch {}
  };
  return { ctx, node, close };
}

async function recordFloat32(durationSec) {
  const { ctx, node, close } = await _openMic();
  const chunks = [];
  node.port.onmessage = e => chunks.push(e.data);
  await new Promise(r => setTimeout(r, durationSec * 1000));
  await close();
  let total = 0; for (const c of chunks) total += c.length;
  const flat = new Float32Array(total);
  let off = 0; for (const c of chunks) { flat.set(c, off); off += c.length; }
  return { samples: flat, sampleRate: ctx.sampleRate };
}

async function startStream(onChunk, chunkMs = 120) {
  const { ctx, node, close } = await _openMic();
  const sr = ctx.sampleRate;
  const samplesPerChunk = Math.round(sr * chunkMs / 1000);
  let buf = new Float32Array(0);
  node.port.onmessage = e => {
    const data = e.data;
    const merged = new Float32Array(buf.length + data.length);
    merged.set(buf, 0); merged.set(data, buf.length);
    buf = merged;
    while (buf.length >= samplesPerChunk) {
      const out = buf.slice(0, samplesPerChunk);
      buf = buf.slice(samplesPerChunk);
      onChunk(out, sr);
    }
  };
  return () => { close(); };
}

// ---------- recording overlay ----------
async function recordWithOverlay(kind) {
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const inner = document.createElement("div");
  inner.className = "overlay-inner";
  overlay.appendChild(inner);
  document.body.appendChild(overlay);

  try {
    // Countdown
    for (let c = 3; c >= 1; c--) {
      inner.innerHTML = `<div class="count">${c}</div>`;
      await new Promise(r => setTimeout(r, COUNTDOWN_MS_PER_NUMBER));
    }
    // Recording
    inner.innerHTML = `<div class="recording-label">● Recording…</div>
                       <div class="muted" style="margin-top:14px;">Say it now. ${RECORD_SECONDS} seconds.</div>`;
    const { samples, sampleRate } = await recordFloat32(RECORD_SECONDS);
    // Save
    inner.innerHTML = `<div class="saving-label">Saving…</div>`;
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    await api(`/api/projects/${encodeURIComponent(currentProject)}/record?kind=${kind}&sr=${Math.round(sampleRate)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: buf,
    });
    inner.innerHTML = `<div style="color:var(--ok); font-weight:700; font-size:20px;">✓ saved</div>`;
    await new Promise(r => setTimeout(r, 400));
  } finally {
    overlay.remove();
  }
}

// ---------- rendering ----------
function dots(count, recommended) {
  const out = document.createDocumentFragment();
  const n = Math.max(count, recommended);
  for (let i = 0; i < n; i++) {
    const d = document.createElement("div");
    d.className = "dot" + (i < count ? " filled" : "");
    out.appendChild(d);
  }
  return out;
}

function clipList(target, kind, clips) {
  target.innerHTML = "";
  if (clips.length === 0) {
    target.innerHTML = '<div class="muted" style="font-size:12px; padding: 8px 0;">no clips yet</div>';
    return;
  }
  const list = document.createElement("div"); list.className = "clip-list";
  for (const fn of clips) {
    const row = document.createElement("div"); row.className = "clip-row";
    const name = document.createElement("span"); name.textContent = fn;
    const audio = document.createElement("audio"); audio.controls = true;
    audio.src = `/api/projects/${encodeURIComponent(currentProject)}/clip/${kind}/${encodeURIComponent(fn)}`;
    const del = document.createElement("button"); del.className = "danger"; del.textContent = "✕";
    del.onclick = async () => {
      if (!confirm(`delete ${fn}?`)) return;
      await api(`/api/projects/${encodeURIComponent(currentProject)}/clip`, {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, filename: fn })
      });
      await refresh();
    };
    row.append(name, audio, del);
    list.appendChild(row);
  }
  target.appendChild(list);
}

async function loadEnv() {
  env = await api("/api/projects");
  const ttsBadge = env.tts_available
    ? "TTS ready"
    : (env.voice_downloaded && !env.piper_importable)
      ? "TTS broken"
      : "no TTS";
  $("#env-line").innerHTML =
    `<b>${env.cuda ? "GPU" : "CPU"}</b> · ${ttsBadge}`;

  let ttsLine;
  if (env.tts_available) {
    ttsLine = "downloaded and importable";
  } else if (env.voice_downloaded && !env.piper_importable) {
    ttsLine = `voice downloaded but <b>piper-tts is not importable</b> from this Python. ` +
              `Run <code>python -m heed.cli doctor</code> in the same terminal for details.`;
  } else if (!env.voice_downloaded) {
    ttsLine = "not downloaded - run <code>heed download-tts</code>";
  } else {
    ttsLine = "unavailable";
  }
  $("#env-detail").innerHTML =
    `Workspace: <span style="font-family:ui-monospace,Consolas,monospace; word-break:break-all;">${env.workspace}</span><br>` +
    `Device: ${env.cuda ? "CUDA available" : "CPU only"}<br>` +
    `TTS: ${ttsLine}`;
  $("#set-tts").checked = env.tts_available;
  $("#set-tts").disabled = !env.tts_available;
  if (env.tts_available) {
    $("#tts-status").textContent = "Multi-speaker voice ready. Recommended.";
  } else if (env.voice_downloaded && !env.piper_importable) {
    $("#tts-status").innerHTML =
      "Voice file present but <b>piper-tts cannot be imported</b>. " +
      "Run <code>python -m heed.cli doctor</code> to diagnose.";
  } else {
    $("#tts-status").textContent =
      "Run `python -m heed.cli download-tts` and `pip install piper-tts` to enable.";
  }
}

async function loadProjects() {
  const sel = $("#project-select");
  sel.innerHTML = "";
  for (const p of env.projects) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = `"${p.phrase}"  (${p.name})${p.has_model ? "  ✓" : ""}`;
    sel.appendChild(opt);
  }
  if (env.projects.length === 0) {
    showNewProjectForm(true);
    document.querySelectorAll("section").forEach(s => {
      if (s.id !== "new-project-section") s.classList.add("hidden");
    });
    return;
  }
  // restore last selection from localStorage if possible
  const saved = localStorage.getItem("heed-project");
  if (saved && env.projects.find(p => p.name === saved)) sel.value = saved;
  currentProject = sel.value;
  localStorage.setItem("heed-project", currentProject);
  await refresh();
}

function showNewProjectForm(show) {
  $("#new-project-section").classList.toggle("hidden", !show);
}

async function refresh() {
  if (!currentProject) return;
  currentInfo = await api(`/api/projects/${encodeURIComponent(currentProject)}`);
  // reveal all the right sections
  ["step-pos", "step-neg", "step-train"].forEach(id =>
    $("#" + id).classList.remove("hidden"));
  $("#step-listen").classList.toggle("hidden", !currentInfo.has_model);
  $("#step-export").classList.toggle("hidden", !currentInfo.has_model);
  $("#step-models").classList.toggle("hidden", !currentInfo.has_model);
  if (currentInfo.has_model) {
    refreshExportInfo();
    refreshModelsList();
  }
  $("#new-project-section").classList.add("hidden");

  // Hint with phrase
  $("#hint-pos").innerHTML = `Say <b>"${currentInfo.phrase}"</b> once per take. <b>3 seconds per take</b> - speak naturally and don't rush.`;

  // dots
  const dpos = $("#dots-pos"); dpos.innerHTML = "";
  dpos.appendChild(dots(currentInfo.positives.length, RECOMMENDED_CLIPS));
  const dneg = $("#dots-neg"); dneg.innerHTML = "";
  dneg.appendChild(dots(currentInfo.negatives.length, RECOMMENDED_CLIPS));

  // clip lists (inside details)
  clipList($("#list-pos"), "positive", currentInfo.positives);
  clipList($("#list-neg"), "negative", currentInfo.negatives);

  // train hint
  const enough = currentInfo.positives.length >= 4 && currentInfo.negatives.length >= 4;
  $("#train-btn").disabled = !enough;
  if (!enough) {
    $("#train-status").textContent = "Record at least 4 of each to enable training.";
  } else if (currentInfo.has_model) {
    const m = currentInfo.model_info || {};
    const kokoroBit = (m.kokoro_positives_used > 0)
      ? ` + ${m.kokoro_positives_used} Kokoro`
      : ``;
    const ttsBadge = (m.tts_positives_used > 0 || m.kokoro_positives_used > 0)
      ? `<span style="color:var(--ok);">TTS on (${m.tts_positives_used||0} Piper${kokoroBit}, ${m.tts_distractor_phrases||0} distractors)</span>`
      : `<span style="color:var(--danger);">TTS was OFF</span>`;
    // User-voice-only metric - the most directly relevant gauge. Val acc is
    // misleading (dominated by TTS samples); this one tells you "does the
    // model trigger when YOU say it" from your training recordings.
    let userVoiceBadge = "";
    if (m.n_user_positives_eval > 0) {
      const tpr = m.user_voice_tpr || 0;
      const fpr = m.user_voice_fpr || 0;
      const tprColor = tpr >= 0.9 ? "var(--ok)" : tpr >= 0.6 ? "var(--warn, #f59e0b)" : "var(--danger)";
      const fprColor = fpr <= 0.1 ? "var(--ok)" : fpr <= 0.3 ? "var(--warn, #f59e0b)" : "var(--danger)";
      userVoiceBadge = ` · <span style="color:${tprColor};">your voice TPR ${(tpr*100).toFixed(0)}%</span>` +
                       ` <span style="color:${fprColor};">FPR ${(fpr*100).toFixed(0)}%</span>`;
    }
    $("#train-status").innerHTML =
      `<span style="color:var(--ok);">model ready</span> · threshold ${(m.threshold||0).toFixed(2)} · ` +
      `${m.n_params||"?"} params · trained in ${(m.seconds||0).toFixed(1)}s · ${ttsBadge}${userVoiceBadge}`;
  } else {
    $("#train-status").textContent = "";
  }

  // "Will train with…" plan above the button
  renderTrainPlan();
  renderQualityTips();

  // TTS cache (only show if we have any samples)
  await renderTtsCache();

  // Phonetic neighbors + ambient
  await renderNeighbors();
  await renderHardNegs();
  await renderAmbient();
}

function renderQualityTips() {
  if (!currentInfo) return;
  const el = $("#quality-tips");
  if (!el) return;
  const nPos = currentInfo.positives.length;
  const nNeg = currentInfo.negatives.length;
  const tips = [];

  // What's already done well
  const done = [];
  if (nPos >= 8) done.push(`✓ ${nPos} wake-word recordings`);
  if (nNeg >= 8) done.push(`✓ ${nNeg} distractor recordings`);
  if (env.tts_available) done.push("✓ TTS multi-speaker augmentation");

  // What would help (in priority order)
  if (nPos < 20) {
    tips.push(`<b>Record more "${currentInfo.phrase}"</b> &mdash; you have <b>${nPos}</b>, target <b>15-25</b>. Each new recording with deliberate variation (close to mic, far away, soft voice, fast, slow, with background noise) is worth roughly <b>50 synthetic samples</b>.`);
  }
  tips.push(`<b>Step 2b - phonetic neighbors</b>: record 5-8 of "hey siri / hey google / hey fetch / hi andre" in <i>your own voice</i>. This is the single biggest fix for "fires on any 'hey X'".`);
  tips.push(`<b>Step 2c - room ambient</b>: 5 seconds of silence. Synthesized TTS samples get mixed with this so they sound recorded by your actual mic.`);
  tips.push(`<b>Always-on augmentation</b> &mdash; the trainer now applies SpecAugment (mel masking), parametric room reverb (RIR), a 6-class noise pool (white/pink/brown/hum/fan/babble), and spectral-envelope matching (TTS audio EQ-shifted toward your mic's spectrum). All on by default; toggle in ⚙ settings.`);

  let html = "";
  if (done.length) html += `<div style="color: var(--ok); margin-bottom: 10px;">${done.join("&nbsp;&nbsp;&middot;&nbsp;&nbsp;")}</div>`;
  html += `<ul style="margin: 0; padding-left: 20px;">${tips.map(t => `<li style="margin-bottom: 6px;">${t}</li>`).join("")}</ul>`;
  el.innerHTML = html;
}

async function renderNeighbors() {
  if (!currentProject) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/suggested_neighbors`);
    const section = $("#step-neighbors");
    if (!data.suggestions || data.suggestions.length === 0) {
      section.classList.add("hidden"); return;
    }
    section.classList.remove("hidden");
    const list = $("#neighbors-list");
    list.innerHTML = "";
    for (const phrase of data.suggestions.slice(0, 8)) {
      const safe = phrase.toLowerCase().replace(/[^a-z0-9]/g, "_").slice(0, 40);
      const count = (data.recorded[safe] || []).length;
      const row = document.createElement("div");
      row.style.cssText =
        "display:flex; align-items:center; gap:10px; padding:8px 10px; " +
        "border:1px solid var(--line); border-radius:6px; margin-bottom:6px;";
      const phraseEl = document.createElement("span");
      phraseEl.style.cssText = "flex:1; font-family:ui-monospace,Consolas,monospace;";
      phraseEl.innerHTML = `"${phrase}"`;
      const status = document.createElement("span");
      status.className = "muted";
      status.style.fontSize = "12px";
      status.textContent = count > 0 ? `✓ recorded (${count})` : "○ pending";
      status.style.color = count > 0 ? "var(--ok)" : "var(--muted)";
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.style.padding = "6px 12px";
      btn.textContent = count > 0 ? "+1 more" : "● Record";
      btn.onclick = () => doRecordNeighbor(phrase);
      row.append(phraseEl, status, btn);
      list.appendChild(row);
    }
  } catch (e) {
    $("#step-neighbors").classList.add("hidden");
  }
}

async function doRecordNeighbor(phrase) {
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const inner = document.createElement("div"); inner.className = "overlay-inner";
  overlay.appendChild(inner); document.body.appendChild(overlay);
  try {
    for (let c = 3; c >= 1; c--) {
      inner.innerHTML = `<div class="count">${c}</div><div class="muted" style="margin-top:14px;">Say: "${phrase}"</div>`;
      await new Promise(r => setTimeout(r, 700));
    }
    inner.innerHTML = `<div class="recording-label">● Recording…</div>
      <div class="muted" style="margin-top:14px;">Say "${phrase}" now.</div>`;
    const { samples, sampleRate } = await recordFloat32(3.0);
    inner.innerHTML = `<div class="saving-label">Saving…</div>`;
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    await api(
      `/api/projects/${encodeURIComponent(currentProject)}/record_neighbor?phrase=${encodeURIComponent(phrase)}&sr=${Math.round(sampleRate)}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf }
    );
    inner.innerHTML = `<div style="color:var(--ok); font-weight:700; font-size:20px;">✓ saved</div>`;
    await new Promise(r => setTimeout(r, 350));
  } catch (e) {
    inner.innerHTML = `<div style="color:var(--danger)">${e.message}</div>`;
    await new Promise(r => setTimeout(r, 1500));
  } finally {
    overlay.remove();
    await refresh();
  }
}

// ----- Hard-negative library (breathing / typing / mouth / etc.) -----------

const HARD_NEG_CATEGORIES = [
  { key: "breathing",        label: "Breathing (normal)",          prompt: "Breathe in and out at normal pace. NO words, just breath.", duration: 8.0 },
  { key: "mouth_sounds",     label: "Mouth / throat sounds",       prompt: "Sniff, cough, clear throat, smack lips, swallow. Mix it up.", duration: 8.0 },
  { key: "typing",           label: "Typing / clicking",           prompt: "Type, click the mouse, scroll, rustle papers. Be natural.", duration: 8.0 },
  { key: "mumbling",         label: "Quiet mumbling / humming",    prompt: "Mumble or hum quietly - anything except your wake phrase.", duration: 8.0 },
  { key: "loud_breath",      label: "Heavy breath / sigh / yawn",  prompt: "One or two deliberate sighs / yawns / loud exhales.", duration: 5.0 },
  { key: "distant_speech",   label: "Distant / quiet you-speech",  prompt: "Talk randomly (NOT the wake word) like you're chatting to someone. Move slightly away from the mic.", duration: 8.0 },
];

async function renderHardNegs() {
  if (!currentProject) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/hard_negative_status`);
    const section = $("#step-hardneg");
    section.classList.remove("hidden");
    const list = $("#hardneg-list");
    list.innerHTML = "";
    for (const cat of HARD_NEG_CATEGORIES) {
      const count = data.counts[cat.key] || 0;
      const card = document.createElement("div");
      card.style.cssText =
        "padding: 12px; border: 1px solid var(--line); border-radius: 6px; " +
        "display: flex; flex-direction: column; gap: 6px;";
      const titleRow = document.createElement("div");
      titleRow.style.cssText = "display:flex; justify-content:space-between; align-items:center;";
      titleRow.innerHTML =
        `<b>${cat.label}</b>` +
        (count > 0
          ? `<span class="muted" style="font-size:12px; color:var(--ok)">✓ recorded (${count})</span>`
          : `<span class="muted" style="font-size:12px;">○ pending</span>`);
      const promptEl = document.createElement("div");
      promptEl.className = "muted";
      promptEl.style.cssText = "font-size: 12px; line-height: 1.4;";
      promptEl.textContent = cat.prompt;
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.style.cssText = "align-self: flex-start; padding: 6px 14px; margin-top: 4px;";
      btn.textContent = count > 0 ? `+1 more (${cat.duration}s)` : `● Record ${cat.duration}s`;
      btn.onclick = () => doRecordHardNeg(cat);
      card.append(titleRow, promptEl, btn);
      list.appendChild(card);
    }
  } catch (e) {
    $("#step-hardneg").classList.add("hidden");
  }
}

async function doRecordHardNeg(cat) {
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const inner = document.createElement("div"); inner.className = "overlay-inner";
  overlay.appendChild(inner); document.body.appendChild(overlay);
  try {
    for (let c = 3; c >= 1; c--) {
      inner.innerHTML = `<div class="count">${c}</div>` +
        `<div class="muted" style="margin-top:14px; max-width:400px; text-align:center;">${cat.prompt}</div>`;
      await new Promise(r => setTimeout(r, 700));
    }
    inner.innerHTML = `<div class="recording-label">● Recording ${cat.duration}s…</div>` +
      `<div class="muted" style="margin-top:14px; max-width:400px; text-align:center;">${cat.prompt}</div>`;
    const { samples, sampleRate } = await recordFloat32(cat.duration);
    inner.innerHTML = `<div class="saving-label">Saving…</div>`;
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    await api(
      `/api/projects/${encodeURIComponent(currentProject)}/record_hard_negative?category=${encodeURIComponent(cat.key)}&sr=${Math.round(sampleRate)}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf }
    );
    inner.innerHTML = `<div style="color:var(--ok); font-weight:700; font-size:20px;">✓ saved</div>`;
    await new Promise(r => setTimeout(r, 350));
  } catch (e) {
    inner.innerHTML = `<div style="color:var(--danger)">${e.message}</div>`;
    await new Promise(r => setTimeout(r, 1500));
  } finally {
    overlay.remove();
    await refresh();
  }
}

async function renderAmbient() {
  if (!currentProject) return;
  try {
    const s = await api(`/api/projects/${encodeURIComponent(currentProject)}/ambient`);
    if (s.present) {
      $("#ambient-status").innerHTML =
        `<span style="color:var(--ok);">✓ captured</span> · ${s.duration_s.toFixed(1)}s · ` +
        `<a href="/api/projects/${encodeURIComponent(currentProject)}/ambient/audio" target="_blank">listen</a>`;
    } else {
      $("#ambient-status").innerHTML = '<span class="muted">○ not captured yet</span>';
    }
  } catch (e) {
    $("#ambient-status").textContent = "-";
  }
}

async function doRecordAmbient() {
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const inner = document.createElement("div"); inner.className = "overlay-inner";
  overlay.appendChild(inner); document.body.appendChild(overlay);
  try {
    inner.innerHTML = `<div class="count">3</div>
      <div class="muted" style="margin-top:14px;">Stay silent. Recording room ambient.</div>`;
    await new Promise(r => setTimeout(r, 700));
    for (let c = 2; c >= 1; c--) {
      inner.innerHTML = `<div class="count">${c}</div><div class="muted" style="margin-top:14px;">Stay silent.</div>`;
      await new Promise(r => setTimeout(r, 700));
    }
    inner.innerHTML = `<div class="recording-label">● Recording 5 s of silence…</div>`;
    const { samples, sampleRate } = await recordFloat32(5.0);
    inner.innerHTML = `<div class="saving-label">Saving…</div>`;
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    const res = await api(
      `/api/projects/${encodeURIComponent(currentProject)}/ambient?sr=${Math.round(sampleRate)}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf }
    );
    inner.innerHTML = `<div style="color:var(--ok); font-weight:700; font-size:20px;">✓ ${res.duration_s.toFixed(1)}s saved</div>`;
    await new Promise(r => setTimeout(r, 400));
  } catch (e) {
    inner.innerHTML = `<div style="color:var(--danger)">${e.message}</div>`;
    await new Promise(r => setTimeout(r, 1500));
  } finally {
    overlay.remove();
    await renderAmbient();
  }
}

async function renderTtsCache() {
  let cache;
  try {
    cache = await api(`/api/projects/${encodeURIComponent(currentProject)}/tts_cache`);
  } catch (e) { return; }
  const hasAny = ((cache.positives || []).length > 0) ||
                 (Object.keys(cache.negatives || {}).length > 0);
  const section = $("#step-tts-cache");
  if (section) section.classList.toggle("hidden", !hasAny);
  if (!hasAny) return;

  const phraseEl = $("#tts-cache-phrase");
  if (phraseEl) phraseEl.textContent = currentInfo.phrase;
  const posSummary = $("#tts-pos-summary");
  if (posSummary) posSummary.textContent =
    `Positives - ${cache.positives.length} synthetic "${currentInfo.phrase}" clips`;

  const posDiv = $("#tts-pos-list");
  if (!posDiv) return;  // page is stale, abort rest
  posDiv.innerHTML = "";
  // Cap display to 30 for browser perf; the rest are on disk anyway
  const visiblePos = cache.positives.slice(0, 30);
  for (const fn of visiblePos) {
    const row = document.createElement("div"); row.className = "clip-row";
    row.appendChild(Object.assign(document.createElement("span"), { textContent: fn }));
    const audio = document.createElement("audio"); audio.controls = true;
    audio.src = `/api/projects/${encodeURIComponent(currentProject)}/tts_clip/positive/${encodeURIComponent(fn)}`;
    row.appendChild(audio);
    posDiv.appendChild(row);
  }
  if (cache.positives.length > visiblePos.length) {
    const more = document.createElement("div");
    more.className = "muted"; more.style.padding = "8px";
    more.textContent = `+ ${cache.positives.length - visiblePos.length} more on disk under tts_cache/positive/`;
    posDiv.appendChild(more);
  }

  // Negatives - group by distractor phrase
  const negKeys = Object.keys(cache.negatives || {});
  const negSummary = $("#tts-neg-summary");
  if (negSummary) negSummary.textContent =
    `Hard-negative phrases - ${negKeys.length} distractor` +
    (negKeys.length === 1 ? "" : "s");
  const negDiv = $("#tts-neg-list");
  if (!negDiv) return;
  negDiv.innerHTML = "";
  for (const phraseKey of negKeys) {
    const block = document.createElement("details");
    block.style.marginBottom = "8px";
    const sum = document.createElement("summary");
    sum.style.color = "var(--muted)";
    sum.textContent = `${phraseKey}  (${cache.negatives[phraseKey].length} clips)`;
    block.appendChild(sum);
    const list = document.createElement("div"); list.className = "clip-list";
    list.style.marginTop = "8px";
    for (const fn of cache.negatives[phraseKey].slice(0, 8)) {
      const row = document.createElement("div"); row.className = "clip-row";
      row.appendChild(Object.assign(document.createElement("span"), { textContent: fn }));
      const audio = document.createElement("audio"); audio.controls = true;
      audio.src = `/api/projects/${encodeURIComponent(currentProject)}/tts_clip/negative/${encodeURIComponent(phraseKey)}/${encodeURIComponent(fn)}`;
      row.appendChild(audio);
      list.appendChild(row);
    }
    block.appendChild(list);
    negDiv.appendChild(block);
  }
}

function renderTrainPlan() {
  if (!currentInfo) return;
  const ttsOn = $("#set-tts").checked && env.tts_available;
  const ttsCount = parseInt($("#set-tts-count").value, 10) || 0;
  const kokoroCount = parseInt($("#set-kokoro-count").value, 10) || 0;
  const autoDist = $("#set-autodist").checked;
  const nPos = currentInfo.positives.length;
  const nNeg = currentInfo.negatives.length;

  let html;
  if (ttsOn && ttsCount > 0) {
    const nDistractors = autoDist ? 20 : 0;
    let plan =
      `<div style="color:var(--ok); font-weight:600; margin-bottom:4px;">✓ Cross-speaker training` +
      (kokoroCount > 0 ? ` + cross-TTS-family` : ``) +
      `</div>` +
      `<div class="muted">${nPos} of your wake-word clips · ${nNeg} of your distractors · ` +
      `<b style="color:var(--fg);">${ttsCount} Piper voices</b> saying "${currentInfo.phrase}"`;
    if (kokoroCount > 0) {
      plan += ` · <b style="color:var(--fg);">${kokoroCount} Kokoro voices</b> (different family)`;
    }
    if (autoDist) {
      plan += ` · <b style="color:var(--fg);">${nDistractors}×15 hard-negative phrases</b>`;
    }
    plan += `</div>`;
    html = plan;
  } else {
    let why = "";
    if (!env.tts_available) {
      if (env.voice_downloaded && !env.piper_importable) {
        why = `<a href="#" id="open-settings-link">piper-tts isn't importable from this Python</a> - open ⚙ for diagnostics`;
      } else if (!env.voice_downloaded) {
        why = `voice model not downloaded - run <code>heed download-tts</code>`;
      } else {
        why = `TTS is not enabled`;
      }
    } else {
      why = `TTS unchecked in ⚙ settings`;
    }
    html =
      `<div style="color:var(--danger); font-weight:600; margin-bottom:4px;">⚠ Speaker-locked training (no TTS)</div>` +
      `<div class="muted">Only your ${nPos} recordings will be used as positives. ` +
      `The model will be sensitive to your specific tone and may false-trigger on similar-sounding speech. ` +
      `<br>${why}.</div>`;
  }
  $("#train-plan").innerHTML = html;
  const link = $("#open-settings-link");
  if (link) link.onclick = e => { e.preventDefault(); $("#drawer").classList.add("open"); };
}

async function doRecord(kind) {
  $("#rec-pos").disabled = $("#rec-neg").disabled = true;
  try {
    await recordWithOverlay(kind);
    await refresh();
  } catch (e) {
    alert("Recording failed: " + e.message);
  } finally {
    $("#rec-pos").disabled = $("#rec-neg").disabled = false;
  }
}

async function doTrain() {
  $("#train-btn").disabled = true;
  $("#train-log-wrap").classList.remove("hidden");
  $("#train-log-wrap").open = true;
  $("#train-status").textContent = "starting…";
  $("#train-log").textContent = "";
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: parseInt($("#set-epochs").value, 10),
        tts_pos: $("#set-tts").checked ? parseInt($("#set-tts-count").value, 10) : 0,
        kokoro_pos: parseInt($("#set-kokoro-count").value, 10) || 0,
        force_regenerate_tts: $("#set-force-regen-tts").checked,
        model_size: $("#set-model-size").value,
        auto_distractors: $("#set-autodist").checked,
        partial_negatives: $("#set-partials").checked,
        specaugment: $("#set-specaug").checked,
        use_rir: $("#set-rir").checked,
        use_parametric_noise: $("#set-noisepool").checked,
        spectral_matching: $("#set-specmatch").checked,
        loss_function: $("#set-focal").checked ? "focal" : "bce",
        end_aligned_variants: $("#set-endalign").checked,
        target_fpr: parseFloat($("#set-fpr").value),
        device: "auto",
      })
    });
  } catch (e) {
    $("#train-status").innerHTML = '<span style="color:var(--danger)">' + e.message + '</span>';
    $("#train-btn").disabled = false;
    return;
  }
  const started = Date.now();
  while (true) {
    const s = await api(`/api/projects/${encodeURIComponent(currentProject)}/train_status`);
    $("#train-log").textContent = s.logs.join("\n");
    $("#train-log").scrollTop = $("#train-log").scrollHeight;
    if (s.state === "running") {
      $("#train-status").textContent = `running… (${Math.round((Date.now() - started) / 1000)}s)`;
    }
    if (s.state === "done") {
      $("#train-status").innerHTML = `<span style="color:var(--ok)">✓ done</span> in ${(s.elapsed||0).toFixed(1)}s`;
      $("#train-btn").disabled = false;
      await refresh();
      return;
    }
    if (s.state === "error") {
      $("#train-status").innerHTML = `<span style="color:var(--danger)">error - see log</span>`;
      $("#train-btn").disabled = false;
      return;
    }
    await new Promise(r => setTimeout(r, 500));
  }
}

let liveLogPollTimer = null;
let liveLogLastT = 0;
let liveLogRows = [];

async function doListenStart() {
  $("#listen-start").disabled = true;
  let triggers = 0;
  liveLogLastT = 0;
  liveLogRows = [];
  $("#listen-log-rows").innerHTML = "";
  try {
    const thrVal = $("#set-thr").value;
    const backend = $("#listen-backend").value;
    const body = { backend };
    if (thrVal) body.threshold = parseFloat(thrVal);
    const r = await api(`/api/projects/${encodeURIComponent(currentProject)}/live_start`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    liveSession = r.session;
    const backendLabel = ({pytorch:"PyTorch", onnx_fp32:"ONNX fp32", onnx_int8:"ONNX int8"})[r.backend] || r.backend;
    $("#listen-hint").innerHTML =
      `Say <b>"${r.phrase}"</b>. Threshold ${r.threshold.toFixed(2)}. Backend: <code>${backendLabel}</code>.`;
    $("#trigger-count").textContent = "0";
    $("#prob-val").textContent = "0.00";
    $("#prob-fill").style.width = "0%";
    $("#listen-stop").classList.remove("hidden");
    $("#listen-log-wrap").classList.remove("hidden");
    $("#listen-log-copy").classList.remove("hidden");

    // Poll the per-frame log buffer every 250 ms and render new rows
    if (liveLogPollTimer) clearInterval(liveLogPollTimer);
    liveLogPollTimer = setInterval(async () => {
      if (!liveSession) return;
      try {
        const log = await api(
          `/api/projects/${encodeURIComponent(currentProject)}/live_log?session=${liveSession}&since=${liveLogLastT}`
        );
        const thr = log.threshold;
        const newRows = [];
        for (const e of log.entries) {
          liveLogLastT = Math.max(liveLogLastT, e.t);
          liveLogRows.push(e);
          const trigCol = e.triggered
            ? `<span style="color:var(--ok)">TRIG</span>`
            : (e.prob > thr ? `<span style="color:var(--warn, #f59e0b)">over</span>` : "");
          const gateCol = e.gated
            ? `<span style="color:var(--muted)">${e.reason || "block"}</span>`
            : `<span style="color:var(--ok)">pass</span>`;
          const probColor = e.prob > thr ? "var(--ok)" : "var(--fg)";
          newRows.push(`
            <div style="display: grid; grid-template-columns: 60px 70px 70px 90px 60px 60px 60px 70px 1fr; gap: 6px; padding: 2px 0; border-bottom: 1px solid #1a1d24;">
              <div>${e.t.toFixed(1)}</div>
              <div style="color:${probColor};">${e.prob.toFixed(3)}</div>
              <div>${e.ema.toFixed(3)}</div>
              <div>${e.rms_dbfs.toFixed(1)}</div>
              <div>${e.band}%</div>
              <div>${e.rumble}%</div>
              <div>${e.hiss}%</div>
              <div>${trigCol}</div>
              <div>${gateCol}</div>
            </div>
          `);
        }
        if (newRows.length) {
          const rows = $("#listen-log-rows");
          rows.insertAdjacentHTML("beforeend", newRows.join(""));
          // keep ~300 rows visible
          while (rows.children.length > 300) rows.removeChild(rows.firstChild);
          // auto-scroll if user near bottom
          const wrap = $("#listen-log-wrap");
          if (wrap.scrollTop + wrap.clientHeight + 60 >= wrap.scrollHeight) {
            wrap.scrollTop = wrap.scrollHeight;
          }
        }
      } catch (e) {}
    }, 250);

    liveStop = await startStream(async (samples, sr) => {
      if (!liveSession) return;
      try {
        const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
        const res = await api(`/api/projects/${encodeURIComponent(currentProject)}/live_chunk?session=${liveSession}&sr=${Math.round(sr)}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: buf
        });
        const probFill = $("#prob-fill");
        probFill.style.width = (Math.round(res.prob * 100)) + "%";
        $("#prob-val").textContent = res.prob.toFixed(2);
        if (res.diag) {
          $("#prob-rms").textContent =
            res.diag.rms_dbfs != null ? res.diag.rms_dbfs.toFixed(1) + " dBFS" : "-";
          $("#prob-band").textContent =
            res.diag.band_frac != null ? (res.diag.band_frac * 100).toFixed(0) + " %" : "-";
        }
        $("#prob-gate").innerHTML = res.gated
          ? `<span style="color:var(--danger)">BLOCKED (${res.diag?.reason || ""})</span>`
          : `<span style="color:var(--ok)">pass</span>`;
        if (res.triggered) {
          triggers++;
          $("#trigger-count").textContent = triggers;
          const wrap = $("#prob-wrap");
          wrap.classList.remove("flash"); void wrap.offsetWidth;
          wrap.classList.add("flash");
        }
      } catch (e) {}
    });
  } catch (e) {
    alert(e.message);
    $("#listen-start").disabled = false;
  }
}

async function doListenStop() {
  if (liveLogPollTimer) { clearInterval(liveLogPollTimer); liveLogPollTimer = null; }
  if (liveStop) { liveStop(); liveStop = null; }
  if (liveSession) {
    try {
      await api(`/api/projects/${encodeURIComponent(currentProject)}/live_stop?session=${liveSession}`, { method: "POST" });
    } catch (e) {}
    liveSession = null;
  }
  $("#listen-start").disabled = false;
  $("#listen-stop").classList.add("hidden");
}

function copyListenLog() {
  // Produce a clean text-table the user can paste back to chat
  const lines = ["t,prob,ema,rms_dbfs,voice_pct,rumble_pct,hiss_pct,triggered,gated,reason"];
  for (const e of liveLogRows) {
    lines.push([e.t, e.prob, e.ema, e.rms_dbfs, e.band, e.rumble, e.hiss,
                e.triggered ? 1 : 0, e.gated ? 1 : 0, e.reason || ""].join(","));
  }
  const text = lines.join("\n");
  navigator.clipboard.writeText(text).then(() => {
    const btn = $("#listen-log-copy");
    const orig = btn.textContent;
    btn.textContent = `copied ${liveLogRows.length} rows`;
    setTimeout(() => { btn.textContent = orig; }, 1500);
  }).catch(() => {
    // fallback: show in a textarea so user can manually copy
    const w = window.open("", "_blank");
    w.document.body.innerHTML = "<pre>" + text + "</pre>";
  });
}

// ---------- Model slots ----------

async function refreshModelsList() {
  if (!currentProject) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(currentProject)}/models`);
    const list = $("#models-list");
    if (!data.slots || data.slots.length === 0) {
      list.innerHTML = `<div class="muted" style="font-size:13px;">No saved slots yet - train a model and a slot will be created automatically.</div>`;
      return;
    }
    let html = "";
    for (const s of data.slots) {
      const sizeKb = (s.size_bytes / 1024).toFixed(1);
      const ts = new Date(s.mtime * 1000).toLocaleString();
      const tpr = (s.user_voice_tpr != null) ? ` · your-voice TPR <b>${(s.user_voice_tpr*100).toFixed(0)}%</b>` : "";
      const fpr = (s.user_voice_fpr != null) ? ` FPR <b>${(s.user_voice_fpr*100).toFixed(0)}%</b>` : "";
      const params = s.n_params ? ` · ${s.n_params} params` : "";
      const seconds = s.seconds ? ` · trained in ${s.seconds.toFixed(1)}s` : "";
      const badge = s.is_active
        ? `<span style="color:var(--ok); font-weight:600; font-size:12px;">✓ ACTIVE</span>`
        : `<button class="ghost" data-slot="${s.name}" data-action="activate" style="padding: 4px 12px; font-size: 12px;">Activate</button>`;
      const delBtn = s.is_active
        ? ``
        : `<button class="ghost" data-slot="${s.name}" data-action="delete" style="padding: 4px 12px; font-size: 12px; color: var(--danger);">Delete</button>`;
      html += `
        <div style="display: grid; grid-template-columns: 90px 1fr auto auto; gap: 14px; align-items: center;
                    padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px;
                    margin-bottom: 8px; background: ${s.is_active ? 'rgba(34,197,94,0.05)' : 'transparent'};">
          <code style="font-weight: 600;">${s.name}</code>
          <div style="font-size: 13px;">
            <div>${sizeKb} KB${params}${seconds}</div>
            <div class="muted" style="font-size: 12px; margin-top: 2px;">${ts}${tpr}${fpr}</div>
          </div>
          ${badge}
          ${delBtn}
        </div>
      `;
    }
    list.innerHTML = html;
    // wire buttons
    list.querySelectorAll('button[data-slot]').forEach(btn => {
      btn.onclick = async () => {
        const slot = btn.dataset.slot;
        const action = btn.dataset.action;
        btn.disabled = true;
        try {
          if (action === "activate") {
            await api(`/api/projects/${encodeURIComponent(currentProject)}/models/${encodeURIComponent(slot)}/activate`,
                      { method: "POST" });
          } else if (action === "delete") {
            if (!confirm(`Delete slot "${slot}"? This only removes the saved slot, not the currently-active model.`)) {
              btn.disabled = false;
              return;
            }
            await api(`/api/projects/${encodeURIComponent(currentProject)}/models/${encodeURIComponent(slot)}`,
                      { method: "DELETE" });
          }
          await refresh();  // re-render everything (active model changed → train badge, etc.)
        } catch (e) {
          alert(e.message);
          btn.disabled = false;
        }
      };
    });
  } catch (e) {
    $("#models-list").innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
}

// ---------- ONNX export ----------

function fmtBytes(n) {
  if (n == null) return "-";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(2) + " MB";
}

async function refreshExportInfo() {
  if (!currentProject) return;
  try {
    const info = await api(`/api/projects/${encodeURIComponent(currentProject)}/export/info`);
    const wrap = $("#export-files");
    if (!info.exists || !info.files.length) {
      wrap.classList.add("hidden");
      return;
    }
    const list = $("#export-files-list");
    list.innerHTML = info.files.map(f => {
      const dlURL = `/api/projects/${encodeURIComponent(currentProject)}/export/download/${encodeURIComponent(f.name)}`;
      const mtime = new Date(f.mtime * 1000).toLocaleString();
      return `<div style="display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--line); font-size: 13px;">
        <code style="flex: 1;">${f.name}</code>
        <span class="muted" style="font-size: 12px;">${fmtBytes(f.size_bytes)} · ${mtime}</span>
        <a href="${dlURL}" download style="color: var(--accent); text-decoration: none;">↓ download</a>
      </div>`;
    }).join("");
    wrap.classList.remove("hidden");
  } catch (e) {
    // silent - section just won't show files
  }
}

async function doExport() {
  if (!currentProject) return;
  const btn = $("#export-btn");
  const status = $("#export-status");
  btn.disabled = true;
  status.innerHTML = "exporting…";
  try {
    const res = await api(`/api/projects/${encodeURIComponent(currentProject)}/export`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ int8: true }),
    });
    const fp32Bits = `<b style="color:var(--ok);">fp32 ${fmtBytes(res.onnx_size_bytes)}</b> (max-abs vs PyTorch: ${res.max_abs_error_fp32.toExponential(2)})`;
    const int8Bits = res.int8_size_bytes
      ? ` · <b style="color:var(--ok);">int8 ${fmtBytes(res.int8_size_bytes)}</b> (${(100 * res.int8_size_bytes / res.onnx_size_bytes).toFixed(0)}% of fp32, max-abs ${res.max_abs_error_int8.toExponential(2)})`
      : "";
    status.innerHTML = `✓ exported · ${res.n_params} params · ${fp32Bits}${int8Bits}`;
    await refreshExportInfo();
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

async function doSendToMobile() {
  if (!currentProject) return;
  const btn = $("#send-mobile-btn");
  const status = $("#export-status");
  btn.disabled = true;
  status.innerHTML = "exporting + copying into the mobile demo…";
  try {
    const slot = parseInt(($("#send-slot") || {}).value ?? "0", 10) || 0;
    const res = await api(`/api/projects/${encodeURIComponent(currentProject)}/send-to-mobile`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slot }),
    });
    const skip = (res.skipped && res.skipped.length)
      ? ` · <span style="color:var(--muted)">skipped ${res.skipped.join(", ")}</span>` : "";
    status.innerHTML = `✓ "${res.phrase}" → slot ${(res.slot ?? 0) + 1} (${res.sent.length} files)${skip}. `
      + `Reload Metro (press <b>r</b>) on the phone, then pick it in the word row.`;
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

async function doExportVerify() {
  if (!currentProject) return;
  const btn = $("#export-verify-btn");
  const status = $("#export-status");
  const result = $("#export-verify-result");
  btn.disabled = true;
  status.innerHTML = "scoring user clips through PyTorch, ONNX fp32, ONNX int8…";
  result.classList.add("hidden");
  try {
    const res = await api(`/api/projects/${encodeURIComponent(currentProject)}/export/verify`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const s = res.summary;
    const okFp32 = s.fp32_max_abs_diff < 1e-3 && s.fp32_trigger_disagreements === 0;
    const okInt8 = s.has_int8 === false || (s.int8_max_abs_diff < 0.05 && s.int8_trigger_disagreements === 0);
    const verdictColor = (okFp32 && okInt8) ? "var(--ok)" : "var(--warn, #f59e0b)";
    const verdictLabel = (okFp32 && okInt8)
      ? "✓ ONNX exports behave identically to PyTorch"
      : "⚠ ONNX exports diverge - review below";

    let html = `
      <div style="margin-bottom: 12px; padding: 10px 12px; background: #0d1015; border-radius: 6px;">
        <div style="color: ${verdictColor}; font-weight: 600;">${verdictLabel}</div>
        <div class="muted" style="font-size: 12px; margin-top: 6px; line-height: 1.6;">
          float32 ONNX: max Δ ${s.fp32_max_abs_diff.toExponential(2)} ·
          mean Δ ${s.fp32_mean_abs_diff.toExponential(2)} ·
          trigger disagreements ${s.fp32_trigger_disagreements}/${s.n_total}<br>
          ${res.has_int8 ? `int8 ONNX: max Δ ${s.int8_max_abs_diff.toExponential(2)} · mean Δ ${s.int8_mean_abs_diff.toExponential(2)} · trigger disagreements ${s.int8_trigger_disagreements}/${s.n_total}` : "int8 not exported"}
        </div>
      </div>
    `;
    function tableSection(title, rows) {
      if (!rows.length) return "";
      let h = `<div style="font-size: 13px; color: var(--muted); margin: 10px 0 6px;">${title}</div>
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
          <thead><tr style="border-bottom: 1px solid var(--line); color: var(--muted);">
            <th style="text-align:left;padding:4px 6px;">file</th>
            <th style="text-align:right;padding:4px 6px;">PyTorch</th>
            <th style="text-align:right;padding:4px 6px;">ONNX fp32</th>
            <th style="text-align:right;padding:4px 6px;">ONNX int8</th>
          </tr></thead><tbody>`;
      for (const r of rows) {
        const ptCol = r.pt_trigger ? "var(--ok)" : "var(--muted)";
        const fpCol = (r.fp32_trigger === r.pt_trigger) ? "var(--ok)" : "var(--danger)";
        const i8Col = (r.int8_trigger === null) ? "var(--muted)" :
                      (r.int8_trigger === r.pt_trigger) ? "var(--ok)" : "var(--danger)";
        h += `<tr style="border-bottom: 1px solid #1a1d24;">
          <td style="padding:4px 6px;font-family:ui-monospace,Consolas,monospace;">${r.filename}</td>
          <td style="padding:4px 6px;text-align:right;font-family:ui-monospace,Consolas,monospace;color:${ptCol};">${r.pytorch.toFixed(4)}${r.pt_trigger ? " ✓" : ""}</td>
          <td style="padding:4px 6px;text-align:right;font-family:ui-monospace,Consolas,monospace;color:${fpCol};">${r.onnx_fp32.toFixed(4)}${r.fp32_trigger ? " ✓" : ""}</td>
          <td style="padding:4px 6px;text-align:right;font-family:ui-monospace,Consolas,monospace;color:${i8Col};">${r.onnx_int8 != null ? r.onnx_int8.toFixed(4) + (r.int8_trigger ? " ✓" : "") : "-"}</td>
        </tr>`;
      }
      h += `</tbody></table>`;
      return h;
    }
    html += tableSection(`Positives (should trigger, threshold ${res.threshold.toFixed(2)})`, res.positives);
    html += tableSection("Negatives (should NOT trigger)", res.negatives);
    result.innerHTML = html;
    result.classList.remove("hidden");
    status.textContent = "";
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

// ---------- wiring ----------
$("#rec-pos").onclick = () => doRecord("positive");
$("#rec-neg").onclick = () => doRecord("negative");
$("#train-btn").onclick = doTrain;
$("#listen-start").onclick = doListenStart;
$("#export-btn").onclick = doExport;
$("#send-mobile-btn").onclick = doSendToMobile;
$("#export-verify-btn").onclick = doExportVerify;
$("#listen-stop").onclick = doListenStop;
$("#listen-log-copy").onclick = copyListenLog;
$("#test-once-btn").onclick = doTestOnce;
$("#rec-ambient").onclick = doRecordAmbient;
$("#holdout-btn").onclick = doHoldoutTest;
$("#crosstts-btn").onclick = doCrossTtsTest;
$("#selftest-btn").onclick = doSelfTest;

async function doHoldoutTest() {
  if (!currentProject || !currentInfo?.has_model) return;
  const btn = $("#holdout-btn");
  const status = $("#holdout-status");
  const result = $("#holdout-result");
  btn.disabled = true;
  status.textContent = "synthesizing held-out voices + running model…";
  result.classList.add("hidden");
  result.innerHTML = "";
  try {
    const res = await api(
      `/api/projects/${encodeURIComponent(currentProject)}/holdout_test`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
    );

    const verdictColor =
      res.verdict === "generalizes" ? "var(--ok)" :
      res.verdict === "speaker-locked" ? "var(--danger)" : "var(--warn, #f59e0b)";
    const verdictLabel =
      res.verdict === "generalizes" ? "✓ generalizes to unseen voices" :
      res.verdict === "speaker-locked" ? "✗ model appears speaker-locked" :
      "⚠ borderline";

    let html = `
      <div style="margin-bottom: 12px; padding: 10px 12px; background: #0d1015;
                  border-radius: 6px;">
        <div style="color: ${verdictColor}; font-weight: 600;">${verdictLabel}</div>
        <div class="muted" style="font-size: 12px; margin-top: 4px;">
          ${res.n_pos_fire}/${res.n_positives} held-out positives triggered ·
          ${res.n_neg_fire}/${res.n_negatives} false triggers ·
          threshold ${res.threshold.toFixed(2)}
        </div>
      </div>
      <div style="font-size: 13px; color: var(--muted); margin-bottom: 6px;">
        Held-out positives - "${res.phrase}" by 3 speakers never seen at training
      </div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>
    `;
    for (const p of res.positives) {
      const pass = p.score > res.threshold;
      const color = pass ? "var(--ok)" : "var(--danger)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/holdout_clip/${encodeURIComponent(p.filename)}`;
      html += `
        <tr style="border-bottom: 1px solid var(--line);">
          <td style="padding: 6px;">spk ${p.speaker_id}</td>
          <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace;">"${p.phrase}"</td>
          <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${p.score.toFixed(3)}</td>
          <td style="padding: 6px; text-align: right; color: ${color};">${pass ? "✓" : "✗"}</td>
          <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
        </tr>
      `;
    }
    html += `</tbody></table>
      <div style="font-size: 13px; color: var(--muted); margin: 14px 0 6px;">
        Held-out negatives - false-trigger phrases by the same speakers (should NOT fire)
      </div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>
    `;
    for (const n of res.negatives) {
      const fired = n.score > res.threshold;
      const color = fired ? "var(--danger)" : "var(--ok)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/holdout_clip/${encodeURIComponent(n.filename)}`;
      html += `
        <tr style="border-bottom: 1px solid var(--line);">
          <td style="padding: 6px;">spk ${n.speaker_id}</td>
          <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace;">"${n.phrase}"</td>
          <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${n.score.toFixed(3)}</td>
          <td style="padding: 6px; text-align: right; color: ${color};">${fired ? "✗ false trigger" : "✓"}</td>
          <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
        </tr>
      `;
    }
    html += `</tbody></table>`;
    result.innerHTML = html;
    result.classList.remove("hidden");
    status.textContent = "";
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

async function doCrossTtsTest() {
  if (!currentProject || !currentInfo?.has_model) return;
  const btn = $("#crosstts-btn");
  const status = $("#crosstts-status");
  const result = $("#crosstts-result");
  btn.disabled = true;
  status.textContent = "synthesizing held-out Kokoro voices + running model…";
  result.classList.add("hidden");
  result.innerHTML = "";
  try {
    const res = await api(
      `/api/projects/${encodeURIComponent(currentProject)}/cross_tts_test`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
    );

    const verdictColor =
      res.verdict === "generalizes" ? "var(--ok)" :
      res.verdict === "tts-family-locked" ? "var(--danger)" : "var(--warn, #f59e0b)";
    const verdictLabel =
      res.verdict === "generalizes" ? "✓ generalizes across TTS families" :
      res.verdict === "tts-family-locked" ? "✗ model is locked to the Piper acoustic family" :
      "⚠ borderline - some Kokoro voices triggered, others didn't";

    let html = `
      <div style="margin-bottom: 12px; padding: 10px 12px; background: #0d1015;
                  border-radius: 6px;">
        <div style="color: ${verdictColor}; font-weight: 600;">${verdictLabel}</div>
        <div class="muted" style="font-size: 12px; margin-top: 4px;">
          ${res.n_pos_fire}/${res.n_positives} held-out Kokoro positives triggered ·
          ${res.n_neg_fire}/${res.n_negatives} false triggers ·
          threshold ${res.threshold.toFixed(2)}
        </div>
      </div>
      <div style="font-size: 13px; color: var(--muted); margin-bottom: 6px;">
        Kokoro positives - "${res.phrase}" by 3 voices from a different TTS family
      </div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>
    `;
    for (const p of res.positives) {
      const pass = p.score > res.threshold;
      const color = pass ? "var(--ok)" : "var(--danger)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/cross_tts_clip/${encodeURIComponent(p.filename)}`;
      html += `
        <tr style="border-bottom: 1px solid var(--line);">
          <td style="padding: 6px;">${p.voice_id}</td>
          <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace;">"${p.phrase}"</td>
          <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${p.score.toFixed(3)}</td>
          <td style="padding: 6px; text-align: right; color: ${color};">${pass ? "✓" : "✗"}</td>
          <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
        </tr>
      `;
    }
    html += `</tbody></table>
      <div style="font-size: 13px; color: var(--muted); margin: 14px 0 6px;">
        Kokoro negatives - false-trigger phrases by the same voices (should NOT fire)
      </div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>
    `;
    for (const n of res.negatives) {
      const fired = n.score > res.threshold;
      const color = fired ? "var(--danger)" : "var(--ok)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/cross_tts_clip/${encodeURIComponent(n.filename)}`;
      html += `
        <tr style="border-bottom: 1px solid var(--line);">
          <td style="padding: 6px;">${n.voice_id}</td>
          <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace;">"${n.phrase}"</td>
          <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${n.score.toFixed(3)}</td>
          <td style="padding: 6px; text-align: right; color: ${color};">${fired ? "✗ false trigger" : "✓"}</td>
          <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
        </tr>
      `;
    }
    html += `</tbody></table>`;
    result.innerHTML = html;
    result.classList.remove("hidden");
    status.textContent = "";
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

async function doSelfTest() {
  if (!currentProject || !currentInfo?.has_model) return;
  const btn = $("#selftest-btn");
  const status = $("#selftest-status");
  const result = $("#selftest-result");
  btn.disabled = true;
  status.textContent = "scoring your recordings…";
  result.classList.add("hidden");
  result.innerHTML = "";
  try {
    const res = await api(
      `/api/projects/${encodeURIComponent(currentProject)}/self_test`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
    );

    const verdictColor =
      res.verdict === "user-voice-ready" ? "var(--ok)" :
      res.verdict === "model does NOT recognize your voice" ? "var(--danger)" :
      "var(--warn, #f59e0b)";

    let html = `
      <div style="margin-bottom: 12px; padding: 10px 12px; background: #0d1015;
                  border-radius: 6px;">
        <div style="color: ${verdictColor}; font-weight: 600;">${res.verdict}</div>
        <div class="muted" style="font-size: 12px; margin-top: 4px;">
          TPR ${(res.tpr*100).toFixed(0)}% (${res.positives.filter(p=>p.ok).length}/${res.n_positives} positives triggered) ·
          FPR ${(res.fpr*100).toFixed(0)}% (${res.negatives.filter(n=>!n.ok).length}/${res.n_negatives} false triggers) ·
          threshold ${res.threshold.toFixed(2)}
        </div>
        <div class="muted" style="font-size: 12px; margin-top: 8px;">
          ${res.tpr >= 0.9 && res.fpr <= 0.1
            ? `The model recognizes your training recordings well. If Test Once still fails, your test-time voice differs from training (try speaking with the same prosody / mic distance / volume as when you recorded positives).`
            : res.tpr < 0.5
            ? `Model failed to learn your voice from training data - investigate before retraining (check recording quality, listen to your positives).`
            : `Borderline. Re-record positives with more variety (deliberately fast/slow/soft/loud) or add 5-10 more.`}
        </div>
      </div>
      <div style="font-size: 13px; color: var(--muted); margin-bottom: 6px;">Your positives (should trigger):</div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>
    `;
    for (const p of res.positives) {
      const color = p.ok ? "var(--ok)" : "var(--danger)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/clip/positive/${encodeURIComponent(p.filename)}`;
      html += `<tr style="border-bottom: 1px solid var(--line);">
        <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace; font-size: 12px;">${p.filename}</td>
        <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${p.score.toFixed(3)}</td>
        <td style="padding: 6px; text-align: right; color: ${color};">${p.ok ? "✓" : "✗"}</td>
        <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
      </tr>`;
    }
    html += `</tbody></table>
      <div style="font-size: 13px; color: var(--muted); margin: 14px 0 6px;">Your negatives (should NOT trigger):</div>
      <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
        <tbody>`;
    for (const n of res.negatives) {
      const color = n.ok ? "var(--ok)" : "var(--danger)";
      const audioURL = `/api/projects/${encodeURIComponent(currentProject)}/clip/negative/${encodeURIComponent(n.filename)}`;
      html += `<tr style="border-bottom: 1px solid var(--line);">
        <td style="padding: 6px; font-family: ui-monospace,Consolas,monospace; font-size: 12px;">${n.filename}</td>
        <td style="padding: 6px; text-align: right; font-family: ui-monospace,Consolas,monospace; color: ${color};">${n.score.toFixed(3)}</td>
        <td style="padding: 6px; text-align: right; color: ${color};">${n.ok ? "✓" : "✗ false trigger"}</td>
        <td style="padding: 6px;"><audio controls preload="none" style="width: 200px; height: 24px;" src="${audioURL}"></audio></td>
      </tr>`;
    }
    html += `</tbody></table>`;
    result.innerHTML = html;
    result.classList.remove("hidden");
    status.textContent = "";
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

async function doTestOnce() {
  if (!currentProject || !currentInfo?.has_model) return;
  const btn = $("#test-once-btn");
  const status = $("#test-once-status");
  btn.disabled = true;
  status.textContent = "";
  // Countdown overlay (reuse the same one)
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const inner = document.createElement("div"); inner.className = "overlay-inner";
  overlay.appendChild(inner); document.body.appendChild(overlay);
  try {
    for (let c = 3; c >= 1; c--) {
      inner.innerHTML = `<div class="count">${c}</div>`;
      await new Promise(r => setTimeout(r, 700));
    }
    inner.innerHTML = `<div class="recording-label">● Recording…</div>
      <div class="muted" style="margin-top:14px;">Say the phrase now. 3 seconds.</div>`;
    const { samples, sampleRate } = await recordFloat32(3.0);
    inner.innerHTML = `<div class="saving-label">Scoring…</div>`;
    const buf = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    const res = await api(
      `/api/projects/${encodeURIComponent(currentProject)}/test_take?sr=${Math.round(sampleRate)}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf }
    );
    overlay.remove();

    const wrap = $("#test-once-result");
    wrap.classList.remove("hidden");
    $("#t1-prob").textContent = res.prob.toFixed(3);
    $("#t1-fill").style.width = (Math.round(res.prob * 100)) + "%";
    $("#t1-decision").innerHTML = res.would_trigger
      ? `<span style="color:var(--ok)">✓ TRIGGER  (prob ${res.prob.toFixed(2)} ≥ threshold ${res.threshold.toFixed(2)})</span>`
      : `<span style="color:var(--danger)">✗ no trigger  (prob ${res.prob.toFixed(2)} < threshold ${res.threshold.toFixed(2)})</span>`;
    $("#t1-rms").textContent = res.rms_dbfs != null ? `${res.rms_dbfs.toFixed(1)} dBFS` : "-";
    const bandPct = res.band_frac != null ? Math.round(res.band_frac * 100) : null;
    const rumblePct = res.rumble_frac != null ? Math.round(res.rumble_frac * 100) : null;
    const hissPct = res.hiss_frac != null ? Math.round(res.hiss_frac * 100) : null;
    const bandColor = bandPct == null ? "" :
                      bandPct >= 50 ? "var(--ok)" :
                      bandPct >= 20 ? "var(--warn, #f59e0b)" :
                      "var(--danger)";
    $("#t1-band").innerHTML = bandPct != null
      ? `<span style="color:${bandColor}">${bandPct} % (gate pass: ${res.gate_pass ? "yes" : "no"})</span>`
      : "-";
    $("#t1-rumble").innerHTML = rumblePct != null
      ? `<span style="color:${rumblePct > 50 ? 'var(--danger)' : rumblePct > 20 ? 'var(--warn, #f59e0b)' : 'var(--muted)'}">${rumblePct} %</span>`
      : "-";
    $("#t1-hiss").innerHTML = hissPct != null
      ? `<span style="color:${hissPct > 30 ? 'var(--warn, #f59e0b)' : 'var(--muted)'}">${hissPct} %</span>`
      : "-";

    // Diagnostic hint: if band breakdown is unhealthy, explain what to do.
    const hint = $("#t1-spectrum-hint");
    if (bandPct != null && bandPct < 30 && rumblePct > 50) {
      hint.classList.remove("hidden");
      hint.innerHTML = `<b style="color:var(--danger)">Mic capture is rumble-dominated.</b> ` +
        `Most of the energy is below 100 Hz (AC hum, HVAC, mic handling, USB power). ` +
        `Train + infer apply an 80 Hz high-pass filter automatically, but if the rumble ` +
        `is THIS strong it may saturate after normalization. ` +
        `Try: (1) speak closer to the mic, (2) move away from HVAC/fans, (3) switch to a different mic, ` +
        `(4) check Windows audio enhancements aren't doing weird things (Sound Settings → Mic properties → ` +
        `Advanced → uncheck "Allow apps/system effects").`;
    } else if (bandPct != null && bandPct < 30 && hissPct > 30) {
      hint.classList.remove("hidden");
      hint.innerHTML = `<b style="color:var(--warn, #f59e0b)">Mic capture is hiss-dominated.</b> ` +
        `Energy is concentrated above 7 kHz (mic self-noise, electronic interference, or ` +
        `excessive Windows audio enhancement). Try: (1) increase mic gain, (2) speak louder, ` +
        `(3) disable Windows audio enhancements.`;
    } else if (bandPct != null && bandPct < 30) {
      hint.classList.remove("hidden");
      hint.innerHTML = `<b style="color:var(--warn, #f59e0b)">Voice band is low</b> (${bandPct} %). ` +
        `Speak closer/louder and try again.`;
    } else {
      hint.classList.add("hidden");
    }

    $("#t1-audio").src = `/api/projects/${encodeURIComponent(currentProject)}/test_take/${encodeURIComponent(res.filename)}`;
    status.innerHTML = res.would_trigger
      ? `<span style="color:var(--ok)">saved → ${res.filename}</span>`
      : `<span style="color:var(--danger)">model said no - review below</span>`;
  } catch (e) {
    overlay.remove();
    status.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
  btn.disabled = false;
}

$("#project-select").onchange = async e => {
  currentProject = e.target.value;
  localStorage.setItem("heed-project", currentProject);
  await doListenStop();
  await refresh();
};

$("#new-project").onclick = () => showNewProjectForm(true);

$("#new-create").onclick = async () => {
  const name = $("#new-name").value.trim();
  const phrase = $("#new-phrase").value.trim();
  if (!name || !phrase) { alert("name and phrase are required"); return; }
  try {
    await api("/api/projects", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, phrase })
    });
    $("#new-name").value = ""; $("#new-phrase").value = "";
    await loadEnv(); await loadProjects();
    $("#project-select").value = name;
    currentProject = name;
    localStorage.setItem("heed-project", name);
    await refresh();
  } catch (e) { alert(e.message); }
};

$("#upload-pos-link").onclick = e => { e.preventDefault(); $("#upload-pos").click(); };
$("#upload-neg-link").onclick = e => { e.preventDefault(); $("#upload-neg").click(); };
$("#upload-pos").onchange = e => doUpload("positive", e.target.files);
$("#upload-neg").onchange = e => doUpload("negative", e.target.files);
async function doUpload(kind, files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("file", f);
  try {
    await api(`/api/projects/${encodeURIComponent(currentProject)}/upload?kind=${kind}`,
      { method: "POST", body: fd });
    await refresh();
  } catch (e) { alert(e.message); }
}

$("#open-settings").onclick = () => $("#drawer").classList.add("open");
$("#close-settings").onclick = () => $("#drawer").classList.remove("open");

// Reactively update the train-plan when settings change.
for (const id of ["set-tts", "set-tts-count", "set-autodist", "set-kokoro-count"]) {
  document.getElementById(id).addEventListener("change", renderTrainPlan);
  document.getElementById(id).addEventListener("input", renderTrainPlan);
}

(async () => {
  await loadEnv();
  await loadProjects();
})();
</script>
</body>
</html>
"""
