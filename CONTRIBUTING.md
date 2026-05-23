# Contributing to Heed

Thanks for your interest. PRs and issues are welcome.

## Ground rules

- **Keep the smoke test green.** Before opening a PR, run `heed smoke` (or
  `python -m heed.cli smoke`). It should print `smoke test PASSED`. CI runs it on
  every push and PR.
- **Match the surrounding code.** Avoid reformatting churn. The codebase is ruff
  and black friendly, though CI does not enforce a formatter yet.
- **Add tests that capture a real invariant.** Good targets are preprocessing
  parity (Python vs JS), streaming versus batch equality, ONNX and TFLite versus
  PyTorch, and the CMN and HPF properties. See `tests/`.
- **No personal data or large binaries in commits.** No `.wav`, no `model.pt`,
  no TTS voice files. `.gitignore` already blocks the usual offenders.

## Dev setup

```bash
pip install -e ".[all]"     # full toolchain (UI, TTS, export)
heed doctor                 # check torch, onnxruntime, and TTS
heed smoke
```

## Good first issues

- Reference preprocessing in another language (Swift, Kotlin, Dart, Rust) that
  matches `wake.json` and the Python and JS references bit-for-bit.
- A `heed plot` command for probability-over-time, log-mel, and score histograms.
- A pretrained pack phrase, trained speaker-independent and validated cross-speaker.

## Scope

Heed is two things. It is a custom wake-word trainer, and it ships a few
ready-made models. Both produce the same artifact, an ONNX or TFLite model plus
a `wake.json` preprocessing contract, that runs anywhere. Changes that keep the
"train anywhere, deploy anywhere, no cloud" promise are the ones we want.
