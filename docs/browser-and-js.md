# Browser and JavaScript

The browser example is a complete client-side wake-word detector and the
reference implementation of the preprocessing chain for any JavaScript or native
port. All audio stays on the device, and there is no server.

The full guide lives with the example:

- [`examples/inference_browser/README.md`](../examples/inference_browser/README.md)

In short:

- It loads ONNX Runtime Web from a CDN and the model from a relative path, so the
  folder is a drop-anywhere static site. It ships with a working "hey doc" model.
- `preprocessing.js` is the canonical JS port of `heed/audio.py`: a
  state-preserving high-pass biquad cascade, STFT, sparse mel filterbank, log, and
  CMN. It is checked against the Python reference bit-for-bit by
  `verify-preprocessing.mjs`, which CI runs on every change.
- `wakeword.js` wraps the preprocessor, the ONNX session, and the trigger logic
  (energy gate, EMA, hysteresis, refractory) into one object the UI drives frame
  by frame.

To run your own word, drop `wake.onnx` and `wake.json` next to `index.html` and
serve the folder. See [Export and deploy](export-and-deploy.md).
