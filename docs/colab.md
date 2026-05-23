# Train in Colab

Two notebooks train a model on a free GPU with nothing installed locally. Open
either in Colab, set the runtime to GPU, and run the cells top to bottom.

## Two notebooks

- [`heed_train_colab.ipynb`](../notebooks/heed_train_colab.ipynb): the quick path.
  Type a phrase and it synthesizes the data, trains, and exports. No microphone,
  no recording. The model is speaker-independent by design, which is what you want
  when one model should work for anyone.
- [`heed_train_your_voice_colab.ipynb`](../notebooks/heed_train_your_voice_colab.ipynb):
  the personal path. Upload or record a few clips of your own voice, train with
  synthetic speakers mixed in for robustness, then test it on your own voice right
  in the notebook before downloading.

## What they produce

Both export `wake.onnx`, `wake.int8.onnx`, and `wake.json` (plus `wake.tflite` if
you install `litert-torch`), zipped for download. Run the result in the browser,
on a phone, or in Python; see [Export and deploy](export-and-deploy.md).

## Notes

- The notebooks `pip install heed-wakeword` from PyPI.
- Add `--kokoro-pos 150` to the train step (after `heed download-kokoro`) for a
  second TTS family and stronger cross-engine robustness.
- Browser recording in the personal notebook needs microphone permission in the
  Colab output frame. If it does not work in your browser, the upload cell does
  the same job.
