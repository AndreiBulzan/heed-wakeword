# Train in Colab

The notebook at [`notebooks/heed_train_colab.ipynb`](../notebooks/heed_train_colab.ipynb)
trains a model with no local install. Open it in Colab (the README has an Open in
Colab badge), set the runtime to GPU, and run the cells top to bottom.

## What it does

1. Installs `heed-wakeword[tts,export]`.
2. You type a wake phrase.
3. Downloads the multi-speaker TTS voice (LibriTTS-R, 904 speakers).
4. Synthesizes seed positives and phonetic-neighbor negatives.
5. Trains, synthesizing many more speakers and hard negatives on top, and prints a
   held-out cross-speaker evaluation so you can see whether it generalizes.
6. Exports `wake.onnx`, `wake.int8.onnx`, `wake.json` (and `wake.tflite` if you
   install `litert-torch`), and offers them as a download.

## When to use it

This path trains purely on synthetic voices, so the model is speaker-independent
by design: good when you want one model that works for anyone. For the best
accuracy on your own microphone and room, record a handful of positives in your
own voice in the [studio](studio.md) and retrain.

## Notes

- The notebook installs from PyPI, so it works once `heed-wakeword` is published.
  Before then, change the install cell to
  `pip install "heed-wakeword[tts,export] @ git+https://github.com/AndreiBulzan/heed-wakeword"`.
- Add `--kokoro-pos 150` to the train cell (after `heed download-kokoro`) for a
  second TTS family and stronger cross-engine robustness.
