// AudioWorkletProcessor that buffers ~100 ms of input and posts chunks
// to the main thread. Runs in the audio rendering thread (off the main
// thread) so jank-free even under load.

class WakeWordCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // ~100 ms at 16 kHz = 1600 samples. AudioWorklet renders 128 samples
    // per process() call by default, so we accumulate ~12 calls per chunk.
    this.chunkSize = 1600;
    this.buffer = new Float32Array(this.chunkSize);
    this.written = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];  // mono — first channel only
    for (let i = 0; i < ch.length; i++) {
      this.buffer[this.written++] = ch[i];
      if (this.written >= this.chunkSize) {
        this.port.postMessage(this.buffer);
        this.buffer = new Float32Array(this.chunkSize);
        this.written = 0;
      }
    }
    return true;
  }
}

registerProcessor("wake-word-capture", WakeWordCaptureProcessor);
