// react-native-live-audio-stream emits base64-encoded 16-bit signed PCM.
// We need Float32 in [-1, 1] for the wake-word preprocessor.
//
// This file is the small bridge between the audio module's output format
// and what `StreamingPreprocessor.step()` expects.

// Reusable scratch for the decoded bytes — refilled each call and never
// retained past decodePcm16, so one growable buffer replaces a per-chunk
// Uint8Array allocation (less GC pressure in the always-on audio loop).
let _byteBuf = new Uint8Array(0);
function _ensureBytes(n) {
  if (_byteBuf.length < n) _byteBuf = new Uint8Array(n);
  return _byteBuf;
}

let _b64Lookup = null;
function _lookupTable() {
  if (_b64Lookup) return _b64Lookup;
  _b64Lookup = new Uint8Array(256);
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (let i = 0; i < alphabet.length; i++) _b64Lookup[alphabet.charCodeAt(i)] = i;
  return _b64Lookup;
}

/** Decode a base64 string into the reusable byte buffer; returns the byte
 *  count. RN's atob may be missing in some Hermes versions, so we provide a
 *  self-contained fallback. */
function base64ToBytes(b64) {
  if (typeof globalThis.atob === "function") {
    const bin = globalThis.atob(b64);
    const n = bin.length;
    const bytes = _ensureBytes(n);
    for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    return n;
  }
  // Fallback: decode base64 manually
  const lookup = _lookupTable();
  let len = b64.length;
  while (len > 0 && b64[len - 1] === "=") len--;
  const bytes = _ensureBytes(Math.floor((len * 3) / 4));
  let outIdx = 0, accum = 0, accumBits = 0;
  for (let i = 0; i < len; i++) {
    accum = (accum << 6) | lookup[b64.charCodeAt(i)];
    accumBits += 6;
    if (accumBits >= 8) {
      accumBits -= 8;
      bytes[outIdx++] = (accum >> accumBits) & 0xff;
    }
  }
  return outIdx;
}

/** Convert base64 16-bit PCM (little-endian) → Float32Array in [-1, 1].
 *  The returned Float32Array is allocated fresh (it's queued for later
 *  processing); only the intermediate byte buffer is reused across calls. */
export function decodePcm16(b64) {
  const nBytes = base64ToBytes(b64);
  const bytes = _byteBuf;
  const n = nBytes >> 1; // 2 bytes per sample
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // little-endian int16
    let v = (bytes[2 * i] | (bytes[2 * i + 1] << 8));
    if (v & 0x8000) v -= 0x10000; // sign extend
    out[i] = v / 32768;
  }
  return out;
}
