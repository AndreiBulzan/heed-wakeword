// heed — React Native wake-word demo (Android + iOS via Expo).
//
// Flow:
//   mic (16 kHz PCM, ~100 ms chunks) → decodePcm16 → WakeWordDetector.step
//     → updates UI (probability bar, trigger log, latency)
//
// Requires a development build (Expo Go cannot load native modules like
// onnxruntime-react-native). See README for `expo run:android` and EAS Build
// commands.

import { Asset } from "expo-asset";
import { StatusBar } from "expo-status-bar";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import LiveAudioStream from "react-native-live-audio-stream";

import { decodePcm16 } from "./src/audio-decode.js";
import { N_FRAMES, N_MELS } from "./src/preprocessing.js";
import { WakeWordDetector } from "./src/wakeword.js";

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 1600;  // ~100 ms at 16 kHz

// Wake-word "slots": each is a trained word bundled in 3 runtime formats. Files
// are static requires (Metro), so overwriting a slot's files via the web UI
// "Send to mobile" + a Metro reload swaps the word with NO native rebuild.
// Slots are generic — ship pretrained models in some, leave others for users'
// own; rearrange freely. Each chip's label is read live from the slot's wake.json.
const SLOTS = [
  { key: "0", meta: require("./assets/slot0.json"), onnx: require("./assets/slot0.onnx"), int8: require("./assets/slot0.int8.onnx"), tflite: require("./assets/slot0.tflite") },
  { key: "1", meta: require("./assets/slot1.json"), onnx: require("./assets/slot1.onnx"), int8: require("./assets/slot1.int8.onnx"), tflite: require("./assets/slot1.tflite") },
  { key: "2", meta: require("./assets/slot2.json"), onnx: require("./assets/slot2.onnx"), int8: require("./assets/slot2.int8.onnx"), tflite: require("./assets/slot2.tflite") },
  { key: "3", meta: require("./assets/slot3.json"), onnx: require("./assets/slot3.onnx"), int8: require("./assets/slot3.int8.onnx"), tflite: require("./assets/slot3.tflite") },
  // Slot 5 is the "custom" slot: ships as a placeholder, overwritten by the
  // studio's "Send to mobile -> slot 5" with a model you trained yourself.
  { key: "4", meta: require("./assets/slot4.json"), onnx: require("./assets/slot4.onnx"), int8: require("./assets/slot4.int8.onnx"), tflite: require("./assets/slot4.tflite") },
];
const FORMATS = [
  { key: "fp32", label: "ONNX fp32" },
  { key: "int8", label: "ONNX int8" },
  { key: "tflite", label: "TFLite" },
];

// Build an inference runtime for (slot, format). Both engines expose the same
// { run(mel) -> Promise<logit> } interface, so the detector is engine-agnostic.
async function makeRuntime(slot, format) {
  if (format === "tflite") {
    // Lazy-import so the app still runs ONNX even when the TFLite native module
    // isn't in this build (needs `npm i react-native-fast-tflite` + a rebuild).
    // Throws if the native side is missing; selectFormat() catches it.
    const { loadTensorflowModel } = await import("react-native-fast-tflite");
    const model = await loadTensorflowModel(slot.tflite); // CPU delegate (default)
    return {
      async run(mel) { const outputs = await model.run([mel]); return outputs[0][0]; },
      release() {},
    };
  }
  // ONNX (fp32 / int8) via onnxruntime-react-native.
  const mod = format === "int8" ? slot.int8 : slot.onnx;
  const asset = Asset.fromModule(mod);
  await asset.downloadAsync();
  const modelPath = (asset.localUri ?? asset.uri).replace(/^file:\/\//, "");
  const session = await InferenceSession.create(modelPath);
  return {
    async run(mel) {
      const t = new Tensor("float32", mel, [1, N_MELS, N_FRAMES]);
      const r = await session.run({ mel: t });
      return r.logit.data[0];
    },
    release() { return session.release?.(); },
  };
}

export default function App() {
  const [status, setStatus] = useState("loading model…");
  const [phrase, setPhrase] = useState("");
  const [threshold, setThreshold] = useState(0);
  const [prob, setProb] = useState(0);
  const [ema, setEma] = useState(0);
  const [gateLabel, setGateLabel] = useState("—");
  const [latency, setLatency] = useState(0);
  const [triggers, setTriggers] = useState([]);
  const [listening, setListening] = useState(false);
  const [slotKey, setSlotKey] = useState("0");
  const [formatKey, setFormatKey] = useState("fp32");

  const detectorRef = useRef(null);
  const runtimeRef = useRef(null);
  const latencyBufRef = useRef([]);
  const lastUiUpdateRef = useRef(0);
  const latestResultRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isProcessingRef = useRef(false);

  // Load a (slot, format) pair: the wake word + the runtime to run it on. Each
  // slot carries its own wake.json (phrase + threshold + preprocessing).
  async function loadModel(sk, fk) {
    const slot = SLOTS.find((s) => s.key === sk) ?? SLOTS[0];
    const fmt = FORMATS.find((f) => f.key === fk) ?? FORMATS[0];
    const meta = slot.meta; // require()d JSON object
    setStatus(`loading "${meta.phrase || "model"}" (${fmt.label})…`);
    try { await runtimeRef.current?.release?.(); } catch (_) {}
    const runtime = await makeRuntime(slot, fk);
    runtimeRef.current = runtime;
    detectorRef.current = new WakeWordDetector(runtime, meta);
    setPhrase(meta.phrase ?? "");
    setThreshold(meta.threshold ?? 0.5);
    setStatus(`"${meta.phrase}" · ${fmt.label} — press Start`);
  }

  // Load the default model on mount.
  useEffect(() => {
    (async () => {
      try {
        await loadModel(slotKey, formatKey);
      } catch (e) {
        console.error(e);
        setStatus(`error loading model: ${e.message}`);
        Alert.alert(
          "Model load failed",
          `Did you copy the model files into assets/? Error: ${e.message}`
        );
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Switch wake word (slot) or runtime (format): stop, release, reload. On
  // failure (e.g. TFLite native module not in this build), revert so the app
  // keeps working.
  async function selectSlot(sk) {
    if (sk === slotKey || !detectorRef.current) return;
    const prev = slotKey;
    if (listening) stop();
    setSlotKey(sk);
    try {
      await loadModel(sk, formatKey);
    } catch (e) {
      Alert.alert("Couldn't load that word", String(e?.message ?? e));
      setSlotKey(prev);
      try { await loadModel(prev, formatKey); } catch (_) {}
    }
  }

  async function selectFormat(fk) {
    if (fk === formatKey || !detectorRef.current) return;
    const prev = formatKey;
    if (listening) stop();
    setFormatKey(fk);
    try {
      await loadModel(slotKey, fk);
    } catch (e) {
      Alert.alert(
        "Runtime unavailable",
        fk === "tflite"
          ? "TFLite needs the native module: `npm i react-native-fast-tflite` then `eas build`. ONNX works now."
          : String(e?.message ?? e)
      );
      setFormatKey(prev);
      try { await loadModel(slotKey, prev); } catch (_) {}
    }
  }

  async function ensureMicPermission() {
    if (Platform.OS === "android") {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: "Microphone access",
          message: "heed needs the mic to detect the wake word on-device.",
          buttonPositive: "OK",
        }
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    }
    // iOS prompts automatically on first audio start (Info.plist string set in app.json)
    return true;
  }

  async function start() {
    if (!detectorRef.current) {
      Alert.alert("Not ready", "Model isn't loaded yet.");
      return;
    }
    const ok = await ensureMicPermission();
    if (!ok) {
      Alert.alert("Mic denied", "Wake word can't run without microphone access.");
      return;
    }

    LiveAudioStream.init({
      sampleRate: SAMPLE_RATE,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6,  // VOICE_RECOGNITION on Android; ignored on iOS
      bufferSize: CHUNK_SAMPLES * 2,  // 2 bytes/sample for int16
    });
    // Diagnostic counters
    let _chunkCount = 0;
    let _lastChunkLog = Date.now();

    // Drain the audio queue: process all queued chunks (concatenated) one at
    // a time. Single-flight — only one step() runs at any moment. Audio that
    // arrives during processing accumulates; on the next drain it gets
    // concatenated into one bigger chunk so no audio is lost.
    const drainQueue = async () => {
      if (isProcessingRef.current) return;
      isProcessingRef.current = true;
      while (audioQueueRef.current.length > 0) {
        const chunks = audioQueueRef.current;
        audioQueueRef.current = [];
        // Common case (keeping up): exactly one chunk — feed it directly, no
        // merge allocation. Only allocate + concat when chunks actually backed
        // up during a slow step. (step() copies the input internally anyway.)
        let merged;
        if (chunks.length === 1) {
          merged = chunks[0];
        } else {
          let total = 0;
          for (const c of chunks) total += c.length;
          merged = new Float32Array(total);
          let off = 0;
          for (const c of chunks) { merged.set(c, off); off += c.length; }
        }

        try {
          const result = await detectorRef.current.step(merged);

          if (!result.gated) {
            console.log(
              `[step] chunks_merged=${chunks.length} samples=${merged.length} ` +
              `prep=${result.latencyPrepMs}ms infer=${result.latencyInferMs}ms ` +
              `prob=${result.prob.toFixed(3)} triggered=${result.triggered}`
            );
          }

          latestResultRef.current = result;
          latencyBufRef.current.push(result.latencyMs);
          if (latencyBufRef.current.length > 50) latencyBufRef.current.shift();

          // Throttle UI updates
          const nowMs = Date.now();
          if (nowMs - lastUiUpdateRef.current >= 100 || result.triggered) {
            lastUiUpdateRef.current = nowMs;
            setProb(result.prob);
            setEma(result.ema);
            setGateLabel(result.gated ? "skipped (silence)" : "passed");
            const sorted = latencyBufRef.current.slice().sort((a, b) => a - b);
            setLatency(sorted[Math.floor(sorted.length / 2)] ?? 0);
            if (result.triggered) {
              const ts = new Date().toLocaleTimeString();
              setTriggers(prev => [{
                ts, prob: result.prob, ema: result.ema, id: Date.now()
              }, ...prev.slice(0, 19)]);
            }
          }
        } catch (e) {
          console.warn("step error", e);
        }
      }
      isProcessingRef.current = false;
    };

    LiveAudioStream.on("data", (base64) => {
      const tArrival = Date.now();
      const pcm = decodePcm16(base64);
      audioQueueRef.current.push(pcm);
      _chunkCount++;
      if (tArrival - _lastChunkLog > 2000) {
        const elapsed = (tArrival - _lastChunkLog) / 1000;
        console.log(
          `[audio] ${_chunkCount} chunks in last ${elapsed.toFixed(1)}s ` +
          `(${(_chunkCount / elapsed).toFixed(1)}/s, ` +
          `queue=${audioQueueRef.current.length})`
        );
        _chunkCount = 0;
        _lastChunkLog = tArrival;
      }
      // Fire-and-forget; drainQueue serializes internally
      drainQueue();
    });
    LiveAudioStream.start();
    setListening(true);
    setStatus("listening");
  }

  function stop() {
    LiveAudioStream.stop();
    detectorRef.current?.reset();
    setListening(false);
    setStatus("stopped");
    setProb(0);
    setEma(0);
    setGateLabel("—");
  }

  const probPct = Math.min(100, prob * 100);
  const threshPct = Math.min(100, threshold * 100);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <StatusBar style="light" />
      <Text style={styles.title}>Heed Wake Word</Text>
      <Text style={styles.subtitle}>
        Audio stays on-device. Nothing is sent anywhere.
      </Text>

      <View style={styles.panel}>
        <Text style={styles.text}>
          {phrase ? `Listening for: "${phrase}"` : status}
        </Text>
        <View style={styles.row}>
          <TouchableOpacity
            style={[styles.btn, listening && styles.btnDisabled]}
            onPress={start}
            disabled={listening}
          >
            <Text style={styles.btnText}>Start</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, styles.btnStop, !listening && styles.btnDisabled]}
            onPress={stop}
            disabled={!listening}
          >
            <Text style={styles.btnText}>Stop</Text>
          </TouchableOpacity>
          <Text style={[styles.dim, styles.statusText]} numberOfLines={2}>{status}</Text>
        </View>
        <View style={styles.modelRow}>
          <Text style={styles.dim}>word:</Text>
          {SLOTS.map((s) => (
            <TouchableOpacity
              key={s.key}
              style={[styles.chip, slotKey === s.key && styles.chipActive]}
              onPress={() => selectSlot(s.key)}
            >
              <Text style={[styles.chipText, slotKey === s.key && styles.chipTextActive]}>
                {s.meta.phrase || `slot ${s.key}`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        <View style={styles.modelRow}>
          <Text style={styles.dim}>runtime:</Text>
          {FORMATS.map((f) => (
            <TouchableOpacity
              key={f.key}
              style={[styles.chip, formatKey === f.key && styles.chipActive]}
              onPress={() => selectFormat(f.key)}
            >
              <Text style={[styles.chipText, formatKey === f.key && styles.chipTextActive]}>
                {f.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.panel}>
        <View style={styles.bar}>
          <View style={[styles.barFill, { width: `${probPct}%` }, prob > threshold && styles.barFillTrig]} />
          <View style={[styles.marker, { left: `${threshPct}%` }]} />
        </View>
        <View style={styles.labelRow}>
          <Text style={styles.dim}>0.0</Text>
          <Text style={styles.dim}>threshold: {threshold.toFixed(3)}</Text>
          <Text style={styles.dim}>1.0</Text>
        </View>
        <View style={styles.statGrid}>
          <Text style={styles.statLabel}>raw probability</Text>
          <Text style={styles.statValue}>{prob.toFixed(3)}</Text>
          <Text style={styles.statLabel}>smoothed (EMA)</Text>
          <Text style={styles.statValue}>{ema.toFixed(3)}</Text>
          <Text style={styles.statLabel}>energy gate</Text>
          <Text style={styles.statValue}>{gateLabel}</Text>
          <Text style={styles.statLabel}>inference (median, ms)</Text>
          <Text style={styles.statValue}>{latency.toFixed(1)}</Text>
        </View>
      </View>

      <View style={[styles.panel, styles.logPanel]}>
        <Text style={styles.text}>Triggers</Text>
        <View style={styles.log}>
          {triggers.length === 0 ? (
            <Text style={styles.dim}>(none yet — say the phrase)</Text>
          ) : (
            triggers.map(t => (
              <Text key={t.id} style={styles.logLine}>
                {t.ts}  TRIGGER  (prob={t.prob.toFixed(3)} ema={t.ema.toFixed(3)})
              </Text>
            ))
          )}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0f1419" },
  content: { padding: 16, paddingTop: 48, paddingBottom: 32 },
  statusText: { flex: 1 },
  title: { color: "#e6edf3", fontSize: 22, fontWeight: "600" },
  subtitle: { color: "#7d8590", fontSize: 13, marginTop: 4, marginBottom: 14 },
  panel: { backgroundColor: "#161b22", borderColor: "#30363d", borderWidth: 1,
           borderRadius: 8, padding: 14, marginBottom: 12 },
  logPanel: { minHeight: 100 },
  text: { color: "#e6edf3", fontSize: 15, marginBottom: 8 },
  dim: { color: "#7d8590", fontSize: 13 },
  modelRow: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: 12, flexWrap: "wrap" },
  chip: { backgroundColor: "#21262d", borderColor: "#30363d", borderWidth: 1,
          borderRadius: 14, paddingVertical: 6, paddingHorizontal: 12 },
  chipActive: { backgroundColor: "#1f6feb", borderColor: "#1f6feb" },
  chipText: { color: "#7d8590", fontSize: 12 },
  chipTextActive: { color: "#fff", fontWeight: "600" },
  row: { flexDirection: "row", alignItems: "center", gap: 10, flexWrap: "wrap" },
  btn: { backgroundColor: "#58a6ff", paddingVertical: 10, paddingHorizontal: 18,
         borderRadius: 6, marginRight: 6 },
  btnStop: { backgroundColor: "#444c56" },
  btnDisabled: { opacity: 0.4 },
  btnText: { color: "#fff", fontWeight: "600" },
  bar: { height: 28, backgroundColor: "#21262d", borderRadius: 4, position: "relative" },
  barFill: { height: "100%", backgroundColor: "#58a6ff", borderRadius: 4 },
  barFillTrig: { backgroundColor: "#56d364" },
  marker: { position: "absolute", top: -4, bottom: -4, width: 2, backgroundColor: "#7d8590" },
  labelRow: { flexDirection: "row", justifyContent: "space-between", marginTop: 4 },
  statGrid: { flexDirection: "row", flexWrap: "wrap", marginTop: 12 },
  statLabel: { color: "#7d8590", width: "60%", fontSize: 13, marginBottom: 4 },
  statValue: { color: "#e6edf3", width: "40%", fontSize: 13, marginBottom: 4, textAlign: "right" },
  log: { marginTop: 6 },
  logLine: { color: "#56d364", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
             fontSize: 12, marginBottom: 2 },
});
