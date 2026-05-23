// Metro needs to know that .onnx files are binary assets (not source).
// Without this, `require("./assets/wake.onnx")` fails to bundle.
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push("onnx");
config.resolver.assetExts.push("ort");
config.resolver.assetExts.push("tflite");

module.exports = config;
