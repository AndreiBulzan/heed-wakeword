# Mobile (iOS and Android)

The React Native demo runs on-device wake-word detection on both platforms. It
ships five word slots with live word switching and on-device switching between
ONNX float32, ONNX int8, and TFLite.

The full setup, build, and troubleshooting guide lives with the project:

- [`examples/inference_react_native/README.md`](../examples/inference_react_native/README.md)

In short:

- Android works from any OS, including Windows, over a USB cable
  (`npx expo run:android`). No Mac needed.
- iOS builds in the EAS cloud (`eas build --profile development --platform ios`),
  so you also do not need a Mac.
- On-device timing is about 15 to 20 ms of preprocessing plus 1 to 15 ms of
  inference per 100 ms of audio. Enabling the NNAPI (Android) or Core ML (iOS)
  delegate cuts power further for always-on use.

To put your own word on a phone, train in the [studio](studio.md) and press Send
to mobile, or copy your exported files into a slot under
`examples/inference_react_native/assets/`. See [Export and deploy](export-and-deploy.md).
