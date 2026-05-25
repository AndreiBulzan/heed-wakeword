# Security policy

Heed runs entirely on-device. It records and processes audio locally, trains and
runs models locally, and makes no network calls of its own. The only times it
touches the network are `pip install` and the optional voice downloads
(`heed download-tts`, `heed download-kokoro`), which fetch public package and
model files. No audio or telemetry leaves your machine.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use
GitHub's private vulnerability reporting on this repository (the Security tab,
then "Report a vulnerability"). Include the affected version and steps to
reproduce.

We aim to acknowledge a report within a few days, and we will credit you in the
fix unless you ask otherwise.

## Supported versions

Heed is pre-1.0. Fixes ship on the latest release, so please upgrade to the newest
`heed-wakeword` before reporting and verify the issue still reproduces there.
