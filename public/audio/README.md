# Browser audio engine assets

These files are copied from the pinned `js-synthesizer` npm dependency by
`scripts/copy-audio-engine.mjs` before development and production builds.

- `js-synthesizer` 1.13.0: BSD-3-Clause
- bundled FluidSynth 2.4.6 build: LGPL-2.1-or-later

The app uses FluidSynth through AudioWorklet on secure origins and keeps a
larger ScriptProcessor buffer as a compatibility fallback. Mobile browsers are
more reliable over HTTPS because AudioWorklet is unavailable on plain HTTP.
