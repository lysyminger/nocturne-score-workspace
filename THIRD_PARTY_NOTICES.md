# Third-party notices

Nocturne uses third-party software and media assets. Those components remain
under their own licenses; this file is a practical summary and does not replace
the license text shipped by each dependency.

## Browser and score playback

| Component | Version | License | Purpose |
| --- | --- | --- | --- |
| [alphaTab](https://github.com/CoderLine/alphaTab) | 1.8.4 | MPL-2.0 | Guitar Pro/MusicXML parsing, notation rendering and playback events |
| [alphaTab Vite plugin](https://github.com/CoderLine/alphaTab) | 1.8.4 | MPL-2.0 | Worker bundling and build-time score assets |
| [js-synthesizer](https://github.com/jet2jet/js-synthesizer) | 1.13.0 | BSD-3-Clause | FluidSynth bindings for Web Audio |
| [FluidSynth](https://github.com/FluidSynth/fluidsynth) | 2.4.6 browser build | LGPL-2.1-or-later | SoundFont synthesis |
| [React](https://github.com/facebook/react) | 19.2.8 | MIT | User interface |
| [Lucide](https://github.com/lucide-icons/lucide) | 1.33.0 | ISC | Interface icons |

The alphaTab package supplies Bravura notation fonts and a SONiVOX SoundFont
with their accompanying notices. The Vite plugin copies those assets into the
build output. `scripts/copy-audio-engine.mjs` copies the pinned FluidSynth and
js-synthesizer browser files together with their license texts before a build.

## Server and media processing

The Python environment includes FastAPI, Uvicorn, Pillow, NumPy, OpenCV,
yt-dlp, httpx and pytest under their respective upstream licenses. Runtime
deployments also depend on separately installed FFmpeg and Tesseract. Optional
five-line staff recognition uses a separately installed Audiveris distribution.

Consult `package-lock.json`, `backend/requirements.txt` and each installed
package for the exact dependency graph and complete license text.

## User-provided content

No license in this repository grants rights to third-party videos, audio,
thumbnails or sheet music processed with the application. Users are responsible
for obtaining the permissions required for downloading, transforming, storing
or sharing their source material.
