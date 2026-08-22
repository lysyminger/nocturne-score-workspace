import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = resolve(projectRoot, "node_modules/js-synthesizer");
const outputRoot = resolve(projectRoot, "public/audio");

const assets = [
  ["externals/libfluidsynth-2.4.6-with-libsndfile.js", "libfluidsynth-2.4.6-with-libsndfile.js"],
  ["dist/js-synthesizer.worklet.min.js", "js-synthesizer.worklet.min.js"],
  ["LICENSE", "LICENSE.js-synthesizer.txt"],
  ["externals/LICENSE.fluidsynth.txt", "LICENSE.fluidsynth.txt"]
];

await mkdir(outputRoot, { recursive: true });
await Promise.all(
  assets.map(([source, destination]) =>
    copyFile(resolve(packageRoot, source), resolve(outputRoot, destination))
  )
);
