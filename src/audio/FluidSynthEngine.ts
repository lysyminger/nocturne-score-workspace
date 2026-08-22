import type * as alphaTab from "@coderline/alphatab";
import type { ISynthesizer, SynthesizerSettings } from "js-synthesizer";

const SOUND_FONT_URL = "/soundfont/sonivox.sf2";
const WORKLET_FLUIDSYNTH_URL = "/audio/libfluidsynth-2.4.6-with-libsndfile.js";
const WORKLET_PLAYER_URL = "/audio/js-synthesizer.worklet.min.js";
const SCRIPT_PROCESSOR_FRAMES = 8192;
const POSITION_UPDATE_INTERVAL = 50;
const TEMPO_EVENT_TYPE = 81;
const DEFAULT_MICROSECONDS_PER_QUARTER = 500_000;

const SYNTH_SETTINGS: SynthesizerSettings = {
  chorusActive: true,
  initialGain: 0.42,
  midiBankSelect: "gm",
  polyphony: 256,
  reverbActive: true,
  reverbDamp: 0.25,
  reverbLevel: 0.35,
  reverbRoomSize: 0.24,
  reverbWidth: 0.8
};

export type FluidSynthMode = "audio-worklet" | "stable-buffer";

type TempoSegment = {
  tick: number;
  timeMs: number;
  microsecondsPerQuarter: number;
};

type EngineCallbacks = {
  onPosition: (timeMs: number) => void;
  onPlaybackChange: (playing: boolean) => void;
  onReady: (mode: FluidSynthMode) => void;
  onStatus: (message: string) => void;
};

type SynthBackend = {
  mode: FluidSynthMode;
  node: AudioNode;
  synth: Pick<
    ISynthesizer,
    | "addSMFDataToPlayer"
    | "close"
    | "closePlayer"
    | "isPlayerPlaying"
    | "loadSFont"
    | "playPlayer"
    | "retrievePlayerCurrentTick"
    | "seekPlayer"
    | "stopPlayer"
    | "waitForPlayerStopped"
  >;
};

class MidiTimeline {
  readonly durationMs: number;
  readonly endTick: number;
  private readonly division: number;
  private readonly segments: TempoSegment[];

  constructor(file: alphaTab.midi.MidiFile) {
    this.division = Math.max(1, file.division);
    this.endTick = file.events.reduce((end, event) => Math.max(end, event.tick), 0);

    const tempoChanges = file.events
      .filter((event) => event.type === TEMPO_EVENT_TYPE)
      .map((event) => ({
        tick: event.tick,
        microsecondsPerQuarter: (event as alphaTab.midi.TempoChangeEvent)
          .microSecondsPerQuarterNote
      }))
      .sort((left, right) => left.tick - right.tick);

    const normalized = new Map<number, number>();
    normalized.set(0, DEFAULT_MICROSECONDS_PER_QUARTER);
    for (const change of tempoChanges) {
      if (change.microsecondsPerQuarter > 0) {
        normalized.set(change.tick, change.microsecondsPerQuarter);
      }
    }

    this.segments = [];
    let previousTick = 0;
    let previousTime = 0;
    let previousTempo = DEFAULT_MICROSECONDS_PER_QUARTER;
    for (const [tick, tempo] of [...normalized.entries()].sort((left, right) => left[0] - right[0])) {
      if (tick > previousTick) {
        previousTime += this.ticksToMilliseconds(tick - previousTick, previousTempo);
      }
      this.segments.push({ tick, timeMs: previousTime, microsecondsPerQuarter: tempo });
      previousTick = tick;
      previousTempo = tempo;
    }

    this.durationMs = this.tickToMilliseconds(this.endTick);
  }

  tickToMilliseconds(tick: number): number {
    const safeTick = Math.max(0, tick);
    const segment = this.findSegmentForTick(safeTick);
    return segment.timeMs + this.ticksToMilliseconds(
      safeTick - segment.tick,
      segment.microsecondsPerQuarter
    );
  }

  millisecondsToTick(timeMs: number): number {
    const safeTime = Math.max(0, timeMs);
    const segment = this.findSegmentForTime(safeTime);
    const elapsed = safeTime - segment.timeMs;
    return Math.round(segment.tick + elapsed * this.division * 1000 / segment.microsecondsPerQuarter);
  }

  private ticksToMilliseconds(ticks: number, microsecondsPerQuarter: number): number {
    return ticks * microsecondsPerQuarter / this.division / 1000;
  }

  private findSegmentForTick(tick: number): TempoSegment {
    for (let index = this.segments.length - 1; index >= 0; index -= 1) {
      if (this.segments[index].tick <= tick) return this.segments[index];
    }
    return this.segments[0];
  }

  private findSegmentForTime(timeMs: number): TempoSegment {
    for (let index = this.segments.length - 1; index >= 0; index -= 1) {
      if (this.segments[index].timeMs <= timeMs) return this.segments[index];
    }
    return this.segments[0];
  }
}

let soundFontPromise: Promise<ArrayBuffer> | null = null;
let scriptRuntimePromise: Promise<typeof import("js-synthesizer").Synthesizer> | null = null;
let fluidRuntimePromise: Promise<unknown> | null = null;

function loadSoundFont(): Promise<ArrayBuffer> {
  if (!soundFontPromise) {
    const request = fetch(SOUND_FONT_URL).then(async (response) => {
      if (!response.ok) throw new Error(`SoundFont HTTP ${response.status}`);
      return response.arrayBuffer();
    });
    soundFontPromise = request;
    void request.catch(() => {
      if (soundFontPromise === request) soundFontPromise = null;
    });
  }
  return soundFontPromise;
}

async function loadScriptRuntime(): Promise<typeof import("js-synthesizer").Synthesizer> {
  if (!scriptRuntimePromise) {
    scriptRuntimePromise = Promise.all([
      import("js-synthesizer"),
      loadFluidRuntime()
    ]).then(async ([library, runtime]) => {
      library.Synthesizer.initializeWithFluidSynthModule(runtime);
      await library.waitForReady();
      library.disableLogging(library.LogLevel.Error);
      return library.Synthesizer;
    });
    void scriptRuntimePromise.catch(() => {
      scriptRuntimePromise = null;
    });
  }
  return scriptRuntimePromise;
}

function loadFluidRuntime(): Promise<unknown> {
  const existing = (globalThis as typeof globalThis & { Module?: unknown }).Module;
  if (isFluidRuntime(existing)) return Promise.resolve(existing);
  if (!fluidRuntimePromise) {
    const request = new Promise<unknown>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = WORKLET_FLUIDSYNTH_URL;
      script.async = true;
      script.onload = () => {
        const runtime = (globalThis as typeof globalThis & { Module?: unknown }).Module;
        if (isFluidRuntime(runtime)) resolve(runtime);
        else reject(new Error("FluidSynth 浏览器运行时未正确初始化"));
      };
      script.onerror = () => reject(new Error("FluidSynth 浏览器运行时下载失败"));
      document.head.append(script);
    });
    fluidRuntimePromise = request;
    void request.catch(() => {
      if (fluidRuntimePromise === request) fluidRuntimePromise = null;
    });
  }
  return fluidRuntimePromise;
}

function isFluidRuntime(value: unknown): value is { addFunction: unknown; addOnPostRun: unknown } {
  if (!value || typeof value !== "object") return false;
  const runtime = value as { addFunction?: unknown; addOnPostRun?: unknown };
  return typeof runtime.addFunction === "function" && typeof runtime.addOnPostRun === "function";
}

export class FluidSynthEngine {
  private backend: SynthBackend | null = null;
  private callbacks: EngineCallbacks;
  private context: AudioContext | null = null;
  private disposed = false;
  private gainNode: GainNode | null = null;
  private loadVersion = 0;
  private masterVolumeValue = 1;
  private midiTimeline: MidiTimeline | null = null;
  private playbackRateValue = 1;
  private playing = false;
  private playVersion = 0;
  private pollBusy = false;
  private pollTimer: number | null = null;
  private positionMs = 0;
  private ready = false;

  constructor(callbacks: EngineCallbacks) {
    this.callbacks = callbacks;
  }

  get backingTrackDuration(): number {
    return this.midiTimeline?.durationMs ?? 0;
  }

  get playbackRate(): number {
    return this.playbackRateValue;
  }

  set playbackRate(value: number) {
    this.playbackRateValue = Number.isFinite(value) && value > 0 ? value : 1;
  }

  get masterVolume(): number {
    return this.masterVolumeValue;
  }

  set masterVolume(value: number) {
    this.masterVolumeValue = Math.min(1, Math.max(0, value));
    const context = this.context;
    const gainNode = this.gainNode;
    if (context && gainNode) {
      gainNode.gain.setTargetAtTime(this.masterVolumeValue, context.currentTime, 0.015);
    }
  }

  async loadMidi(file: alphaTab.midi.MidiFile): Promise<void> {
    const version = ++this.loadVersion;
    this.ready = false;
    this.positionMs = 0;
    this.midiTimeline = new MidiTimeline(file);
    this.callbacks.onPosition(0);
    this.callbacks.onStatus("正在启动 FluidSynth…");

    await this.releaseBackend();
    if (!this.isCurrent(version)) return;

    const context = new AudioContext({ latencyHint: "playback" });
    const gainNode = context.createGain();
    gainNode.gain.value = this.masterVolumeValue;
    gainNode.connect(context.destination);

    let backend: SynthBackend | null = null;
    try {
      backend = await this.createBackend(context, gainNode);
      if (!this.isCurrent(version)) {
        await this.disposeCreatedBackend(backend, gainNode, context);
        return;
      }

      this.callbacks.onStatus("正在载入采样音色…");
      const soundFont = await loadSoundFont();
      if (!this.isCurrent(version)) {
        await this.disposeCreatedBackend(backend, gainNode, context);
        return;
      }

      const midiBytes = file.toBinary();
      const midiBuffer = midiBytes.buffer.slice(
        midiBytes.byteOffset,
        midiBytes.byteOffset + midiBytes.byteLength
      ) as ArrayBuffer;
      await backend.synth.loadSFont(soundFont.slice(0));
      await backend.synth.addSMFDataToPlayer(midiBuffer);
      if (!this.isCurrent(version)) {
        await this.disposeCreatedBackend(backend, gainNode, context);
        return;
      }

      this.backend = backend;
      this.context = context;
      this.gainNode = gainNode;
      this.ready = true;
      this.callbacks.onReady(backend.mode);
    } catch (error) {
      if (backend) await this.disposeCreatedBackend(backend, gainNode, context);
      else {
        gainNode.disconnect();
        await context.close().catch(() => undefined);
      }
      if (this.isCurrent(version)) throw error;
    }
  }

  play(): void {
    void this.startPlayback();
  }

  pause(): void {
    this.pausePlayback();
  }

  seekTo(timeMs: number): void {
    const timeline = this.midiTimeline;
    const backend = this.backend;
    this.positionMs = Math.min(Math.max(0, timeMs), this.backingTrackDuration);
    if (timeline && backend) {
      backend.synth.seekPlayer(timeline.millisecondsToTick(this.positionMs));
    }
    this.callbacks.onPosition(this.positionMs);
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.loadVersion += 1;
    this.playVersion += 1;
    this.stopPolling();
    void this.releaseBackend();
  }

  private async createBackend(context: AudioContext, gainNode: GainNode): Promise<SynthBackend> {
    if (globalThis.isSecureContext && context.audioWorklet) {
      try {
        return await this.createWorkletBackend(context, gainNode);
      } catch (error) {
        console.warn("FluidSynth AudioWorklet 不可用，切换到稳定缓冲模式。", error);
      }
    }
    return this.createStableBufferBackend(context, gainNode);
  }

  private async createWorkletBackend(
    context: AudioContext,
    gainNode: GainNode
  ): Promise<SynthBackend> {
    await context.audioWorklet.addModule(WORKLET_FLUIDSYNTH_URL);
    await context.audioWorklet.addModule(WORKLET_PLAYER_URL);
    const { AudioWorkletNodeSynthesizer } = await import("js-synthesizer");
    const synth = new AudioWorkletNodeSynthesizer();
    const node = synth.createAudioNode(context, SYNTH_SETTINGS);
    node.connect(gainNode);
    return { mode: "audio-worklet", node, synth };
  }

  private async createStableBufferBackend(
    context: AudioContext,
    gainNode: GainNode
  ): Promise<SynthBackend> {
    const Synthesizer = await loadScriptRuntime();
    const synth = new Synthesizer();
    synth.init(context.sampleRate, SYNTH_SETTINGS);
    const node = synth.createAudioNode(context, SCRIPT_PROCESSOR_FRAMES);
    node.connect(gainNode);
    return { mode: "stable-buffer", node, synth };
  }

  private async startPlayback(): Promise<void> {
    const backend = this.backend;
    const context = this.context;
    const timeline = this.midiTimeline;
    if (!this.ready || !backend || !context || !timeline || this.playing || this.disposed) return;

    const version = ++this.playVersion;
    if (this.positionMs >= this.backingTrackDuration - 5) this.seekTo(0);

    try {
      await context.resume();
      if (version !== this.playVersion || this.disposed) return;
      backend.synth.seekPlayer(timeline.millisecondsToTick(this.positionMs));
      await backend.synth.playPlayer();
      if (version !== this.playVersion || this.disposed) {
        backend.synth.stopPlayer();
        return;
      }
      this.playing = true;
      this.callbacks.onPlaybackChange(true);
      this.startPolling();
      void backend.synth.waitForPlayerStopped().then(() => {
        if (version === this.playVersion && this.playing && !this.disposed) {
          this.finishPlayback();
        }
      });
    } catch (error) {
      if (version === this.playVersion && !this.disposed) {
        this.playing = false;
        this.callbacks.onPlaybackChange(false);
        this.callbacks.onStatus(`播放启动失败：${errorMessage(error)}`);
      }
    }
  }

  private pausePlayback(): void {
    this.playVersion += 1;
    this.playing = false;
    this.stopPolling();
    const backend = this.backend;
    const timeline = this.midiTimeline;
    if (backend) {
      backend.synth.stopPlayer();
      if (timeline) backend.synth.seekPlayer(timeline.millisecondsToTick(this.positionMs));
    }
    this.callbacks.onPlaybackChange(false);
  }

  private finishPlayback(): void {
    this.playing = false;
    this.stopPolling();
    this.positionMs = this.backingTrackDuration;
    this.callbacks.onPosition(this.positionMs);
    this.callbacks.onPlaybackChange(false);
  }

  private startPolling(): void {
    this.stopPolling();
    this.pollTimer = window.setInterval(() => void this.updatePosition(), POSITION_UPDATE_INTERVAL);
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.pollBusy = false;
  }

  private async updatePosition(): Promise<void> {
    const backend = this.backend;
    const timeline = this.midiTimeline;
    if (!this.playing || this.pollBusy || !backend || !timeline || this.disposed) return;
    this.pollBusy = true;
    const version = this.playVersion;
    try {
      const tick = await backend.synth.retrievePlayerCurrentTick();
      if (version !== this.playVersion || !this.playing || this.disposed) return;
      this.positionMs = Math.min(timeline.tickToMilliseconds(tick), this.backingTrackDuration);
      this.callbacks.onPosition(this.positionMs);
      if (!backend.synth.isPlayerPlaying() && tick >= timeline.endTick) this.finishPlayback();
    } catch (error) {
      if (!this.disposed) console.warn("FluidSynth 播放位置读取失败。", error);
    } finally {
      this.pollBusy = false;
    }
  }

  private isCurrent(version: number): boolean {
    return !this.disposed && version === this.loadVersion;
  }

  private async releaseBackend(): Promise<void> {
    const backend = this.backend;
    const gainNode = this.gainNode;
    const context = this.context;
    this.backend = null;
    this.gainNode = null;
    this.context = null;
    this.ready = false;
    this.pausePlayback();
    if (backend && gainNode && context) {
      await this.disposeCreatedBackend(backend, gainNode, context);
    }
  }

  private async disposeCreatedBackend(
    backend: SynthBackend,
    gainNode: GainNode,
    context: AudioContext
  ): Promise<void> {
    try {
      backend.synth.stopPlayer();
      backend.synth.closePlayer();
      backend.synth.close();
    } catch (error) {
      console.warn("FluidSynth 释放资源时出现警告。", error);
    }
    backend.node.disconnect();
    gainNode.disconnect();
    await context.close().catch(() => undefined);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "未知错误";
}
