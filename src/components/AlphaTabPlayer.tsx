import { useEffect, useRef, useState } from "react";
import * as alphaTab from "@coderline/alphatab";
import { CircleStop, Download, LoaderCircle, Pause, Play } from "lucide-react";
import { FluidSynthEngine, type FluidSynthMode } from "../audio/FluidSynthEngine";

type Props = {
  scoreUrl: string;
  scrollElement: HTMLElement | null;
  masterVolume: number;
  fileBaseName: string;
};

type ExportKind = "gp" | "midi" | "wav";

const WAV_SAMPLE_RATE = 44_100;
const MAX_WAV_DURATION_MS = 6 * 60 * 1000;

function safeFileName(value: string) {
  return value
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/[. ]+$/g, "")
    .slice(0, 100) || "nocturne-score";
}

function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function binaryBlob(bytes: Uint8Array, mediaType: string) {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new Blob([copy.buffer], { type: mediaType });
}

function encodePcm16Chunk(samples: Float32Array) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const normalized = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, normalized < 0 ? normalized * 0x8000 : normalized * 0x7fff, true);
  }
  return buffer;
}

function encodePcm16Wav(chunks: ArrayBuffer[], sampleCount: number, sampleRate: number) {
  const channels = 2;
  const bytesPerSample = 2;
  const dataBytes = sampleCount * bytesPerSample;
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const writeAscii = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);
  return new Blob([header, ...chunks], { type: "audio/wav" });
}

export function AlphaTabPlayer({ scoreUrl, scrollElement, masterVolume, fileBaseName }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const engineRef = useRef<FluidSynthEngine | null>(null);
  const scoreRef = useRef<alphaTab.model.Score | null>(null);
  const midiRef = useRef<alphaTab.midi.MidiFile | null>(null);
  const [ready, setReady] = useState(false);
  const [scoreReady, setScoreReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState("正在载入乐谱…");
  const [engineMode, setEngineMode] = useState<FluidSynthMode | null>(null);
  const [exporting, setExporting] = useState<ExportKind | null>(null);
  const [exportStatus, setExportStatus] = useState("");

  useEffect(() => {
    if (!hostRef.current) return;

    let externalOutput: alphaTab.synth.IExternalMediaSynthOutput | null = null;
    let active = true;
    let engineLoadingStarted = false;
    scoreRef.current = null;
    midiRef.current = null;
    setScoreReady(false);
    const engine = new FluidSynthEngine({
      onPosition: (timeMs) => externalOutput?.updatePosition(timeMs),
      onPlaybackChange: (isPlaying) => active && setPlaying(isPlaying),
      onReady: (mode) => {
        if (!active) return;
        setEngineMode(mode);
        setReady(true);
        setLoading("");
      },
      onStatus: (message) => active && setLoading(message)
    });
    engine.masterVolume = masterVolume;
    engineRef.current = engine;

    const api = new alphaTab.AlphaTabApi(hostRef.current, {
      core: {
        fontDirectory: "/font/"
      },
      display: {
        layoutMode: alphaTab.LayoutMode.Page,
        staveProfile: alphaTab.StaveProfile.TabMixed
      },
      player: {
        playerMode: alphaTab.PlayerMode.EnabledExternalMedia,
        ...(scrollElement ? { scrollElement } : {})
      }
    });
    apiRef.current = api;

    const attachEngine = () => {
      const output = api.player?.output as alphaTab.synth.IExternalMediaSynthOutput | undefined;
      if (!output || !("handler" in output)) return;
      externalOutput = output;
      output.handler = engine;
    };

    attachEngine();
    api.renderStarted.on(() => !engineLoadingStarted && setLoading("正在排版乐谱…"));
    api.renderFinished.on(() => !engineLoadingStarted && setLoading("正在生成演奏数据…"));
    api.playerReady.on(attachEngine);
    api.scoreLoaded.on((score) => {
      try {
        scoreRef.current = score;
        setScoreReady(true);
        engineLoadingStarted = true;
        const midi = new alphaTab.midi.MidiFile();
        const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midi, true);
        const generator = new alphaTab.midi.MidiFileGenerator(score, api.settings, handler);
        generator.generate();
        midiRef.current = midi;
        void engine.loadMidi(midi).catch((error: unknown) => {
          if (!active) return;
          const message = error instanceof Error ? error.message : "未知错误";
          setReady(false);
          setEngineMode(null);
          setLoading(`音色引擎载入失败：${message}`);
        });
      } catch (error) {
        if (!active) return;
        const message = error instanceof Error ? error.message : "未知错误";
        setReady(false);
        setEngineMode(null);
        setLoading(`演奏数据生成失败：${message}`);
      }
    });
    api.playerStateChanged.on((event) => setPlaying(event.state === alphaTab.synth.PlayerState.Playing));
    api.error.on((error) => {
      if (!active) return;
      setReady(false);
      setLoading(`乐谱载入失败：${error.message}`);
    });
    api.load(scoreUrl);

    return () => {
      active = false;
      scoreRef.current = null;
      midiRef.current = null;
      if (externalOutput) externalOutput.handler = undefined;
      engine.destroy();
      if (engineRef.current === engine) engineRef.current = null;
      api.destroy();
      apiRef.current = null;
    };
  }, [scoreUrl, scrollElement]);

  useEffect(() => {
    if (engineRef.current) engineRef.current.masterVolume = masterVolume;
  }, [masterVolume]);

  async function exportScore(kind: ExportKind) {
    if (exporting) return;
    const api = apiRef.current;
    const score = scoreRef.current;
    const midi = midiRef.current;
    if (!api || !score) {
      setExportStatus("乐谱尚未载入完成");
      return;
    }

    const baseName = safeFileName(fileBaseName);
    setExporting(kind);
    try {
      if (kind === "gp") {
        setExportStatus("正在生成 Guitar Pro 7 文件…");
        const bytes = new alphaTab.exporter.Gp7Exporter().export(score, api.settings);
        downloadBlob(binaryBlob(bytes, "application/octet-stream"), `${baseName}.gp`);
        setExportStatus("Guitar Pro 7 已导出");
        return;
      }

      if (kind === "midi") {
        if (!midi) throw new Error("MIDI 演奏数据尚未准备完成");
        setExportStatus("正在生成 MIDI…");
        downloadBlob(binaryBlob(midi.toBinary(), "audio/midi"), `${baseName}.mid`);
        setExportStatus("MIDI 已导出");
        return;
      }

      setExportStatus("正在载入导出音色…");
      const soundFontResponse = await fetch("/soundfont/sonivox.sf2", { credentials: "same-origin" });
      if (!soundFontResponse.ok) throw new Error("导出音色载入失败");
      const options = new alphaTab.synth.AudioExportOptions();
      options.soundFonts = [new Uint8Array(await soundFontResponse.arrayBuffer())];
      options.sampleRate = WAV_SAMPLE_RATE;
      options.masterVolume = Math.max(0, Math.min(3, masterVolume));
      options.metronomeVolume = 0;
      options.useSyncPoints = false;

      const exporter = await api.exportAudio(options);
      const chunks: ArrayBuffer[] = [];
      let sampleCount = 0;
      try {
        while (true) {
          const chunk = await exporter.render(500);
          if (!chunk) break;
          if (chunk.endTime > MAX_WAV_DURATION_MS) {
            throw new Error("浏览器 WAV 导出目前限制为 6 分钟，请改用 GP7 或 MIDI");
          }
          sampleCount += chunk.samples.length;
          chunks.push(encodePcm16Chunk(chunk.samples));
          const progress = chunk.endTime > 0 ? Math.min(100, Math.round(chunk.currentTime / chunk.endTime * 100)) : 0;
          setExportStatus(`正在合成 WAV · ${progress}%`);
        }
      } finally {
        exporter.destroy();
      }
      if (!chunks.length) throw new Error("没有生成可导出的音频样本");
      downloadBlob(encodePcm16Wav(chunks, sampleCount, options.sampleRate), `${baseName}.wav`);
      setExportStatus("WAV 合成音频已导出");
    } catch (error) {
      setExportStatus(`导出失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setExporting(null);
    }
  }

  return (
    <div className="alpha-player">
      <div ref={hostRef} className="alpha-host" />
      <div className="alpha-controls glass-bar">
        <button
          className="icon-button strong"
          type="button"
          disabled={!ready}
          onClick={() => apiRef.current?.playPause()}
          aria-label={playing ? "暂停" : "播放"}
        >
          {playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
        </button>
        <button
          className="icon-button"
          type="button"
          disabled={!ready}
          onClick={() => apiRef.current?.stop()}
          aria-label="停止"
        >
          <CircleStop size={18} />
        </button>
        <div className="alpha-export-actions" aria-label="导出乐谱">
          <button type="button" disabled={!scoreReady || Boolean(exporting)} onClick={() => void exportScore("gp")} title="导出 Guitar Pro 7 .gp 文件"><Download size={13} /> GP7</button>
          <button type="button" disabled={!midiRef.current || Boolean(exporting)} onClick={() => void exportScore("midi")} title="导出标准 MIDI 文件"><Download size={13} /> MIDI</button>
          <button type="button" disabled={!scoreReady || Boolean(exporting)} onClick={() => void exportScore("wav")} title="用当前谱面与音色合成 WAV"><Download size={13} /> WAV</button>
        </div>
        <span className="alpha-status">
          {(!ready || Boolean(exporting)) && <LoaderCircle size={14} className="spin" />}
          {exportStatus || (ready
            ? engineMode === "audio-worklet"
              ? "FluidSynth · 低延迟引擎"
              : "FluidSynth · 稳定缓冲引擎"
            : loading)}
        </span>
      </div>
    </div>
  );
}
