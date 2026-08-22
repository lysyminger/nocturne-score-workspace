import { useEffect, useRef, useState } from "react";
import * as alphaTab from "@coderline/alphatab";
import { CircleStop, LoaderCircle, Pause, Play } from "lucide-react";
import { FluidSynthEngine, type FluidSynthMode } from "../audio/FluidSynthEngine";

type Props = {
  scoreUrl: string;
  scrollElement: HTMLElement | null;
  masterVolume: number;
};

export function AlphaTabPlayer({ scoreUrl, scrollElement, masterVolume }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const engineRef = useRef<FluidSynthEngine | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState("正在载入乐谱…");
  const [engineMode, setEngineMode] = useState<FluidSynthMode | null>(null);

  useEffect(() => {
    if (!hostRef.current) return;

    let externalOutput: alphaTab.synth.IExternalMediaSynthOutput | null = null;
    let active = true;
    let engineLoadingStarted = false;
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
        engineLoadingStarted = true;
        const midi = new alphaTab.midi.MidiFile();
        const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midi, true);
        const generator = new alphaTab.midi.MidiFileGenerator(score, api.settings, handler);
        generator.generate();
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
        <span className="alpha-status">
          {!ready && <LoaderCircle size={14} className="spin" />}
          {ready
            ? engineMode === "audio-worklet"
              ? "FluidSynth · 低延迟引擎"
              : "FluidSynth · 稳定缓冲引擎"
            : loading}
        </span>
      </div>
    </div>
  );
}
