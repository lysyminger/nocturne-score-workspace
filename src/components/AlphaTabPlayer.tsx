import { useEffect, useMemo, useRef, useState } from "react";
import * as alphaTab from "@coderline/alphatab";
import {
  Check,
  CircleStop,
  Command,
  Download,
  ImageIcon,
  Keyboard,
  LoaderCircle,
  Pause,
  PencilLine,
  Plus,
  Play,
  Save,
  ScanLine,
  Undo2,
  Video
} from "lucide-react";
import type {
  RecognitionDiagnostics,
  RecognitionEvent,
  RecognitionMeasure,
  SyncPoint,
  TabTechnique,
  VideoFrame
} from "../types";
import { FluidSynthEngine, type FluidSynthMode } from "../audio/FluidSynthEngine";
import { FloatingReference, type ReferenceMode } from "./FloatingReference";

type RecognitionChange = { measure: number; events: RecognitionEvent[] };

type Props = {
  scoreUrl: string;
  scrollElement: HTMLElement | null;
  masterVolume: number;
  fileBaseName: string;
  pdfUrl: string | null;
  videoUrl: string | null;
  syncPoints: SyncPoint[];
  videoFrames: VideoFrame[];
  recognition: RecognitionDiagnostics | null;
  editingDisabled: boolean;
  onFocusMeasure: (measure: number) => void;
  onDirtyChange: (dirty: boolean) => void;
  onSaveScore: (file: File) => Promise<void>;
  onSaveRecognition: (changes: RecognitionChange[]) => Promise<void>;
  onRetryRecognition: (measure: number) => Promise<void>;
  onAppendMeasure: () => Promise<void>;
};

type ExportKind = "gp" | "midi" | "wav";
type EditCommand = {
  id: TabTechnique;
  label: string;
  mark: string;
  shortcut: string;
  requiresPair?: boolean;
};

type SelectionMarker = {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type ScoreCursor = {
  beat: alphaTab.model.Beat;
  onset: number;
  string: number;
  marker: SelectionMarker;
};

type MeasureMarker = SelectionMarker & { measure: number };

type NoteState = {
  note: alphaTab.model.Note;
  beat: alphaTab.model.Beat;
  present: boolean;
  fret: number;
  string: number;
  bendType: alphaTab.model.BendType;
  bendPoints: Array<{ offset: number; value: number }> | null;
  isHammerPullOrigin: boolean;
  hammerPullOrigin: alphaTab.model.Note | null;
  hammerPullDestination: alphaTab.model.Note | null;
  isSlurDestination: boolean;
  slurOrigin: alphaTab.model.Note | null;
  slurDestination: alphaTab.model.Note | null;
  harmonicType: alphaTab.model.HarmonicType;
  harmonicValue: number;
  isLetRing: boolean;
  isPalmMute: boolean;
  isDead: boolean;
  slideOutType: alphaTab.model.SlideOutType;
  slideTarget: alphaTab.model.Note | null;
  slideOrigin: alphaTab.model.Note | null;
  isTieDestination: boolean;
  tieOrigin: alphaTab.model.Note | null;
  tieDestination: alphaTab.model.Note | null;
  vibrato: alphaTab.model.VibratoType;
};

type BeatState = {
  beat: alphaTab.model.Beat;
  voice: alphaTab.model.Voice;
  index: number;
  present: boolean;
  duration: alphaTab.model.Duration;
  dots: number;
};

type HistoryEntry = {
  label: string;
  states: NoteState[];
  beatStates: BeatState[];
  selection: alphaTab.model.Note[];
  recognition: RecognitionMeasure[] | null;
  dirtyRecognitionMeasures: number[];
};

type VideoSyncAnchor = { tick: number; timeSeconds: number };
type PlaybackPosition = { tick: number; endTick: number; endTime: number };
type RestPlan = {
  beats: alphaTab.model.Beat[];
  delta: number;
  explicitDuration: number;
  availableDuration: number;
  insertIndex: number;
  voice: alphaTab.model.Voice;
};

const WAV_SAMPLE_RATE = 44_100;
const MAX_WAV_DURATION_MS = 6 * 60 * 1000;
const EIGHTH_NOTE_TICKS = 480;
const MEASURE_EIGHTHS = 8;
const DIRECT_DURATIONS = [8, 4, 2, 1, 0.5] as const;

const EDIT_COMMANDS: EditCommand[] = [
  { id: "legato", label: "连音", mark: "⌒", shortcut: "L", requiresPair: true },
  { id: "slide", label: "滑音", mark: "/", shortcut: "S", requiresPair: true },
  { id: "hammer_on", label: "击弦", mark: "H", shortcut: "H", requiresPair: true },
  { id: "pull_off", label: "勾弦", mark: "P", shortcut: "P", requiresPair: true },
  { id: "bend", label: "推弦", mark: "B", shortcut: "B" },
  { id: "vibrato", label: "颤音", mark: "~", shortcut: "V" },
  { id: "harmonic", label: "泛音", mark: "◇", shortcut: "N" },
  { id: "palm_mute", label: "闷音", mark: "PM", shortcut: "M" },
  { id: "let_ring", label: "延音", mark: "LR", shortcut: "I" },
  { id: "dead_note", label: "死音", mark: "×", shortcut: "X" }
];

function safeFileName(value: string) {
  return value.trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/[. ]+$/g, "").slice(0, 100) || "nocturne-score";
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
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
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

function createMidi(score: alphaTab.model.Score, settings: alphaTab.Settings) {
  const midi = new alphaTab.midi.MidiFile();
  const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midi, true);
  new alphaTab.midi.MidiFileGenerator(score, settings, handler).generate();
  return midi;
}

function flattenNotes(score: alphaTab.model.Score) {
  const notes: alphaTab.model.Note[] = [];
  for (const track of score.tracks) for (const staff of track.staves) for (const bar of staff.bars) {
    for (const voice of bar.voices) for (const beat of voice.beats) notes.push(...beat.notes);
  }
  return notes.sort((left, right) =>
    left.beat.voice.bar.masterBar.index - right.beat.voice.bar.masterBar.index
    || left.beat.absolutePlaybackStart - right.beat.absolutePlaybackStart
    || left.string - right.string
  );
}

function flattenBeats(score: alphaTab.model.Score) {
  const beats: alphaTab.model.Beat[] = [];
  for (const track of score.tracks) for (const staff of track.staves) for (const bar of staff.bars) {
    for (const voice of bar.voices) beats.push(...voice.beats);
  }
  return beats;
}

function relatedNotes(notes: alphaTab.model.Note[]) {
  const related = [...notes];
  for (const note of notes) {
    for (const candidate of [
      note.hammerPullOrigin,
      note.hammerPullDestination,
      note.slurOrigin,
      note.slurDestination,
      note.slideOrigin,
      note.slideTarget,
      note.tieOrigin,
      note.tieDestination
    ]) if (candidate) related.push(candidate);
  }
  return uniqueNotes(related);
}

function harmonicValueForFret(fret: number) {
  if (fret === 2) return 2.4;
  if (fret === 3) return 3.2;
  if ([4, 5, 7, 9, 12, 16, 17, 19, 24].includes(fret)) return fret;
  if (fret === 8) return 8.2;
  if (fret === 10) return 9.6;
  if (fret === 14 || fret === 15) return 14.7;
  if (fret === 21 || fret === 22) return 21.7;
  return 12;
}

function prepareScoreForFinish(score: alphaTab.model.Score) {
  const notes = flattenNotes(score);
  const expectedHammerTargets = new Map(notes.map((note) => [note, note.hammerPullDestination]));

  for (const beat of flattenBeats(score)) {
    beat.noteStringLookup.clear();
    beat.noteValueLookup.clear();
    beat.isLetRing = false;
    beat.isPalmMute = false;
    beat.isEffectSlurOrigin = false;
    beat.effectSlurOrigin = null;
    beat.effectSlurDestination = null;
    beat.notes.forEach((note, index) => {
      note.beat = beat;
      note.index = index;
      if (note.isStringed) beat.noteStringLookup.set(note.string, note);
      beat.noteValueLookup.set(note.realValue, note);
    });
  }

  for (const note of notes) {
    if (note.isHammerPullOrigin) {
      const expected = expectedHammerTargets.get(note);
      const actual = alphaTab.model.Note.findHammerPullDestination(note);
      if (expected && actual !== expected) note.isHammerPullOrigin = false;
    }
    note.hammerPullOrigin = null;
    note.hammerPullDestination = null;
    note.isEffectSlurOrigin = false;
    note.hasEffectSlur = false;
    note.effectSlurOrigin = null;
    note.effectSlurDestination = null;
    note.letRingDestination = null;
    note.palmMuteDestination = null;
    note.maxBendPoint = note.bendPoints?.reduce<alphaTab.model.BendPoint | null>(
      (maximum, point) => !maximum || point.value > maximum.value ? point : maximum,
      null
    ) ?? null;
  }
}

function detachNoteRelations(note: alphaTab.model.Note) {
  if (note.hammerPullOrigin) {
    note.hammerPullOrigin.isHammerPullOrigin = false;
    note.hammerPullOrigin.hammerPullDestination = null;
    note.hammerPullOrigin = null;
  }
  if (note.hammerPullDestination) {
    if (note.hammerPullDestination.hammerPullOrigin === note) note.hammerPullDestination.hammerPullOrigin = null;
    note.hammerPullDestination = null;
  }
  note.isHammerPullOrigin = false;
  if (note.slideOrigin) {
    note.slideOrigin.slideTarget = null;
    note.slideOrigin.slideOutType = alphaTab.model.SlideOutType.None;
    note.slideOrigin = null;
  }
  if (note.slideTarget) {
    if (note.slideTarget.slideOrigin === note) note.slideTarget.slideOrigin = null;
    note.slideTarget = null;
  }
  note.slideOutType = alphaTab.model.SlideOutType.None;
  if (note.slurOrigin) {
    note.slurOrigin.slurDestination = null;
    note.slurOrigin = null;
  }
  if (note.slurDestination) {
    note.slurDestination.slurOrigin = null;
    note.slurDestination.isSlurDestination = false;
    note.slurDestination = null;
  }
  note.isSlurDestination = false;
  if (note.tieOrigin) {
    note.tieOrigin.tieDestination = null;
    note.tieOrigin = null;
  }
  if (note.tieDestination) {
    note.tieDestination.tieOrigin = null;
    note.tieDestination.isTieDestination = false;
    note.tieDestination = null;
  }
  note.isTieDestination = false;
}

function clearTechniqueFromNote(note: alphaTab.model.Note, technique: TabTechnique) {
  if (technique === "legato") {
    const destination = note.slurDestination;
    note.slurDestination = null;
    if (destination?.slurOrigin === note) {
      destination.slurOrigin = null;
      destination.isSlurDestination = false;
    }
  } else if (technique === "slide") {
    const destination = note.slideTarget;
    note.slideOutType = alphaTab.model.SlideOutType.None;
    note.slideTarget = null;
    if (destination?.slideOrigin === note) destination.slideOrigin = null;
  } else if (technique === "hammer_on" || technique === "pull_off") {
    const destination = note.hammerPullDestination;
    note.isHammerPullOrigin = false;
    note.hammerPullDestination = null;
    if (destination?.hammerPullOrigin === note) destination.hammerPullOrigin = null;
  } else if (technique === "bend") {
    note.bendType = alphaTab.model.BendType.None;
    note.bendPoints = null;
    note.maxBendPoint = null;
  } else if (technique === "vibrato") note.vibrato = alphaTab.model.VibratoType.None;
  else if (technique === "harmonic") {
    note.harmonicType = alphaTab.model.HarmonicType.None;
    note.harmonicValue = 0;
  } else if (technique === "palm_mute") note.isPalmMute = false;
  else if (technique === "let_ring") note.isLetRing = false;
  else if (technique === "dead_note") note.isDead = false;
}

function uniqueNotes(notes: alphaTab.model.Note[]) {
  return [...new Map(notes.map((note) => [note.id, note])).values()];
}

function uniqueBeats(notes: alphaTab.model.Note[]) {
  return [...new Map(notes.map((note) => [note.beat.id, note.beat])).values()];
}

function beatDurationEighths(beat: alphaTab.model.Beat) {
  const base = beatBaseDurationEighths(beat);
  if (beat.dots === 1) return base * 1.5;
  if (beat.dots >= 2) return base * 1.75;
  return base;
}

function beatBaseDurationEighths(beat: alphaTab.model.Beat) {
  return beat.duration > 0 ? 8 / beat.duration : 8;
}

function alphaDuration(eighths: number) {
  if (eighths === 8) return alphaTab.model.Duration.Whole;
  if (eighths === 4) return alphaTab.model.Duration.Half;
  if (eighths === 2) return alphaTab.model.Duration.Quarter;
  if (eighths === 1) return alphaTab.model.Duration.Eighth;
  return alphaTab.model.Duration.Sixteenth;
}

function directDurationLabel(eighths: number) {
  return ({ 8: "1/1", 4: "1/2", 2: "1/4", 1: "1/8", 0.5: "1/16" } as Record<number, string>)[eighths] ?? `${eighths}/8`;
}

const REST_DURATION_SHAPES = [
  { units: 8, duration: alphaTab.model.Duration.Whole, dots: 0 },
  { units: 7, duration: alphaTab.model.Duration.Half, dots: 2 },
  { units: 6, duration: alphaTab.model.Duration.Half, dots: 1 },
  { units: 4, duration: alphaTab.model.Duration.Half, dots: 0 },
  { units: 3.5, duration: alphaTab.model.Duration.Quarter, dots: 2 },
  { units: 3, duration: alphaTab.model.Duration.Quarter, dots: 1 },
  { units: 2, duration: alphaTab.model.Duration.Quarter, dots: 0 },
  { units: 1.75, duration: alphaTab.model.Duration.Eighth, dots: 2 },
  { units: 1.5, duration: alphaTab.model.Duration.Eighth, dots: 1 },
  { units: 1, duration: alphaTab.model.Duration.Eighth, dots: 0 },
  { units: 0.875, duration: alphaTab.model.Duration.Sixteenth, dots: 2 },
  { units: 0.75, duration: alphaTab.model.Duration.Sixteenth, dots: 1 },
  { units: 0.5, duration: alphaTab.model.Duration.Sixteenth, dots: 0 },
  { units: 0.375, duration: alphaTab.model.Duration.ThirtySecond, dots: 1 },
  { units: 0.25, duration: alphaTab.model.Duration.ThirtySecond, dots: 0 },
  { units: 0.125, duration: alphaTab.model.Duration.SixtyFourth, dots: 0 }
] as const;

function restShapes(units: number) {
  const shapes: Array<(typeof REST_DURATION_SHAPES)[number]> = [];
  let remaining = Math.round(units * 8) / 8;
  for (const shape of REST_DURATION_SHAPES) {
    while (remaining + 1e-9 >= shape.units) {
      shapes.push(shape);
      remaining = Math.round((remaining - shape.units) * 8) / 8;
    }
  }
  return shapes;
}

function noteMeasure(note: alphaTab.model.Note, recognition: RecognitionDiagnostics | null) {
  return (recognition?.summary.start_measure ?? 1) + note.beat.voice.bar.masterBar.index;
}

function cloneRecognitionMeasures(recognition: RecognitionDiagnostics | null) {
  return recognition ? recognition.measures.map((measure) => ({
    ...measure,
    events: measure.events.map((event) => ({ ...event, notes: event.notes.map((note) => ({ ...note })) }))
  })) : null;
}

function cloneMeasureDraft(measures: RecognitionMeasure[] | null) {
  return measures?.map((measure) => ({
    ...measure,
    events: measure.events.map((event) => ({ ...event, notes: event.notes.map((note) => ({ ...note })) }))
  })) ?? null;
}

function captureNoteState(note: alphaTab.model.Note): NoteState {
  return {
    note,
    beat: note.beat,
    present: note.beat.notes.includes(note),
    fret: note.fret,
    string: note.string,
    bendType: note.bendType,
    bendPoints: note.bendPoints?.map((point) => ({ offset: point.offset, value: point.value })) ?? null,
    isHammerPullOrigin: note.isHammerPullOrigin,
    hammerPullOrigin: note.hammerPullOrigin,
    hammerPullDestination: note.hammerPullDestination,
    isSlurDestination: note.isSlurDestination,
    slurOrigin: note.slurOrigin,
    slurDestination: note.slurDestination,
    harmonicType: note.harmonicType,
    harmonicValue: note.harmonicValue,
    isLetRing: note.isLetRing,
    isPalmMute: note.isPalmMute,
    isDead: note.isDead,
    slideOutType: note.slideOutType,
    slideTarget: note.slideTarget,
    slideOrigin: note.slideOrigin,
    isTieDestination: note.isTieDestination,
    tieOrigin: note.tieOrigin,
    tieDestination: note.tieDestination,
    vibrato: note.vibrato
  };
}

function captureBeatState(beat: alphaTab.model.Beat): BeatState {
  return {
    beat,
    voice: beat.voice,
    index: beat.voice.beats.indexOf(beat),
    present: beat.voice.beats.includes(beat),
    duration: beat.duration,
    dots: beat.dots
  };
}

function restoreNoteState(state: NoteState) {
  const { note, beat } = state;
  if (state.present && !beat.notes.includes(note)) beat.addNote(note);
  if (!state.present && beat.notes.includes(note)) beat.removeNote(note);
  note.fret = state.fret;
  note.string = state.string;
  note.bendType = state.bendType;
  note.bendPoints = null;
  note.maxBendPoint = null;
  for (const point of state.bendPoints ?? []) note.addBendPoint(new alphaTab.model.BendPoint(point.offset, point.value));
  note.isHammerPullOrigin = state.isHammerPullOrigin;
  note.hammerPullOrigin = state.hammerPullOrigin;
  note.hammerPullDestination = state.hammerPullDestination;
  note.isSlurDestination = state.isSlurDestination;
  note.slurOrigin = state.slurOrigin;
  note.slurDestination = state.slurDestination;
  note.harmonicType = state.harmonicType;
  note.harmonicValue = state.harmonicValue;
  note.isLetRing = state.isLetRing;
  note.isPalmMute = state.isPalmMute;
  note.isDead = state.isDead;
  note.slideOutType = state.slideOutType;
  note.slideTarget = state.slideTarget;
  note.slideOrigin = state.slideOrigin;
  note.isTieDestination = state.isTieDestination;
  note.tieOrigin = state.tieOrigin;
  note.tieDestination = state.tieDestination;
  note.vibrato = state.vibrato;
}

function restoreBeatState(state: BeatState) {
  const currentIndex = state.voice.beats.indexOf(state.beat);
  if (state.present && currentIndex < 0) {
    state.voice.beats.splice(Math.min(state.index, state.voice.beats.length), 0, state.beat);
    state.beat.voice = state.voice;
  } else if (!state.present && currentIndex >= 0) state.voice.beats.splice(currentIndex, 1);
  state.beat.duration = state.duration;
  state.beat.dots = state.dots;
}

function formatTime(seconds: number) {
  const safe = Math.max(0, seconds);
  return `${String(Math.floor(safe / 60)).padStart(2, "0")}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

function firstBeatForMeasure(score: alphaTab.model.Score, measureIndex: number) {
  for (const track of score.tracks) for (const staff of track.staves) {
    const bar = staff.bars[measureIndex];
    if (!bar) continue;
    for (const voice of bar.voices) if (voice.beats.length) return voice.beats[0];
  }
  return null;
}

function buildVideoSyncAnchors(score: alphaTab.model.Score, syncPoints: SyncPoint[], recognition: RecognitionDiagnostics | null) {
  const firstMeasure = recognition?.summary.start_measure ?? 1;
  const candidates = syncPoints.map((point) => {
    const beat = firstBeatForMeasure(score, point.measure_number - firstMeasure);
    return beat ? { tick: beat.absolutePlaybackStart, timeSeconds: point.time_seconds } : null;
  }).filter((anchor): anchor is VideoSyncAnchor => Boolean(anchor)).sort((left, right) => left.tick - right.tick);
  const anchors: VideoSyncAnchor[] = [];
  for (const candidate of candidates) {
    const previous = anchors.at(-1);
    if (previous && (candidate.tick <= previous.tick || candidate.timeSeconds <= previous.timeSeconds)) continue;
    anchors.push(candidate);
  }
  return anchors;
}

function syncSegment(anchors: VideoSyncAnchor[], tick: number) {
  let left = anchors[0];
  let right = anchors[1];
  for (let index = 0; index < anchors.length - 1; index += 1) {
    left = anchors[index];
    right = anchors[index + 1];
    if (tick <= right.tick) break;
  }
  return { left, right };
}

function videoTimeAtTick(anchors: VideoSyncAnchor[], tick: number) {
  if (anchors.length < 2) return null;
  const { left, right } = syncSegment(anchors, tick);
  const progress = (tick - left.tick) / Math.max(1, right.tick - left.tick);
  return Math.max(0, left.timeSeconds + progress * (right.timeSeconds - left.timeSeconds));
}

function videoRateAtTick(anchors: VideoSyncAnchor[], tick: number, endTick: number, endTime: number) {
  if (anchors.length < 2 || endTick <= 0 || endTime <= 0) return 1;
  const { left, right } = syncSegment(anchors, tick);
  const scoreSeconds = (right.tick - left.tick) * (endTime / 1000) / endTick;
  const videoSeconds = right.timeSeconds - left.timeSeconds;
  return scoreSeconds > 0 && videoSeconds > 0 ? Math.min(2, Math.max(0.5, videoSeconds / scoreSeconds)) : 1;
}

function isNativeControlTarget(target: EventTarget | null) {
  return Boolean((target as HTMLElement | null)?.closest("input, textarea, select, button, a, summary, [contenteditable='true'], audio, video"));
}

function isTypingTarget(target: EventTarget | null) {
  return isNativeControlTarget(target) || Boolean((target as HTMLElement | null)?.closest("[data-shortcut-scope]"));
}

export function AlphaTabPlayer({
  scoreUrl,
  scrollElement,
  masterVolume,
  fileBaseName,
  pdfUrl,
  videoUrl,
  syncPoints,
  videoFrames,
  recognition,
  editingDisabled,
  onFocusMeasure,
  onDirtyChange,
  onSaveScore,
  onSaveRecognition,
  onRetryRecognition,
  onAppendMeasure
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const apiRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const engineRef = useRef<FluidSynthEngine | null>(null);
  const scoreRef = useRef<alphaTab.model.Score | null>(null);
  const midiRef = useRef<alphaTab.midi.MidiFile | null>(null);
  const selectedNotesRef = useRef<alphaTab.model.Note[]>([]);
  const selectionAnchorRef = useRef<alphaTab.model.Note | null>(null);
  const scoreCursorRef = useRef<ScoreCursor | null>(null);
  const entryDurationRef = useRef<number>(1);
  const dragBaseSelectionRef = useRef<alphaTab.model.Note[]>([]);
  const pointerModifiersRef = useRef({ additive: false, range: false });
  const referenceModeRef = useRef<ReferenceMode | null>(null);
  const videoSyncEnabledRef = useRef(false);
  const videoSyncAnchorsRef = useRef<VideoSyncAnchor[]>([]);
  const recognitionRef = useRef(recognition);
  const syncPointsRef = useRef(syncPoints);
  const playingRef = useRef(false);
  const recognitionDraftRef = useRef<RecognitionMeasure[] | null>(cloneRecognitionMeasures(recognition));
  const dirtyRecognitionMeasuresRef = useRef(new Set<number>());
  const digitBufferRef = useRef<{ value: string; time: number } | null>(null);
  const digitHistoryEntryRef = useRef<HistoryEntry | null>(null);
  const historyBaseDirtyRef = useRef(false);
  const midiRebuildTimerRef = useRef<number | null>(null);
  const lastPlaybackPositionRef = useRef<PlaybackPosition>({ tick: 0, endTick: 0, endTime: 0 });
  const videoSyncInitializedRef = useRef(false);
  const commandReturnFocusRef = useRef<HTMLElement | null>(null);
  const editingDisabledRef = useRef(editingDisabled);
  const savingRef = useRef(false);
  const onFocusMeasureRef = useRef(onFocusMeasure);
  const preservedScrollTopRef = useRef(0);
  const restoringScrollRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [scoreReady, setScoreReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState("正在载入乐谱…");
  const [engineMode, setEngineMode] = useState<FluidSynthMode | null>(null);
  const [exporting, setExporting] = useState<ExportKind | null>(null);
  const [exportStatus, setExportStatus] = useState("");
  const [editStatus, setEditStatus] = useState("单击音符开始编辑");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [selectionMarkers, setSelectionMarkers] = useState<SelectionMarker[]>([]);
  const [scoreCursor, setScoreCursor] = useState<ScoreCursor | null>(null);
  const [measureMarkers, setMeasureMarkers] = useState<MeasureMarker[]>([]);
  const [entryDuration, setEntryDurationState] = useState(1);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [focusedMeasure, setFocusedMeasure] = useState(recognition?.summary.start_measure ?? 1);
  const [referenceMode, setReferenceMode] = useState<ReferenceMode | null>(null);
  const [videoSyncEnabled, setVideoSyncEnabled] = useState(syncPoints.length >= 2);
  const [validSyncAnchorCount, setValidSyncAnchorCount] = useState(0);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [commandQuery, setCommandQuery] = useState("");
  const [shortcutHelp, setShortcutHelp] = useState(false);
  const [barsPerRow, setBarsPerRow] = useState<3 | 4>(() => localStorage.getItem("nocturne-bars-per-row") === "3" ? 3 : 4);
  const syncAvailable = validSyncAnchorCount >= 2;

  useEffect(() => {
    recognitionRef.current = recognition;
    if (dirty) return;
    recognitionDraftRef.current = cloneRecognitionMeasures(recognition);
    dirtyRecognitionMeasuresRef.current.clear();
  }, [dirty, recognition]);
  useEffect(() => { referenceModeRef.current = referenceMode; }, [referenceMode]);
  useEffect(() => { editingDisabledRef.current = editingDisabled; }, [editingDisabled]);
  useEffect(() => {
    if (!scrollElement) return;
    preservedScrollTopRef.current = scrollElement.scrollTop;
    const rememberScroll = () => {
      if (!restoringScrollRef.current) preservedScrollTopRef.current = scrollElement.scrollTop;
    };
    scrollElement.addEventListener("scroll", rememberScroll, { passive: true });
    return () => scrollElement.removeEventListener("scroll", rememberScroll);
  }, [scrollElement]);
  useEffect(() => { onFocusMeasureRef.current = onFocusMeasure; }, [onFocusMeasure]);
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);
  useEffect(() => { videoSyncEnabledRef.current = videoSyncEnabled; }, [videoSyncEnabled]);
  useEffect(() => { syncPointsRef.current = syncPoints; }, [syncPoints]);
  useEffect(() => { if (videoSyncInitializedRef.current && !syncAvailable) setVideoSyncEnabled(false); }, [syncAvailable]);
  useEffect(() => {
    if (!editingDisabled) return;
    setCommandPaletteOpen(false);
    setSelection([]);
    clearScoreCursor();
    setEditStatus("当前谱面暂时只读；仍可播放和导出");
  }, [editingDisabled]);

  const referenceImage = useMemo(() => {
    if (!videoFrames.length) return null;
    const targetTime = recognition?.measures.find((measure) => measure.number === focusedMeasure)?.source_time;
    const frame = typeof targetTime === "number"
      ? [...videoFrames].sort((left, right) => Math.abs(left.time_seconds - targetTime) - Math.abs(right.time_seconds - targetTime))[0]
      : videoFrames[0];
    return frame ? {
      url: frame.url,
      label: `第 ${focusedMeasure} 小节 · 源帧 ${frame.source_frame}`,
      timeLabel: `${formatTime(frame.time_seconds)} · ${frame.time_seconds.toFixed(3)}s`
    } : null;
  }, [focusedMeasure, recognition, videoFrames]);

  const filteredCommands = useMemo(() => {
    const query = commandQuery.trim().toLowerCase();
    return EDIT_COMMANDS.filter((command) => !query || command.label.includes(query) || command.id.includes(query) || command.shortcut.toLowerCase() === query);
  }, [commandQuery]);

  function openCommandPalette() {
    if (editingDisabledRef.current || savingRef.current) return;
    commandReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setCommandQuery("");
    setCommandPaletteOpen(true);
  }

  function closeCommandPalette() {
    setCommandPaletteOpen(false);
    window.requestAnimationFrame(() => commandReturnFocusRef.current?.focus({ preventScroll: true }));
  }

  function trapCommandPaletteFocus(event: React.KeyboardEvent<HTMLElement>) {
    if (event.key !== "Tab") return;
    const focusable = [...event.currentTarget.querySelectorAll<HTMLElement>("input, button:not(:disabled), [href], [tabindex]:not([tabindex='-1'])")];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1)!;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function updateSelectionMarkers() {
    const api = apiRef.current;
    const host = hostRef.current;
    const lookup = api?.renderer.boundsLookup;
    if (!api || !host || !lookup) return setSelectionMarkers([]);
    const markers: SelectionMarker[] = [];
    for (const note of selectedNotesRef.current) {
      const bounds = lookup.findBeat(note.beat)?.notes?.find((candidate) => candidate.note.id === note.id)?.noteHeadBounds;
      if (!bounds) continue;
      markers.push({ id: note.id, x: host.offsetLeft + bounds.x - 6, y: host.offsetTop + bounds.y - 5, width: Math.max(18, bounds.w + 12), height: Math.max(18, bounds.h + 10) });
    }
    setSelectionMarkers(markers);

    const draft = recognitionDraftRef.current;
    const firstMeasure = recognitionRef.current?.summary.start_measure ?? 1;
    if (!draft) setMeasureMarkers([]);
    else setMeasureMarkers(draft.flatMap((measure) => {
      const masterBounds = lookup.findMasterBarByIndex(measure.number - firstMeasure);
      if (!masterBounds) return [];
      const capacity = (measure.time_signature?.numerator ?? 4) * 8 / (measure.time_signature?.denominator ?? 4);
      const events = [...measure.events].sort((left, right) => left.onset_eighths - right.onset_eighths);
      let cursor = 0;
      let invalid = false;
      for (const event of events) {
        if (event.onset_eighths > cursor + 1e-9 || event.onset_eighths < cursor - 1e-9 || event.duration_eighths <= 0) invalid = true;
        cursor = Math.max(cursor, event.onset_eighths + event.duration_eighths);
      }
      invalid ||= Math.abs(cursor - capacity) > 1e-9;
      if (!invalid) return [];
      const bounds = masterBounds.lineAlignedBounds ?? masterBounds.realBounds;
      return [{
        id: measure.number,
        measure: measure.number,
        x: host.offsetLeft + bounds.x,
        y: host.offsetTop + bounds.y,
        width: bounds.w,
        height: bounds.h
      }];
    }));
  }

  function clearScoreCursor() {
    scoreCursorRef.current = null;
    setScoreCursor(null);
  }

  function setEntryDuration(value: number) {
    entryDurationRef.current = value;
    setEntryDurationState(value);
    if (scoreCursorRef.current) setEditStatus(`新音时值 ${directDurationLabel(value)} · 输入 0–9 写入当前弦`);
  }

  function selectScoreCellFromPointer(event: PointerEvent) {
    if (editingDisabledRef.current || savingRef.current) return;
    const api = apiRef.current;
    const host = hostRef.current;
    const lookup = api?.renderer.boundsLookup;
    if (!api || !host || !lookup) return;
    const hostBounds = host.getBoundingClientRect();
    const x = event.clientX - hostBounds.left;
    const y = event.clientY - hostBounds.top;
    const beat = lookup.getBeatAtPos(x, y);
    if (!beat || lookup.getNoteAtPos(beat, x, y)) return;
    const beatBounds = lookup.findBeat(beat);
    if (!beatBounds) return;
    const barBounds = beatBounds.barBounds;
    const stringCount = barBounds.bar.staff.tuning.length || 6;
    if (!stringCount) return;
    const staffBounds = barBounds.visualBounds.h > 0 ? barBounds.visualBounds : barBounds.realBounds;
    const relativeY = Math.max(0, Math.min(0.999, (y - staffBounds.y) / Math.max(1, staffBounds.h)));
    const visualString = Math.max(1, Math.min(stringCount, Math.floor(relativeY * stringCount) + 1));
    const alphaString = stringCount + 1 - visualString;
    const beatStart = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
    const duration = beatDurationEighths(beat);
    const relativeX = beat.isRest
      ? Math.max(0, Math.min(0.999, (x - beatBounds.realBounds.x) / Math.max(1, beatBounds.realBounds.w)))
      : 0;
    const onset = Math.round((beatStart + Math.floor(relativeX * duration / 0.5) * 0.5) * 2) / 2;
    const positionRatio = duration > 0 ? (onset - beatStart) / duration : 0;
    const marker: SelectionMarker = {
      id: -1,
      x: host.offsetLeft + beatBounds.realBounds.x + beatBounds.realBounds.w * positionRatio - 9,
      y: host.offsetTop + staffBounds.y + staffBounds.h * ((visualString - 0.5) / stringCount) - 9,
      width: 18,
      height: 18
    };
    const cursor = { beat, onset, string: alphaString, marker };
    scoreCursorRef.current = cursor;
    setScoreCursor(cursor);
    setSelection([]);
    const measure = (recognitionRef.current?.summary.start_measure ?? 1) + beat.voice.bar.masterBar.index;
    setFocusedMeasure(measure);
    onFocusMeasureRef.current(measure);
    setEditStatus(`第 ${measure} 小节 · ${visualString} 弦 · 空拍 ${directDurationLabel(entryDurationRef.current)} · 输入品位`);
  }

  function setSelection(notes: alphaTab.model.Note[], anchor = notes.at(-1) ?? null) {
    const next = uniqueNotes(notes);
    selectedNotesRef.current = next;
    selectionAnchorRef.current = anchor;
    setSelectedIds(next.map((note) => note.id));
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    if (anchor) {
      const measure = noteMeasure(anchor, recognitionRef.current);
      setFocusedMeasure(measure);
      onFocusMeasureRef.current(measure);
    }
    setEditStatus(next.length ? `已选 ${next.length} 个音 · 可直接输入品位或技巧快捷键` : "选择已清空");
    window.requestAnimationFrame(updateSelectionMarkers);
  }

  function noteRange(start: alphaTab.model.Note, end: alphaTab.model.Note) {
    const score = scoreRef.current;
    if (!score) return [end];
    if (start.beat.voice.index !== end.beat.voice.index || start.beat.voice.bar.staff !== end.beat.voice.bar.staff) return [end];
    const notes = flattenNotes(score).filter((note) =>
      note.beat.voice.index === start.beat.voice.index
      && note.beat.voice.bar.staff === start.beat.voice.bar.staff
    );
    const startIndex = notes.findIndex((note) => note.id === start.id);
    const endIndex = notes.findIndex((note) => note.id === end.id);
    return startIndex < 0 || endIndex < 0 ? [end] : notes.slice(Math.min(startIndex, endIndex), Math.max(startIndex, endIndex) + 1);
  }

  function handleNotePointerDown(note: alphaTab.model.Note) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    clearScoreCursor();
    const current = selectedNotesRef.current;
    if (pointerModifiersRef.current.range && selectionAnchorRef.current) {
      setSelection(noteRange(selectionAnchorRef.current, note), selectionAnchorRef.current);
    } else if (pointerModifiersRef.current.additive) {
      const exists = current.some((candidate) => candidate.id === note.id);
      setSelection(exists ? current.filter((candidate) => candidate.id !== note.id) : [...current, note], note);
    } else setSelection([note], note);
    dragBaseSelectionRef.current = pointerModifiersRef.current.additive ? [...selectedNotesRef.current] : [];
  }

  function handleNotePointerMove(note: alphaTab.model.Note) {
    if (editingDisabledRef.current || savingRef.current) return;
    const anchor = selectionAnchorRef.current;
    if (anchor) setSelection([...dragBaseSelectionRef.current, ...noteRange(anchor, note)], anchor);
  }

  function setPlaybackState(value: boolean) {
    playingRef.current = value;
    setPlaying(value);
    const video = videoRef.current;
    if (!video || referenceModeRef.current !== "video" || !videoSyncEnabledRef.current) return;
    if (value) void video.play().catch(() => undefined);
    else video.pause();
  }

  useEffect(() => {
    if (!hostRef.current) return;
    const host = hostRef.current;
    let externalOutput: alphaTab.synth.IExternalMediaSynthOutput | null = null;
    let active = true;
    let engineLoadingStarted = false;
    scoreRef.current = null;
    midiRef.current = null;
    selectedNotesRef.current = [];
    setSelectedIds([]);
    setScoreReady(false);
    setReady(false);
    setEngineMode(null);
    setLoading("正在载入乐谱…");
    restoringScrollRef.current = true;
    setDirty(false);
    setHistory([]);
    historyBaseDirtyRef.current = false;
    if (midiRebuildTimerRef.current !== null) window.clearTimeout(midiRebuildTimerRef.current);
    midiRebuildTimerRef.current = null;
    videoSyncInitializedRef.current = false;
    setValidSyncAnchorCount(0);

    const engine = new FluidSynthEngine({
      onPosition: (timeMs) => externalOutput?.updatePosition(timeMs),
      onPlaybackChange: (isPlaying) => active && setPlaybackState(isPlaying),
      onReady: (mode) => { if (active) { setEngineMode(mode); setReady(true); setLoading(""); } },
      onStatus: (message) => active && setLoading(message)
    });
    engine.masterVolume = masterVolume;
    engineRef.current = engine;

    const api = new alphaTab.AlphaTabApi(host, {
      core: { fontDirectory: "/font/", includeNoteBounds: true },
      display: { layoutMode: alphaTab.LayoutMode.Page, staveProfile: alphaTab.StaveProfile.TabMixed, barsPerRow },
      player: { playerMode: alphaTab.PlayerMode.EnabledExternalMedia, ...(scrollElement ? { scrollElement } : {}) }
    });
    apiRef.current = api;
    const capturePointerModifiers = (event: PointerEvent) => {
      host.focus({ preventScroll: true });
      pointerModifiersRef.current = { additive: event.ctrlKey || event.metaKey, range: event.shiftKey };
      selectScoreCellFromPointer(event);
    };
    host.addEventListener("pointerdown", capturePointerModifiers, true);
    const attachEngine = () => {
      const output = api.player?.output as alphaTab.synth.IExternalMediaSynthOutput | undefined;
      if (output && "handler" in output) { externalOutput = output; output.handler = engine; }
    };

    attachEngine();
    api.renderStarted.on(() => !engineLoadingStarted && setLoading("正在排版乐谱…"));
    api.renderFinished.on(() => window.requestAnimationFrame(() => {
      updateSelectionMarkers();
      if (scrollElement && restoringScrollRef.current) {
        scrollElement.scrollTop = preservedScrollTopRef.current;
        window.requestAnimationFrame(() => { restoringScrollRef.current = false; });
      }
    }));
    api.renderFinished.on(() => !engineLoadingStarted && setLoading("正在生成演奏数据…"));
    api.playerReady.on(attachEngine);
    api.scoreLoaded.on((score) => {
      try {
        scoreRef.current = score;
        const anchors = buildVideoSyncAnchors(score, syncPointsRef.current, recognitionRef.current);
        videoSyncAnchorsRef.current = anchors;
        setValidSyncAnchorCount(anchors.length);
        if (!videoSyncInitializedRef.current) {
          videoSyncInitializedRef.current = true;
          setVideoSyncEnabled(anchors.length >= 2);
        }
        setScoreReady(true);
        engineLoadingStarted = true;
        const midi = createMidi(score, api.settings);
        midiRef.current = midi;
        void engine.loadMidi(midi).catch((error: unknown) => {
          if (!active) return;
          setReady(false);
          setEngineMode(null);
          setLoading(`音色引擎载入失败：${error instanceof Error ? error.message : "未知错误"}`);
        });
      } catch (error) {
        if (!active) return;
        setReady(false);
        setEngineMode(null);
        setLoading(`演奏数据生成失败：${error instanceof Error ? error.message : "未知错误"}`);
      }
    });
    api.noteMouseDown.on(handleNotePointerDown);
    api.noteMouseMove.on(handleNotePointerMove);
    api.activeBeatsChanged.on((event) => {
      const beat = event.activeBeats[0];
      if (!beat) return;
      const measure = (recognitionRef.current?.summary.start_measure ?? 1) + beat.voice.bar.masterBar.index;
      setFocusedMeasure(measure);
      onFocusMeasureRef.current(measure);
    });
    api.playerPositionChanged.on((event) => {
      lastPlaybackPositionRef.current = { tick: event.currentTick, endTick: event.endTick, endTime: event.endTime };
      const video = videoRef.current;
      if (!video || !videoSyncEnabledRef.current || referenceModeRef.current !== "video") return;
      const target = videoTimeAtTick(videoSyncAnchorsRef.current, event.currentTick);
      if (target === null) return;
      video.playbackRate = videoRateAtTick(videoSyncAnchorsRef.current, event.currentTick, event.endTick, event.endTime);
      if (event.isSeek || !playingRef.current || Math.abs(video.currentTime - target) > 0.32) video.currentTime = target;
    });
    api.playerStateChanged.on((event) => setPlaybackState(event.state === alphaTab.synth.PlayerState.Playing));
    api.error.on((error) => { if (active) { setReady(false); setLoading(`乐谱载入失败：${error.message}`); } });
    api.load(scoreUrl);
    const resize = () => window.requestAnimationFrame(updateSelectionMarkers);
    window.addEventListener("resize", resize);
    return () => {
      active = false;
      scoreRef.current = null;
      midiRef.current = null;
      host.removeEventListener("pointerdown", capturePointerModifiers, true);
      window.removeEventListener("resize", resize);
      if (midiRebuildTimerRef.current !== null) window.clearTimeout(midiRebuildTimerRef.current);
      midiRebuildTimerRef.current = null;
      if (externalOutput) externalOutput.handler = undefined;
      engine.destroy();
      if (engineRef.current === engine) engineRef.current = null;
      api.destroy();
      apiRef.current = null;
    };
  }, [scoreUrl, scrollElement]);

  useEffect(() => {
    localStorage.setItem("nocturne-bars-per-row", String(barsPerRow));
    const api = apiRef.current;
    if (!api || api.settings.display.barsPerRow === barsPerRow) return;
    api.settings.display.barsPerRow = barsPerRow;
    api.updateSettings();
    setEditStatus(`已固定为每行 ${barsPerRow} 小节`);
  }, [barsPerRow]);

  useEffect(() => { if (engineRef.current) engineRef.current.masterVolume = masterVolume; }, [masterVolume]);
  useEffect(() => {
    const score = scoreRef.current;
    if (score) {
      const anchors = buildVideoSyncAnchors(score, syncPoints, recognition);
      videoSyncAnchorsRef.current = anchors;
      setValidSyncAnchorCount(anchors.length);
      if (!videoSyncInitializedRef.current) {
        videoSyncInitializedRef.current = true;
        setVideoSyncEnabled(anchors.length >= 2);
      }
    }
  }, [recognition, syncPoints]);

  function historyEntry(label: string, notes = selectedNotesRef.current): HistoryEntry {
    const related = relatedNotes(uniqueNotes(notes));
    return {
      label,
      states: related.map(captureNoteState),
      beatStates: uniqueBeats(related).map(captureBeatState),
      selection: [...selectedNotesRef.current],
      recognition: cloneMeasureDraft(recognitionDraftRef.current),
      dirtyRecognitionMeasures: [...dirtyRecognitionMeasuresRef.current]
    };
  }

  function followingRestPlan(beat: alphaTab.model.Beat, delta: number): RestPlan {
    const voice = beat.voice;
    const start = voice.beats.indexOf(beat) + 1;
    let end = start;
    while (end < voice.beats.length && voice.beats[end].isRest) end += 1;
    const beats = voice.beats.slice(start, end);
    const explicitDuration = beats.reduce((sum, candidate) => sum + beatDurationEighths(candidate), 0);
    const masterBar = voice.bar.masterBar;
    const capacity = masterBar.timeSignatureNumerator * 8 / masterBar.timeSignatureDenominator;
    const voiceDuration = voice.beats.reduce((sum, candidate) => sum + beatDurationEighths(candidate), 0);
    const implicitDuration = end === voice.beats.length ? Math.max(0, capacity - voiceDuration) : 0;
    return { beats, delta, explicitDuration, availableDuration: explicitDuration + implicitDuration, insertIndex: start, voice };
  }

  function rememberBeat(entry: HistoryEntry, beat: alphaTab.model.Beat) {
    if (!entry.beatStates.some((state) => state.beat === beat)) entry.beatStates.push(captureBeatState(beat));
  }

  function applyRestPlan(plan: RestPlan, entry: HistoryEntry) {
    const implicitDuration = Math.max(0, plan.availableDuration - plan.explicitDuration);
    const explicitDelta = Math.max(0, plan.delta - implicitDuration);
    if (explicitDelta <= 1e-9) return;
    const targetDuration = Math.max(0, plan.explicitDuration - Math.min(explicitDelta, plan.explicitDuration));
    const shapes = restShapes(targetDuration);
    for (const beat of plan.beats) rememberBeat(entry, beat);
    for (let index = 0; index < shapes.length; index += 1) {
      const shape = shapes[index];
      let beat = plan.beats[index];
      if (!beat) {
        beat = new alphaTab.model.Beat();
        beat.voice = plan.voice;
        entry.beatStates.push({
          beat,
          voice: plan.voice,
          index: plan.insertIndex + index,
          present: false,
          duration: beat.duration,
          dots: beat.dots
        });
        plan.voice.beats.splice(plan.insertIndex + index, 0, beat);
      }
      beat.duration = shape.duration;
      beat.dots = shape.dots;
    }
    for (const beat of plan.beats.slice(shapes.length)) {
      const index = plan.voice.beats.indexOf(beat);
      if (index >= 0) plan.voice.beats.splice(index, 1);
    }
  }

  function recognitionString(note: alphaTab.model.Note, alphaTabString = note.string) {
    const stringCount = note.beat.voice.bar.staff.tuning.length || 6;
    return stringCount + 1 - alphaTabString;
  }

  function recognitionNote(note: alphaTab.model.Note, originalString = note.string) {
    const measures = recognitionDraftRef.current;
    if (!measures || !recognition) return null;
    const measure = measures.find((candidate) => candidate.number === noteMeasure(note, recognition));
    const onset = Math.round(note.beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
    const event = measure?.events.find((candidate) => candidate.onset_eighths === onset);
    const target = event?.notes.find((candidate) => candidate.string === recognitionString(note, originalString));
    return measure && event && target ? { measure, event, note: target } : null;
  }

  function markRecognitionChanged(note: alphaTab.model.Note, originalString = note.string) {
    if (!recognition) return;
    dirtyRecognitionMeasuresRef.current.add(noteMeasure(note, recognition));
    const target = recognitionNote(note, originalString);
    if (target) dirtyRecognitionMeasuresRef.current.add(target.measure.number);
  }

  function rebuildAfterEdit(label: string, changedNotes: alphaTab.model.Note[], markDirty = true) {
    const api = apiRef.current;
    const score = scoreRef.current;
    const engine = engineRef.current;
    if (!api || !score || !engine) return;
    const firstChangedMasterBar = Math.min(...changedNotes.map((note) => note.beat.voice.bar.masterBar.index));
    prepareScoreForFinish(score);
    score.finish(api.settings);
    api.render({ reuseViewport: true, firstChangedMasterBar: Number.isFinite(firstChangedMasterBar) ? firstChangedMasterBar : 0 });
    midiRef.current = null;
    setReady(false);
    setLoading("正在合并编辑并重建演奏数据…");
    if (midiRebuildTimerRef.current !== null) window.clearTimeout(midiRebuildTimerRef.current);
    midiRebuildTimerRef.current = window.setTimeout(() => {
      midiRebuildTimerRef.current = null;
      try {
        api.loadMidiForScore();
        const midi = createMidi(score, api.settings);
        midiRef.current = midi;
        void engine.loadMidi(midi).catch((error: unknown) => {
          setReady(false);
          setLoading(`演奏数据重建失败：${error instanceof Error ? error.message : "未知错误"}`);
        });
      } catch (error) {
        setReady(false);
        setLoading(`演奏数据重建失败：${error instanceof Error ? error.message : "未知错误"}`);
      }
    }, 180);
    if (markDirty) setDirty(true);
    setEditStatus(`${label} · ${markDirty ? "Ctrl/⌘S 保存" : "已恢复"}`);
    window.requestAnimationFrame(updateSelectionMarkers);
  }

  function commitEdit(label: string, entry: HistoryEntry, changedNotes = selectedNotesRef.current) {
    setHistory((current) => {
      if (current.length >= 50) historyBaseDirtyRef.current = true;
      return [...current, entry].slice(-50);
    });
    rebuildAfterEdit(`${label}已应用`, changedNotes);
  }

  function notesForCommand(command: EditCommand) {
    const selected = [...selectedNotesRef.current].sort((left, right) => left.beat.absolutePlaybackStart - right.beat.absolutePlaybackStart);
    if (!command.requiresPair) return selected;
    if (selected.length !== 2 || selected[0].string !== selected[1].string) {
      setEditStatus(`${command.label}需要选择同一弦上的两个音`);
      return [];
    }
    const [origin, destination] = selected;
    if (origin.beat.voice.index !== destination.beat.voice.index || origin.beat.voice.bar.staff !== destination.beat.voice.bar.staff || origin.beat.absolutePlaybackStart >= destination.beat.absolutePlaybackStart) {
      setEditStatus(`${command.label}只能连接同一声部中先后出现的两个音`);
      return [];
    }
    const expectedDestination = command.id === "hammer_on" || command.id === "pull_off"
      ? alphaTab.model.Note.findHammerPullDestination(origin)
      : alphaTab.model.Note.nextNoteOnSameLine(origin);
    if (expectedDestination !== destination) {
      setEditStatus(`${command.label}只能连接同一弦上的下一个音`);
      return [];
    }
    if (command.id === "hammer_on" && destination.fret <= origin.fret) {
      setEditStatus("击弦目标品位必须高于起点；下降音请使用勾弦");
      return [];
    }
    if (command.id === "pull_off" && destination.fret >= origin.fret) {
      setEditStatus("勾弦目标品位必须低于起点；上升音请使用击弦");
      return [];
    }
    if (command.id === "legato" && ((origin.slurDestination && origin.slurDestination !== destination) || (destination.slurOrigin && destination.slurOrigin !== origin))) {
      setEditStatus("其中一个音已经连接到其他连音，请先关闭原连接");
      return [];
    }
    if (command.id === "slide" && ((origin.slideTarget && origin.slideTarget !== destination) || (destination.slideOrigin && destination.slideOrigin !== origin))) {
      setEditStatus("其中一个音已经连接到其他滑音，请先关闭原连接");
      return [];
    }
    if ((command.id === "hammer_on" || command.id === "pull_off") && ((origin.hammerPullDestination && origin.hammerPullDestination !== destination) || (destination.hammerPullOrigin && destination.hammerPullOrigin !== origin))) {
      setEditStatus("其中一个音已经连接到其他击勾弦，请先关闭原连接");
      return [];
    }
    return selected;
  }

  function applyCommand(command: EditCommand) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    const selected = notesForCommand(command);
    if (!selected.length) { if (!command.requiresPair) setEditStatus("请先在谱面上选择音符"); return; }
    const entry = historyEntry(command.label, selected);
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    const recognitionTargets = (command.requiresPair ? [selected[0]] : selected).map((note) => ({
      note,
      target: recognitionDraftRef.current ? recognitionNote(note) : null
    }));
    if (recognitionDraftRef.current && recognitionTargets.some(({ target }) => !target)) {
      return setEditStatus("这个音符无法安全映射回保存草稿；请在已展开的六线网格中修改");
    }
    if (recognitionDraftRef.current) {
      for (const { note, target } of recognitionTargets) {
        const previousTechnique = target?.note.technique;
        if (previousTechnique && previousTechnique !== command.id) clearTechniqueFromNote(note, previousTechnique);
      }
    }
    let enabled = true;
    if (command.id === "legato") {
      const [origin, destination] = selected;
      enabled = destination.slurOrigin !== origin;
      origin.slurDestination = enabled ? destination : null;
      destination.slurOrigin = enabled ? origin : null;
      destination.isSlurDestination = enabled;
    } else if (command.id === "slide") {
      const [origin, destination] = selected;
      enabled = origin.slideTarget !== destination;
      origin.slideOutType = enabled ? alphaTab.model.SlideOutType.Shift : alphaTab.model.SlideOutType.None;
      origin.slideTarget = enabled ? destination : null;
      destination.slideOrigin = enabled ? origin : null;
    } else if (command.id === "hammer_on" || command.id === "pull_off") {
      const [origin, destination] = selected;
      enabled = !(origin.isHammerPullOrigin && origin.hammerPullDestination === destination);
      clearTechniqueFromNote(origin, command.id);
      if (enabled) {
        origin.isHammerPullOrigin = true;
        origin.hammerPullDestination = destination;
        destination.hammerPullOrigin = origin;
      }
    } else if (command.id === "bend") {
      enabled = !selected.every((note) => note.bendType !== alphaTab.model.BendType.None);
      for (const note of selected) {
        note.bendType = enabled ? alphaTab.model.BendType.Bend : alphaTab.model.BendType.None;
        note.bendPoints = null;
        note.maxBendPoint = null;
        if (enabled) {
          note.addBendPoint(new alphaTab.model.BendPoint(0, 0));
          note.addBendPoint(new alphaTab.model.BendPoint(60, 4));
        }
      }
    } else if (command.id === "vibrato") {
      enabled = !selected.every((note) => note.vibrato !== alphaTab.model.VibratoType.None);
      for (const note of selected) note.vibrato = enabled ? alphaTab.model.VibratoType.Slight : alphaTab.model.VibratoType.None;
    } else if (command.id === "harmonic") {
      enabled = !selected.every((note) => note.harmonicType !== alphaTab.model.HarmonicType.None);
      for (const note of selected) {
        note.harmonicType = enabled ? alphaTab.model.HarmonicType.Natural : alphaTab.model.HarmonicType.None;
        note.harmonicValue = enabled ? harmonicValueForFret(note.fret) : 0;
      }
    } else if (command.id === "palm_mute") {
      enabled = !selected.every((note) => note.isPalmMute);
      for (const note of selected) note.isPalmMute = enabled;
    } else if (command.id === "let_ring") {
      enabled = !selected.every((note) => note.isLetRing);
      for (const note of selected) note.isLetRing = enabled;
    } else if (command.id === "dead_note") {
      enabled = !selected.every((note) => note.isDead);
      for (const note of selected) note.isDead = enabled;
    }
    if (recognitionDraftRef.current) {
      for (const { note, target } of recognitionTargets) {
        if (target) target.note.technique = enabled ? command.id : undefined;
        markRecognitionChanged(note);
      }
    }
    commitEdit(command.label, entry, selected);
    closeCommandPalette();
  }

  function recognitionMeasureForBeat(beat: alphaTab.model.Beat) {
    const measures = recognitionDraftRef.current;
    if (!measures || !recognitionRef.current) return null;
    const number = (recognitionRef.current.summary.start_measure ?? 1) + beat.voice.bar.masterBar.index;
    return measures.find((candidate) => candidate.number === number) ?? null;
  }

  function recognitionEventsWithInsertedNote(
    measure: RecognitionMeasure,
    onset: number,
    duration: number,
    string: number,
    fret: number
  ) {
    const end = onset + duration;
    const exact = measure.events.find((event) => !event.rest && event.onset_eighths === onset);
    if (exact) {
      if (exact.notes.some((note) => note.string === string)) return null;
      return measure.events.map((event) => event === exact
        ? { ...event, notes: [...event.notes, { string, fret }].sort((left, right) => left.string - right.string) }
        : event);
    }
    const events: RecognitionEvent[] = [];
    for (const event of measure.events) {
      const eventEnd = event.onset_eighths + event.duration_eighths;
      const overlaps = event.onset_eighths < end - 1e-9 && eventEnd > onset + 1e-9;
      if (!overlaps) {
        events.push(event);
        continue;
      }
      if (!event.rest) return null;
      const leading = onset - event.onset_eighths;
      const trailing = eventEnd - end;
      if (leading > 1e-9) events.push({ onset_eighths: event.onset_eighths, duration_eighths: leading, notes: [], rest: true });
      if (trailing > 1e-9) events.push({ onset_eighths: end, duration_eighths: trailing, notes: [], rest: true });
    }
    events.push({ onset_eighths: onset, duration_eighths: duration, notes: [{ string, fret }] });
    return events.sort((left, right) => left.onset_eighths - right.onset_eighths);
  }

  function newBeatForEdit(voice: alphaTab.model.Voice, entry: HistoryEntry, index: number, duration: alphaTab.model.Duration, dots = 0) {
    const beat = new alphaTab.model.Beat();
    beat.voice = voice;
    beat.duration = duration;
    beat.dots = dots;
    entry.beatStates.push({ beat, voice, index, present: false, duration: beat.duration, dots: beat.dots });
    return beat;
  }

  function insertFretAtScoreCursor(digit: string) {
    const cursor = scoreCursorRef.current;
    if (!cursor) return false;
    const beat = cursor.beat;
    const measure = recognitionMeasureForBeat(beat);
    if (!measure) {
      setEditStatus("当前空拍没有可保存的识别草稿");
      return true;
    }
    const fret = Number(digit);
    const voice = beat.voice;
    const stringCount = voice.bar.staff.tuning.length || 6;
    const recognitionStringNumber = stringCount + 1 - cursor.string;
    const beatStart = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
    const duration = beat.isRest ? entryDurationRef.current : beatDurationEighths(beat);
    const beatEnd = beatStart + beatDurationEighths(beat);
    if (cursor.onset + duration > beatEnd + 1e-9) {
      setEditStatus(`当前休止拍只剩 ${(beatEnd - cursor.onset).toFixed(1)} 个八分单位，无法写入 ${directDurationLabel(duration)}`);
      return true;
    }
    const nextEvents = recognitionEventsWithInsertedNote(measure, cursor.onset, duration, recognitionStringNumber, fret);
    if (!nextEvents) {
      setEditStatus("这个位置与已有音符重叠，先缩短时值或选择其他拍位");
      return true;
    }
    const entry = historyEntry("添加音符", []);
    let noteBeat = beat;
    if (beat.isRest) {
      const originalIndex = voice.beats.indexOf(beat);
      if (originalIndex < 0) return true;
      rememberBeat(entry, beat);
      voice.beats.splice(originalIndex, 1);
      const leading = Math.max(0, cursor.onset - beatStart);
      const trailing = Math.max(0, beatEnd - cursor.onset - duration);
      const replacements: alphaTab.model.Beat[] = [];
      for (const shape of restShapes(leading)) replacements.push(newBeatForEdit(voice, entry, originalIndex + replacements.length, shape.duration, shape.dots));
      noteBeat = newBeatForEdit(voice, entry, originalIndex + replacements.length, alphaDuration(duration));
      replacements.push(noteBeat);
      for (const shape of restShapes(trailing)) replacements.push(newBeatForEdit(voice, entry, originalIndex + replacements.length, shape.duration, shape.dots));
      voice.beats.splice(originalIndex, 0, ...replacements);
    } else if (beat.getNoteOnString(cursor.string)) {
      setEditStatus("当前弦位已经有音符；可直接选中后输入新品位");
      return true;
    }
    const note = new alphaTab.model.Note();
    note.beat = noteBeat;
    note.string = cursor.string;
    note.fret = fret;
    entry.states.push(captureNoteState(note));
    noteBeat.addNote(note);
    measure.events = nextEvents;
    dirtyRecognitionMeasuresRef.current.add(measure.number);
    clearScoreCursor();
    commitEdit(`已在第 ${measure.number} 小节 ${recognitionStringNumber} 弦写入 ${fret} 品`, entry, [note]);
    setSelection([note], note);
    digitBufferRef.current = { value: digit, time: Date.now() };
    digitHistoryEntryRef.current = entry;
    return true;
  }

  function cursorAt(beat: alphaTab.model.Beat, onset: number, string: number) {
    const api = apiRef.current;
    const host = hostRef.current;
    const bounds = api?.renderer.boundsLookup?.findBeat(beat);
    if (!api || !host || !bounds) return null;
    const staffBounds = bounds.barBounds.visualBounds.h > 0 ? bounds.barBounds.visualBounds : bounds.barBounds.realBounds;
    const stringCount = beat.voice.bar.staff.tuning.length || 6;
    const visualString = stringCount + 1 - string;
    const start = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
    const ratio = Math.max(0, Math.min(1, (onset - start) / Math.max(0.5, beatDurationEighths(beat))));
    return {
      beat,
      onset,
      string,
      marker: {
        id: -1,
        x: host.offsetLeft + bounds.realBounds.x + bounds.realBounds.w * ratio - 9,
        y: host.offsetTop + staffBounds.y + staffBounds.h * ((visualString - 0.5) / stringCount) - 9,
        width: 18,
        height: 18
      }
    } satisfies ScoreCursor;
  }

  function moveScoreCursor(horizontal: number, vertical: number) {
    const current = scoreCursorRef.current;
    if (!current) return;
    const stringCount = current.beat.voice.bar.staff.tuning.length || 6;
    let string = Math.max(1, Math.min(stringCount, current.string - vertical));
    let beat = current.beat;
    let onset = current.onset;
    if (horizontal) {
      const targetOnset = onset + horizontal * entryDurationRef.current;
      const beatStart = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
      const beatEnd = beatStart + beatDurationEighths(beat);
      if (targetOnset >= beatStart && targetOnset < beatEnd) onset = targetOnset;
      else {
        const targetBeat = horizontal > 0 ? beat.nextBeat : beat.previousBeat;
        if (!targetBeat) return setEditStatus("已经到达谱面边界");
        beat = targetBeat;
        const targetStart = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
        onset = horizontal > 0 ? targetStart : Math.max(targetStart, targetStart + beatDurationEighths(beat) - entryDurationRef.current);
      }
    }
    const next = cursorAt(beat, onset, string);
    if (!next) return;
    scoreCursorRef.current = next;
    setScoreCursor(next);
    const visualString = stringCount + 1 - string;
    setEditStatus(`${visualString} 弦 · 空拍 ${directDurationLabel(entryDurationRef.current)} · 输入品位或 Enter 写入休止`);
  }

  function selectCursorString(visualString: number) {
    const current = scoreCursorRef.current;
    if (!current) return;
    const stringCount = current.beat.voice.bar.staff.tuning.length || 6;
    const next = cursorAt(current.beat, current.onset, stringCount + 1 - visualString);
    if (!next) return;
    scoreCursorRef.current = next;
    setScoreCursor(next);
    setEditStatus(`${visualString} 弦 · 空拍 ${directDurationLabel(entryDurationRef.current)} · 输入品位`);
  }

  function commitRestAtScoreCursor() {
    const cursor = scoreCursorRef.current;
    if (!cursor) return false;
    const beat = cursor.beat;
    if (!beat.isRest) return false;
    const measure = recognitionMeasureForBeat(beat);
    if (!measure) return true;
    const duration = entryDurationRef.current;
    const beatStart = Math.round(beat.playbackStart / EIGHTH_NOTE_TICKS * 2) / 2;
    const beatEnd = beatStart + beatDurationEighths(beat);
    const end = cursor.onset + duration;
    if (end > beatEnd + 1e-9) {
      setEditStatus("当前空拍剩余时值不足，先缩短新音时值");
      return true;
    }
    const events: RecognitionEvent[] = [];
    for (const event of measure.events) {
      const eventEnd = event.onset_eighths + event.duration_eighths;
      const overlaps = event.onset_eighths < end - 1e-9 && eventEnd > cursor.onset + 1e-9;
      if (!overlaps) { events.push(event); continue; }
      if (!event.rest) {
        setEditStatus("这个位置已经有音符，不能写入休止");
        return true;
      }
      if (event.onset_eighths < cursor.onset) events.push({ onset_eighths: event.onset_eighths, duration_eighths: cursor.onset - event.onset_eighths, notes: [], rest: true });
      if (eventEnd > end) events.push({ onset_eighths: end, duration_eighths: eventEnd - end, notes: [], rest: true });
    }
    events.push({ onset_eighths: cursor.onset, duration_eighths: duration, notes: [], rest: true });
    const entry = historyEntry("写入休止", []);
    const voice = beat.voice;
    const originalIndex = voice.beats.indexOf(beat);
    rememberBeat(entry, beat);
    voice.beats.splice(originalIndex, 1);
    const replacements: alphaTab.model.Beat[] = [];
    for (const shape of restShapes(cursor.onset - beatStart)) replacements.push(newBeatForEdit(voice, entry, originalIndex + replacements.length, shape.duration, shape.dots));
    replacements.push(newBeatForEdit(voice, entry, originalIndex + replacements.length, alphaDuration(duration)));
    for (const shape of restShapes(beatEnd - end)) replacements.push(newBeatForEdit(voice, entry, originalIndex + replacements.length, shape.duration, shape.dots));
    voice.beats.splice(originalIndex, 0, ...replacements);
    measure.events = events.sort((left, right) => left.onset_eighths - right.onset_eighths);
    dirtyRecognitionMeasuresRef.current.add(measure.number);
    clearScoreCursor();
    commitEdit(`第 ${measure.number} 小节休止已实体化`, entry, []);
    return true;
  }

  function writeFretDigit(digit: string) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    if (insertFretAtScoreCursor(digit)) return;
    const selected = selectedNotesRef.current.filter((note) => note.isStringed);
    if (!selected.length) return setEditStatus("请在谱面空拍或已有音符上单击，再输入品位");
    const recognitionTargets = new Map(selected.map((note) => [
      note,
      recognitionDraftRef.current ? recognitionNote(note) : null
    ]));
    if (recognitionDraftRef.current && [...recognitionTargets.values()].some((target) => !target)) {
      return setEditStatus("这个音符无法安全映射回保存草稿；请在已展开的六线网格中输入");
    }
    const now = Date.now();
    const previous = digitBufferRef.current;
    const continuing = Boolean(previous && digitHistoryEntryRef.current && now - previous.time < 900);
    let value = digit;
    if (previous && now - previous.time < 900) {
      const appended = `${previous.value}${digit}`.replace(/^0+(?=\d)/, "");
      value = Number(appended) <= 36 ? appended : digit;
    }
    digitBufferRef.current = { value, time: now };
    const fret = Number(value);
    const entry = continuing && digitHistoryEntryRef.current ? digitHistoryEntryRef.current : historyEntry("品位", selected);
    if (!continuing) digitHistoryEntryRef.current = entry;
    for (const note of selected) {
      const target = recognitionTargets.get(note);
      note.fret = fret;
      if (note.harmonicType !== alphaTab.model.HarmonicType.None) note.harmonicValue = harmonicValueForFret(fret);
      if (target) target.note.fret = fret;
      markRecognitionChanged(note);
    }
    const hammerOrigins = uniqueNotes(selected.flatMap((note) => [note.isHammerPullOrigin ? note : null, note.hammerPullOrigin].filter((candidate): candidate is alphaTab.model.Note => Boolean(candidate))));
    for (const origin of hammerOrigins) {
      const destination = origin.hammerPullDestination;
      const target = recognitionNote(origin);
      if (destination?.fret === origin.fret) {
        clearTechniqueFromNote(origin, "hammer_on");
        if (target && (target.note.technique === "hammer_on" || target.note.technique === "pull_off")) target.note.technique = undefined;
        markRecognitionChanged(origin);
      } else if (destination && target && (target.note.technique === "hammer_on" || target.note.technique === "pull_off")) {
        target.note.technique = destination.fret > origin.fret ? "hammer_on" : "pull_off";
        markRecognitionChanged(origin);
      }
    }
    if (continuing) rebuildAfterEdit(`品位 ${fret}已应用`, selected);
    else commitEdit(`品位 ${fret}`, entry, selected);
  }

  function moveNotesAcrossStrings(direction: number) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    const selected = selectedNotesRef.current.filter((note) => note.isStringed);
    if (!selected.length) return setEditStatus("请先选择六线谱音符");
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    const selectedSet = new Set(selected);
    const proposals = new Map<alphaTab.model.Note, { originalString: number; nextString: number; nextFret: number }>();
    for (const note of selected) {
      if (note.harmonicType !== alphaTab.model.HarmonicType.None) continue;
      const originalString = note.string;
      const nextString = originalString + direction;
      const staff = note.beat.voice.bar.staff;
      if (nextString < 1 || nextString > staff.tuning.length) continue;
      const nextFret = note.calculateRealValue(false, false) - staff.capo - alphaTab.model.Note.getStringTuning(staff, nextString);
      if (nextFret < 0 || nextFret > 36) continue;
      proposals.set(note, { originalString, nextString, nextFret });
    }
    const duplicateTargets = new Set<string>();
    const seenTargets = new Set<string>();
    for (const [note, proposal] of proposals) {
      const key = `${note.beat.id}:${proposal.nextString}`;
      if (seenTargets.has(key)) duplicateTargets.add(key);
      seenTargets.add(key);
    }
    const accepted = new Map(proposals);
    for (const [note, proposal] of accepted) {
      const key = `${note.beat.id}:${proposal.nextString}`;
      if (duplicateTargets.has(key)) accepted.delete(note);
    }
    let removedCollision = true;
    while (removedCollision) {
      removedCollision = false;
      for (const [note, proposal] of accepted) {
        const occupant = note.beat.getNoteOnString(proposal.nextString);
        if (occupant && occupant !== note && (!selectedSet.has(occupant) || !accepted.has(occupant))) {
          accepted.delete(note);
          removedCollision = true;
        }
      }
    }
    if (!accepted.size) return setEditStatus(selected.some((note) => note.harmonicType !== alphaTab.model.HarmonicType.None)
      ? "自然泛音不会自动换弦，请先关闭泛音或手动校对音高"
      : "相邻弦没有保持同音高的可用品位");
    if (recognitionDraftRef.current && [...accepted].some(([note, proposal]) => !recognitionNote(note, proposal.originalString))) {
      return setEditStatus("换弦无法安全映射回保存草稿；请在已展开的六线网格中移动");
    }
    const projectedString = (note: alphaTab.model.Note) => accepted.get(note)?.nextString ?? note.string;
    const allNotes = flattenNotes(scoreRef.current!);
    const affectedLanes = [...accepted].map(([note, proposal]) => ({
      staff: note.beat.voice.bar.staff,
      voice: note.beat.voice.index,
      strings: new Set([proposal.originalString, proposal.nextString])
    }));
    const relationOrigins = allNotes.filter((note) =>
      (note.hammerPullDestination || note.slurDestination || note.slideTarget)
      && affectedLanes.some((lane) =>
        lane.staff === note.beat.voice.bar.staff
        && lane.voice === note.beat.voice.index
        && lane.strings.has(projectedString(note))
      )
    );
    const entry = historyEntry("换弦", uniqueNotes([...selected, ...relationOrigins]));
    const predictedNextOnLine = (origin: alphaTab.model.Note) => allNotes.find((candidate) =>
      candidate !== origin
      && candidate.beat.voice.index === origin.beat.voice.index
      && candidate.beat.voice.bar.staff === origin.beat.voice.bar.staff
      && candidate.beat.absolutePlaybackStart > origin.beat.absolutePlaybackStart
      && projectedString(candidate) === projectedString(origin)
    ) ?? null;
    const clearedRelationOrigins: alphaTab.model.Note[] = [];
    for (const origin of relationOrigins) {
      const relations: Array<{ technique: TabTechnique; destination: alphaTab.model.Note | null }> = [
        { technique: "hammer_on", destination: origin.hammerPullDestination },
        { technique: "legato", destination: origin.slurDestination },
        { technique: "slide", destination: origin.slideTarget }
      ];
      for (const relation of relations) {
        const destination = relation.destination;
        if (!destination) continue;
        const remainsValid = projectedString(origin) === projectedString(destination) && predictedNextOnLine(origin) === destination;
        if (remainsValid) continue;
        clearedRelationOrigins.push(origin);
        clearTechniqueFromNote(origin, relation.technique);
        const target = recognitionNote(origin);
        if (target?.note.technique && (
          target.note.technique === relation.technique
          || (relation.technique === "hammer_on" && target.note.technique === "pull_off")
        )) target.note.technique = undefined;
        markRecognitionChanged(origin);
      }
    }
    let moved = 0;
    for (const [note, proposal] of accepted) {
      const { originalString, nextString, nextFret } = proposal;
      const target = recognitionNote(note, originalString);
      note.string = nextString;
      note.fret = nextFret;
      if (target) { target.note.string = recognitionString(note, nextString); target.note.fret = nextFret; }
      markRecognitionChanged(note, originalString);
      moved += 1;
    }
    const skipped = selected.length - moved;
    commitEdit(
      `已将 ${moved} 个音换弦并保持音高${skipped ? `，跳过 ${skipped} 个不可安全移动的音` : ""}`,
      entry,
      uniqueNotes([...selected, ...clearedRelationOrigins])
    );
  }

  function setSelectedBeatDuration(eighths: number) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    const selected = selectedNotesRef.current;
    const beats = uniqueBeats(selected);
    if (!beats.length) return setEditStatus("请先选择一个或多个音符节拍");
    const selectedBeats = new Set(beats);
    for (const beat of beats) {
      const masterBar = beat.voice.bar.masterBar;
      const capacity = masterBar.timeSignatureNumerator * 8 / masterBar.timeSignatureDenominator;
      const total = beat.voice.beats.reduce((sum, candidate) => sum + (selectedBeats.has(candidate) ? eighths : beatDurationEighths(candidate)), 0);
      if (total > capacity + 1e-9) return setEditStatus(`第 ${masterBar.index + 1} 小节会超过拍号容量，未修改`);
    }
    const recognitionTargets = beats.map((beat) => {
      const note = beat.notes[0];
      return note ? recognitionNote(note) : null;
    });
    if (recognitionDraftRef.current && recognitionTargets.some((target) => !target)) {
      return setEditStatus("这个节拍无法安全映射回识别草稿，请在精确网格中修改");
    }
    for (const target of recognitionTargets) {
      if (!target) continue;
      const next = target.measure.events
        .filter((event) => event.onset_eighths > target.event.onset_eighths)
        .sort((left, right) => left.onset_eighths - right.onset_eighths)[0];
      if (target.event.onset_eighths + eighths > (next?.onset_eighths ?? MEASURE_EIGHTHS)) {
        return setEditStatus("这个时值会与后一个节拍重叠，未修改");
      }
    }
    const entry = historyEntry("音符时值", selected);
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    for (const beat of beats) {
      beat.duration = alphaDuration(eighths);
      beat.dots = 0;
    }
    for (const target of recognitionTargets) if (target) {
      target.event.duration_eighths = eighths;
      dirtyRecognitionMeasuresRef.current.add(target.measure.number);
    }
    commitEdit(`时值 ${directDurationLabel(eighths)}`, entry, selected);
  }

  function applySelectedDots(double = false) {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    const selected = selectedNotesRef.current;
    const beats = uniqueBeats(selected);
    if (!beats.length) {
      return setEditStatus("请先选择一个或多个音符节拍，再按小键盘 .");
    }
    const dots = double ? 2 : beats.every((beat) => beat.dots === 1) ? 0 : 1;
    const factor = dots === 2 ? 1.75 : dots === 1 ? 1.5 : 1;
    const durations = new Map(beats.map((beat) => [beat, beatBaseDurationEighths(beat) * factor]));
    const restPlans = beats
      .map((beat) => followingRestPlan(beat, durations.get(beat)! - beatDurationEighths(beat)))
      .filter((plan) => plan.delta > 1e-9);
    for (const plan of restPlans) {
      if (plan.availableDuration + 1e-9 < plan.delta) {
        return setEditStatus(`${dots === 2 ? "双附点" : "附点"}前没有足够休止时值，未修改`);
      }
    }
    const recognitionTargets = beats.map((beat) => {
      const note = beat.notes[0];
      return { beat, target: note ? recognitionNote(note) : null };
    });
    if (recognitionDraftRef.current && recognitionTargets.some(({ target }) => !target)) {
      return setEditStatus("这个节拍无法安全映射回识别草稿，请在精确网格中修改");
    }
    for (const { beat, target } of recognitionTargets) {
      if (!target) continue;
      const next = target.measure.events
        .filter((event) => event.onset_eighths > target.event.onset_eighths)
        .sort((left, right) => left.onset_eighths - right.onset_eighths)[0];
      if (target.event.onset_eighths + durations.get(beat)! > (next?.onset_eighths ?? MEASURE_EIGHTHS)) {
        return setEditStatus(`${dots === 2 ? "双附点" : "附点"}会与后一个节拍重叠，未修改`);
      }
    }
    const label = dots === 2 ? "双附点" : dots === 1 ? "附点" : "移除附点";
    const entry = historyEntry(label, selected);
    const changed = beats.some((beat) => beat.dots !== dots);
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    for (const plan of restPlans) applyRestPlan(plan, entry);
    for (const beat of beats) beat.dots = dots;
    for (const { beat, target } of recognitionTargets) if (target) {
      target.event.duration_eighths = durations.get(beat)!;
      dirtyRecognitionMeasuresRef.current.add(target.measure.number);
    }
    if (changed) commitEdit(label, entry, selected);
    else setEditStatus(dots === 2 ? "所选节拍已经是双附点" : dots === 1 ? "所选节拍已经是单附点" : "所选节拍没有附点");
  }

  function adjustSelectedBeatDuration(shorter: boolean) {
    if (scoreCursorRef.current) {
      const index = DIRECT_DURATIONS.findIndex((value) => value === entryDurationRef.current);
      const nextIndex = Math.max(0, Math.min(DIRECT_DURATIONS.length - 1, (index < 0 ? 3 : index) + (shorter ? 1 : -1)));
      setEntryDuration(DIRECT_DURATIONS[nextIndex]);
      return;
    }
    const currentBeat = selectedNotesRef.current[0]?.beat;
    if (!currentBeat) return setEditStatus("请先选择一个或多个音符节拍");
    const current = beatDurationEighths(currentBeat);
    const index = DIRECT_DURATIONS.findIndex((value) => value <= current + 1e-9);
    const nextIndex = Math.max(0, Math.min(DIRECT_DURATIONS.length - 1, (index < 0 ? 2 : index) + (shorter ? 1 : -1)));
    setSelectedBeatDuration(DIRECT_DURATIONS[nextIndex]);
  }

  function moveScoreSelection(horizontal: number, vertical: number, extend: boolean) {
    const current = selectedNotesRef.current.at(-1) ?? selectionAnchorRef.current;
    if (!current) return;
    let target: alphaTab.model.Note | null = null;
    if (horizontal) {
      let beat: alphaTab.model.Beat | null = horizontal > 0 ? current.beat.nextBeat : current.beat.previousBeat;
      for (let count = 0; beat && count < 128; count += 1) {
        if (beat.voice.index === current.beat.voice.index && beat.voice.bar.staff === current.beat.voice.bar.staff && beat.notes.length) {
          target = beat.notes.find((note) => note.string === current.string) ?? beat.notes[0];
          break;
        }
        beat = horizontal > 0 ? beat.nextBeat : beat.previousBeat;
      }
    } else if (vertical) {
      const candidates = current.beat.notes
        .filter((note) => vertical < 0 ? note.string < current.string : note.string > current.string)
        .sort((left, right) => Math.abs(left.string - current.string) - Math.abs(right.string - current.string));
      target = candidates[0] ?? null;
    }
    if (!target) return setEditStatus("这个方向没有其他可选音符");
    if (extend) {
      const anchor = selectionAnchorRef.current ?? current;
      setSelection(noteRange(anchor, target), anchor);
    } else setSelection([target], target);
  }

  function deleteSelectedNotes() {
    if (editingDisabledRef.current || savingRef.current) return setEditStatus(editingDisabledRef.current ? "请先完成当前识别或精确校对" : "正在保存，请稍候");
    const selected = selectedNotesRef.current;
    if (!selected.length) return;
    if (recognitionDraftRef.current && selected.some((note) => !recognitionNote(note))) {
      return setEditStatus("删除无法安全映射回保存草稿；请在已展开的六线网格中删除");
    }
    const entry = historyEntry("删除音符", selected);
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    for (const note of selected) {
      const linkedOrigins = uniqueNotes([note.hammerPullOrigin, note.slideOrigin, note.slurOrigin].filter((candidate): candidate is alphaTab.model.Note => Boolean(candidate)));
      for (const origin of linkedOrigins) {
        const originTarget = recognitionNote(origin);
        if (originTarget?.note.technique && ["legato", "slide", "hammer_on", "pull_off"].includes(originTarget.note.technique)) {
          originTarget.note.technique = undefined;
          markRecognitionChanged(origin);
        }
      }
      const target = recognitionNote(note);
      if (target) {
        target.event.notes = target.event.notes.filter((candidate) => candidate !== target.note);
        if (!target.event.notes.length) target.measure.events = target.measure.events.filter((candidate) => candidate !== target.event);
      }
      markRecognitionChanged(note);
      detachNoteRelations(note);
      note.beat.removeNote(note);
    }
    commitEdit(`已删除 ${selected.length} 个音`, entry, selected);
    setSelection([]);
  }

  function undoLastEdit() {
    if (editingDisabledRef.current || savingRef.current) return;
    const entry = history.at(-1);
    if (!entry) return;
    digitBufferRef.current = null;
    digitHistoryEntryRef.current = null;
    for (const state of entry.states) restoreNoteState(state);
    for (const state of entry.beatStates) restoreBeatState(state);
    recognitionDraftRef.current = cloneMeasureDraft(entry.recognition);
    dirtyRecognitionMeasuresRef.current = new Set(entry.dirtyRecognitionMeasures);
    const remaining = history.slice(0, -1);
    setHistory(remaining);
    setDirty(remaining.length > 0 || historyBaseDirtyRef.current);
    setSelection(entry.selection);
    rebuildAfterEdit(`已撤销“${entry.label}”`, entry.selection, remaining.length > 0 || historyBaseDirtyRef.current);
  }

  async function saveChanges() {
    if (!dirty || savingRef.current) return;
    const score = scoreRef.current;
    const api = apiRef.current;
    if (!score || !api) return;
    savingRef.current = true;
    setSaving(true);
    setEditStatus("正在保存当前校对版本…");
    try {
      const recognitionDraft = recognitionDraftRef.current;
      const dirtyMeasures = [...dirtyRecognitionMeasuresRef.current].sort((left, right) => left - right);
      if (recognitionDraft && dirtyMeasures.length) {
        await onSaveRecognition(dirtyMeasures.map((measure) => ({ measure, events: recognitionDraft.find((candidate) => candidate.number === measure)?.events ?? [] })));
      } else {
        const bytes = new alphaTab.exporter.Gp7Exporter().export(score, api.settings);
        const file = new File([binaryBlob(bytes, "application/octet-stream")], `${safeFileName(fileBaseName)}.gp`, { type: "application/octet-stream" });
        await onSaveScore(file);
      }
      setHistory([]);
      historyBaseDirtyRef.current = false;
      digitBufferRef.current = null;
      digitHistoryEntryRef.current = null;
      dirtyRecognitionMeasuresRef.current.clear();
      setDirty(false);
      setEditStatus("已保存到私人曲库");
    } catch (error) {
      setEditStatus(`保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }

  useEffect(() => {
    const handler = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        if (commandPaletteOpen) closeCommandPalette();
        else { setSelection([]); clearScoreCursor(); }
        return;
      }
      if (event.code === "Space" && !isNativeControlTarget(event.target)) { event.preventDefault(); if (ready) apiRef.current?.playPause(); return; }
      if (isTypingTarget(event.target)) return;
      const commandKey = event.ctrlKey || event.metaKey;
      if (event.key === "?") { event.preventDefault(); setShortcutHelp((value) => !value); return; }
      if (editingDisabledRef.current || savingRef.current) return;
      if (commandKey && event.key.toLowerCase() === "e") { event.preventDefault(); openCommandPalette(); return; }
      if (commandKey && event.key.toLowerCase() === "s") { event.preventDefault(); void saveChanges(); return; }
      if (commandKey && event.key.toLowerCase() === "z") { event.preventDefault(); undoLastEdit(); return; }
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) && scoreCursorRef.current) {
        event.preventDefault();
        moveScoreCursor(
          event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0,
          event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0
        );
        return;
      }
      if (event.altKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) { event.preventDefault(); moveNotesAcrossStrings(event.key === "ArrowUp" ? -1 : 1); return; }
      if (!event.altKey && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) && selectedNotesRef.current.length) {
        event.preventDefault();
        moveScoreSelection(
          event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0,
          event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0,
          event.shiftKey
        );
        return;
      }
      if (!commandKey && (event.key === "+" || event.key === "=")) { event.preventDefault(); adjustSelectedBeatDuration(true); return; }
      if (!commandKey && (event.key === "-" || event.key === "_")) { event.preventDefault(); adjustSelectedBeatDuration(false); return; }
      if (event.code === "NumpadDecimal" || event.key === "Decimal" || event.key === ".") { event.preventDefault(); applySelectedDots(commandKey); return; }
      if (/^\d$/.test(event.key)) { event.preventDefault(); writeFretDigit(event.key); return; }
      if (event.key === "Enter" || event.key.toLowerCase() === "r") { if (commitRestAtScoreCursor()) event.preventDefault(); return; }
      if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteSelectedNotes(); return; }
      const command = EDIT_COMMANDS.find((candidate) => candidate.shortcut.toLowerCase() === event.key.toLowerCase());
      if (command) { event.preventDefault(); applyCommand(command); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  function openReference(mode: ReferenceMode) {
    if (mode === "image" && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.playbackRate = 1;
    }
    setReferenceMode(mode);
    if (mode === "video") window.requestAnimationFrame(() => {
      const video = videoRef.current;
      if (!video || !videoSyncEnabledRef.current) return;
      const position = lastPlaybackPositionRef.current;
      const target = videoTimeAtTick(videoSyncAnchorsRef.current, position.tick);
      if (target !== null) video.currentTime = target;
      video.playbackRate = videoRateAtTick(videoSyncAnchorsRef.current, position.tick, position.endTick, position.endTime);
      if (playingRef.current) void video.play().catch(() => undefined);
      else video.pause();
    });
  }

  function changeVideoSync(enabled: boolean) {
    if (enabled && !syncAvailable) return;
    setVideoSyncEnabled(enabled);
    const video = videoRef.current;
    if (!video) return;
    if (!enabled) {
      video.playbackRate = 1;
      return;
    }
    const position = lastPlaybackPositionRef.current;
    const target = videoTimeAtTick(videoSyncAnchorsRef.current, position.tick);
    if (target !== null) video.currentTime = target;
    video.playbackRate = videoRateAtTick(videoSyncAnchorsRef.current, position.tick, position.endTick, position.endTime);
    if (playingRef.current) void video.play().catch(() => undefined);
    else video.pause();
  }

  async function exportScore(kind: ExportKind) {
    if (exporting) return;
    const api = apiRef.current;
    const score = scoreRef.current;
    const midi = midiRef.current;
    if (!api || !score) return setExportStatus("乐谱尚未载入完成");
    const baseName = safeFileName(fileBaseName);
    setExporting(kind);
    try {
      if (kind === "gp") {
        setExportStatus("正在生成 Guitar Pro 7 文件…");
        downloadBlob(binaryBlob(new alphaTab.exporter.Gp7Exporter().export(score, api.settings), "application/octet-stream"), `${baseName}.gp`);
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
          if (chunk.endTime > MAX_WAV_DURATION_MS) throw new Error("浏览器 WAV 导出目前限制为 6 分钟，请改用 GP7 或 MIDI");
          sampleCount += chunk.samples.length;
          chunks.push(encodePcm16Chunk(chunk.samples));
          setExportStatus(`正在合成 WAV · ${chunk.endTime > 0 ? Math.min(100, Math.round(chunk.currentTime / chunk.endTime * 100)) : 0}%`);
        }
      } finally { exporter.destroy(); }
      if (!chunks.length) throw new Error("没有生成可导出的音频样本");
      downloadBlob(encodePcm16Wav(chunks, sampleCount, options.sampleRate), `${baseName}.wav`);
      setExportStatus("WAV 合成音频已导出");
    } catch (error) {
      setExportStatus(`导出失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally { setExporting(null); }
  }

  const selectedBeats = uniqueBeats(selectedNotesRef.current);
  const selectedBeatDurations = selectedBeats.map(beatBaseDurationEighths);
  const selectedBeatDuration = selectedBeatDurations.length && selectedBeatDurations.every((value) => value === selectedBeatDurations[0])
    ? selectedBeatDurations[0]
    : null;
  const selectedBeatDots = selectedBeats.length && selectedBeats.every((beat) => beat.dots === selectedBeats[0].dots)
    ? selectedBeats[0].dots
    : null;
  const cursorStringCount = scoreCursor ? scoreCursor.beat.voice.bar.staff.tuning.length || 6 : 0;
  const cursorVisualString = scoreCursor ? cursorStringCount + 1 - scoreCursor.string : 0;
  const activeDuration = scoreCursor ? entryDuration : selectedBeatDuration;

  return (
    <div className="alpha-player score-studio">
      <div className="score-studio-commandbar">
        <div className={`selection-readout ${selectedIds.length ? "active" : ""}`}>
          <span>{selectedIds.length ? <Check size={15} /> : <PencilLine size={15} />}</span>
          <div><strong>{selectedIds.length ? `已选 ${selectedIds.length} 个音` : scoreCursor ? `${cursorVisualString} 弦 · 新音 ${directDurationLabel(entryDuration)}` : "谱面即编辑器"}</strong><small role="status" aria-live="polite">{editStatus}</small></div>
        </div>
        <div className="notation-command-strip" aria-label="演奏技巧快捷命令">
          {EDIT_COMMANDS.map((command) => (
            <button type="button" key={command.id} disabled={editingDisabled || saving || !selectedIds.length || Boolean(command.requiresPair && selectedIds.length !== 2)} onClick={() => applyCommand(command)} title={`${command.label} · ${command.shortcut}`}>
              <b>{command.mark}</b><span>{command.label}</span><kbd>{command.shortcut}</kbd>
            </button>
          ))}
        </div>
        <div className="score-studio-actions">
          <button type="button" disabled={editingDisabled || saving || !history.length} onClick={undoLastEdit} title="撤销 Ctrl/⌘Z"><Undo2 size={15} /><span>撤销</span></button>
          <button type="button" disabled={editingDisabled || saving} onClick={openCommandPalette} title="命令面板 Ctrl/⌘E"><Command size={15} /><span>命令</span></button>
          <button type="button" aria-label="打开悬浮视频" disabled={!videoUrl} className={referenceMode === "video" ? "active" : ""} onClick={() => openReference("video")}><Video size={15} /><span>视频</span></button>
          <button type="button" aria-label="打开悬浮原帧" disabled={!referenceImage} className={referenceMode === "image" ? "active" : ""} onClick={() => openReference("image")}><ImageIcon size={15} /><span>原帧</span></button>
          {recognition && <button type="button" disabled={editingDisabled || saving || dirty} onClick={() => void onRetryRecognition(focusedMeasure)} title={dirty ? "请先保存当前谱面修改" : `重新识别第 ${focusedMeasure} 小节`}><ScanLine size={15} /><span>重识别</span></button>}
          {recognition && <button type="button" disabled={editingDisabled || saving || dirty} onClick={() => void onAppendMeasure()} title={dirty ? "请先保存当前谱面修改" : "在乐谱末尾添加小节"}><Plus size={15} /><span>加小节</span></button>}
          <button type="button" aria-label={saving ? "正在保存乐谱" : "保存乐谱"} className={dirty ? "save-needed" : ""} disabled={editingDisabled || !dirty || saving} onClick={() => void saveChanges()}><Save size={15} /><span>{saving ? "保存中" : "保存"}</span></button>
          <details className="score-export-menu">
            <summary aria-label="导出乐谱"><Download size={15} /><span>导出</span></summary>
            <div>
              {pdfUrl && <a href={`${pdfUrl}?download=1`} download><span>PDF</span><small>源图合成版本</small></a>}
              <button type="button" disabled={!scoreReady || Boolean(exporting)} onClick={() => void exportScore("gp")}><span>GP7</span><small>Guitar Pro 7</small></button>
              <button type="button" disabled={!midiRef.current || Boolean(exporting)} onClick={() => void exportScore("midi")}><span>MIDI</span><small>标准演奏数据</small></button>
              <button type="button" disabled={!scoreReady || Boolean(exporting)} onClick={() => void exportScore("wav")}><span>WAV</span><small>当前音色合成</small></button>
            </div>
          </details>
        </div>
      </div>

      <div className="score-structure-toolbar">
        <div className="bars-per-row-control" aria-label="每行小节数">
          <span>每行小节</span>
          {[3, 4].map((count) => <button type="button" key={count} className={barsPerRow === count ? "active" : ""} onClick={() => setBarsPerRow(count as 3 | 4)} aria-pressed={barsPerRow === count}>{count}</button>)}
        </div>
        <div className="direct-duration-control" aria-label="所选节拍时值">
          <span>{scoreCursor ? "新音时值" : "所选时值"}</span>
          {DIRECT_DURATIONS.map((duration) => <button type="button" key={duration} className={activeDuration === duration ? "active" : ""} disabled={editingDisabled || saving || (!selectedIds.length && !scoreCursor)} onClick={() => scoreCursor ? setEntryDuration(duration) : setSelectedBeatDuration(duration)} aria-pressed={activeDuration === duration}>{directDurationLabel(duration)}</button>)}
          <button type="button" className={`dot-duration-button ${selectedBeatDots ? "active" : ""}`} disabled={editingDisabled || saving || !selectedIds.length} onClick={() => applySelectedDots(false)} aria-pressed={Boolean(selectedBeatDots)} title=". 切换单附点；Ctrl/⌘ + . 设双附点">{selectedBeatDots === 2 ? "··" : "·"} 附点</button>
          <i><kbd>+</kbd>/<kbd>−</kbd> 时值 · 小键盘 <kbd>.</kbd> 附点</i>
        </div>
        {scoreCursor && <div className="score-string-picker" aria-label="当前编辑弦"><span>弦</span>{Array.from({ length: cursorStringCount }, (_, index) => index + 1).map((string) => <button type="button" key={string} className={cursorVisualString === string ? "active" : ""} onClick={() => selectCursorString(string)}>{string}</button>)}</div>}
      </div>

      <div className="studio-keyboard-hint">
        <button type="button" onClick={() => setShortcutHelp((value) => !value)}><Keyboard size={13} /> 快捷键 <kbd>?</kbd></button>
        <span><kbd>Space</kbd> 播放/暂停</span><span><kbd>Ctrl/⌘ 点击</kbd> 离散多选</span><span><kbd>Shift 点击/方向键</kbd> 连续选择</span><span><kbd>0–9</kbd> 批量品位</span><span><kbd>.</kbd> 单附点开关</span><span><kbd>Ctrl/⌘ .</kbd> 双附点</span><span><kbd>↑↓←→</kbd> 移动选区</span><span><kbd>Alt ↑↓</kbd> 保持音高换弦</span><span><kbd>Ctrl/⌘ S</kbd> 保存</span>
      </div>
      {shortcutHelp && <div className="shortcut-help-panel"><strong>编辑快捷键</strong><span>拖过音符可扩展选区；<kbd>.</kbd> 切换单附点，<kbd>Ctrl/⌘ .</kbd> 设双附点；<kbd>Esc</kbd> 清空；<kbd>Delete</kbd> 删除；<kbd>Ctrl/⌘ Z</kbd> 撤销。</span><span>{EDIT_COMMANDS.map((command) => `${command.shortcut} ${command.label}`).join(" · ")}</span></div>}

      <div ref={hostRef} className="alpha-host" tabIndex={0} aria-disabled={editingDisabled || saving} aria-label={editingDisabled ? "当前只读的可播放乐谱" : saving ? "正在保存的只读乐谱" : "可直接在空拍和音符上编辑的乐谱"} />
      <div className="score-selection-layer" aria-hidden="true">
        {measureMarkers.map((marker) => <i className="measure-error-marker" key={`measure-${marker.measure}`} style={{ left: marker.x, top: marker.y, width: marker.width, height: marker.height }}><b>{marker.measure}</b></i>)}
        {selectionMarkers.map((marker) => <i key={marker.id} style={{ left: marker.x, top: marker.y, width: marker.width, height: marker.height }} />)}
        {scoreCursor && <i className="score-cell-cursor" style={{ left: scoreCursor.marker.x, top: scoreCursor.marker.y, width: scoreCursor.marker.width, height: scoreCursor.marker.height }} />}
      </div>

      <div className="alpha-controls glass-bar">
        <button className="icon-button strong" type="button" disabled={!ready} onClick={() => apiRef.current?.playPause()} aria-label={playing ? "暂停" : "播放"}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button>
        <button className="icon-button" type="button" disabled={!ready} onClick={() => apiRef.current?.stop()} aria-label="停止"><CircleStop size={18} /></button>
        <span className="alpha-status" role="status" aria-live="polite">{(!ready || Boolean(exporting)) && <LoaderCircle size={14} className="spin" />}{exportStatus || (ready ? (engineMode === "audio-worklet" ? "FluidSynth · 低延迟引擎" : "FluidSynth · 稳定缓冲引擎") : loading)}</span>
      </div>

      {referenceMode && <FloatingReference mode={referenceMode} videoUrl={videoUrl} image={referenceImage} videoRef={videoRef} syncAvailable={syncAvailable} syncEnabled={videoSyncEnabled} onModeChange={openReference} onSyncChange={changeVideoSync} onClose={() => setReferenceMode(null)} />}

      {commandPaletteOpen && (
        <div className="score-command-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) closeCommandPalette(); }}>
          <section className="score-command-palette" role="dialog" aria-modal="true" aria-label="乐谱命令面板" onKeyDown={trapCommandPaletteFocus}>
            <header><Command size={18} /><div><strong>命令</strong><span>作用于当前 {selectedIds.length} 个音</span></div><kbd>Esc</kbd></header>
            <input autoFocus value={commandQuery} onChange={(event) => setCommandQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && filteredCommands[0]) { event.preventDefault(); applyCommand(filteredCommands[0]); } }} placeholder="输入技巧名称，例如：滑音、bend、M…" aria-label="搜索乐谱命令" />
            <div className="score-command-results">
              {filteredCommands.map((command, index) => (
                <button type="button" key={command.id} disabled={editingDisabled || saving || !selectedIds.length || Boolean(command.requiresPair && selectedIds.length !== 2)} onClick={() => applyCommand(command)}>
                  <b>{command.mark}</b><span><strong>{command.label}</strong><small>{command.requiresPair ? "需要同一弦上的两个音" : "应用到全部所选音"}</small></span><kbd>{command.shortcut}</kbd>{index === 0 && <em>Enter</em>}
                </button>
              ))}
              {!filteredCommands.length && <p>没有匹配命令</p>}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
