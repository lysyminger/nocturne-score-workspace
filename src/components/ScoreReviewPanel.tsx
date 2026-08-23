import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Clock3, Minus, Plus, RefreshCw, Save, Trash2 } from "lucide-react";
import type { Project, RecognitionDiagnostics, RecognitionEvent, RecognitionMeasure, RecognitionNote, TabTechnique } from "../types";

type Props = {
  project: Project;
  diagnostics: RecognitionDiagnostics;
  measureNumber: number;
  busy: boolean;
  retrying: boolean;
  embedded?: boolean;
  onMeasureChange: (measure: number) => void;
  onDirtyChange: (dirty: boolean) => void;
  onSave: (measure: number, events: RecognitionEvent[]) => Promise<void>;
  onRetryRecognition: (measure: number) => Promise<RecognitionMeasure>;
};

type SelectedCell = { onset: number; string: number };
type TimeSelection = { anchor: number; focus: number };

const GRID_STEP = 0.5;
const GRID_SLOTS = 16;
const MEASURE_UNITS = 8;
const DURATION_OPTIONS = [0.5, 1, 1.5, 2, 3, 4, 6, 8];
const POWER_OF_TWO_DURATIONS = [0.5, 1, 2, 4, 8];
const TECHNIQUES: Array<{ id: TabTechnique; label: string; mark: string; shortcut: string }> = [
  { id: "legato", label: "连音", mark: "⌒", shortcut: "L" },
  { id: "slide", label: "滑音", mark: "/", shortcut: "S" },
  { id: "hammer_on", label: "击弦", mark: "H", shortcut: "H" },
  { id: "pull_off", label: "勾弦", mark: "P", shortcut: "P" },
  { id: "bend", label: "推弦", mark: "B", shortcut: "B" },
  { id: "vibrato", label: "颤音", mark: "~", shortcut: "V" },
  { id: "harmonic", label: "泛音", mark: "◇", shortcut: "N" },
  { id: "palm_mute", label: "闷音", mark: "PM", shortcut: "M" },
  { id: "let_ring", label: "延音", mark: "LR", shortcut: "I" },
  { id: "dead_note", label: "死音", mark: "×", shortcut: "X" }
];

function cloneEvents(measure: RecognitionMeasure | undefined): RecognitionEvent[] {
  return (measure?.events ?? []).map((event) => ({
    ...event,
    notes: event.notes.map((note) => ({ ...note }))
  }));
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function durationLabel(units: number) {
  return ({
    0.5: "1/16",
    1: "1/8",
    1.5: "附点 1/8",
    2: "1/4",
    3: "附点 1/4",
    4: "1/2",
    6: "附点 1/2",
    8: "全音符"
  } as Record<number, string>)[units] ?? `${units}/8`;
}

function beatLabel(onset: number) {
  const slot = Math.round(onset / GRID_STEP);
  return `${Math.floor(slot / 4) + 1}${["", "e", "&", "a"][slot % 4]}`;
}

export function ScoreReviewPanel({ project, diagnostics, measureNumber, busy, retrying, embedded = false, onMeasureChange, onDirtyChange, onSave, onRetryRecognition }: Props) {
  const startMeasure = diagnostics.summary.start_measure ?? diagnostics.measures[0]?.number ?? 1;
  const endMeasure = diagnostics.summary.end_measure ?? diagnostics.measures.at(-1)?.number ?? startMeasure;
  const currentMeasure = diagnostics.measures.find((item) => item.number === measureNumber);
  const [events, setEvents] = useState<RecognitionEvent[]>(() => cloneEvents(currentMeasure));
  const [dirty, setDirty] = useState(false);
  const [selectedCell, setSelectedCell] = useState<SelectedCell>({ onset: 0, string: 1 });
  const [timeSelection, setTimeSelection] = useState<TimeSelection>({ anchor: 0, focus: 0 });
  const [entryDuration, setEntryDuration] = useState(1);
  const [armedTechnique, setArmedTechnique] = useState<TabTechnique | null>(null);
  const [editStatus, setEditStatus] = useState("方向键移动；点击空格后输入品位即可手动打谱");
  const digitBufferRef = useRef<{ key: string; value: string; time: number } | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const saveInFlightRef = useRef(false);
  const dragSelectionRef = useRef<{ pointerId: number; anchor: number } | null>(null);

  useEffect(() => {
    const nextEvents = cloneEvents(currentMeasure);
    setEvents(nextEvents);
    setDirty(false);
    const firstNote = nextEvents[0]?.notes[0];
    const firstOnset = nextEvents[0]?.onset_eighths ?? 0;
    setSelectedCell({ onset: firstOnset, string: firstNote?.string ?? 1 });
    setTimeSelection({ anchor: firstOnset, focus: firstOnset });
    setEntryDuration(nextEvents[0]?.duration_eighths ?? 1);
    setEditStatus("方向键移动；点击空格后输入品位即可手动打谱");
    digitBufferRef.current = null;
  }, [measureNumber, diagnostics]);

  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);

  const sourceFrame = useMemo(() => {
    const diagnosticFrame = diagnostics.frames.find((frame) => {
      const start = frame.start_measure;
      return start !== null && start <= measureNumber && measureNumber < start + Math.max(1, frame.raw_measure_labels.length);
    });
    const targetTime = currentMeasure?.source_time || diagnosticFrame?.time_seconds;
    if (targetTime === undefined) return project.video_frames[0] ?? null;
    return [...project.video_frames].sort(
      (left, right) => Math.abs(left.time_seconds - targetTime) - Math.abs(right.time_seconds - targetTime)
    )[0] ?? null;
  }, [currentMeasure?.source_time, diagnostics.frames, measureNumber, project.video_frames]);

  const selectedEventIndex = events.findIndex((event) => event.onset_eighths === selectedCell.onset);
  const selectedNote = selectedEventIndex >= 0
    ? events[selectedEventIndex].notes.find((note) => note.string === selectedCell.string)
    : undefined;
  const selectedTechnique = selectedNote?.technique ?? armedTechnique;
  const selectedStart = Math.min(timeSelection.anchor, timeSelection.focus);
  const selectedEnd = Math.max(timeSelection.anchor, timeSelection.focus);
  const selectedEventIndexes = events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => selectedStart <= event.onset_eighths && event.onset_eighths <= selectedEnd)
    .map(({ index }) => index);
  const selectedEvents = selectedEventIndexes.map((index) => events[index]);
  const selectionVisualEnd = selectedEvents.length
    ? Math.max(selectedEnd, ...selectedEvents.map((event) => event.onset_eighths + event.duration_eighths - GRID_STEP))
    : selectedEnd;
  const selectedSpan = selectedEnd - selectedStart + GRID_STEP;
  const selectedDuration = selectedEvents.reduce((total, event) => total + event.duration_eighths, 0);

  function markChanged(nextEvents: RecognitionEvent[]) {
    if (busy || saveInFlightRef.current) return;
    setEvents([...nextEvents].sort((left, right) => left.onset_eighths - right.onset_eighths));
    setDirty(true);
  }

  function updateEvent(index: number, patch: Partial<RecognitionEvent>) {
    markChanged(events.map((event, eventIndex) => eventIndex === index ? { ...event, ...patch } : event));
  }

  function updateNote(eventIndex: number, noteIndex: number, patch: Partial<RecognitionNote>) {
    markChanged(events.map((event, currentEventIndex) => currentEventIndex === eventIndex
      ? { ...event, notes: event.notes.map((note, currentNoteIndex) => currentNoteIndex === noteIndex ? { ...note, ...patch } : note) }
      : event));
  }

  function editableOnset(onset: number) {
    const covering = events.find((event) => event.onset_eighths <= onset && onset < event.onset_eighths + event.duration_eighths);
    return covering?.onset_eighths ?? onset;
  }

  function adjacentOnset(current: number, direction: number) {
    if (!direction) return current;
    if (direction > 0) {
      const covering = events.find((event) => event.onset_eighths <= current && current < event.onset_eighths + event.duration_eighths);
      const raw = covering ? covering.onset_eighths + covering.duration_eighths : current + GRID_STEP;
      return raw >= MEASURE_UNITS ? current : editableOnset(raw);
    }
    return editableOnset(Math.max(0, current - GRID_STEP));
  }

  function selectCell(onset: number, string: number, extend = false) {
    const targetOnset = editableOnset(onset);
    setSelectedCell({ onset: targetOnset, string });
    setTimeSelection((current) => extend
      ? { ...current, focus: targetOnset }
      : { anchor: targetOnset, focus: targetOnset });
    const targetEvent = events.find((event) => event.onset_eighths === targetOnset);
    if (targetEvent) setEntryDuration(targetEvent.duration_eighths);
    window.requestAnimationFrame(() => gridRef.current?.focus());
  }

  function beginCellSelection(event: React.PointerEvent<HTMLButtonElement>, onset: number, string: number) {
    if (busy || saveInFlightRef.current) return;
    const targetOnset = editableOnset(onset);
    const anchor = event.shiftKey ? timeSelection.anchor : targetOnset;
    dragSelectionRef.current = { pointerId: event.pointerId, anchor };
    gridRef.current?.setPointerCapture(event.pointerId);
    setSelectedCell({ onset: targetOnset, string });
    setTimeSelection({ anchor, focus: targetOnset });
    const targetEvent = events.find((item) => item.onset_eighths === targetOnset);
    if (targetEvent) setEntryDuration(targetEvent.duration_eighths);
    event.preventDefault();
  }

  function moveCellSelection(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragSelectionRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>("[data-grid-onset]");
    const onset = Number(target?.dataset.gridOnset);
    if (!Number.isFinite(onset)) return;
    setTimeSelection({ anchor: drag.anchor, focus: editableOnset(onset) });
  }

  function endCellSelection(event: React.PointerEvent<HTMLDivElement>) {
    if (dragSelectionRef.current?.pointerId !== event.pointerId) return;
    dragSelectionRef.current = null;
    if (gridRef.current?.hasPointerCapture(event.pointerId)) gridRef.current.releasePointerCapture(event.pointerId);
    window.requestAnimationFrame(() => gridRef.current?.focus());
  }

  function validEvents(candidate: RecognitionEvent[]) {
    const ordered = [...candidate].sort((left, right) => left.onset_eighths - right.onset_eighths);
    for (let index = 0; index < ordered.length; index += 1) {
      const event = ordered[index];
      const next = ordered[index + 1];
      if (event.onset_eighths < 0 || event.onset_eighths + event.duration_eighths > MEASURE_UNITS) return false;
      if (next && event.onset_eighths + event.duration_eighths > next.onset_eighths) return false;
    }
    return true;
  }

  function setCellFret(onset: number, string: number, fret: number) {
    const targetOnset = editableOnset(onset);
    const eventIndex = events.findIndex((event) => event.onset_eighths === targetOnset);
    if (eventIndex >= 0) {
      const noteIndex = events[eventIndex].notes.findIndex((note) => note.string === string);
      const notes: RecognitionNote[] = noteIndex >= 0
        ? events[eventIndex].notes.map((note, index) => index === noteIndex ? { ...note, fret } : note)
        : [...events[eventIndex].notes, { string, fret, ...(armedTechnique ? { technique: armedTechnique } : {}) }]
            .sort((left, right) => left.string - right.string);
      markChanged(events.map((event, index) => index === eventIndex ? { ...event, notes } : event));
    } else {
      const nextEvents = [...events, {
        onset_eighths: targetOnset,
        duration_eighths: entryDuration,
        notes: [{ string, fret, ...(armedTechnique ? { technique: armedTechnique } : {}) }]
      }];
      if (!validEvents(nextEvents)) {
        setEditStatus("新音符会与后一个节拍重叠，请先缩短时值");
        return;
      }
      markChanged(nextEvents);
    }
    setSelectedCell({ onset: targetOnset, string });
    setTimeSelection({ anchor: targetOnset, focus: targetOnset });
    setEditStatus(`已在 ${beatLabel(targetOnset)} 输入 ${string} 弦 ${fret} 品`);
  }

  function deleteSelectedNote() {
    if (selectedEventIndex < 0) return;
    const deleted = selectedNote;
    const remaining = events[selectedEventIndex].notes.filter((note) => note.string !== selectedCell.string);
    markChanged(remaining.length
      ? events.map((event, index) => index === selectedEventIndex ? { ...event, notes: remaining } : event)
      : events.filter((_, index) => index !== selectedEventIndex));
    digitBufferRef.current = null;
    if (deleted) setEditStatus(`已删除 ${beatLabel(selectedCell.onset)} 的 ${selectedCell.string} 弦 ${deleted.fret} 品`);
  }

  function applyTechnique(technique: TabTechnique) {
    if (busy || saveInFlightRef.current) return;
    const rangeSelection = selectedEventIndexes.length > 1 || selectedStart !== selectedEnd;
    const targets = rangeSelection
      ? selectedEvents.flatMap((event) => event.notes)
      : selectedNote ? [selectedNote] : [];
    const removing = targets.length > 0 && targets.every((note) => note.technique === technique);
    setArmedTechnique(removing ? null : technique);
    if (targets.length) {
      const selected = new Set(selectedEventIndexes);
      markChanged(events.map((event, eventIndex) => selected.has(eventIndex)
        ? { ...event, notes: event.notes.map((note) => rangeSelection || note.string === selectedCell.string ? { ...note, technique: removing ? undefined : technique } : note) }
        : event));
      setEditStatus(`${removing ? "已移除" : "已应用"} ${TECHNIQUES.find((item) => item.id === technique)?.label} · ${targets.length} 个音`);
    }
    window.requestAnimationFrame(() => gridRef.current?.focus());
  }

  function writeDigit(digit: string) {
    const key = `${measureNumber}:${selectedCell.onset}:${selectedCell.string}`;
    const now = Date.now();
    const previous = digitBufferRef.current;
    let value = digit;
    if (previous && previous.key === key && now - previous.time < 900) {
      const appended = `${previous.value}${digit}`.replace(/^0+(?=\d)/, "");
      value = Number(appended) <= 36 ? appended : digit;
    }
    digitBufferRef.current = { key, value, time: now };
    setCellFret(selectedCell.onset, selectedCell.string, Number(value));
  }

  function applyDuration(duration: number) {
    setEntryDuration(duration);
    if (!selectedEventIndexes.length) {
      setEditStatus(`新输入音符将使用 ${durationLabel(duration)}`);
      return;
    }
    const selected = new Set(selectedEventIndexes);
    const nextEvents = events.map((event, index) => selected.has(index) ? { ...event, duration_eighths: duration } : event);
    if (!validEvents(nextEvents)) {
      setEditStatus("这个时值会与后一个音重叠或越过小节线，未应用");
      return;
    }
    markChanged(nextEvents);
    setEditStatus(`已把 ${selected.size} 个节拍改为 ${durationLabel(duration)}`);
  }

  function adjustDuration(shorter: boolean) {
    const current = selectedEventIndex >= 0 ? events[selectedEventIndex].duration_eighths : entryDuration;
    const candidates = POWER_OF_TWO_DURATIONS.filter((value) => shorter ? value < current : value > current);
    const next = shorter ? candidates.at(-1) : candidates[0];
    applyDuration(next ?? current);
  }

  function makeSelectionRest() {
    if (!selectedEventIndexes.length) {
      setEditStatus("当前已经是休止位置");
      return;
    }
    const selected = new Set(selectedEventIndexes);
    markChanged(events.filter((_, index) => !selected.has(index)));
    digitBufferRef.current = null;
    setEditStatus(`已把所选 ${selectedEvents.length} 个节拍（${durationLabel(selectedDuration)}）设为休止`);
  }

  function moveSelectedEvent(horizontal: number) {
    if (selectedEventIndex < 0) return setEditStatus("当前空位没有可移动的音符");
    const target = clamp(events[selectedEventIndex].onset_eighths + horizontal * GRID_STEP, 0, MEASURE_UNITS - GRID_STEP);
    const nextEvents = events.map((event, index) => index === selectedEventIndex ? { ...event, onset_eighths: target } : event);
    if (!validEvents(nextEvents)) return setEditStatus("目标位置会与其他音符重叠，未移动");
    markChanged(nextEvents);
    setSelectedCell((current) => ({ ...current, onset: target }));
    setTimeSelection({ anchor: target, focus: target });
    setEditStatus(`音符已移动到 ${beatLabel(target)}`);
  }

  function moveSelectedNote(vertical: number) {
    if (selectedEventIndex < 0 || !selectedNote) return setEditStatus("请先选择一个音符再换弦");
    const targetString = clamp(selectedCell.string + vertical, 1, 6);
    if (targetString === selectedCell.string) return;
    if (events[selectedEventIndex].notes.some((note) => note.string === targetString)) return setEditStatus(`${targetString} 弦同一时刻已有音符`);
    const noteIndex = events[selectedEventIndex].notes.findIndex((note) => note.string === selectedCell.string);
    updateNote(selectedEventIndex, noteIndex, { string: targetString });
    setSelectedCell((current) => ({ ...current, string: targetString }));
    setEditStatus(`音符已移到 ${targetString} 弦；品位保持 ${selectedNote.fret}`);
  }

  function moveSelection(horizontal: number, vertical: number, extend = false) {
    const onset = horizontal ? adjacentOnset(selectedCell.onset, horizontal) : selectedCell.onset;
    const string = clamp(selectedCell.string + vertical, 1, 6);
    const targetOnset = editableOnset(onset);
    setSelectedCell({ onset: targetOnset, string });
    setTimeSelection((current) => extend && horizontal
      ? { ...current, focus: targetOnset }
      : { anchor: targetOnset, focus: targetOnset });
    digitBufferRef.current = null;
  }

  function handleGridKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (busy || saveInFlightRef.current) {
      if (event.code !== "Space") {
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }
    const handle = () => {
      event.preventDefault();
      event.stopPropagation();
    };
    if (/^\d$/.test(event.key)) {
      handle();
      writeDigit(event.key);
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      handle();
      if (event.altKey) {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") moveSelectedEvent(event.key === "ArrowLeft" ? -1 : 1);
        else moveSelectedNote(event.key === "ArrowUp" ? -1 : 1);
        return;
      }
      moveSelection(
        event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0,
        event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0,
        event.shiftKey
      );
      return;
    }
    if (event.key === "+" || event.key === "=") {
      handle();
      adjustDuration(true);
      return;
    }
    if (event.key === "-" || event.key === "_") {
      handle();
      adjustDuration(false);
      return;
    }
    if (event.key === "Delete") {
      handle();
      deleteSelectedNote();
      return;
    }
    if (event.key === "Backspace" && selectedNote) {
      handle();
      setCellFret(selectedCell.onset, selectedCell.string, Math.floor(selectedNote.fret / 10));
      digitBufferRef.current = null;
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      handle();
      void saveDraft();
      return;
    }
    if (event.key.toLowerCase() === "r") {
      handle();
      makeSelectionRest();
      return;
    }
    const shortcut = TECHNIQUES.find((technique) => technique.shortcut.toLowerCase() === event.key.toLowerCase());
    if (shortcut) {
      handle();
      applyTechnique(shortcut.id);
    } else if (event.key === "Escape") {
      event.stopPropagation();
      setArmedTechnique(null);
    }
  }

  function changeMeasure(nextMeasure: number) {
    if (!dirty || window.confirm("当前小节还有未保存修改。确定切换并丢弃这些修改吗？")) onMeasureChange(nextMeasure);
  }

  async function saveDraft() {
    if (!dirty || busy || saveInFlightRef.current) return;
    saveInFlightRef.current = true;
    try {
      await onSave(measureNumber, events);
    } finally {
      saveInFlightRef.current = false;
    }
  }

  async function retryMeasureRecognition() {
    if (busy || saveInFlightRef.current) return;
    if (dirty && !window.confirm("重新识别会替换当前小节尚未保存的草稿。确定继续吗？")) return;
    try {
      const proposal = await onRetryRecognition(measureNumber);
      const nextEvents = cloneEvents(proposal);
      setEvents(nextEvents);
      setDirty(true);
      const firstNote = nextEvents[0]?.notes[0];
      const firstOnset = nextEvents[0]?.onset_eighths ?? 0;
      setSelectedCell({ onset: firstOnset, string: firstNote?.string ?? 1 });
      setTimeSelection({ anchor: firstOnset, focus: firstOnset });
      setEntryDuration(nextEvents[0]?.duration_eighths ?? 1);
      setEditStatus(nextEvents.length
        ? `已重新识别出 ${nextEvents.length} 个节拍，请对照后保存`
        : "本次没有识别到音符，已作为全小节休止草稿；请确认后保存");
    } catch {
      // The workspace notice presents the API error.
    }
  }

  function addEvent() {
    const lastEnd = events.reduce((end, event) => Math.max(end, event.onset_eighths + event.duration_eighths), 0);
    if (lastEnd >= MEASURE_UNITS) return;
    const duration = Math.min(entryDuration, MEASURE_UNITS - lastEnd);
    markChanged([...events, { onset_eighths: lastEnd, duration_eighths: duration, notes: [{ string: selectedCell.string, fret: 0 }] }]);
    setSelectedCell({ onset: lastEnd, string: selectedCell.string });
    setTimeSelection({ anchor: lastEnd, focus: lastEnd });
  }

  function addNote(eventIndex: number) {
    const used = new Set(events[eventIndex].notes.map((note) => note.string));
    const nextString = [1, 2, 3, 4, 5, 6].find((value) => !used.has(value));
    if (!nextString) return;
    updateEvent(eventIndex, { notes: [...events[eventIndex].notes, { string: nextString, fret: 0 }] });
  }

  return (
    <div className={`score-review-shell ${embedded ? "embedded" : ""}`}>
      {!embedded && <section className="review-source-panel">
        <div className="review-panel-heading">
          <div><span>ORIGINAL FRAME</span><h3>原视频帧对照</h3></div>
          {sourceFrame && <div className="frame-time-badge"><Clock3 size={13} /> {formatTime(sourceFrame.time_seconds)} · {sourceFrame.time_seconds.toFixed(3)}s</div>}
        </div>
        {sourceFrame ? (
          <figure className="review-frame">
            <img src={sourceFrame.url} alt={`第 ${measureNumber} 小节对应的视频切片`} draggable={false} />
            <figcaption><span>源帧 {sourceFrame.source_frame}</span><span>切片 #{sourceFrame.sort_order + 1}</span></figcaption>
          </figure>
        ) : <div className="review-empty-frame">没有找到这个小节对应的原始切片</div>}
        <p className="review-guidance">左边保留原图证据；右边可细化到十六分音符、拖动框选时值并直接按数字键打谱。空白时间会导出为休止符；重新识别只生成草稿，保存后才会重建 MusicXML。</p>
      </section>}

      <section className="review-editor-panel" aria-busy={busy}>
        <fieldset className="review-editor-lock" disabled={busy}>
        <div className="measure-navigator">
          <button type="button" disabled={measureNumber <= startMeasure} onClick={() => changeMeasure(measureNumber - 1)} aria-label="上一小节"><ChevronLeft size={17} /></button>
          <label><span>小节</span><input type="number" min={startMeasure} max={endMeasure} value={measureNumber} onChange={(event) => changeMeasure(clamp(Number(event.target.value), startMeasure, endMeasure))} /></label>
          <small>{startMeasure}–{endMeasure}</small>
          <button type="button" disabled={measureNumber >= endMeasure} onClick={() => changeMeasure(measureNumber + 1)} aria-label="下一小节"><ChevronRight size={17} /></button>
        </div>

        <div className="tab-edit-workbench">
          <aside className="technique-toolbar" aria-label="演奏技巧工具栏">
            <div className="technique-toolbar-title"><span>TOOLS</span><strong>技巧</strong></div>
            {TECHNIQUES.map((technique) => (
              <button type="button" key={technique.id} className={selectedTechnique === technique.id ? "active" : ""} onClick={() => applyTechnique(technique.id)} title={`${technique.label} · 快捷键 ${technique.shortcut}`} aria-pressed={selectedTechnique === technique.id}>
                <b>{technique.mark}</b><span>{technique.label}</span><kbd>{technique.shortcut}</kbd>
              </button>
            ))}
          </aside>

          <div className="tab-grid-area">
            <div className="tab-grid-heading">
              <div><span>16TH-NOTE GRID</span><strong>拖动选择时值，数字键输入品位</strong></div>
              <small>{selectedNote ? `${selectedCell.string} 弦 · ${selectedNote.fret} 品 · ${durationLabel(events[selectedEventIndex].duration_eighths)}` : `${selectedCell.string} 弦 · 空位 · 新音 ${durationLabel(entryDuration)}`}</small>
            </div>
            <div className="rhythm-edit-bar" aria-label="音符时值控制">
              <div className="duration-stepper">
                <button type="button" onClick={() => adjustDuration(false)} title="延长音符，Guitar Pro 快捷键 -"><Minus size={13} /><span>延长</span><kbd>−</kbd></button>
                <button type="button" onClick={() => adjustDuration(true)} title="缩短音符，Guitar Pro 快捷键 +"><Plus size={13} /><span>缩短</span><kbd>+</kbd></button>
              </div>
              <div className="duration-palette">
                {POWER_OF_TWO_DURATIONS.map((value) => (
                  <button type="button" key={value} className={entryDuration === value ? "active" : ""} onClick={() => applyDuration(value)} aria-pressed={entryDuration === value}>{durationLabel(value)}</button>
                ))}
              </div>
              <button type="button" className="rest-command" onClick={makeSelectionRest}><b>𝄽</b><span>休止</span><kbd>R</kbd></button>
              <button type="button" className="retry-measure-command" disabled={busy || retrying} onClick={() => void retryMeasureRecognition()}><RefreshCw size={13} className={retrying ? "spin" : ""} /><span>{retrying ? "识别中" : "重识别本小节"}</span></button>
            </div>
            <div className="string-picker" aria-label="选择吉他弦">
              <span>弦</span>
              {[1, 2, 3, 4, 5, 6].map((string) => <button type="button" key={string} className={selectedCell.string === string ? "active" : ""} onClick={() => { setSelectedCell((current) => ({ ...current, string })); window.requestAnimationFrame(() => gridRef.current?.focus()); }} aria-pressed={selectedCell.string === string}>{string}</button>)}
              <small>{selectedEvents.length ? `已选 ${selectedEvents.length} 个节拍 · ${durationLabel(selectedDuration)}` : selectedSpan > GRID_STEP ? `已框选 ${selectedSpan}/8` : beatLabel(selectedCell.onset)}</small>
            </div>
            <div
              className="tab-entry-grid"
              ref={gridRef}
              tabIndex={0}
              data-shortcut-scope="review"
              onKeyDown={handleGridKeyDown}
              onPointerMove={moveCellSelection}
              onPointerUp={endCellSelection}
              onPointerCancel={endCellSelection}
              aria-label="十六分音符六线 TAB 键盘编辑器"
            >
              <div className="tab-corner">TAB</div>
              {Array.from({ length: GRID_SLOTS }, (_, slot) => {
                const onset = slot * GRID_STEP;
                return <div className={`tab-beat-label ${selectedStart <= onset && onset <= selectionVisualEnd ? "time-selected" : ""}`} key={`beat-${slot}`}>{beatLabel(onset)}</div>;
              })}
              {Array.from({ length: 6 }, (_, row) => row + 1).map((string) => (
                <div className="tab-string-row" key={`string-${string}`}>
                  <span className="tab-string-label">{string}</span>
                  {Array.from({ length: GRID_SLOTS }, (_, slot) => {
                    const onset = slot * GRID_STEP;
                    const exactEvent = events.find((event) => event.onset_eighths === onset);
                    const coveringEvent = events.find((event) => event.onset_eighths < onset && onset < event.onset_eighths + event.duration_eighths);
                    const note = exactEvent?.notes.find((item) => item.string === string);
                    const technique = note?.technique ? TECHNIQUES.find((item) => item.id === note.technique) : undefined;
                    const selected = selectedCell.onset === onset && selectedCell.string === string;
                    const timeSelected = selectedStart <= onset && onset <= selectionVisualEnd;
                    return (
                      <button
                        type="button"
                        key={`${string}-${slot}`}
                        data-grid-onset={onset}
                        className={`${selected ? "selected" : ""} ${timeSelected ? "time-selected" : ""} ${note ? "has-note" : ""} ${coveringEvent ? "sustain-cell" : ""}`}
                        onPointerDown={(event) => beginCellSelection(event, onset, string)}
                        onClick={(event) => { if (event.detail === 0) selectCell(onset, string, event.shiftKey); }}
                        aria-label={`${string} 弦，${beatLabel(onset)} 十六分位置${note ? `，${note.fret} 品` : coveringEvent ? "，延音范围" : "，空位"}`}
                      >
                        {note && <><b>{note.fret}</b>{technique && <i>{technique.mark}</i>}</>}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="grid-edit-status" role="status" aria-live="polite">{editStatus}</div>
            <div className="keyboard-hints"><span><kbd>0–9</kbd> 品位</span><span><kbd>↑↓←→</kbd> 游标</span><span><kbd>Shift ←→</kbd> 扩展选区</span><span><kbd>Alt ↑↓←→</kbd> 移动音符</span><span><kbd>+ / −</kbd> 缩短 / 延长</span><span><kbd>R</kbd> 休止</span><span><kbd>Ctrl S</kbd> 保存</span></div>
          </div>
        </div>

        <details className="advanced-event-editor">
          <summary>精确节奏与和弦事件</summary>
          <div className="event-editor-list">
            {events.length ? events.map((event, eventIndex) => (
              <article className="event-editor" key={`${measureNumber}-${eventIndex}`}>
                <header><strong>事件 {eventIndex + 1}</strong><div>
                  <label><span>起点</span><select value={event.onset_eighths} onChange={(change) => updateEvent(eventIndex, { onset_eighths: Number(change.target.value) })}>{Array.from({ length: GRID_SLOTS }, (_, slot) => { const value = slot * GRID_STEP; return <option value={value} key={value}>{beatLabel(value)}</option>; })}</select></label>
                  <label><span>时值</span><select value={event.duration_eighths} onChange={(change) => updateEvent(eventIndex, { duration_eighths: Number(change.target.value) })}>{DURATION_OPTIONS.map((value) => <option value={value} key={value}>{durationLabel(value)}</option>)}</select></label>
                  <button type="button" className="event-delete" onClick={() => markChanged(events.filter((_, index) => index !== eventIndex))} aria-label={`删除事件 ${eventIndex + 1}`}><Trash2 size={13} /></button>
                </div></header>
                <div className="note-editor-grid">
                  {event.notes.map((note, noteIndex) => (
                    <div className="note-editor" key={`${eventIndex}-${noteIndex}`}>
                      <label><span>弦</span><input type="number" min="1" max="6" value={note.string} onChange={(change) => updateNote(eventIndex, noteIndex, { string: Number(change.target.value) })} /></label>
                      <label><span>品</span><input type="number" min="0" max="36" value={note.fret} onChange={(change) => updateNote(eventIndex, noteIndex, { fret: Number(change.target.value) })} /></label>
                      <button type="button" disabled={event.notes.length <= 1} onClick={() => updateEvent(eventIndex, { notes: event.notes.filter((_, index) => index !== noteIndex) })} aria-label="删除这个音"><Trash2 size={12} /></button>
                    </div>
                  ))}
                  <button type="button" className="add-note-button" disabled={event.notes.length >= 6} onClick={() => addNote(eventIndex)}><Plus size={13} /> 和弦音</button>
                </div>
              </article>
            )) : <div className="empty-measure"><strong>当前是空小节</strong><span>点击六线格并输入数字即可补回音符。</span></div>}
          </div>
        </details>

        <footer className="review-actions">
          <button type="button" className="secondary-button" onClick={addEvent}><Plus size={15} /> 添加事件</button>
          <button type="button" className="primary-button" disabled={!dirty || busy} onClick={() => void saveDraft()}><Save size={15} /> {busy ? "正在保存…" : "保存并重建乐谱"}</button>
        </footer>
        </fieldset>
      </section>
    </div>
  );
}
