import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Clock3, Plus, Save, Trash2 } from "lucide-react";
import type { Project, RecognitionDiagnostics, RecognitionEvent, RecognitionMeasure, RecognitionNote, TabTechnique } from "../types";

type Props = {
  project: Project;
  diagnostics: RecognitionDiagnostics;
  measureNumber: number;
  busy: boolean;
  onMeasureChange: (measure: number) => void;
  onSave: (measure: number, events: RecognitionEvent[]) => Promise<void>;
};

type SelectedCell = { onset: number; string: number };

const DURATION_OPTIONS = [1, 2, 3, 4, 6, 8];
const TECHNIQUES: Array<{ id: TabTechnique; label: string; mark: string; shortcut: string }> = [
  { id: "legato", label: "连音", mark: "⌒", shortcut: "L" },
  { id: "slide", label: "滑音", mark: "/", shortcut: "S" },
  { id: "hammer_on", label: "击弦", mark: "H", shortcut: "H" },
  { id: "pull_off", label: "勾弦", mark: "P", shortcut: "P" },
  { id: "bend", label: "推弦", mark: "B", shortcut: "B" },
  { id: "vibrato", label: "颤音", mark: "~", shortcut: "V" },
  { id: "harmonic", label: "泛音", mark: "◇", shortcut: "N" },
  { id: "palm_mute", label: "闷音", mark: "PM", shortcut: "M" },
  { id: "let_ring", label: "延音", mark: "LR", shortcut: "R" },
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

export function ScoreReviewPanel({ project, diagnostics, measureNumber, busy, onMeasureChange, onSave }: Props) {
  const startMeasure = diagnostics.summary.start_measure ?? diagnostics.measures[0]?.number ?? 1;
  const endMeasure = diagnostics.summary.end_measure ?? diagnostics.measures.at(-1)?.number ?? startMeasure;
  const currentMeasure = diagnostics.measures.find((item) => item.number === measureNumber);
  const [events, setEvents] = useState<RecognitionEvent[]>(() => cloneEvents(currentMeasure));
  const [dirty, setDirty] = useState(false);
  const [selectedCell, setSelectedCell] = useState<SelectedCell>({ onset: 0, string: 1 });
  const [armedTechnique, setArmedTechnique] = useState<TabTechnique | null>(null);
  const digitBufferRef = useRef<{ key: string; value: string; time: number } | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const nextEvents = cloneEvents(currentMeasure);
    setEvents(nextEvents);
    setDirty(false);
    const firstNote = nextEvents[0]?.notes[0];
    setSelectedCell({ onset: nextEvents[0]?.onset_eighths ?? 0, string: firstNote?.string ?? 1 });
    digitBufferRef.current = null;
  }, [measureNumber, diagnostics]);

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

  function markChanged(nextEvents: RecognitionEvent[]) {
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

  function selectCell(onset: number, string: number) {
    setSelectedCell({ onset: editableOnset(onset), string });
    window.requestAnimationFrame(() => gridRef.current?.focus());
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
      markChanged([...events, {
        onset_eighths: targetOnset,
        duration_eighths: 1,
        notes: [{ string, fret, ...(armedTechnique ? { technique: armedTechnique } : {}) }]
      }]);
    }
    setSelectedCell({ onset: targetOnset, string });
  }

  function deleteSelectedNote() {
    if (selectedEventIndex < 0) return;
    const remaining = events[selectedEventIndex].notes.filter((note) => note.string !== selectedCell.string);
    markChanged(remaining.length
      ? events.map((event, index) => index === selectedEventIndex ? { ...event, notes: remaining } : event)
      : events.filter((_, index) => index !== selectedEventIndex));
    digitBufferRef.current = null;
  }

  function applyTechnique(technique: TabTechnique) {
    const removing = selectedNote?.technique === technique;
    setArmedTechnique(removing ? null : technique);
    if (selectedEventIndex >= 0 && selectedNote) {
      const noteIndex = events[selectedEventIndex].notes.findIndex((note) => note.string === selectedCell.string);
      updateNote(selectedEventIndex, noteIndex, { technique: removing ? undefined : technique });
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

  function moveSelection(horizontal: number, vertical: number) {
    const onset = clamp(selectedCell.onset + horizontal, 0, 7);
    const string = clamp(selectedCell.string + vertical, 1, 6);
    setSelectedCell({ onset: editableOnset(onset), string });
    digitBufferRef.current = null;
  }

  function handleGridKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (/^\d$/.test(event.key)) {
      event.preventDefault();
      writeDigit(event.key);
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      moveSelection(
        event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0,
        event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0
      );
      return;
    }
    if (event.key === "Delete") {
      event.preventDefault();
      deleteSelectedNote();
      return;
    }
    if (event.key === "Backspace" && selectedNote) {
      event.preventDefault();
      setCellFret(selectedCell.onset, selectedCell.string, Math.floor(selectedNote.fret / 10));
      digitBufferRef.current = null;
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      if (dirty && !busy) void onSave(measureNumber, events);
      return;
    }
    const shortcut = TECHNIQUES.find((technique) => technique.shortcut.toLowerCase() === event.key.toLowerCase());
    if (shortcut) {
      event.preventDefault();
      applyTechnique(shortcut.id);
    } else if (event.key === "Escape") {
      setArmedTechnique(null);
    }
  }

  function addEvent() {
    const lastEnd = events.reduce((end, event) => Math.max(end, event.onset_eighths + event.duration_eighths), 0);
    if (lastEnd >= 8) return;
    markChanged([...events, { onset_eighths: lastEnd, duration_eighths: 1, notes: [{ string: 1, fret: 0 }] }]);
    setSelectedCell({ onset: lastEnd, string: 1 });
  }

  function addNote(eventIndex: number) {
    const used = new Set(events[eventIndex].notes.map((note) => note.string));
    const nextString = [1, 2, 3, 4, 5, 6].find((value) => !used.has(value));
    if (!nextString) return;
    updateEvent(eventIndex, { notes: [...events[eventIndex].notes, { string: nextString, fret: 0 }] });
  }

  return (
    <div className="score-review-shell">
      <section className="review-source-panel">
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
        <p className="review-guidance">左边保留原图证据；右边点击六线格后直接按数字键修改品位。保存会重建 MusicXML，不改动原始切片和完整 PDF。</p>
      </section>

      <section className="review-editor-panel">
        <div className="measure-navigator">
          <button type="button" disabled={measureNumber <= startMeasure} onClick={() => onMeasureChange(measureNumber - 1)} aria-label="上一小节"><ChevronLeft size={17} /></button>
          <label><span>小节</span><input type="number" min={startMeasure} max={endMeasure} value={measureNumber} onChange={(event) => onMeasureChange(clamp(Number(event.target.value), startMeasure, endMeasure))} /></label>
          <small>{startMeasure}–{endMeasure}</small>
          <button type="button" disabled={measureNumber >= endMeasure} onClick={() => onMeasureChange(measureNumber + 1)} aria-label="下一小节"><ChevronRight size={17} /></button>
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
              <div><span>8TH-NOTE GRID</span><strong>点击弦格，键盘输入品位</strong></div>
              <small>{selectedNote ? `${selectedCell.string} 弦 · ${selectedNote.fret} 品` : `${selectedCell.string} 弦 · 空位`}</small>
            </div>
            <div className="tab-entry-grid" ref={gridRef} tabIndex={0} onKeyDown={handleGridKeyDown} aria-label="六线 TAB 键盘编辑器">
              <div className="tab-corner">TAB</div>
              {Array.from({ length: 8 }, (_, onset) => <div className="tab-beat-label" key={`beat-${onset}`}>{Math.floor(onset / 2) + 1}{onset % 2 ? "&" : ""}</div>)}
              {Array.from({ length: 6 }, (_, row) => row + 1).map((string) => (
                <div className="tab-string-row" key={`string-${string}`}>
                  <span className="tab-string-label">{string}</span>
                  {Array.from({ length: 8 }, (_, onset) => {
                    const exactEvent = events.find((event) => event.onset_eighths === onset);
                    const coveringEvent = events.find((event) => event.onset_eighths < onset && onset < event.onset_eighths + event.duration_eighths);
                    const note = exactEvent?.notes.find((item) => item.string === string);
                    const technique = note?.technique ? TECHNIQUES.find((item) => item.id === note.technique) : undefined;
                    const selected = selectedCell.onset === onset && selectedCell.string === string;
                    return (
                      <button type="button" key={`${string}-${onset}`} className={`${selected ? "selected" : ""} ${note ? "has-note" : ""} ${coveringEvent ? "sustain-cell" : ""}`} onClick={() => selectCell(onset, string)} aria-label={`${string} 弦，第 ${onset + 1} 个八分位置${note ? `，${note.fret} 品` : ""}`}>
                        {note && <><b>{note.fret}</b>{technique && <i>{technique.mark}</i>}</>}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="keyboard-hints"><span><kbd>0–9</kbd> 品位</span><span><kbd>↑↓←→</kbd> 移动</span><span><kbd>Del</kbd> 删除</span><span><kbd>Ctrl S</kbd> 保存</span></div>
          </div>
        </div>

        <details className="advanced-event-editor">
          <summary>精确节奏与和弦事件</summary>
          <div className="event-editor-list">
            {events.length ? events.map((event, eventIndex) => (
              <article className="event-editor" key={`${measureNumber}-${eventIndex}`}>
                <header><strong>事件 {eventIndex + 1}</strong><div>
                  <label><span>起点</span><select value={event.onset_eighths} onChange={(change) => updateEvent(eventIndex, { onset_eighths: Number(change.target.value) })}>{Array.from({ length: 8 }, (_, value) => <option value={value} key={value}>{value + 1}/8</option>)}</select></label>
                  <label><span>时值</span><select value={event.duration_eighths} onChange={(change) => updateEvent(eventIndex, { duration_eighths: Number(change.target.value) })}>{DURATION_OPTIONS.map((value) => <option value={value} key={value}>{value}/8</option>)}</select></label>
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
          <button type="button" className="primary-button" disabled={!dirty || busy} onClick={() => onSave(measureNumber, events)}><Save size={15} /> {busy ? "正在保存…" : "保存并重建乐谱"}</button>
        </footer>
      </section>
    </div>
  );
}
