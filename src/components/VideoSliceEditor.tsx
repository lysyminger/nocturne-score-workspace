import {
  FormEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState
} from "react";
import { Crop, LoaderCircle, RotateCcw, Scissors, TimerReset } from "lucide-react";
import type { CropRegion, VideoAnalysis, VideoAnalysisRequest } from "../types";


const FULL_FRAME: CropRegion = {
  crop_x: 0,
  crop_y: 0,
  crop_width: 1,
  crop_height: 1
};

const MAX_FRAMES = 180;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function rounded(value: number, digits = 4) {
  return Number(value.toFixed(digits));
}

type DragState = {
  startX: number;
  startY: number;
  previous: CropRegion;
};

export function VideoSliceEditor({
  videoUrl,
  durationHint,
  analysis,
  busy,
  onAnalyze
}: {
  videoUrl: string;
  durationHint: number;
  analysis: VideoAnalysis | null;
  busy: boolean;
  onAnalyze: (request: VideoAnalysisRequest) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [duration, setDuration] = useState(Math.max(0, durationHint));
  const [currentTime, setCurrentTime] = useState(0);
  const [startTime, setStartTime] = useState(analysis?.start_seconds ?? 0);
  const [endTime, setEndTime] = useState(analysis?.end_seconds ?? Math.max(0, durationHint));
  const [frameInterval, setFrameInterval] = useState(analysis?.frame_interval ?? 60);
  const [crop, setCrop] = useState<CropRegion>(analysis ?? FULL_FRAME);
  const [cropMode, setCropMode] = useState(false);

  useEffect(() => {
    if (!analysis) return;
    setStartTime(analysis.start_seconds);
    setEndTime(analysis.end_seconds);
    setFrameInterval(analysis.frame_interval);
    setCrop({
      crop_x: analysis.crop_x,
      crop_y: analysis.crop_y,
      crop_width: analysis.crop_width,
      crop_height: analysis.crop_height
    });
  }, [analysis?.analysis_id]);

  function updateDuration(value: number) {
    if (!Number.isFinite(value) || value <= 0) return;
    setDuration(value);
    setEndTime((current) => (!analysis && (current <= 0 || current > value) ? value : Math.min(current, value)));
  }

  function seek(value: number) {
    const target = clamp(value, 0, duration || value);
    setCurrentTime(target);
    if (videoRef.current) videoRef.current.currentTime = target;
  }

  function normalizedPoint(event: ReactPointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
      y: clamp((event.clientY - bounds.top) / bounds.height, 0, 1)
    };
  }

  function beginCrop(event: ReactPointerEvent<HTMLDivElement>) {
    if (!cropMode) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    videoRef.current?.pause();
    const point = normalizedPoint(event);
    dragRef.current = { startX: point.x, startY: point.y, previous: crop };
    setCrop({ crop_x: point.x, crop_y: point.y, crop_width: 0, crop_height: 0 });
  }

  function moveCrop(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || !cropMode) return;
    const point = normalizedPoint(event);
    setCrop({
      crop_x: rounded(Math.min(drag.startX, point.x)),
      crop_y: rounded(Math.min(drag.startY, point.y)),
      crop_width: rounded(Math.abs(point.x - drag.startX)),
      crop_height: rounded(Math.abs(point.y - drag.startY))
    });
  }

  function finishCrop(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setCrop((current) =>
      current.crop_width < 0.02 || current.crop_height < 0.02 ? drag.previous : current
    );
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (invalidRange || invalidCrop || estimatedFrames > MAX_FRAMES) return;
    onAnalyze({
      start_seconds: rounded(startTime, 3),
      end_seconds: rounded(endTime, 3),
      frame_interval: Math.max(1, Math.round(frameInterval)),
      ...crop
    });
  }

  const sourceFps = analysis?.source_fps || 30;
  const estimatedFrames =
    endTime > startTime && frameInterval > 0
      ? Math.floor(((endTime - startTime) * sourceFps) / frameInterval) + 1
      : 0;
  const invalidRange = endTime - startTime < 0.25 || startTime < 0 || (duration > 0 && endTime > duration + 0.1);
  const invalidCrop = crop.crop_width < 0.02 || crop.crop_height < 0.02;
  const selectionStyle = {
    left: `${crop.crop_x * 100}%`,
    top: `${crop.crop_y * 100}%`,
    width: `${crop.crop_width * 100}%`,
    height: `${crop.crop_height * 100}%`
  };

  return (
    <form className="video-slice-editor" onSubmit={submit}>
      <div className="video-edit-canvas">
        <div className={`video-frame-shell ${cropMode ? "is-cropping" : ""}`}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            preload="metadata"
            playsInline
            onLoadedMetadata={(event) => updateDuration(event.currentTarget.duration)}
            onDurationChange={(event) => updateDuration(event.currentTarget.duration)}
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          />
          <div
            className="crop-interaction-layer"
            aria-label="拖动框选谱面区域"
            onPointerDown={beginCrop}
            onPointerMove={moveCrop}
            onPointerUp={finishCrop}
            onPointerCancel={finishCrop}
          >
            <div className="crop-selection" style={selectionStyle}>
              <span>谱面分析区域</span>
              <i className="corner corner-a" />
              <i className="corner corner-b" />
              <i className="corner corner-c" />
              <i className="corner corner-d" />
            </div>
          </div>
        </div>

        <div className="video-scrub-row">
          <time>{currentTime.toFixed(1)}s</time>
          <input
            type="range"
            min="0"
            max={Math.max(duration, 0.1)}
            step="0.05"
            value={Math.min(currentTime, Math.max(duration, 0.1))}
            onChange={(event) => seek(Number(event.target.value))}
            aria-label="视频时间"
          />
          <time>{duration.toFixed(1)}s</time>
        </div>

        <div className="video-mark-actions">
          <button type="button" onClick={() => setStartTime(rounded(currentTime, 3))}>
            <TimerReset size={15} /> 设为开始
          </button>
          <button type="button" onClick={() => setEndTime(rounded(currentTime, 3))}>
            <TimerReset size={15} /> 设为结束
          </button>
          <button
            type="button"
            className={cropMode ? "active" : ""}
            aria-pressed={cropMode}
            onClick={() => setCropMode((value) => !value)}
          >
            <Crop size={15} /> {cropMode ? "完成框选" : "框选谱面"}
          </button>
          <button type="button" onClick={() => setCrop(FULL_FRAME)}>
            <RotateCcw size={14} /> 全画面
          </button>
        </div>
      </div>

      <aside className="slice-settings">
        <div className="slice-heading">
          <span><Scissors size={17} /></span>
          <div><h3>切片分析</h3><p>先选时间，再框住谱面</p></div>
        </div>

        <div className="slice-time-grid">
          <label>
            <span>开始秒数</span>
            <input
              type="number"
              min="0"
              max={Math.max(0, endTime - 0.25)}
              step="any"
              value={startTime}
              onChange={(event) => setStartTime(Number(event.target.value))}
            />
          </label>
          <label>
            <span>结束秒数</span>
            <input
              type="number"
              min={startTime + 0.25}
              max={duration || undefined}
              step="any"
              value={endTime}
              onChange={(event) => setEndTime(Number(event.target.value))}
            />
          </label>
        </div>

        <label className="frame-interval-field">
          <span>每隔多少帧取 1 张</span>
          <input
            type="number"
            min="1"
            max="3600"
            step="1"
            value={frameInterval}
            onChange={(event) => setFrameInterval(Number(event.target.value))}
          />
          <small>
            当前按 {sourceFps.toFixed(2)} fps 估算，约 {estimatedFrames} 张候选帧
          </small>
        </label>

        <div className="crop-readout">
          <span>选区</span>
          <strong>{Math.round(crop.crop_width * 100)}% × {Math.round(crop.crop_height * 100)}%</strong>
          <small>左 {Math.round(crop.crop_x * 100)}% · 上 {Math.round(crop.crop_y * 100)}%</small>
        </div>

        {invalidRange && <p className="slice-validation">请选择至少 0.25 秒、且不超过视频时长的范围。</p>}
        {invalidCrop && <p className="slice-validation">请拖出一个足够大的谱面区域。</p>}
        {estimatedFrames > MAX_FRAMES && (
          <p className="slice-validation">最多生成 {MAX_FRAMES} 张，请缩短时间或增大帧间隔。</p>
        )}

        <button
          className="primary-button full analyze-button"
          type="submit"
          disabled={busy || invalidRange || invalidCrop || estimatedFrames < 1 || estimatedFrames > MAX_FRAMES}
        >
          {busy ? <LoaderCircle size={17} className="spin" /> : <Scissors size={17} />}
          {busy ? "正在分析视频…" : `开始分析约 ${estimatedFrames} 张`}
        </button>
        <p className="slice-footnote">这一版先输出可检查的裁剪帧，不会假装已经完成自动拼谱。</p>
      </aside>
    </form>
  );
}
