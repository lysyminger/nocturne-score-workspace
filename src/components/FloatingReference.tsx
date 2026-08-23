import { useRef, useState, type PointerEvent, type RefObject } from "react";
import { GripHorizontal, ImageIcon, Link2, Maximize2, Minimize2, Unlink, Video, X } from "lucide-react";

export type ReferenceMode = "video" | "image";

type ReferenceImage = {
  url: string;
  label: string;
  timeLabel: string;
};

type Props = {
  mode: ReferenceMode;
  videoUrl: string | null;
  image: ReferenceImage | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  syncAvailable: boolean;
  syncEnabled: boolean;
  onModeChange: (mode: ReferenceMode) => void;
  onSyncChange: (enabled: boolean) => void;
  onClose: () => void;
};

type DragState = {
  pointerId: number;
  offsetX: number;
  offsetY: number;
};

export function FloatingReference({
  mode,
  videoUrl,
  image,
  videoRef,
  syncAvailable,
  syncEnabled,
  onModeChange,
  onSyncChange,
  onClose
}: Props) {
  const panelRef = useRef<HTMLElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [maximized, setMaximized] = useState(false);

  function startDrag(event: PointerEvent<HTMLElement>) {
    if (maximized || event.button !== 0 || window.innerWidth <= 720) return;
    if ((event.target as HTMLElement).closest("button")) return;
    const panel = panelRef.current;
    if (!panel) return;
    const bounds = panel.getBoundingClientRect();
    dragRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - bounds.left,
      offsetY: event.clientY - bounds.top
    };
    setPosition({ x: bounds.left, y: bounds.top });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveDrag(event: PointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    const panel = panelRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !panel) return;
    const margin = 12;
    const x = Math.min(
      Math.max(margin, event.clientX - drag.offsetX),
      Math.max(margin, window.innerWidth - panel.offsetWidth - margin)
    );
    const y = Math.min(
      Math.max(margin, event.clientY - drag.offsetY),
      Math.max(margin, window.innerHeight - panel.offsetHeight - margin)
    );
    setPosition({ x, y });
  }

  function stopDrag(event: PointerEvent<HTMLElement>) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const shownMode = mode === "video" && videoUrl ? "video" : "image";

  return (
    <aside
      ref={panelRef}
      className={`floating-reference ${maximized ? "maximized" : ""}`}
      style={!maximized && position ? { left: position.x, top: position.y, right: "auto" } : undefined}
      aria-label="悬浮校对参考"
    >
      <header
        className="floating-reference-header"
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={stopDrag}
        onPointerCancel={stopDrag}
      >
        <GripHorizontal size={17} className="reference-grip" />
        <div>
          <strong>{shownMode === "video" ? "同步视频" : "识别原帧"}</strong>
          <span>{shownMode === "video" ? (syncEnabled ? "跟随谱面时间轴" : "自由检查模式") : image?.timeLabel || "当前小节证据"}</span>
        </div>
        <div className="reference-window-actions">
          <button type="button" onClick={() => setMaximized((value) => !value)} aria-label={maximized ? "还原参考窗" : "放大参考窗"}>
            {maximized ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
          <button type="button" onClick={onClose} aria-label="关闭参考窗"><X size={16} /></button>
        </div>
      </header>

      <div className="reference-mode-switch" role="group" aria-label="参考内容">
        <button type="button" className={shownMode === "video" ? "active" : ""} aria-pressed={shownMode === "video"} disabled={!videoUrl} onClick={() => onModeChange("video")}><Video size={14} /> 视频</button>
        <button type="button" className={shownMode === "image" ? "active" : ""} aria-pressed={shownMode === "image"} disabled={!image} onClick={() => onModeChange("image")}><ImageIcon size={14} /> 原帧</button>
        {shownMode === "video" && (
          <button
            type="button"
            className={`reference-sync-toggle ${syncEnabled ? "active" : ""}`}
            aria-pressed={syncEnabled}
            disabled={!syncAvailable}
            onClick={() => onSyncChange(!syncEnabled)}
            title={syncAvailable ? "让视频跟随谱面播放和定位" : "至少需要两个小节和时间都递增的有效同步点"}
          >
            {syncEnabled ? <Link2 size={14} /> : <Unlink size={14} />}
            {syncEnabled ? "已同步" : "自由播放"}
          </button>
        )}
      </div>

      <div className="reference-media">
        {shownMode === "video" && videoUrl ? (
          <video ref={videoRef} src={videoUrl} controls muted playsInline preload="metadata" aria-label="悬浮校对视频" />
        ) : image ? (
          <figure>
            <img src={image.url} alt={image.label} draggable={false} />
            <figcaption><span>{image.label}</span><time>{image.timeLabel}</time></figcaption>
          </figure>
        ) : (
          <div className="reference-empty">当前小节没有可用原帧</div>
        )}
      </div>
    </aside>
  );
}
