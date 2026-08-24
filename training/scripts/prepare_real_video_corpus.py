from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.tab_recognition import FrameInput, parse_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从服务器视频切片提取真实品位候选，运行现有模型并生成待人工核验页面。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("training/data/real-video"),
        help="从服务器同步下来的项目根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/real-video-token-review"),
        help="候选小图、清单和校对页输出目录",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("training/models/fret-ocr-inference"),
        help="已导出的 PaddleOCR 品位识别模型目录",
    )
    parser.add_argument("--stride", type=int, default=4, help="每个分析目录每隔多少帧抽一帧")
    parser.add_argument("--frame-seconds", type=float, default=2.0, help="相邻切片的秒数间隔")
    parser.add_argument("--padding", type=int, default=4, help="品位候选框四周扩展像素")
    parser.add_argument("--batch-size", type=int, default=256, help="模型推理批量")
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default="gpu", help="现有模型推理设备"
    )
    parser.add_argument(
        "--max-frames-per-run",
        type=int,
        default=0,
        help="每个分析目录最多处理多少帧，0 表示不限制",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_directories(source: Path) -> list[Path]:
    directories = {path.parent for path in source.rglob("frame-0001.*") if path.is_file()}
    return sorted(directories, key=lambda path: path.as_posix().lower())


def load_project_metadata(source: Path) -> dict[str, dict]:
    projects: dict[str, dict] = {}
    for manifest_path in source.rglob("projects.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = payload.get("projects", {})
        if isinstance(values, dict):
            projects.update(
                (str(project_id), value)
                for project_id, value in values.items()
                if isinstance(value, dict)
            )
    return projects


def frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def selected_frames(directory: Path, stride: int, limit: int) -> list[Path]:
    frames = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and path.stem.startswith("frame-")
        ),
        key=frame_number,
    )
    selected = frames[::stride]
    if frames and selected and selected[-1] != frames[-1]:
        selected.append(frames[-1])
    return selected[:limit] if limit > 0 else selected


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = box
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(width, x + box_width + padding),
        min(height, y + box_height + padding),
    )


def token_feature(glyphs: list) -> np.ndarray:
    canvas = np.zeros((28, 44), dtype=np.float32)
    if len(glyphs) == 1:
        canvas[:, 12:32] = glyphs[0].feature
    else:
        for index, glyph in enumerate(glyphs[:2]):
            left = index * 24
            canvas[:, left : left + 20] = glyph.feature
    return canvas


class FretRecognizer:
    characters = "0123456789"

    def __init__(self, model_dir: Path, device: str) -> None:
        from paddle import inference

        model_file = model_dir / "inference.json"
        params_file = model_dir / "inference.pdiparams"
        if not model_file.is_file() or not params_file.is_file():
            raise FileNotFoundError(f"找不到推理模型：{model_dir}")
        config = inference.Config(str(model_file), str(params_file))
        if device == "gpu":
            config.enable_use_gpu(512, 0)
        else:
            config.disable_gpu()
        config.switch_ir_optim(True)
        self.predictor = inference.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        target_height, target_width = 32, 100
        height, width = image.shape[:2]
        resized_width = min(target_width, max(1, math.ceil(target_height * width / max(height, 1))))
        resized = cv2.resize(image, (resized_width, target_height)).astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5
        canvas = np.zeros((3, target_height, target_width), dtype=np.float32)
        canvas[:, :, :resized_width] = resized
        return canvas

    def predict(self, paths: list[Path], batch_size: int) -> list[tuple[str, float]]:
        predictions: list[tuple[str, float]] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"无法读取候选小图：{path}")
                images.append(self.preprocess(image))
            tensor = np.stack(images)
            self.input_handle.reshape(tensor.shape)
            self.input_handle.copy_from_cpu(tensor)
            self.predictor.run()
            output = self.output_handle.copy_to_cpu()
            for sequence in output:
                indices = np.argmax(sequence, axis=1)
                previous = -1
                text: list[str] = []
                confidences: list[float] = []
                for position, index_value in enumerate(indices):
                    index = int(index_value)
                    if index != 0 and index != previous and index - 1 < len(self.characters):
                        text.append(self.characters[index - 1])
                        confidences.append(float(sequence[position, index]))
                    previous = index
                predictions.append(("".join(text), float(np.mean(confidences)) if confidences else 0.0))
        return predictions


def build_review_html(rows: list[dict], output: Path) -> None:
    public_rows = [
        {
            "id": row["id"],
            "image": row["normalized_image"],
            "original": row["original_image"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "source": row["source_frame"],
            "sourceId": row["source_id"],
            "time": row["time_seconds"],
            "string": row["string"],
            "layout": row["layout"],
            "polarity": row["polarity"],
            "members": row["member_images"],
            "memberCount": row["member_count"],
            "agreement": row["agreement"],
        }
        for row in rows
    ]
    data = json.dumps(public_rows, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实视频品位簇校对</title><style>
:root{color-scheme:dark;font-family:Inter,"SF Pro Display","PingFang SC",sans-serif;background:#09090d;color:#f5f3ff}
*{box-sizing:border-box}body{margin:0}header{position:sticky;top:0;z-index:4;padding:14px 20px;background:#111018eF;backdrop-filter:blur(22px);border-bottom:1px solid #292536}
.top{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.muted{color:#9d96ae;font-size:13px}button,input{font:inherit}button{border:1px solid #4f426d;background:#201a2d;color:#f8f3ff;border-radius:10px;padding:8px 12px;cursor:pointer}button.primary{background:#7857db;border-color:#8e70ef}
.grid{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}.card{border:1px solid #292536;background:#111018;border-radius:13px;padding:9px}.card.verified{border-color:#8a66ef;box-shadow:0 0 0 1px #8a66ef55 inset}
.crop{height:72px;width:100%;object-fit:contain;background:#f7f7f7;border-radius:8px;image-rendering:auto}.meta{font-size:11px;color:#9d96ae;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.edit{display:flex;gap:6px;margin-top:7px}.edit input{width:52px;border:1px solid #4f426d;background:#09090d;color:white;border-radius:8px;padding:7px;text-align:center}.suggest{flex:1;padding:7px}.pager{margin-left:auto}.status{min-width:150px}
</style></head><body><header><div class="top"><strong>真实视频品位簇校对</strong><span class="muted status" id="status"></span><button id="accept">接受本页高一致建议</button><button class="primary" id="export">导出已核验 TSV</button><span class="pager"><button id="prev">上一页</button> <span id="page"></span> <button id="next">下一页</button></span></div><div class="muted">一张代表图对应同字体的多个候选。可填 0–36；x 表示死音；- 表示误检。修改自动保存在浏览器本机。</div></header><main class="grid" id="grid"></main><script>
const rows=__ROWS__,size=160,key='nocturne-real-token-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{}');let page=0;
const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function persist(){localStorage.setItem(key,JSON.stringify(saved))}function valid(v){return /^(?:(?:[0-9]|[12][0-9]|3[0-6])|x|-)$/.test(v)}
function render(){const first=page*size,last=Math.min(rows.length,first+size),grid=document.querySelector('#grid');grid.innerHTML='';for(let i=first;i<last;i++){const r=rows[i],value=saved[r.id]??'',card=document.createElement('article');card.className='card'+(valid(value)?' verified':'');card.innerHTML=`<img class="crop" src="${esc(r.image)}" title="点击切换原图/归一化图"><div class="meta" title="${esc(r.source)}">${r.memberCount} 个 · 一致 ${(r.agreement*100).toFixed(0)}% · ${esc(r.sourceId)} · ${r.time.toFixed(1)}s · ${r.layout} · ${r.polarity} · ${r.string}弦</div><div class="edit"><button class="suggest" title="采用模型建议">建议 ${esc(r.prediction||'—')} · ${(r.confidence*100).toFixed(1)}%</button><input maxlength="2" value="${esc(value)}" placeholder="值/x/-"></div>`;const img=card.querySelector('img');let original=false;img.onclick=()=>{original=!original;img.src=original?r.original:r.image};card.querySelector('.suggest').onclick=()=>{if(r.prediction){saved[r.id]=r.prediction;persist();render()}};const input=card.querySelector('input');input.oninput=()=>{const v=input.value.trim().toLowerCase();if(v==='')delete saved[r.id];else saved[r.id]=v;persist();card.classList.toggle('verified',valid(v));status()};grid.appendChild(card)}document.querySelector('#page').textContent=`${page+1}/${Math.max(1,Math.ceil(rows.length/size))}`;status()}
function status(){const done=Object.values(saved).filter(valid).length;document.querySelector('#status').textContent=`已核验 ${done}/${rows.length}`}
document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render();scrollTo(0,0)};document.querySelector('#next').onclick=()=>{page=Math.min(Math.ceil(rows.length/size)-1,page+1);render();scrollTo(0,0)};
document.querySelector('#accept').onclick=()=>{const first=page*size,last=Math.min(rows.length,first+size);for(let i=first;i<last;i++){const r=rows[i];if(r.confidence>=.90&&r.agreement>=.90&&valid(r.prediction))saved[r.id]=r.prediction}persist();render()};
document.querySelector('#export').onclick=()=>{const lines=['normalized_image\\tlabel\\tcluster_id\\tsource_frame'];for(const r of rows){const value=saved[r.id];if(!valid(value)||value==='-')continue;for(const image of r.members)lines.push([image,value,r.id,r.source].join('\\t'))}const blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/tab-separated-values'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='verified-labels.tsv';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>""".replace("__ROWS__", data)
    (output / "review.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride 必须大于等于 1")
    source = args.source.resolve()
    output = args.output.resolve()
    project_metadata = load_project_metadata(source)
    normalized_root = output / "images" / "normalized"
    original_root = output / "images" / "original"
    normalized_root.mkdir(parents=True, exist_ok=True)
    original_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    duplicates: list[dict] = []
    seen_frame_hashes: dict[str, str] = {}
    run_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    polarity_counts: Counter[str] = Counter()
    clusters_by_run: dict[str, list[dict]] = {}

    for run_dir in analysis_directories(source):
        try:
            analysis_index = run_dir.parts.index("analysis")
            project_id = run_dir.parts[analysis_index - 1]
        except (ValueError, IndexError):
            project_id = run_dir.parent.parent.name
        analysis_id = run_dir.name
        metadata = project_metadata.get(project_id, {})
        source_id = str(metadata.get("source_id") or project_id)
        frames = selected_frames(run_dir, args.stride, args.max_frames_per_run)
        for frame_path in frames:
            number = frame_number(frame_path)
            time_seconds = max(0, number - 1) * args.frame_seconds
            relative_source = frame_path.relative_to(source).as_posix()
            try:
                source_hash = sha256_file(frame_path)
                duplicate_of = seen_frame_hashes.get(source_hash)
                if duplicate_of is not None:
                    duplicates.append(
                        {
                            "source_frame": relative_source,
                            "duplicate_of": duplicate_of,
                            "source_sha256": source_hash,
                        }
                    )
                    continue
                frame = parse_frame(FrameInput(frame_path, time_seconds, max(0, number - 1)))
                original = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if original is None:
                    raise ValueError("OpenCV 无法读取图片")
                seen_frame_hashes[source_hash] = relative_source
                normalized = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
                height, width = original.shape[:2]
                run_key = f"{project_id}/{analysis_id}"
                run_counts[run_key] += 1
                layout_counts[frame.layout] += 1
                polarity_counts[frame.polarity] += 1
                for token_index, token in enumerate(frame.tokens):
                    boxes = [glyph.box for glyph in token.glyphs if glyph.box is not None]
                    if not boxes or len(token.glyphs) > 2:
                        continue
                    token_box = (
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes),
                        max(box[0] + box[2] for box in boxes) - min(box[0] for box in boxes),
                        max(box[1] + box[3] for box in boxes) - min(box[1] for box in boxes),
                    )
                    left, top, right, bottom = clamp_box(
                        token_box, width, height, args.padding
                    )
                    original_crop = original[top:bottom, left:right]
                    normalized_crop = normalized[top:bottom, left:right]
                    if original_crop.size == 0 or normalized_crop.size == 0:
                        continue
                    identity = (
                        f"{project_id}/{analysis_id}/{frame_path.name}/"
                        f"s{token.string}/t{token_index}"
                    )
                    candidate_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                    clusters = clusters_by_run.setdefault(run_key, [])
                    feature = token_feature(token.glyphs)
                    best_index = -1
                    best_distance = math.inf
                    for cluster_index, cluster in enumerate(clusters):
                        distance = float(np.mean(feature != cluster["prototype"]))
                        if distance < best_distance:
                            best_index, best_distance = cluster_index, distance
                    if best_index >= 0 and best_distance <= 0.075:
                        cluster = clusters[best_index]
                        cluster["count"] += 1
                        cluster["feature_sum"] += feature
                        cluster["prototype"] = (
                            cluster["feature_sum"] >= cluster["count"] / 2
                        ).astype(np.float32)
                    else:
                        best_index = len(clusters)
                        clusters.append(
                            {
                                "count": 1,
                                "feature_sum": feature.copy(),
                                "prototype": feature.copy(),
                            }
                        )
                    cluster_id = hashlib.sha256(
                        f"{run_key}/cluster-{best_index}".encode("utf-8")
                    ).hexdigest()[:20]
                    relative_dir = Path(project_id) / analysis_id
                    normalized_path = normalized_root / relative_dir / f"{candidate_id}.png"
                    original_path = original_root / relative_dir / f"{candidate_id}.png"
                    normalized_path.parent.mkdir(parents=True, exist_ok=True)
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(normalized_path), normalized_crop)
                    cv2.imwrite(str(original_path), original_crop)
                    rows.append(
                        {
                            "id": candidate_id,
                            "project_id": project_id,
                            "source_id": source_id,
                            "analysis_id": analysis_id,
                            "source_frame": relative_source,
                            "source_sha256": source_hash,
                            "time_seconds": time_seconds,
                            "layout": frame.layout,
                            "polarity": frame.polarity,
                            "string": token.string,
                            "token_x": round(float(token.x), 3),
                            "token_index": token_index,
                            "glyph_count": len(token.glyphs),
                            "cluster_id": cluster_id,
                            "box": [left, top, right - left, bottom - top],
                            "normalized_image": normalized_path.relative_to(output).as_posix(),
                            "original_image": original_path.relative_to(output).as_posix(),
                        }
                    )
            except (OSError, ValueError, cv2.error) as exc:
                failures.append({"source_frame": relative_source, "error": str(exc)[:300]})

    recognizer = FretRecognizer(args.model.resolve(), args.device)
    prediction_paths = [output / row["normalized_image"] for row in rows]
    predictions = recognizer.predict(prediction_paths, args.batch_size)
    for row, (prediction, confidence) in zip(rows, predictions, strict=True):
        row["prediction"] = prediction if prediction.isdigit() and int(prediction) <= 36 else ""
        row["confidence"] = round(confidence, 6)
        row["needs_review"] = not row["prediction"] or confidence < 0.985

    members_by_cluster: dict[str, list[dict]] = {}
    for row in rows:
        members_by_cluster.setdefault(row["cluster_id"], []).append(row)
    cluster_rows: list[dict] = []
    for cluster_id, members in members_by_cluster.items():
        votes: dict[str, float] = {}
        counts: Counter[str] = Counter()
        for member in members:
            prediction = member["prediction"]
            if prediction:
                votes[prediction] = votes.get(prediction, 0.0) + member["confidence"]
                counts[prediction] += 1
        winner = max(votes, key=votes.get) if votes else ""
        winner_members = [member for member in members if member["prediction"] == winner]
        representative = max(
            winner_members or members,
            key=lambda member: (member["confidence"], member["box"][2] * member["box"][3]),
        )
        cluster_rows.append(
            {
                **representative,
                "id": cluster_id,
                "prediction": winner,
                "confidence": round(
                    float(np.mean([member["confidence"] for member in winner_members])), 6
                )
                if winner_members
                else 0.0,
                "agreement": round(counts[winner] / len(members), 6) if winner else 0.0,
                "member_count": len(members),
                "member_ids": [member["id"] for member in members],
                "member_images": [member["normalized_image"] for member in members],
            }
        )
    cluster_rows.sort(
        key=lambda row: (
            -row["member_count"],
            row["source_id"],
            row["project_id"],
            row["analysis_id"],
            row["id"],
        )
    )

    with (output / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "review-template.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "representative_image\tlabel\tprediction\tconfidence\tagreement\tmembers"
            "\tcluster_id\tsource_frame\n"
        )
        for row in cluster_rows:
            handle.write(
                f"{row['normalized_image']}\t\t{row['prediction']}\t{row['confidence']:.6f}"
                f"\t{row['agreement']:.6f}\t{row['member_count']}\t{row['id']}"
                f"\t{row['source_frame']}\n"
            )
    with (output / "clusters.jsonl").open("w", encoding="utf-8") as handle:
        for row in cluster_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "duplicates.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    build_review_html(cluster_rows, output)

    confidence_values = [row["confidence"] for row in rows]
    summary = {
        "schema_version": 1,
        "source": str(source),
        "sampling_stride": args.stride,
        "analysis_runs": len(run_counts),
        "source_ids": sorted({row["source_id"] for row in rows}),
        "parsed_frames": sum(run_counts.values()),
        "failed_frames": len(failures),
        "duplicate_frames_skipped": len(duplicates),
        "candidates": len(rows),
        "clusters_to_review": len(cluster_rows),
        "cluster_reduction_ratio": round(1 - len(cluster_rows) / len(rows), 6) if rows else 0.0,
        "top_100_cluster_coverage": round(
            sum(row["member_count"] for row in cluster_rows[:100]) / len(rows), 6
        )
        if rows
        else 0.0,
        "high_confidence_predictions": sum(
            bool(row["prediction"]) and row["confidence"] >= 0.985 for row in rows
        ),
        "mean_prediction_confidence": round(float(np.mean(confidence_values)), 6)
        if confidence_values
        else 0.0,
        "layout_counts": dict(layout_counts),
        "polarity_counts": dict(polarity_counts),
        "runs": dict(run_counts),
        "warning": "prediction 仅是现有合成域模型建议，不是真值；只有人工核验导出的标签才能进入训练集。",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
