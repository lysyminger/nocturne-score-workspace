from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pdf_score_import import render_and_segment_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把私人 PDF 乐谱渲染为页面图与谱表切片。")
    parser.add_argument("--source", type=Path, default=Path(r"D:\document\谱子"))
    parser.add_argument("--output", type=Path, default=Path("training/data/private-pdf"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--match", help="只处理文件名包含此文字的 PDF")
    return parser.parse_args()


def source_split(source_id: str) -> str:
    bucket = int(source_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"源目录不存在: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(source_root.rglob("*.pdf"), key=lambda item: item.name.casefold())
    if args.match:
        files = [item for item in files if args.match in item.name]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit 必须是正整数")
        files = files[: args.limit]

    entries: list[dict] = []
    for index, pdf_path in enumerate(files, start=1):
        source_bytes = pdf_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        source_id = source_sha256[:16]
        source_output = output_root / source_id
        print(f"[{index}/{len(files)}] {pdf_path.name}", flush=True)
        try:
            result = render_and_segment_pdf(pdf_path, source_output)
            entry = {
                "schema_version": 1,
                "source_id": source_id,
                "source_path": str(pdf_path.resolve()),
                "source_sha256": source_sha256,
                "source_size": len(source_bytes),
                "split": source_split(source_id),
                "status": "ok",
                "page_count": len(result.pages),
                "system_count": len(result.systems),
                "layout_counts": result.layout_counts,
                "pages": [
                    {
                        "page_number": page.page_number,
                        "width": page.width,
                        "height": page.height,
                        "image": relative_path(page.path, output_root),
                    }
                    for page in result.pages
                ],
                "systems": [
                    {
                        "page_number": system.page_number,
                        "system_number": system.system_number,
                        "layout": system.layout,
                        "polarity": system.polarity,
                        "image": relative_path(system.path, output_root),
                    }
                    for system in result.systems
                ],
            }
        except Exception as error:  # batch job: one damaged PDF must not stop the corpus
            entry = {
                "schema_version": 1,
                "source_id": source_id,
                "source_path": str(pdf_path.resolve()),
                "source_sha256": source_sha256,
                "source_size": len(source_bytes),
                "split": source_split(source_id),
                "status": "error",
                "error": str(error),
            }
            print(f"  失败: {error}", flush=True)
        source_output.mkdir(parents=True, exist_ok=True)
        (source_output / "source.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entries.append(entry)

    (output_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )
    success_count = sum(entry["status"] == "ok" for entry in entries)
    page_count = sum(entry.get("page_count", 0) for entry in entries)
    system_count = sum(entry.get("system_count", 0) for entry in entries)
    print(f"完成: {success_count}/{len(entries)} 个 PDF，{page_count} 页，{system_count} 个谱表切片")


if __name__ == "__main__":
    main()
