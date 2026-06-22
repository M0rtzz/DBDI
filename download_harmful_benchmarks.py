from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from harmful_benchmark_eval import (
    HARMFUL_BENCH_SPECS,
    PAPER_HARMFUL_BENCHMARKS,
    expand_harmful_benchmark_names,
    load_harmful_benchmark_records,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configure_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(args.hf_hub_download_timeout)))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(args.hf_hub_etag_timeout)))
    if args.hf_cache_dir:
        hf_cache_dir = Path(args.hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir / "datasets"))
    if args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and materialize COMBAT-style harmful benchmarks without importing COMBAT."
    )
    parser.add_argument(
        "--harmful-benchmarks",
        nargs="+",
        default=["all"],
        help="Harmful benchmarks to download. Supports all.",
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=Path("outputs/benchmark_cache"),
        help="Cache directory for direct URL downloads.",
    )
    parser.add_argument(
        "--write-jsonl-dir",
        type=Path,
        default=Path("outputs/downloaded_benchmarks"),
        help="Directory where normalized benchmark JSONL files are written.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-hub-download-timeout", type=int, default=300)
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=60)
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_env(args)

    benchmarks = expand_harmful_benchmark_names(args.harmful_benchmarks)
    args.benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
    args.write_jsonl_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in benchmarks:
        print(f"[download] harmful: {name}", flush=True)
        records_by_name, errors_by_name = load_harmful_benchmark_records(
            [name],
            cache_dir=args.benchmark_cache_dir,
            timeout_seconds=int(args.timeout_seconds),
            max_examples=0,
            seed=0,
            require_external=False,
        )
        records = records_by_name.get(name, [])
        errors = errors_by_name.get(name, [])
        result = {
            "kind": "harmful",
            "name": name,
            "count": len(records),
            "errors": errors,
            "jsonl_path": None,
            "ok": bool(records),
            "spec": HARMFUL_BENCH_SPECS.get(str(name).lower(), {}),
        }
        if records:
            jsonl_path = args.write_jsonl_dir / "harmful" / f"{name}.jsonl"
            _write_jsonl(jsonl_path, records)
            result["jsonl_path"] = str(jsonl_path)
            print(f"[ok] harmful: {name}; records={len(records)}; jsonl={jsonl_path}", flush=True)
        else:
            print(f"[failed] harmful: {name}; errors={errors}", flush=True)
        results.append(result)

    manifest = {
        "results": results,
        "paper_harmful_benchmarks": PAPER_HARMFUL_BENCHMARKS,
        "benchmark_cache_dir": str(args.benchmark_cache_dir),
        "write_jsonl_dir": str(args.write_jsonl_dir),
        "hf_env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"),
            "HF_HUB_ETAG_TIMEOUT": os.environ.get("HF_HUB_ETAG_TIMEOUT"),
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET"),
        },
    }
    manifest_path = args.write_jsonl_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [row for row in results if not row.get("ok")]
    print(f"[summary] ok={len(results) - len(failed)} failed={len(failed)} manifest={manifest_path}", flush=True)
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
