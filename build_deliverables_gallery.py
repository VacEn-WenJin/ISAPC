#!/usr/bin/env python3
"""
Build a simple HTML gallery for FINAL_DELIVERABLES to review normalized spectra
and AIP plots per galaxy.

Outputs: FINAL_DELIVERABLES/index.html
"""
from __future__ import annotations

import csv
import glob
import html
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DELIV = ROOT / "FINAL_DELIVERABLES"


def find_images(gal: str) -> dict[str, list[str]]:
    patterns = {
        "norm": f"**/{gal}*P2P_spectrum_norm_*.png",
        "aip_map": f"**/{gal}*alpha*map*.png",
        "aip_profile": f"**/{gal}*alpha*profile*.png",
        "other": f"**/{gal}*.png",
    }
    results: dict[str, list[str]] = {k: [] for k in patterns}
    for key, pat in patterns.items():
        # glob relative to DELIV, return relative posix paths for HTML
        matches = sorted(
            [str(Path(p).relative_to(DELIV).as_posix()) for p in glob.glob(str(DELIV / pat), recursive=True)]
        )
        results[key] = matches
    # Remove items from "other" that are already in norm/map/profile to avoid duplication
    taken = set(results["norm"]) | set(results["aip_map"]) | set(results["aip_profile"])
    results["other"] = [p for p in results["other"] if p not in taken]
    return results


def read_summary() -> list[dict[str, str]]:
    csv_path = DELIV / "run_summary.csv"
    rows: list[dict[str, str]] = []
    if not csv_path.exists():
        # Fallback: infer galaxies from directory structure
        gals = sorted({Path(p).name.split("_")[0] for p in glob.glob(str(DELIV / "VCC*"))})
        for g in gals:
            rows.append({"galaxy": g, "isapc_status": "?", "aip_map": "?", "aip_profile": "?", "norm_spectra_count": "?", "notes": ""})
        return rows
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_html(rows: list[dict[str, str]]) -> str:
    head = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>ISAPC FINAL_DELIVERABLES Gallery</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
    h1 { font-size: 20px; margin: 8px 0 16px; }
    h2 { font-size: 18px; margin: 24px 0 8px; }
    .meta { color: #555; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
    .card { border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fafafa; }
    .thumb { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }
    .sect { margin-bottom: 16px; }
    .pill { display: inline-block; padding: 2px 6px; border-radius: 999px; background: #eef; color: #225; font-size: 12px; margin-right: 6px; }
    .ok { background: #e9f9ef; color: #165b31; }
    .warn { background: #fff6e5; color: #7a5100; }
  </style>
  </head>
<body>
  <h1>ISAPC FINAL_DELIVERABLES Gallery</h1>
  <div class=\"meta\">Auto-generated from FINAL_DELIVERABLES. Click thumbnails to view full-size images.</div>
  <div class=\"meta\"><a href=\"virgo_cluster_final_gradients.png\">Cluster gradient figure</a> (if present)</div>
"""
    parts = [head]
    for row in rows:
        gal = row.get("galaxy", "").strip()
        if not gal:
            continue
        imgs = find_images(gal)
        isapc = row.get("isapc_status", "?")
        aip_map = row.get("aip_map", "?")
        aip_prof = row.get("aip_profile", "?")
        norm_cnt = row.get("norm_spectra_count", "?")
        notes = row.get("notes", "")
        parts.append(f"<h2 id='{html.escape(gal)}'>{html.escape(gal)}</h2>")
        status = []
        status.append(f"<span class='pill {'ok' if isapc=='ok' else 'warn'}'>ISAPC: {html.escape(isapc)}</span>")
        status.append(f"<span class='pill {'ok' if aip_map in ('1','yes','ok') else 'warn'}'>AIP map: {html.escape(aip_map)}</span>")
        status.append(f"<span class='pill {'ok' if aip_prof in ('1','yes','ok') else 'warn'}'>AIP profile: {html.escape(aip_prof)}</span>")
        status.append(f"<span class='pill'>Norm spectra: {html.escape(str(norm_cnt))} files</span>")
        if notes:
            status.append(f"<span class='pill warn'>Notes: {html.escape(notes)}</span>")
        parts.append("<div class='meta'>" + " ".join(status) + "</div>")

        def section(name: str, files: list[str]):
            if not files:
                parts.append(f"<div class='sect meta'>No {name} found.</div>")
                return
            parts.append(f"<div class='sect'><div class='meta'><b>{html.escape(name)}</b> ({len(files)} files)</div>")
            parts.append("<div class='grid'>")
            for rel in files:
                title = os.path.basename(rel)
                parts.append(
                    f"<a class='card' href='{html.escape(rel)}' target='_blank' rel='noopener'>"
                    f"<img class='thumb' loading='lazy' src='{html.escape(rel)}' alt='{html.escape(title)}'/>"
                    f"<div class='meta'>{html.escape(title)}</div>"
                    f"</a>"
                )
            parts.append("</div></div>")

        section("Normalized spectra (P2P)", imgs["norm"])
        section("AIP maps", imgs["aip_map"])
        section("AIP profiles", imgs["aip_profile"])
        if imgs["other"]:
            section("Other PNGs", imgs["other"])

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    DELIV.mkdir(exist_ok=True)
    rows = read_summary()
    html_out = build_html(rows)
    out_path = DELIV / "index.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
