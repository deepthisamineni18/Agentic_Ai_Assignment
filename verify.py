#!/usr/bin/env python3
"""Cross-platform report verifier for the research pipeline.

Usage:
  python verify.py [output_dir]

This script is equivalent to verify.sh but works on Windows, macOS, and Linux
without requiring a shell interpreter beyond Python.
"""

import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: the 'jsonschema' package is required (see requirements.txt).", file=sys.stderr)
    sys.exit(1)


def main(output_dir: str = "output") -> int:
    output_path = Path(output_dir)
    schema_path = Path("research_pipeline/data/report_schema.json")

    if not output_path.is_dir():
        print(f"ERROR: output directory '{output_dir}' does not exist.", file=sys.stderr)
        return 1

    if not schema_path.exists():
        print(f"ERROR: schema file not found at {schema_path}", file=sys.stderr)
        return 1

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report_files = sorted(output_path.glob("report_*.json"))

    if not report_files:
        print(f"ERROR: no report_*.json files found in {output_path}", file=sys.stderr)
        return 1

    errors: list[str] = []
    seen_report_ids: set[str] = set()
    checked = 0

    for path in report_files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: not valid JSON ({exc})")
            continue

        try:
            jsonschema.validate(instance=report, schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{path.name}: schema violation at {list(exc.path)}: {exc.message}")
            continue

        report_id = report.get("report_id")
        if report_id in seen_report_ids:
            errors.append(f"{path.name}: duplicate report_id '{report_id}'")
        seen_report_ids.add(report_id)

        valid_source_ids = {s.get("source_id") for s in report.get("sources", [])}
        for section in report.get("sections", []):
            for citation in section.get("citations", []):
                if citation not in valid_source_ids:
                    errors.append(
                        f"{path.name}: section '{section.get('heading')}' cites unresolved source_id '{citation}'"
                    )

        confidence = report.get("critique", {}).get("confidence_score")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            errors.append(f"{path.name}: confidence_score {confidence} out of range [0,1]")

        for source in report.get("sources", []):
            rel = source.get("relevance_score")
            if rel is not None and not (0.0 <= rel <= 1.0):
                errors.append(
                    f"{path.name}: relevance_score {rel} out of range [0,1] for source '{source.get('source_id')}'"
                )

        checked += 1

    print(f"Checked {checked}/{len(report_files)} report(s).")

    if errors:
        print(f"\nFAILED — {len(errors)} issue(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        "PASSED — all reports valid, all citations resolve, no duplicate report_ids, "
        "all scores in range."
    )
    return 0


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "output"
    raise SystemExit(main(outdir))
