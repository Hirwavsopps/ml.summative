"""
Fill the three submission links in the report and rebuild the PDF.

The report's first page must carry working links to the GitHub repository, the Google Drive folder
and the video. Those are the only three values in the project that cannot be generated, so they are
parameterised here rather than left as hand-edited placeholders.

Usage
-----
    python src/set_links.py \
        --code  https://github.com/USERNAME/ml-summative \
        --drive https://drive.google.com/drive/folders/XXXXXXXX \
        --video https://youtu.be/XXXXXXXX

After rewriting ``report/content.txt`` it re-renders ``report/Final_Report.pdf`` and re-checks the
word count. Any argument left out keeps its current value.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "report" / "content.txt"

LABELS = {"code": "Code", "drive": "Dataset & models", "video": "Video"}


def current_links(line: str) -> dict[str, str]:
    """Parse '#LINKS Label|url~~Label|url~~Label|url' into {argname: url}."""
    found: dict[str, str] = {}
    payload = line[len("#LINKS "):].strip()
    for item in payload.split("~~"):
        label, _, url = item.partition("|")
        for key, expected in LABELS.items():
            if label.strip().lower() == expected.lower():
                found[key] = url.strip()
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Set the GitHub / Drive / video links in the report.")
    parser.add_argument("--code", help="GitHub repository URL")
    parser.add_argument("--drive", help="Google Drive folder URL (Anyone with the link can view)")
    parser.add_argument("--video", help="Video recording URL")
    parser.add_argument("--no-build", action="store_true", help="Rewrite the source but do not render the PDF")
    args = parser.parse_args()

    if not any((args.code, args.drive, args.video)):
        parser.print_help()
        return 1

    lines = CONTENT.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next((i for i, l in enumerate(lines) if l.startswith("#LINKS ")), None)
    if index is None:
        print("Could not find the #LINKS line in report/content.txt", file=sys.stderr)
        return 1

    links = current_links(lines[index].rstrip("\n"))
    for key in LABELS:
        value = getattr(args, key)
        if value:
            links[key] = value

    ordered = [f"{LABELS[k]}|{links[k]}" for k in ("code", "drive", "video") if k in links]
    lines[index] = "#LINKS " + "~~".join(ordered) + "\n"
    CONTENT.write_text("".join(lines), encoding="utf-8", newline="\n")

    print("Links now in the report header:")
    for item in ordered:
        label, _, url = item.partition("|")
        flag = "  <-- placeholder, not set" if re.search(r"YOUR-|XXXX", url) else ""
        print(f"  {label:<16} {url}{flag}")

    if args.no_build:
        return 0

    result = subprocess.run([sys.executable, str(ROOT / "src" / "build_report_pdf.py")],
                            capture_output=True, text=True, timeout=600)
    print(result.stdout.strip()[-1200:])
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
