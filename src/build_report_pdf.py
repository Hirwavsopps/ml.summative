"""
Render the written report to PDF in IEEE two-column style.

Input  : ``report/content.txt``  -- a small line-oriented markup (see ``docs/MARKUP.md`` notes below)
Output : ``report/Final_Report.pdf``

Why ReportLab rather than LaTeX or an HTML-to-PDF browser: a TeX distribution is not present on the
grading machine, and headless browsers require IPC channels that a confined environment may deny.
ReportLab is a pure-Python layout engine, so the build has no external process dependency.

The script also reports the prose word count.  The count deliberately excludes the title block,
captions, table contents and the reference list, which mirrors the assignment's rule
("3,500-5,000 words excluding references, captions and appendices").
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "report" / "content.txt"
PDF_OUT = ROOT / "report" / "Final_Report.pdf"
FIG_DIR = ROOT / "figures"

# --------------------------------------------------------------------------------------------
# page geometry
# --------------------------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
LM = RM = 15 * mm
TM = 16 * mm
BM = 16 * mm
GUTTER = 6.5 * mm
TEXT_W = PAGE_W - LM - RM
COL_W = (TEXT_W - GUTTER) / 2
FRONT_H = 116 * mm

# --------------------------------------------------------------------------------------------
# fonts: real Times New Roman so that Greek letters, >= and sqrt render
# --------------------------------------------------------------------------------------------
FONT = "Times-Roman"
FONT_B, FONT_I, FONT_BI = "Times-Bold", "Times-Italic", "Times-BoldItalic"


def register_times() -> None:
    global FONT, FONT_B, FONT_I, FONT_BI
    faces = {
        "TNR": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    }
    base = Path(r"C:\Windows\Fonts")
    for family, (reg, bold, ital, boldital) in faces.items():
        paths = [base / reg, base / bold, base / ital, base / boldital]
        if not all(p.exists() for p in paths):
            return
        try:
            pdfmetrics.registerFont(TTFont(family, str(paths[0])))
            pdfmetrics.registerFont(TTFont(family + "-Bold", str(paths[1])))
            pdfmetrics.registerFont(TTFont(family + "-Italic", str(paths[2])))
            pdfmetrics.registerFont(TTFont(family + "-BoldItalic", str(paths[3])))
            from reportlab.pdfbase.pdfmetrics import registerFontFamily
            registerFontFamily(family, normal=family, bold=family + "-Bold",
                               italic=family + "-Italic", boldItalic=family + "-BoldItalic")
            FONT, FONT_B, FONT_I, FONT_BI = family, family + "-Bold", family + "-Italic", family + "-BoldItalic"
            print(f"Font: {family} (embedded)")
        except Exception as exc:                      # fall back to the PDF base-14 fonts
            print(f"Font registration failed ({exc}); using Times-Roman")


# --------------------------------------------------------------------------------------------
# markup parsing
# --------------------------------------------------------------------------------------------
ALLOWED_TAGS = re.compile(r"</?(?:sup|i|b|br)\s*/?>")
PLACEHOLDER = "\x00{}\x00"


def inline(text: str) -> str:
    """Convert our mini-markup to ReportLab's paragraph markup, escaping everything else."""
    holders: list[str] = []

    def stash(match: re.Match) -> str:
        holders.append(match.group(0))
        return PLACEHOLDER.format(len(holders) - 1)

    text = ALLOWED_TAGS.sub(stash, text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # `code` -> monospaced
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.2">\1</font>', text)
    # **bold** first, then *italic* (single asterisks must not match inside a bold span)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    for index, original in enumerate(holders):
        text = text.replace(PLACEHOLDER.format(index), original)
    return text


def parse(path: Path) -> dict:
    doc = {"body": [], "refs": []}
    mode = None
    table = None
    ref = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        if line.startswith("#ENDTABLE"):
            doc["body"].append(("TABLE", table))
            table = None
            continue
        if line.startswith("#ENDREFS"):
            mode = None
            continue

        if line.strip() == "#TABLE":
            table = {"cap": "", "head": [], "rows": [], "note": "", "widths": None, "lastbold": False}
            continue
        if line.startswith("#REFS"):
            mode = "refs"
            continue

        if table is not None:
            for key in ("cap", "head", "row", "note", "widths"):
                if line.startswith(key + ":"):
                    value = line[len(key) + 1:].strip()
                    if key == "cap":
                        table["cap"] = value
                    elif key == "head":
                        table["head"] = [c.strip() for c in value.split("|")]
                    elif key == "row":
                        table["rows"].append([c.strip() for c in value.split("|")])
                    elif key == "note":
                        table["note"] = value
                    elif key == "widths":
                        table["widths"] = [float(c) for c in value.split("|")]
                    break
            if line.startswith("lastbold:"):
                table["lastbold"] = True
            continue

        if mode == "refs":
            doc["refs"].append(line)
            continue

        if line.startswith("#"):
            key, _, value = line[1:].partition(" ")
            value = value.strip()
            if key in ("TITLE", "SUBTITLE", "AUTHOR", "AFFIL", "LINKS", "ABSTRACT", "KEYWORDS"):
                doc[key.lower()] = value
            elif key == "H2":
                doc["body"].append(("H2", value))
            elif key == "H3":
                doc["body"].append(("H3", value))
            elif key == "P":
                doc["body"].append(("P", value))
            elif key == "FIG":
                path_part, _, caption = value.partition("|")
                doc["body"].append(("FIG", path_part.strip(), caption.strip()))
            elif key == "EQ":
                doc["body"].append(("EQ", value))
            elif key == "TABLE_1":
                doc["body"].append(("TABLE", table1_spec()))
            elif key == "TABLE_2":
                doc["body"].append(("TABLE", table2_spec()))
            elif key == "TABLE_3":
                doc["body"].append(("TABLE", table4_spec()))
            elif key == "TABLE_4":
                doc["body"].append(("TABLE", table3_spec()))
            elif key == "PB":
                doc["body"].append(("PB", ""))
            else:
                raise ValueError(f"Unknown directive on line: {line[:60]!r}")
    return doc


# ------------------------------------------------------------------------------------------------
# Result tables are generated from results/*.csv so the paper can never drift from the notebook
# ------------------------------------------------------------------------------------------------
RESULT_DIR = ROOT / "results"
EXP_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]


SHORT_NAME = {
    "E1": "Logistic Regression", "E2": "Random Forest (def.)",
    "E3": "Random Forest (tuned)", "E4": "Hist. Gradient Boost.",
    "E5": "RBF-kernel SVM", "E6": "MLP 64-64",
    "E7": "MLP 128-64-32 +BN/DO", "E8": "Embeddings + MLP",
    "E9": "Best DL + class wts.",
}


def table1_spec() -> dict:
    """Variable dictionary, generated from the notebook's Table I export."""
    import pandas as pd

    df = pd.read_csv(RESULT_DIR / "table1_dataset_summary.csv")
    rows = [[r["Variable"], r["Values"], f"{r['Mean']:.3f}"] for _, r in df.iterrows()]
    return {
        "cap": ("TABLE I. Variable dictionary. Twenty-one predictors plus the target; all are "
                "binary, integer or ordinal-coded, and none is a laboratory measurement. "
                "Values are the coding range and Mean the cohort mean."),
        "head": ["Variable", "Range", "Mean"],
        "rows": rows,
        "widths": [0.48, 0.26, 0.26],
        "boldrows": [],
        "note": "Source: UCI dataset 891, n = 253,680. No missing values in any column.",
        "lastbold": False,
    }


def _experiment_frame():
    """Load table2 and order it E1..E9 (experiment order, not ranking)."""
    import pandas as pd

    df = pd.read_csv(RESULT_DIR / "table2_experiments.csv")
    df["_o"] = df["exp_id"].map({e: i for i, e in enumerate(EXP_ORDER)}).fillna(99)
    return df.sort_values("_o").reset_index(drop=True)


def table2_spec() -> dict:
    """Discrimination quality: the threshold-free metrics."""
    df = _experiment_frame()
    rows = [[
        r["exp_id"], SHORT_NAME.get(r["exp_id"], r["model"]),
        "DL" if r["paradigm"] == "Deep Learning" else "ML",
        f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}",
        f"{r['mcc']:.3f}", f"{r['brier']:.3f}", f"{r['fit_time_s']:.0f}",
    ] for _, r in df.iterrows()]
    best = int(df["pr_auc"].to_numpy().argmax())
    return {
        "cap": ("TABLE II. Discrimination quality of all nine experiments on the held-out test "
                "partition (n = 38,052; 5,302 positives). PR-AUC is the primary criterion fixed in "
                "Section III-F. Fit time is total cost to reach the reported model, including any "
                "hyperparameter search, on a single CPU thread."),
        "head": ["ID", "Model", "Par.", "ROC", "PR-AUC", "MCC", "Brier", "t (s)"],
        "headsize": 6.6,
        "rows": rows,
        "widths": [0.062, 0.296, 0.072, 0.100, 0.132, 0.096, 0.116, 0.126],
        "boldrows": [best],
        "note": ("Bold marks the highest PR-AUC. E5 was trained on a 25,000-row subsample and is not "
                 "like-for-like with E3 and E4."),
        "lastbold": False,
    }


def table4_spec() -> dict:
    """Operating points: the same models at 0.5 and at the validation-tuned threshold."""
    df = _experiment_frame()
    rows = [[
        r["exp_id"], SHORT_NAME.get(r["exp_id"], r["model"]),
        f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
        f"{r['precision_tuned']:.3f}", f"{r['recall_tuned']:.3f}", f"{r['f1_tuned']:.3f}",
    ] for _, r in df.iterrows()]
    best_f1 = int(df["f1_tuned"].to_numpy().argmax())
    return {
        "cap": ("TABLE III. Operating points. The left block fixes the decision threshold at the "
                "conventional 0.5; the right block uses the threshold that maximises F1 on the "
                "validation partition. The difference between the two blocks is larger than any "
                "difference between models."),
        "head": ["ID", "Model", "@0.5 Prec", "@0.5 Rec", "@0.5 F1", "Tuned Prec", "Tuned Rec", "Tuned F1"],
        "rows": rows,
        "widths": [0.058, 0.268, 0.098, 0.098, 0.090, 0.104, 0.104, 0.100],
        "headsize": 6.2,
        "boldrows": [best_f1],
        "note": ("Left block: threshold = 0.5. Right block: threshold tuned on validation, reported "
                 "on test. Bold marks the best tuned F1."),
        "lastbold": False,
    }


def table3_spec() -> dict:
    import pandas as pd

    df = pd.read_csv(RESULT_DIR / "table3_ablation.csv")
    label = {
        "A1 LR | raw features": ("A1", "LR, raw 21 features"),
        "A2 LR | raw + engineered": ("A2", "LR, + 9 engineered"),
        "A3 RF | raw features": ("A3", "RF, raw 21 features"),
        "A4 RF | raw + engineered": ("A4", "RF, + 9 engineered"),
        "A5 HGB | raw features": ("A5", "HGB, raw 21 features"),
        "A6 HGB | raw + engineered": ("A6", "HGB, + 9 engineered"),
        "A7 LR | engineered, unscaled": ("A7", "LR, engineered, unscaled"),
        "A8 HGB | engineered minus 5": ("A8", "HGB, 5 weakest removed"),
        "A9 HGB | duplicates removed": ("A9", "HGB, duplicates removed"),
    }
    rows = []
    for _, r in df.iterrows():
        code, name = label.get(r["run"], ("?", r["run"]))
        rows.append([code, name, str(int(r["features"])), f"{r['roc_auc']:.4f}",
                     f"{r['pr_auc']:.4f}", f"{r['f1']:.4f}", f"{r['recall']:.4f}",
                     f"{r['fit_time_s']:.1f}"])
    return {
        "cap": ("TABLE IV. Controlled ablation. Each model family is trained twice with identical "
                "hyperparameters, changing only the feature set. A7 isolates standardisation, A8 "
                "removes the five lowest-mutual-information features, and A9 refits on a "
                "de-duplicated cohort of 227,908 rows."),
        "head": ["ID", "Configuration", "Feat.", "ROC", "PR-AUC", "F1", "Rec", "t (s)"],
        "rows": rows,
        "widths": [0.060, 0.295, 0.080, 0.105, 0.125, 0.100, 0.100, 0.135],
        "headsize": 6.6,
        "boldrows": [],
        "note": "",
        "lastbold": False,
    }


def word_count(doc: dict) -> int:
    chunks = [doc.get("abstract", "")] + [v for k, *rest in doc["body"] for v in rest if k == "P"]
    total = 0
    for chunk in chunks:
        text = re.sub(r"<[^>]+>", " ", chunk)
        total += len(re.findall(r"[A-Za-z][A-Za-z'\-]*", text))
    return total


# --------------------------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------------------------
register_times()

S_TITLE = ParagraphStyle("t", fontName=FONT, fontSize=19, leading=23, alignment=TA_CENTER,
                         spaceAfter=2.6 * mm)
S_SUB = ParagraphStyle("s", fontName=FONT_I, fontSize=12, leading=15, alignment=TA_CENTER,
                       spaceAfter=4.2 * mm)
S_AUTH = ParagraphStyle("a", fontName=FONT, fontSize=11.5, leading=14, alignment=TA_CENTER,
                        spaceAfter=1.2 * mm)
S_AFF = ParagraphStyle("af", fontName=FONT_I, fontSize=9.3, leading=12, alignment=TA_CENTER,
                       spaceAfter=3.4 * mm)
S_LINK = ParagraphStyle("l", fontName=FONT, fontSize=8.4, leading=10.4, alignment=TA_CENTER)
S_ABS = ParagraphStyle("ab", fontName=FONT, fontSize=9.1, leading=11.9, alignment=TA_JUSTIFY,
                       firstLineIndent=0, spaceAfter=1.8 * mm)
S_KW = ParagraphStyle("kw", fontName=FONT, fontSize=9.0, leading=11.6, alignment=TA_JUSTIFY,
                      firstLineIndent=0, spaceBefore=1.2 * mm)
S_H2 = ParagraphStyle("h2", fontName=FONT, fontSize=9.8, leading=12.2, alignment=TA_CENTER,
                      spaceBefore=3.6 * mm, spaceAfter=1.9 * mm, keepWithNext=1)
S_H3 = ParagraphStyle("h3", fontName=FONT_I, fontSize=9.6, leading=12, alignment=TA_LEFT,
                      spaceBefore=2.6 * mm, spaceAfter=1.1 * mm, keepWithNext=1)
S_P = ParagraphStyle("p", fontName=FONT, fontSize=9.55, leading=12.35, alignment=TA_JUSTIFY,
                     firstLineIndent=4.4 * mm, spaceAfter=1.35 * mm)
S_CAP = ParagraphStyle("c", fontName=FONT, fontSize=8.0, leading=10.1, alignment=TA_JUSTIFY,
                       firstLineIndent=0, spaceBefore=1.3 * mm)
S_TCAP = ParagraphStyle("tc", fontName=FONT, fontSize=8.0, leading=10.1, alignment=TA_LEFT,
                        firstLineIndent=0, spaceAfter=1.3 * mm)
S_CELL = ParagraphStyle("cell", fontName=FONT, fontSize=7.3, leading=8.7, alignment=TA_CENTER)
S_CELLL = ParagraphStyle("celll", fontName=FONT, fontSize=7.3, leading=8.7, alignment=TA_LEFT)
S_CELLH = ParagraphStyle("cellh", fontName=FONT_B, fontSize=7.3, leading=8.7, alignment=TA_CENTER)
S_REF = ParagraphStyle("r", fontName=FONT, fontSize=7.7, leading=9.7, alignment=TA_LEFT,
                       firstLineIndent=-4.6 * mm, leftIndent=4.6 * mm, spaceAfter=0.9 * mm)


# --------------------------------------------------------------------------------------------
# flowables
# --------------------------------------------------------------------------------------------
def front_matter(doc: dict) -> list:
    flow = [
        Paragraph(inline(doc["title"]), S_TITLE),
        Paragraph(inline(doc["subtitle"]), S_SUB),
        Paragraph(inline(doc["author"]), S_AUTH),
        Paragraph(inline(doc["affil"]), S_AFF),
    ]

    # the three required links, on the first page, one per line so a long URL never breaks mid-word
    rows = []
    for item in doc["links"].split("~~"):
        label, _, url = item.partition("|")
        rows.append(Paragraph(
            f'<font name="{FONT_B}" size="8.0">{label.upper()}</font>&nbsp;&nbsp;'
            f'<font size="7.7">{url}</font>', S_LINK))
    links = Table([[r] for r in rows], colWidths=[TEXT_W], hAlign="CENTER")
    links.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("TOPPADDING", (0, 0), (-1, 0), 2.2),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 2.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flow.append(links)
    flow.append(Spacer(1, 3.4 * mm))

    flow.append(Paragraph(
        f'<font name="{FONT_BI}">Abstract</font>&mdash;' + inline(doc["abstract"]), S_ABS))
    flow.append(Paragraph(
        f'<font name="{FONT_BI}">Keywords</font>&mdash;' + inline(doc["keywords"]), S_KW))
    return flow


def make_table(spec: dict) -> list:
    head = spec["head"]
    rows = spec["rows"]
    ncols = len(rows[0])
    widths = spec.get("widths") or [1.0 / ncols] * ncols
    col_widths = [COL_W * w for w in widths * 1] if len(widths) == ncols else [COL_W / ncols] * ncols
    bold = set(spec.get("boldrows", []) or [])

    head_style = S_CELLH if "headsize" not in spec else ParagraphStyle(
        "cellh_var", parent=S_CELLH, fontSize=spec["headsize"], leading=spec["headsize"] + 1.4)
    data = [[Paragraph(inline(c), head_style) for c in head]]
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, cell in enumerate(row):
            style = S_CELLL if c_i == 1 else S_CELL
            text_cell = inline(cell)
            if r_i in bold:
                text_cell = f"<b>{text_cell}</b>"
            cells.append(Paragraph(text_cell, style))
        data.append(cells)

    table = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.05),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.05),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.1),
    ]))

    parts = [Paragraph(inline(spec["cap"]), S_TCAP), table]
    if spec.get("note"):
        parts.append(Paragraph(inline(spec["note"]), S_CAP))
    return [KeepTogether(parts)]


MISSING_FIGURES: list[str] = []


def make_figure(path: str, caption: str) -> list:
    image_path = FIG_DIR / Path(path).name
    if not image_path.exists():
        image_path = ROOT / path
    if not image_path.exists():
        # A figure the notebook has not produced yet: keep the build useful and report it loudly
        MISSING_FIGURES.append(Path(path).name)
        return [Paragraph(f"[missing figure: {Path(path).name}]", S_CAP)]
    iw, ih = ImageReader(str(image_path)).getSize()
    width = COL_W
    height = width * ih / iw
    max_h = 150 * mm   # portrait figures may use most of the column height
    if height > max_h:
        height = max_h
        width = height * iw / ih
    img = Image(str(image_path), width=width, height=height)
    img.hAlign = "CENTER"
    return [KeepTogether([img, Paragraph(inline(caption), S_CAP)])]


def build_story(doc: dict, front: list | None = None) -> list:
    story = list(front) if front is not None else front_matter(doc)
    story.append(FrameBreak())
    story.append(NextPageTemplate("later"))

    for item in doc["body"]:
        kind = item[0]
        if kind == "H2":
            story.append(Paragraph(inline(item[1]), S_H2))
        elif kind == "H3":
            story.append(Paragraph(inline(item[1]), S_H3))
        elif kind == "P":
            story.append(Paragraph(inline(item[1]), S_P))
        elif kind == "FIG":
            story.extend(make_figure(item[1], item[2]))
        elif kind == "TABLE":
            story.extend(make_table(item[1]))
        elif kind == "EQ":
            story.append(Paragraph(item[1], ParagraphStyle("eq", parent=S_P, alignment=TA_CENTER,
                                                           firstLineIndent=0)))
        elif kind == "PB":
            from reportlab.platypus import PageBreak
            story.append(PageBreak())

    if doc["refs"]:
        story.append(Paragraph("References", S_H2))
        for ref in doc["refs"]:
            story.append(Paragraph(inline(ref), S_REF))
    return story


# --------------------------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------------------------
def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.drawCentredString(PAGE_W / 2, 9 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def main() -> int:
    doc = parse(CONTENT)
    words = word_count(doc)
    lo, hi = 3500, 5000
    verdict = "within" if lo <= words <= hi else "OUTSIDE"
    print(f"Prose word count: {words:,} ({verdict} the {lo:,}-{hi:,} requirement)")
    print(f"  body blocks: {len(doc['body'])}, references: {len(doc['refs'])}")

    if "--count" in sys.argv:
        return 0

    top_y = PAGE_H - TM

    # Measure the title block so the full-width frame is exactly tall enough for it.
    # Hard-coding the height is what makes titles and abstracts spill into the column frames.
    # NOTE: `wrap()` returns only the content height -- paragraph spaceBefore/spaceAfter are applied
    # by the *frame*, not included here, so they must be added explicitly or the block is short by
    # the sum of its paragraph spacing (which is what pushed the keywords into column one).
    front = front_matter(doc)
    front_h = 0.0
    for flowable in front:
        _, height = flowable.wrap(TEXT_W, PAGE_H)
        front_h += height + flowable.getSpaceBefore() + flowable.getSpaceAfter()
    front_h += 6 * mm
    if front_h > (top_y - BM) * 0.62:
        print(f"WARNING: front matter is {front_h/mm:.0f} mm tall; consider shortening the abstract.")

    front_frame = Frame(LM, top_y - front_h, TEXT_W, front_h, id="front",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    col_h_first = top_y - front_h - BM
    first = PageTemplate(id="first", onPage=footer, frames=[
        front_frame,
        Frame(LM, BM, COL_W, col_h_first, id="c1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(LM + COL_W + GUTTER, BM, COL_W, col_h_first, id="c2",
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ])
    col_h = top_y - BM
    later = PageTemplate(id="later", onPage=footer, frames=[
        Frame(LM, BM, COL_W, col_h, id="l1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(LM + COL_W + GUTTER, BM, COL_W, col_h, id="l2",
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ])

    if PDF_OUT.exists():
        PDF_OUT.unlink()

    document = BaseDocTemplate(
        str(PDF_OUT), pagesize=A4,
        leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
        title="Non-Invasive Diabetes Risk Screening from Behavioural and Demographic Indicators",
        author="Crispin Hebert Hirwa",
        subject="Introduction to Machine Learning - Summative Assignment",
    )
    document.addPageTemplates([first, later])
    document.build(build_story(doc, front))
    print(f"PDF written: {PDF_OUT} ({PDF_OUT.stat().st_size/1e6:.2f} MB)")
    if MISSING_FIGURES:
        print(f"WARNING: {len(MISSING_FIGURES)} figure(s) missing: {', '.join(MISSING_FIGURES)}")
    else:
        print("All figures embedded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
