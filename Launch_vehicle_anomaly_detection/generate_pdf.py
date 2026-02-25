"""
generate_pdf.py
---------------
Converts the project walkthrough into a styled PDF using reportlab.
Output: walkthrough.pdf  (placed next to this script)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import ListFlowable, ListItem

# ── Output path ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "walkthrough.pdf")

# ── Document setup ───────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT_PATH,
    pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=22*mm, bottomMargin=22*mm,
)

W, H = A4
styles = getSampleStyleSheet()

# ── Custom colour palette ────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0D1B2A")
BLUE    = colors.HexColor("#1E6FBA")
TEAL    = colors.HexColor("#17A589")
ORANGE  = colors.HexColor("#E67E22")
LIGHT   = colors.HexColor("#EBF5FB")
GREY    = colors.HexColor("#566573")
CODEBG  = colors.HexColor("#1E1E2E")
WHITE   = colors.white

# ── Style definitions ────────────────────────────────────────────────────────
def S(name, **kw):
    base = styles["Normal"] if name not in styles else styles[name]
    return ParagraphStyle(name + "_custom", parent=base, **kw)

title_style   = S("Title",    fontSize=26, textColor=WHITE,    alignment=TA_CENTER,
                  fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style= S("Sub",      fontSize=12, textColor=LIGHT,    alignment=TA_CENTER,
                  fontName="Helvetica", spaceAfter=2)
h1_style      = S("H1",       fontSize=16, textColor=WHITE,    fontName="Helvetica-Bold",
                  spaceAfter=6, spaceBefore=14)
h2_style      = S("H2",       fontSize=13, textColor=NAVY,     fontName="Helvetica-Bold",
                  spaceAfter=4, spaceBefore=10,
                  borderPad=4, backColor=LIGHT, borderColor=BLUE,
                  borderWidth=0, leftIndent=4)
h3_style      = S("H3",       fontSize=11, textColor=BLUE,     fontName="Helvetica-Bold",
                  spaceAfter=3, spaceBefore=8)
body_style    = S("Body",     fontSize=9.5, textColor=NAVY,    fontName="Helvetica",
                  spaceAfter=4, leading=15)
code_style    = S("Code",     fontSize=8.5, textColor=colors.HexColor("#A9DC76"),
                  fontName="Courier", backColor=CODEBG,
                  spaceAfter=6, spaceBefore=4,
                  leftIndent=8, rightIndent=8, leading=13,
                  borderPad=6)
note_style    = S("Note",     fontSize=9,  textColor=colors.HexColor("#1A5276"),
                  fontName="Helvetica-Oblique", backColor=colors.HexColor("#D6EAF8"),
                  spaceAfter=4, leftIndent=8, rightIndent=8, leading=13, borderPad=6)
warn_style    = S("Warn",     fontSize=9,  textColor=colors.HexColor("#784212"),
                  fontName="Helvetica-Oblique", backColor=colors.HexColor("#FDEBD0"),
                  spaceAfter=4, leftIndent=8, rightIndent=8, leading=13, borderPad=6)
label_style   = S("Label",    fontSize=8,  textColor=GREY,     fontName="Helvetica-Oblique")
tbl_hdr       = S("TH",       fontSize=9,  textColor=WHITE,    fontName="Helvetica-Bold",
                  alignment=TA_CENTER)
tbl_cell      = S("TD",       fontSize=8.5,textColor=NAVY,     fontName="Helvetica", leading=12)

# ── Helper shortcuts ─────────────────────────────────────────────────────────
def P(text, style=body_style): return Paragraph(text, style)
def SP(n=4):                   return Spacer(1, n*mm)
def HR():                      return HRFlowable(width="100%", thickness=0.5,
                                                  color=colors.HexColor("#AED6F1"), spaceAfter=4)

def section_header(text, style=h1_style):
    """Blue banner for major section headers."""
    tbl = Table([[Paragraph(text, style)]], colWidths=[doc.width])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BLUE),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return KeepTogether([SP(2), tbl, SP(2)])

def data_table(headers, rows, col_widths=None):
    data = [[Paragraph(h, tbl_hdr) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), tbl_cell) for c in row])
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), NAVY),
        ("BACKGROUND",   (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#AED6F1")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl

def bullet_list(items):
    return ListFlowable(
        [ListItem(P(i), leftIndent=12, bulletColor=BLUE) for i in items],
        bulletType="bullet", leftIndent=16, spaceBefore=2, spaceAfter=2
    )

# ── Cover banner ─────────────────────────────────────────────────────────────
cover = Table(
    [[Paragraph("🚀  Launch Vehicle Anomaly Detector", title_style)],
     [Paragraph("Project Walkthrough — Full Learning Guide", subtitle_style)],
     [Paragraph("Days 1 – 4 | Python · NumPy · Pandas · ReportLab", label_style)]],
    colWidths=[doc.width]
)
cover.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), NAVY),
    ("TOPPADDING",    (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ("RIGHTPADDING",  (0,0), (-1,-1), 12),
]))

# ── Build story ───────────────────────────────────────────────────────────────
story = []

story += [cover, SP(4)]

# ── Section 0: The Big Picture ───────────────────────────────────────────────
story.append(section_header("🗺  The Big Picture"))
story.append(P(
    "The goal of this project is to build a system that <b>automatically detects when something is wrong "
    "with a rocket during flight</b>, using only sensor data. The project is organised into modular "
    "day-by-day components that form a clean pipeline from data generation through to anomaly detection."
))
story.append(SP(2))

pipeline_data = [
    ["Day", "File", "Role", "Output"],
    ["1", "day1_generator.py",   "Base telemetry simulator",         "time, altitude, velocity, engine_temp"],
    ["2", "day2_physics.py",     "Physics sensor models",            "fuel_pressure, vibration"],
    ["3", "anomalies.py",        "Fault injectors",                  "corrupted signal arrays"],
    ["3", "assemble_dataset.py", "Training dataset builder",         "data/train_normal.csv"],
    ["4", "make_test.py",        "Labelled test dataset builder",    "data/test_anomalies.csv"],
]
col_w = [20*mm, 52*mm, 58*mm, None]
story.append(data_table(pipeline_data[0], pipeline_data[1:], col_widths=col_w))
story.append(SP(3))

# ── Section 1: day1_generator.py ─────────────────────────────────────────────
story.append(section_header("📄  Day 1 — day1_generator.py"))
story.append(P("<b>What it does:</b> Simulates the basic physical signals a rocket produces during flight."))
story.append(SP(1))
story.append(P("""<b>Function:</b> <font name="Courier">generate_telemetry()</font> — returns a 6,000-row DataFrame (600 s × 10 Hz)."""))
story.append(SP(1))
story.append(data_table(
    ["Column", "Formula", "Real-world meaning"],
    [["time",        "linspace(0, 600, 6000)",         "600 s at 10 readings/sec"],
     ["altitude",    "time² × 0.25",                   "Parabolic climb (constant acceleration)"],
     ["velocity",    "time × 0.5 + noise",             "Linear speed + sensor jitter"],
     ["engine_temp", "300 + time×0.1 + noise",         "Gradually heating engine (Kelvin)"]],
    col_widths=[35*mm, 55*mm, None]
))
story.append(SP(2))
story.append(P("<b>Why time² for altitude?</b>"))
story.append(P(
    "From basic kinematics: <i>s = ½at²</i>.  With a = 0.5 m/s², altitude = 0.25 × t².  "
    "The rocket accelerates, so it climbs faster and faster — a parabolic curve."
))
story.append(P("<b>Why add noise to velocity?</b>"))
story.append(P(
    "Real sensors are never perfect. Adding <font name='Courier'>np.random.normal(0, 1.0, n)</font> "
    "adds tiny random wobbles to mimic real sensor imprecision."
))
story.append(HR())

# ── Section 2: day2_physics.py ───────────────────────────────────────────────
story.append(section_header("📄  Day 2 — day2_physics.py"))
story.append(P("<b>What it does:</b> Adds two more realistic physical signals that require more complex physics modelling."))
story.append(SP(2))

story.append(P("""<b>Function 1 (Jisto):</b> <font name="Courier">simulate_pressure(time_vector)</font>""", h3_style))
story.append(Paragraph(
    '<font name="Courier" color="#A9DC76" backColor="#1E1E2E">  P(t) = 5000 × e^(−0.008 × t)  +  noise</font>',
    code_style
))
story.append(data_table(
    ["Part", "Meaning"],
    [["5000",          "Tank starts full — 5000 pressure units"],
     ["e^(-0.008t)",   "Exponential decay — fuel consumed, pressure drops"],
     ["noise",         "Gaussian jitter N(0, 50) — sensor imprecision"],
     ["At t=600s",     "5000 × e^(-4.8) ≈ 41 units — tank nearly empty ✅"]],
    col_widths=[50*mm, None]
))
story.append(SP(1))
story.append(P(
    "<b>Analogy:</b> Like a balloon slowly deflating — pressure drops fast at first, then slows. "
    "That's exponential decay."
))
story.append(SP(3))

story.append(P("""<b>Function 2 (Devika):</b> <font name="Courier">simulate_vibration(velocity_vector)</font>""", h3_style))
story.append(Paragraph(
    '<font name="Courier" color="#A9DC76" backColor="#1E1E2E">  V(v) = A × exp( −(v − 300)² / (2 × 80²) )  ×  (v / 300)  +  noise</font>',
    code_style
))
story.append(data_table(
    ["Part", "Meaning"],
    [["Max-Q (300 m/s)", "Moment of peak aerodynamic stress on the rocket"],
     ["Gaussian bell",   "Vibration peaks at Max-Q, lower on either side"],
     ["velocity_scale",  "Ensures vibration also scales with overall speed"]],
    col_widths=[50*mm, None]
))
story.append(SP(1))
story.append(P(
    "<b>Analogy:</b> Like turbulence on a plane — worst at a specific speed window, "
    "less severe before and after."
))
story.append(HR())

# ── Section 3: anomalies.py ───────────────────────────────────────────────────
story.append(section_header("📄  Day 3 — anomalies.py"))
story.append(P("<b>What it does:</b> Provides two functions to deliberately corrupt a clean signal, simulating sensor faults or real hardware failures."))
story.append(SP(2))

story.append(P("""<b>Function 1:</b> <font name="Courier">inject_spike(signal, magnitude, probability)</font>""", h3_style))
story.append(P("Randomly adds sudden sharp jumps to individual data points."))
story.append(Paragraph(
    '<font name="Courier" color="#A9DC76" backColor="#1E1E2E">'
    '  Normal:   [100, 101,  99, 100, 100]\n'
    '  Spiked:   [100, 101,  99, 100, 450]  ← sudden spike!</font>',
    code_style
))
story.append(bullet_list([
    "Each sample independently has a <i>probability</i> (e.g. 2%) chance of being spiked",
    "Spike direction is random — can go up OR down",
    "Spike amplitude: Uniform(−magnitude, +magnitude)",
    "<b>Real-world analogy:</b> A loose wire causing a momentary incorrect reading",
]))
story.append(SP(2))

story.append(P("""<b>Function 2:</b> <font name="Courier">inject_drift(signal, drift_factor)</font>""", h3_style))
story.append(P("Adds a slow, cumulative, ever-growing bias starting from a random time point."))
story.append(Paragraph(
    '<font name="Courier" color="#A9DC76" backColor="#1E1E2E">'
    '  Normal:   [100, 101,  99, 100, 101]\n'
    '  Drifted:  [100, 101,  99, 108, 117]  ← creeping upward from index 3\n'
    '                                  bias grows: +8, +16, +24...</font>',
    code_style
))
story.append(bullet_list([
    "Drift starts at a <b>random</b> onset index — simulates gradual sensor degradation",
    "Bias grows linearly: <font name='Courier'>drift_factor × (i − onset)</font>",
    "<b>Real-world analogy:</b> A temperature sensor slowly drifting out of calibration",
]))
story.append(SP(1))
story.append(Paragraph(
    "⚠  IMPORTANT: Both functions always return a <b>copy</b> — the original signal is never modified. This is safe programming practice.",
    warn_style
))
story.append(HR())

# ── Section 4: assemble_dataset.py ───────────────────────────────────────────
story.append(section_header("📄  Day 3 — assemble_dataset.py"))
story.append(P("<b>What it does:</b> Combines Day 1 + Day 2 signals into one clean CSV with <b>no anomalies</b>. This is the training dataset."))
story.append(SP(1))
story.append(data_table(
    ["Step", "Function Called", "Columns Added"],
    [["1", "generate_telemetry()",     "time, altitude, velocity, engine_temp"],
     ["2", "simulate_pressure(time)",  "fuel_pressure"],
     ["3", "simulate_vibration(vel)",  "vibration"],
     ["4", "df.to_csv(...)",           "→ data/train_normal.csv  (6,000 rows, 6 cols)"]],
    col_widths=[15*mm, 60*mm, None]
))
story.append(SP(2))
story.append(Paragraph(
    "ℹ  NOTE: Why no anomalies here?  The detector must first learn what <i>normal</i> looks like before it can spot deviations — "
    "just like a doctor studies healthy scans before identifying disease.",
    note_style
))
story.append(HR())

# ── Section 5: make_test.py ───────────────────────────────────────────────────
story.append(section_header("📄  Day 4 — make_test.py"))
story.append(P(
    "<b>What it does:</b> Generates a labelled test dataset WITH anomalies injected. "
    "Each row is tagged <font name='Courier'>is_anomaly = 1</font> if corrupted, else <font name='Courier'>0</font>."
))
story.append(SP(1))
story.append(data_table(
    ["Step", "Action", "Target Column", "Result"],
    [["1–3", "Same as assemble_dataset.py",   "All 6 columns",  "Clean baseline"],
     ["4",   "inject_spike (2% probability)",  "fuel_pressure",  "~120 rows spiked"],
     ["5",   "inject_drift (factor=0.08)",      "engine_temp",    "~3000 rows drifting"],
     ["6",   "Build is_anomaly mask",            "is_anomaly",     "1 if any column changed"],
     ["7",   "Save to CSV",                      "—",              "data/test_anomalies.csv"]],
    col_widths=[12*mm, 60*mm, 45*mm, None]
))
story.append(SP(2))
story.append(P("<b>The <font name='Courier'>_changed_mask()</font> helper:</b>"))
story.append(P(
    "Compares the original vs. corrupted array element-by-element using "
    "<font name='Courier'>np.isclose()</font> to find exactly which rows were changed. "
    "The union of spike_mask and drift_mask gives the final <font name='Courier'>is_anomaly</font> labels."
))
story.append(HR())

# ── Section 6: Key Concepts ────────────────────────────────────────────────────
story.append(section_header("💡  Key Concepts — Quick Reference"))
story.append(data_table(
    ["Concept", "What it means", "Where used"],
    [["Exponential decay",   "Value drops fast then levels off: e^(−kt)",     "simulate_pressure"],
     ["Gaussian noise",      "Random normal jitter: N(mean, std)",             "All sensor signals"],
     ["Max-Q",               "Peak aerodynamic stress at ~300 m/s",           "simulate_vibration"],
     ["Spike anomaly",       "Sudden, random, isolated bad reading",          "inject_spike"],
     ["Drift anomaly",       "Slow, cumulative sensor degradation",           "inject_drift"],
     ["is_anomaly label",    "0 = normal row,  1 = corrupted row",            "make_test.py"],
     ["Train/Test split",    "train_normal (clean) vs test_anomalies (faults)","Dataset design"]],
    col_widths=[45*mm, 75*mm, None]
))
story.append(HR())

# ── Section 7: Project File Map ───────────────────────────────────────────────
story.append(section_header("🗂  Project File Map"))
story.append(Paragraph(
    '<font name="Courier" color="#A9DC76" backColor="#1E1E2E">'
    'launch_vehicle_anomaly_detection/\n'
    '├── src/\n'
    '│   ├── day1_generator.py      ← Base telemetry (altitude, velocity, temp)\n'
    '│   ├── day2_physics.py        ← Physics sensors (pressure, vibration)\n'
    '│   ├── anomalies.py           ← Fault injection (spike, drift)\n'
    '│   ├── assemble_dataset.py    ← Builds train_normal.csv\n'
    '│   └── make_test.py           ← Builds test_anomalies.csv (labelled)\n'
    '├── data/\n'
    '│   ├── train_normal.csv       ← Clean data for training\n'
    '│   └── test_anomalies.csv     ← Corrupted + labelled for testing\n'
    '└── requirements.txt           ← numpy, pandas, matplotlib, scikit-learn, streamlit</font>',
    code_style
))
story.append(HR())

# ── Section 8: What Comes Next ────────────────────────────────────────────────
story.append(section_header("🔭  What Comes Next"))
story.append(data_table(
    ["Day", "Topic", "Description"],
    [["5", "Z-score Detection",      "Flag readings statistically far from the training mean/std"],
     ["6", "Isolation Forest",       "ML model that isolates unusual data points in feature space"],
     ["7", "Evaluation Metrics",     "Precision, Recall, F1 — compare predictions vs is_anomaly labels"],
     ["8", "Streamlit Dashboard",    "Visual interface to see anomaly detections in real time"]],
    col_widths=[15*mm, 55*mm, None]
))
story.append(SP(4))

# ── Build PDF ─────────────────────────────────────────────────────────────────
doc.build(story)
print(f"✅  PDF saved → {OUT_PATH}")
