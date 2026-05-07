"""Generate presentation slides for TabPFN-2.5 short story."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# Colors
BG_DARK = RGBColor(0x0D, 0x11, 0x17)
BG_CARD = RGBColor(0x16, 0x1B, 0x22)
ACCENT_BLUE = RGBColor(0x58, 0xA6, 0xFF)
ACCENT_GREEN = RGBColor(0x3F, 0xB9, 0x50)
ACCENT_PURPLE = RGBColor(0xBC, 0x8C, 0xFF)
ACCENT_ORANGE = RGBColor(0xF0, 0x88, 0x3E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x8B, 0x94, 0x9E)
LIGHT = RGBColor(0xC9, 0xD1, 0xD9)

W = Inches(13.333)
H = Inches(7.5)

prs = Presentation()
prs.slide_width = W
prs.slide_height = H

def dark_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK

def add_accent_line(slide, left, top, width, color=ACCENT_BLUE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()

def add_text_box(slide, left, top, width, height, text, font_size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = align
    return tf

def add_bullet_list(slide, left, top, width, height, items, font_size=16, color=LIGHT, bullet_color=ACCENT_BLUE):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p.level = 0
    return tf

def add_card(slide, left, top, width, height, color=BG_CARD):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def title_slide(title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_bg(slide)
    add_accent_line(slide, Inches(4.5), Inches(3.2), Inches(4.3), ACCENT_BLUE)
    add_text_box(slide, Inches(1), Inches(1.5), Inches(11), Inches(2), title, 44, WHITE, True, PP_ALIGN.CENTER)
    add_text_box(slide, Inches(2), Inches(3.5), Inches(9), Inches(1.5), subtitle, 20, GRAY, False, PP_ALIGN.CENTER)

def section_slide(number, title, bullets, accent=ACCENT_BLUE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_bg(slide)
    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), f"{number:02d}", 14, accent, True)
    add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), accent)
    add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), title, 32, WHITE, True)
    add_bullet_list(slide, Inches(1.0), Inches(2.3), Inches(11), Inches(4.5), bullets, 18, LIGHT, accent)
    return slide

# === SLIDE 1: TITLE ===
title_slide(
    "Beyond LLMs:\nThe Rise of Tabular Foundation Models",
    "A deep dive into TabPFN-2.5 — the transformer that wants to replace XGBoost\n\nArshan Bhanage  •  Deep Learning Short Story  •  2025"
)

# === SLIDE 2: MOTIVATION ===
section_slide(1, "Motivation", [
    "Foundation models have revolutionized text (GPT), images (DALL·E), and code (Copilot)",
    "But the most common data type in the real world is tabular data — spreadsheets, CSVs, database rows",
    "XGBoost & gradient-boosted trees have dominated tabular ML for a decade",
    "Can foundation models finally challenge this dominance?",
    "TabPFN-2.5 claims: YES — matching 4-hour AutoGluon ensembles in a single forward pass",
], ACCENT_BLUE)

# === SLIDE 3: WHY TABULAR DATA MATTERS ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "02", 14, ACCENT_GREEN, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_GREEN)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Why Tabular Data Matters", 32, WHITE, True)

domains = [
    ("Healthcare", "Patient records, lab results, diagnoses"),
    ("Finance", "Transactions, credit scores, risk models"),
    ("Manufacturing", "Sensor data, quality metrics, supply chain"),
    ("Retail", "Customer data, purchases, inventory"),
]
for i, (dom, desc) in enumerate(domains):
    x = Inches(0.8 + i * 3.1)
    add_card(slide, x, Inches(2.5), Inches(2.8), Inches(2.2))
    add_text_box(slide, x + Inches(0.2), Inches(2.7), Inches(2.4), Inches(0.5), dom, 20, ACCENT_GREEN, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.2), Inches(3.3), Inches(2.4), Inches(1.2), desc, 15, LIGHT, False, PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.8), Inches(5.2), Inches(11), Inches(1),
    "Tabular data powers virtually every business decision on the planet — any improvement has outsized impact", 18, GRAY)

# === SLIDE 4: WHAT ARE FOUNDATION MODELS ===
section_slide(3, "What Are Foundation Models?", [
    "A single model pre-trained on massive data that adapts to many downstream tasks",
    "Text: GPT-4, Claude, Gemini — trained on internet text, generalize to any language task",
    "Vision: CLIP, DALL·E — trained on image-text pairs, generalize to visual understanding",
    "Key idea: learn general capabilities once, apply everywhere without retraining",
    "The question: can this paradigm work for tabular data?",
], ACCENT_PURPLE)

# === SLIDE 5: TABULAR FOUNDATION MODELS ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "04", 14, ACCENT_BLUE, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_BLUE)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "What Are Tabular Foundation Models?", 32, WHITE, True)

steps = [
    ("Pre-train", "Train on millions of\nsynthetic tabular datasets"),
    ("Learn Algorithm", "Model learns HOW to\nlearn from any table"),
    ("In-Context", "Feed new data as context\nin a single forward pass"),
    ("Predict", "Output predictions with\nzero tuning needed"),
]
for i, (t, d) in enumerate(steps):
    x = Inches(0.6 + i * 3.2)
    add_card(slide, x, Inches(2.5), Inches(2.8), Inches(2.5))
    add_text_box(slide, x + Inches(0.2), Inches(2.7), Inches(2.4), Inches(0.5), f"Step {i+1}: {t}", 17, ACCENT_BLUE, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.2), Inches(3.4), Inches(2.4), Inches(1.4), d, 15, LIGHT, False, PP_ALIGN.CENTER)
    if i < 3:
        add_text_box(slide, x + Inches(2.85), Inches(3.3), Inches(0.5), Inches(0.5), "→", 24, ACCENT_BLUE, True, PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.8), Inches(5.5), Inches(11), Inches(1),
    "Traditional ML learns parameters for YOUR data. A TFM has learned HOW to learn from ANY tabular data.", 17, GRAY)

# === SLIDE 6: MAIN PAPER ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "05", 14, ACCENT_ORANGE, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_ORANGE)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Main Paper: TabPFN-2.5", 32, WHITE, True)
add_text_box(slide, Inches(0.8), Inches(2.2), Inches(11), Inches(0.5),
    "TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models", 22, ACCENT_ORANGE, True)
add_text_box(slide, Inches(0.8), Inches(2.8), Inches(11), Inches(0.5),
    "Yu, Jablonski, Hoo, Garg, Robertson, Bühler et al. — Prior Labs & Univ. of Freiburg, 2025", 16, GRAY)

versions = [
    ("TabPFN v1 (2022)", "1K samples, 100 features\nProof of concept"),
    ("TabPFNv2 (2024)", "10K samples, 500 features\nCompetitive with XGBoost"),
    ("TabPFN-2.5 (2025)", "50K samples, 2K features\nBeats tuned XGBoost"),
]
for i, (v, d) in enumerate(versions):
    x = Inches(1.0 + i * 4.0)
    c = [ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN][i]
    add_card(slide, x, Inches(3.8), Inches(3.5), Inches(2.2))
    add_text_box(slide, x + Inches(0.2), Inches(4.0), Inches(3.1), Inches(0.5), v, 18, c, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.2), Inches(4.6), Inches(3.1), Inches(1.2), d, 15, LIGHT, False, PP_ALIGN.CENTER)

# === SLIDE 7: ARCHITECTURE ===
section_slide(6, "Architecture & Method", [
    "Alternating Attention: row-wise (inter-sample) ↔ column-wise (inter-feature) layers",
    "24 transformer layers for classification, 18 for regression",
    "Feature Group Embeddings: groups of features embedded together → faster inference",
    "64 'Thinking' Rows: learned dummy inputs as computational scratch space (inspired by CoT in LLMs)",
    "Pre-trained on synthetic data; Real-TabPFN-2.5 fine-tuned on 43 curated real datasets",
    "Distillation engines: convert to fast MLP or Tree Ensemble for production deployment",
], ACCENT_BLUE)

# === SLIDE 8: BENCHMARKS ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "07", 14, ACCENT_GREEN, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_GREEN)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Benchmarks & Metrics", 32, WHITE, True)

benchmarks = [
    ("TabArena", "51 diverse real-world datasets\nUp to 100K rows, 2K features", ACCENT_BLUE),
    ("Internal Suite", "100+ proprietary datasets\nHealthcare, finance, manufacturing", ACCENT_PURPLE),
    ("RealCause", "Causal inference benchmark\nTreatment effect estimation", ACCENT_ORANGE),
]
for i, (name, desc, c) in enumerate(benchmarks):
    x = Inches(0.6 + i * 4.2)
    add_card(slide, x, Inches(2.3), Inches(3.8), Inches(1.8))
    add_text_box(slide, x + Inches(0.2), Inches(2.5), Inches(3.4), Inches(0.5), name, 20, c, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.2), Inches(3.1), Inches(3.4), Inches(1.0), desc, 15, LIGHT, False, PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11), Inches(0.5), "Metrics", 22, WHITE, True)
add_bullet_list(slide, Inches(1.0), Inches(5.0), Inches(11), Inches(2), [
    "Classification: Accuracy, ROC AUC, Normalized Score (0–1 per dataset)",
    "Regression: RMSE, R²",
    "Causal Inference: PEHE (Precision in Estimating Heterogeneous Effects)",
], 16, LIGHT)

# === SLIDE 9: KEY RESULTS ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "08", 14, ACCENT_GREEN, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_GREEN)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Key Results from the Paper", 32, WHITE, True)

results = [
    ("TabPFN-2.5 (default) > tuned XGBoost, CatBoost, LightGBM", "Single forward pass, zero hyperparameter tuning", ACCENT_GREEN),
    ("TabPFN-2.5 ≈ AutoGluon 1.4 (4-hour extreme mode)", "Matches a massive multi-model ensemble run for 4 hours", ACCENT_BLUE),
    ("Real-TabPFN-2.5 > everything", "Fine-tuned variant sets new SOTA on classification", ACCENT_PURPLE),
    ("Causal inference SOTA on RealCause benchmark", "Best CATE estimation as T-Learner base model", ACCENT_ORANGE),
]
for i, (r, d, c) in enumerate(results):
    y = Inches(2.3 + i * 1.2)
    add_card(slide, Inches(0.8), y, Inches(11.5), Inches(1.0))
    add_text_box(slide, Inches(1.0), y + Inches(0.05), Inches(7), Inches(0.5), r, 17, c, True)
    add_text_box(slide, Inches(1.0), y + Inches(0.5), Inches(10), Inches(0.4), d, 14, GRAY)

# === SLIDE 10: ABLATION ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "09", 14, ACCENT_PURPLE, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_PURPLE)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Ablation Studies", 32, WHITE, True)

ablations = [
    ("Deeper network (24 layers)", "Better accuracy on complex datasets"),
    ("Larger feature groups", "Faster training & inference"),
    ("64 'Thinking' rows", "Extra compute capacity — measurable gains"),
    ("Richer synthetic priors", "Better generalization to diverse data"),
    ("Scale to 50K rows", "5× more real-world problems covered"),
    ("Real-data fine-tuning", "Significant accuracy boost over synthetic-only"),
]
for i, (change, effect) in enumerate(ablations):
    y = Inches(2.3 + i * 0.85)
    c = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_BLUE, ACCENT_GREEN][i]
    add_text_box(slide, Inches(1.0), y, Inches(5), Inches(0.4), f"▸ {change}", 16, c, True)
    add_text_box(slide, Inches(6.5), y, Inches(6), Inches(0.4), f"→ {effect}", 16, LIGHT)

# === SLIDE 11: REPRODUCTION SETUP ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "10", 14, ACCENT_BLUE, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_BLUE)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "My Reproduction Setup", 32, WHITE, True)

add_text_box(slide, Inches(0.8), Inches(2.2), Inches(5.5), Inches(0.5), "Datasets", 20, ACCENT_BLUE, True)
add_bullet_list(slide, Inches(1.0), Inches(2.7), Inches(5), Inches(2), [
    "Breast Cancer Wisconsin — 569 samples, 30 features, 2 classes",
    "Wine — 178 samples, 13 features, 3 classes",
], 15, LIGHT)

add_text_box(slide, Inches(7), Inches(2.2), Inches(5.5), Inches(0.5), "Models", 20, ACCENT_GREEN, True)
add_bullet_list(slide, Inches(7.2), Inches(2.7), Inches(5), Inches(2.5), [
    "Logistic Regression (linear baseline)",
    "Random Forest (200 trees)",
    "XGBoost (200 estimators, lr=0.1)",
    "TabPFN (16 estimators, GPU)",
], 15, LIGHT)

add_text_box(slide, Inches(0.8), Inches(4.8), Inches(11), Inches(0.5), "Protocol", 20, ACCENT_PURPLE, True)
add_bullet_list(slide, Inches(1.0), Inches(5.3), Inches(11), Inches(2), [
    "5-fold stratified cross-validation  •  Metrics: Accuracy & Weighted F1  •  No hyperparameter tuning for any model",
], 15, LIGHT)

# === SLIDE 12: REPRODUCTION RESULTS ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "11", 14, ACCENT_GREEN, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_GREEN)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "My Reproduction Results", 32, WHITE, True)

# Breast Cancer results
add_text_box(slide, Inches(0.8), Inches(2.2), Inches(5.5), Inches(0.5), "Breast Cancer", 20, ACCENT_BLUE, True)
bc = [("Logistic Reg.", "0.974", LIGHT), ("Random Forest", "0.954", LIGHT),
      ("XGBoost", "0.963", LIGHT), ("TabPFN", "0.977 ★", ACCENT_GREEN)]
for i, (m, a, c) in enumerate(bc):
    y = Inches(2.8 + i * 0.55)
    add_text_box(slide, Inches(1.0), y, Inches(2.5), Inches(0.4), m, 15, c, i==3)
    add_text_box(slide, Inches(3.8), y, Inches(1.5), Inches(0.4), a, 15, c, i==3)

# Wine results
add_text_box(slide, Inches(7), Inches(2.2), Inches(5.5), Inches(0.5), "Wine", 20, ACCENT_PURPLE, True)
wi = [("Logistic Reg.", "0.983", LIGHT), ("Random Forest", "0.978", LIGHT),
      ("XGBoost", "0.961", LIGHT), ("TabPFN", "0.983 ★", ACCENT_GREEN)]
for i, (m, a, c) in enumerate(wi):
    y = Inches(2.8 + i * 0.55)
    add_text_box(slide, Inches(7.2), y, Inches(2.5), Inches(0.4), m, 15, c, i==3)
    add_text_box(slide, Inches(10), y, Inches(1.5), Inches(0.4), a, 15, c, i==3)

add_text_box(slide, Inches(0.8), Inches(5.2), Inches(11.5), Inches(1.5),
    "TabPFN achieves the highest accuracy on Breast Cancer and ties for the top on Wine — with zero tuning.\n"
    "Confirms paper's claim: tabular foundation models are strong defaults on small-to-medium datasets.", 16, GRAY)

# === SLIDE 13: WHAT I LEARNED ===
section_slide(12, "What I Learned", [
    "TabPFN-2.5 is not a toy — it's a practical tool ready to replace XGBoost as a default baseline",
    "A single forward pass beating a 4-hour AutoGluon ensemble is genuinely remarkable",
    "'Thinking rows' = chain-of-thought for tabular data — elegant cross-pollination from LLMs",
    "The ecosystem (distillation, API, interpretability) matters as much as the model itself",
    "We are witnessing convergence: text, vision, and now tabular all moving to foundation models",
], ACCENT_PURPLE)

# === SLIDE 14: LIMITATIONS ===
section_slide(13, "Limitations", [
    "Dataset size ceiling: 50K rows max — many production datasets have millions",
    "GPU memory: 24-layer transformer on 50K rows needs serious hardware",
    "Proprietary distillation: fast deployment engines are commercial, not open-source",
    "Interpretability gap: tree models offer SHAP/feature importance; transformers are harder to inspect",
    "Not universal: on very large, well-tuned datasets, XGBoost ensembles may still compete",
], ACCENT_ORANGE)

# === SLIDE 15: CONCLUSION ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "14", 14, ACCENT_GREEN, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), ACCENT_GREEN)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "Conclusion", 32, WHITE, True)

conclusions = [
    ("For Practitioners", "If your dataset has < 50K rows, TabPFN should be your first model to try", ACCENT_BLUE),
    ("For Researchers", "Can we scale to millions of rows? Combine tabular + text in one model?", ACCENT_PURPLE),
    ("The Big Picture", "The age of tabular foundation models is here — XGBoost finally has serious competition", ACCENT_GREEN),
]
for i, (title, desc, c) in enumerate(conclusions):
    y = Inches(2.3 + i * 1.5)
    add_card(slide, Inches(0.8), y, Inches(11.5), Inches(1.3))
    add_text_box(slide, Inches(1.2), y + Inches(0.1), Inches(10), Inches(0.5), title, 20, c, True)
    add_text_box(slide, Inches(1.2), y + Inches(0.6), Inches(10), Inches(0.5), desc, 16, LIGHT)

# === SLIDE 16: REFERENCES ===
slide = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(1), Inches(0.6), "15", 14, GRAY, True)
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(3), GRAY)
add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(1), "References", 32, WHITE, True)

refs = [
    "[1] Yu et al. (2025) TabPFN-2.5 — arXiv:2511.08667",
    "[2] Hollmann et al. (2022) TabPFN — ICLR 2023",
    "[3] Garg et al. (2025) Real-TabPFN — arXiv:2410.04145",
    "[4] Chen & Guestrin (2016) XGBoost — KDD 2016",
    "[5] Erickson et al. (2020) AutoGluon-Tabular",
    "[6] Erickson et al. (2025) TabArena benchmark",
    "[7] Prokhorenkova et al. (2018) CatBoost — NeurIPS",
    "[8] Ke et al. (2017) LightGBM — NeurIPS",
]
add_bullet_list(slide, Inches(1.0), Inches(2.3), Inches(11), Inches(5), refs, 15, LIGHT)

add_text_box(slide, Inches(0.8), Inches(6.2), Inches(11), Inches(0.8),
    "github.com/ArshanBhanage/Deep-Learning---Short-Story", 16, ACCENT_BLUE, False, PP_ALIGN.CENTER)

# Save
out_path = os.path.join(os.path.dirname(__file__), "short_story_slides.pptx")
prs.save(out_path)
print(f"Saved: {out_path}")
