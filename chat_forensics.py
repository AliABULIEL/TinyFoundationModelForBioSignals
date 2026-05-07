"""
chat_forensics.py — Chat-screenshot & screen-recording tampering detector.

Designed for WhatsApp / Messenger / Telegram screenshots and screen recordings,
where the typical fake is re-rendered/edited text in chat bubbles.

Combines four signals:
  1) ML forgery model: aevalone/vit-base-patch16-224-finetuned-forgery (HF pipeline)
  2) JPEG ghost analysis (Farid-style): edited regions match best at a different
     re-save quality than surrounding pixels — strong signal for screenshots
     since WhatsApp re-saves them as JPEG.
  3) Error Level Analysis (ELA): deterministic heatmap of per-pixel compression
     error after a known-quality re-save.
  4) OCR (EasyOCR, Arabic + English): extracts visible chat text so the user
     can sanity-check it against the real conversation.

INSTALL:
    pip install gradio transformers torch pillow opencv-python-headless numpy easyocr

RUN:
    python chat_forensics.py              # launches Gradio UI
    python chat_forensics.py --test       # runs unit tests (no model downloads)

LIMITATIONS:
    - This is heuristic. None of these signals are perfect on screenshots.
    - The ML model is unvalidated on chat content (only 9 HF downloads).
    - Best signal in practice is JPEG-ghost on a single-save WhatsApp screenshot.
    - For screen recordings, runs per-frame image analysis — temporal cues only
      via score variance across sampled frames.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("chat_forensics")

# ---- Constants ------------------------------------------------------------
FORGERY_MODEL_ID = "aevalone/vit-base-patch16-224-finetuned-forgery"
GHOST_QUALITIES = [60, 70, 75, 80, 85, 90, 95]
GHOST_BLOCK = 16  # block size for JPEG-ghost aggregation
ELA_QUALITY = 90
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_SAMPLE_FPS = 1.0

# Verdict thresholds (score in [0,1], higher = more suspicious)
SUSPICIOUS_THRESH = 0.55
CLEAN_THRESH = 0.30


# ---- Data classes ---------------------------------------------------------
@dataclass
class ImageForensics:
    ml_fake_prob: Optional[float]
    ghost_score: float
    ela_score: float
    combined_score: float
    verdict: str
    note: str
    ela_heatmap_path: Optional[str] = None
    ghost_heatmap_path: Optional[str] = None
    ocr_lines: list = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class VideoForensics:
    n_frames_sampled: int
    mean_combined: float
    max_combined: float
    score_variance: float
    verdict: str
    note: str
    sample_frames: list = field(default_factory=list)  # PIL images
    latency_ms: float = 0.0


# ---- Lazy globals ---------------------------------------------------------
_forgery_pipe = None
_ocr_reader = None
_torch_device = None


def get_forgery_model():
    """Lazy-load the HF image-forgery ViT pipeline. Returns None on failure."""
    global _forgery_pipe, _torch_device
    if _forgery_pipe is None:
        try:
            import torch
            from transformers import pipeline
            _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading forgery ML model: %s (device=%s)", FORGERY_MODEL_ID, _torch_device)
            _forgery_pipe = pipeline(
                "image-classification",
                model=FORGERY_MODEL_ID,
                device=0 if _torch_device == "cuda" else -1,
            )
        except Exception as ex:
            log.warning("ML forgery model unavailable: %s", ex)
            _forgery_pipe = False  # sentinel: tried and failed
    return _forgery_pipe if _forgery_pipe else None


def get_ocr():
    """Lazy-load EasyOCR (English + Arabic). Returns None on failure."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            log.info("Loading EasyOCR (en+ar)…")
            _ocr_reader = easyocr.Reader(["en", "ar"], gpu=False, verbose=False)
        except Exception as ex:
            log.warning("EasyOCR unavailable: %s", ex)
            _ocr_reader = False
    return _ocr_reader if _ocr_reader else None


# ---- ELA ------------------------------------------------------------------
def ela_analysis(img: Image.Image, quality: int = ELA_QUALITY) -> tuple[float, np.ndarray]:
    """Error Level Analysis.

    Re-encodes the image at `quality`; returns (score, heatmap[uint8 H×W]).
    Edited regions tend to have higher local error after the re-save.
    """
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    a = np.array(rgb).astype(np.float32)
    b = np.array(rec).astype(np.float32)
    diff = np.abs(a - b).max(axis=2)  # H × W
    if diff.max() > 0:
        norm = (diff / diff.max() * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(diff, dtype=np.uint8)
    # Score: mean error normalized; clipped to plausible range for screenshots
    score = float(np.clip(diff.mean() / 8.0, 0.0, 1.0))
    return score, norm


# ---- JPEG Ghost -----------------------------------------------------------
def jpeg_ghost(img: Image.Image,
               qualities=GHOST_QUALITIES,
               block: int = GHOST_BLOCK) -> tuple[float, np.ndarray]:
    """JPEG-ghost analysis (Farid 2009).

    For each candidate quality Q, re-encode and compute the per-block mean-square
    difference vs original. For every block, find the Q* that minimises diff.
    A clean image has nearly all blocks sharing the same Q*; an edited image has
    a multi-modal distribution of Q* across blocks.

    Returns (score, heatmap):
      score   = fraction of blocks whose Q* differs from the global mode (0..1)
      heatmap = uint8 binary mask, shape (n_blocks_h, n_blocks_w), 255 = anomaly
    """
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.float32)
    h, w = arr.shape[:2]
    h2, w2 = (h // block) * block, (w // block) * block
    if h2 == 0 or w2 == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8)
    arr = arr[:h2, :w2]

    diffs = []
    for q in qualities:
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=q)
        buf.seek(0)
        rec = np.array(Image.open(buf).convert("RGB")).astype(np.float32)[:h2, :w2]
        d = ((arr - rec) ** 2).mean(axis=2)  # H × W per-pixel MSE
        # block-aggregate: shape (h2/block, w2/block)
        nh, nw = h2 // block, w2 // block
        d_blk = d.reshape(nh, block, nw, block).mean(axis=(1, 3))
        diffs.append(d_blk)
    diffs = np.stack(diffs, axis=0)  # (Q, nh, nw)
    min_q = diffs.argmin(axis=0)  # (nh, nw)
    flat = min_q.flatten()
    if flat.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8)
    mode = int(np.bincount(flat).argmax())
    anomaly = (min_q != mode).astype(np.uint8) * 255
    score = float((min_q != mode).mean())
    return score, anomaly


# ---- ML forgery scoring ---------------------------------------------------
def ml_forgery_score(img: Image.Image) -> Optional[float]:
    """Run HF ViT forgery classifier; return P(fake) or None if unavailable."""
    pipe = get_forgery_model()
    if pipe is None:
        return None
    try:
        preds = pipe(img.convert("RGB"), top_k=None)
    except Exception as ex:
        log.warning("ML inference failed: %s", ex)
        return None
    # Find label whose name contains 'fake' / 'forg' / 'tamper' / index 1 fallback
    for p in preds:
        name = str(p["label"]).lower()
        if any(k in name for k in ("fake", "forg", "tamper", "manip")):
            return float(p["score"])
    # Fallback: assume binary [authentic, fake] order by id
    id2 = pipe.model.config.id2label
    if len(id2) == 2:
        # pick the higher index as "fake"
        idx = max(id2.keys())
        target = id2[idx]
        for p in preds:
            if str(p["label"]) == str(target):
                return float(p["score"])
    return 0.5


# ---- OCR ------------------------------------------------------------------
def ocr_image(img_path: Path) -> list[dict]:
    """Return list of {text, conf} from the image. Empty list if OCR unavailable."""
    reader = get_ocr()
    if reader is None:
        return []
    try:
        results = reader.readtext(str(img_path))
        return [{"text": r[1], "conf": float(r[2])} for r in results]
    except Exception as ex:
        log.warning("OCR failed: %s", ex)
        return []


# ---- Combined verdict -----------------------------------------------------
def combine_scores(ml: Optional[float], ghost: float, ela: float) -> tuple[float, str, str]:
    """Weighted combiner.

    Weights tuned for screenshots: JPEG-ghost is the strongest single signal
    (compression-history mismatch is what re-rendered chat bubbles produce).
    ML model is treated as a tie-breaker since it's not validated on chat data.
      ghost: 0.55, ela: 0.25, ml: 0.20
    If ML is unavailable, redistribute weight between ghost (0.7) and ela (0.3).
    """
    if ml is None:
        combined = 0.7 * ghost + 0.3 * ela
        notes = "ML model unavailable; using ghost+ELA only."
    else:
        combined = 0.55 * ghost + 0.25 * ela + 0.20 * ml
        notes = f"Weights: ghost=0.55, ELA=0.25, ML=0.20."

    if combined >= SUSPICIOUS_THRESH:
        verdict = "LIKELY EDITED"
    elif combined <= CLEAN_THRESH:
        verdict = "LIKELY CLEAN"
    else:
        verdict = "INCONCLUSIVE"
    return float(combined), verdict, notes


# ---- Frame iteration ------------------------------------------------------
def iter_video_frames(path: Path, sample_fps: float = VIDEO_SAMPLE_FPS) -> Iterator[Image.Image]:
    """Yield PIL frames at `sample_fps`."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / max(0.1, sample_fps))))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
    finally:
        cap.release()


# ---- Top-level analysis ---------------------------------------------------
def _save_heatmap(arr: np.ndarray, path: Path) -> str:
    """Save a numpy heatmap (H×W uint8) as a colorized PNG."""
    import cv2
    if arr.ndim == 2 and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), arr)
    return str(path)


def analyze_image(path: Path, out_dir: Optional[Path] = None) -> ImageForensics:
    """Run full forensic stack on a single image."""
    t0 = time.perf_counter()
    img = Image.open(path)

    ela_score, ela_map = ela_analysis(img)
    ghost_score, ghost_map = jpeg_ghost(img)
    ml_score = ml_forgery_score(img)
    combined, verdict, note = combine_scores(ml_score, ghost_score, ela_score)
    ocr_lines = ocr_image(path)

    ela_path = ghost_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ela_path = _save_heatmap(ela_map, out_dir / "ela.png")
        # Upscale block-level ghost map for visibility
        import cv2
        gh = cv2.resize(ghost_map, (img.size[0], img.size[1]),
                        interpolation=cv2.INTER_NEAREST)
        ghost_path = _save_heatmap(gh, out_dir / "ghost.png")

    latency = (time.perf_counter() - t0) * 1000.0
    return ImageForensics(
        ml_fake_prob=ml_score,
        ghost_score=ghost_score,
        ela_score=ela_score,
        combined_score=combined,
        verdict=verdict,
        note=note,
        ela_heatmap_path=ela_path,
        ghost_heatmap_path=ghost_path,
        ocr_lines=ocr_lines,
        latency_ms=latency,
    )


def analyze_video(path: Path, sample_fps: float = VIDEO_SAMPLE_FPS) -> VideoForensics:
    """Per-frame image forensics on a sampled video, then aggregate."""
    t0 = time.perf_counter()
    scores = []
    sampled = []
    for fr in iter_video_frames(path, sample_fps):
        sampled.append(fr)
        ela_s, _ = ela_analysis(fr)
        ghost_s, _ = jpeg_ghost(fr)
        ml_s = ml_forgery_score(fr)
        c, _, _ = combine_scores(ml_s, ghost_s, ela_s)
        scores.append(c)
    if not scores:
        raise RuntimeError(f"No frames sampled from {path}")
    arr = np.array(scores)
    mean = float(arr.mean())
    mx = float(arr.max())
    var = float(arr.var())
    if mx >= SUSPICIOUS_THRESH:
        verdict = "LIKELY EDITED"
        note = f"At least one frame is suspicious (max={mx:.2f})."
    elif mean <= CLEAN_THRESH:
        verdict = "LIKELY CLEAN"
        note = f"All frames within clean range (mean={mean:.2f})."
    else:
        verdict = "INCONCLUSIVE"
        note = f"Borderline (mean={mean:.2f}, max={mx:.2f})."
    latency = (time.perf_counter() - t0) * 1000.0
    return VideoForensics(
        n_frames_sampled=len(scores),
        mean_combined=mean,
        max_combined=mx,
        score_variance=var,
        verdict=verdict,
        note=note,
        sample_frames=sampled[:12],
        latency_ms=latency,
    )


# ---- Gradio UI ------------------------------------------------------------
def _verdict_html(verdict: str, score: float) -> str:
    colors = {
        "LIKELY CLEAN": ("#1b5e20", "#c8e6c9"),
        "LIKELY EDITED": ("#b71c1c", "#ffcdd2"),
        "INCONCLUSIVE": ("#f57f17", "#fff9c4"),
    }
    fg, bg = colors.get(verdict, ("#333", "#eee"))
    return (
        f"<div style='padding:20px;border-radius:10px;background:{bg};"
        f"color:{fg};text-align:center;font-weight:700;'>"
        f"<div style='font-size:28px;'>{verdict}</div>"
        f"<div style='font-size:14px;margin-top:6px;'>Suspicion score: {score:.2f}</div>"
        f"</div>"
    )


def _ocr_md(lines: list[dict]) -> str:
    if not lines:
        return "_(OCR unavailable or no text detected.)_"
    rows = "\n".join(f"- ({l['conf']:.2f}) {l['text']}" for l in lines[:30])
    extra = f"\n…and {len(lines)-30} more" if len(lines) > 30 else ""
    return rows + extra


def gradio_callback(file_obj, progress=None):
    import tempfile
    if file_obj is None:
        return _verdict_html("INCONCLUSIVE", 0.0), "Upload a file.", "", None, None, {}
    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    ext = path.suffix.lower()
    out_dir = Path(tempfile.mkdtemp(prefix="chatfx_"))
    try:
        if ext in IMAGE_EXTS:
            if progress is not None:
                progress(0.2, desc="Running forensics…")
            res = analyze_image(path, out_dir)
            details_md = (
                "| Signal | Score |\n|---|---|\n"
                f"| ML forgery P(fake) | {res.ml_fake_prob if res.ml_fake_prob is not None else 'N/A'} |\n"
                f"| JPEG-ghost score | {res.ghost_score:.3f} |\n"
                f"| ELA score | {res.ela_score:.3f} |\n"
                f"| **Combined** | **{res.combined_score:.3f}** |\n"
                f"| Latency | {res.latency_ms:.0f} ms |\n\n"
                f"_{res.note}_"
            )
            ocr_md = _ocr_md(res.ocr_lines)
            verdict_html = _verdict_html(res.verdict, res.combined_score)
            raw = asdict(res)
            return verdict_html, details_md, ocr_md, res.ela_heatmap_path, res.ghost_heatmap_path, raw

        if ext in VIDEO_EXTS:
            if progress is not None:
                progress(0.2, desc="Sampling frames…")
            res = analyze_video(path)
            details_md = (
                "| Signal | Value |\n|---|---|\n"
                f"| Frames sampled | {res.n_frames_sampled} |\n"
                f"| Mean combined | {res.mean_combined:.3f} |\n"
                f"| Max combined | {res.max_combined:.3f} |\n"
                f"| Score variance | {res.score_variance:.3f} |\n"
                f"| Latency | {res.latency_ms:.0f} ms |\n\n"
                f"_{res.note}_"
            )
            verdict_html = _verdict_html(res.verdict, res.mean_combined)
            raw = asdict(res)
            raw.pop("sample_frames", None)
            return verdict_html, details_md, "_(OCR not run on video.)_", None, None, raw

        return (_verdict_html("INCONCLUSIVE", 0.0),
                f"Unsupported extension: {ext}", "", None, None, {})
    except Exception as ex:
        log.exception("analyze failed")
        return (_verdict_html("INCONCLUSIVE", 0.0),
                f"Error: {ex}", "", None, None, {"error": str(ex)})


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Chat Screenshot Forensics") as demo:
        gr.Markdown("# 📱 Chat Screenshot & Video Forensics")
        gr.Markdown(
            "<small>Detects suspected edits in WhatsApp / Messenger / Telegram "
            "screenshots & screen recordings using JPEG-ghost analysis, ELA, "
            "an HF forgery ViT, and Arabic+English OCR. Heuristic, not forensic-grade.</small>"
        )
        with gr.Row():
            with gr.Column():
                f = gr.File(
                    label="Upload screenshot or screen recording",
                    file_types=[".jpg", ".jpeg", ".png", ".webp",
                                ".mp4", ".mov", ".avi"],
                    type="filepath",
                )
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.HTML()
                details = gr.Markdown()
        with gr.Row():
            ela_img = gr.Image(label="ELA heatmap", height=280)
            ghost_img = gr.Image(label="JPEG-ghost heatmap", height=280)
        with gr.Accordion("OCR (extracted chat text)", open=True):
            ocr_md = gr.Markdown()
        with gr.Accordion("Raw output (JSON)", open=False):
            raw = gr.JSON()

        def _on_click(f_, progress=gr.Progress()):
            return gradio_callback(f_, progress)

        btn.click(_on_click, inputs=[f],
                  outputs=[verdict, details, ocr_md, ela_img, ghost_img, raw])
    return demo


# ---- Tests ----------------------------------------------------------------
class _Tests(unittest.TestCase):
    def _solid_jpeg(self, tmp: Path, size=(200, 200), color=(120, 130, 140)) -> Path:
        p = tmp / "solid.jpg"
        Image.new("RGB", size, color).save(p, "JPEG", quality=85)
        return p

    def _spliced_jpeg(self, tmp: Path) -> Path:
        """Create a jpeg, modify a region with different content, re-save → has compression mismatch."""
        p = tmp / "spliced.jpg"
        a = Image.new("RGB", (200, 200), (120, 130, 140))
        a.save(p, "JPEG", quality=70)
        a2 = Image.open(p).convert("RGB")
        # Paste a high-quality patch on top
        patch = Image.new("RGB", (60, 60), (240, 80, 60))
        a2.paste(patch, (70, 70))
        a2.save(p, "JPEG", quality=95)  # different quality → ghost signal
        return p

    def test_ela_returns_score_and_heatmap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._solid_jpeg(Path(td))
            score, heat = ela_analysis(Image.open(p))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertEqual(heat.dtype, np.uint8)
            self.assertEqual(heat.shape, (200, 200))

    def test_ghost_score_solid_low(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._solid_jpeg(Path(td))
            s, hm = jpeg_ghost(Image.open(p))
            # Solid image: most blocks share a single best Q → low score
            self.assertLessEqual(s, 0.5)
            self.assertEqual(hm.dtype, np.uint8)

    def test_ghost_heatmap_shape(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._solid_jpeg(Path(td), size=(160, 160))
            _, hm = jpeg_ghost(Image.open(p), block=16)
            self.assertEqual(hm.shape, (10, 10))

    def test_combine_scores_clean(self):
        s, v, _ = combine_scores(0.05, 0.05, 0.05)
        self.assertEqual(v, "LIKELY CLEAN")
        self.assertLess(s, CLEAN_THRESH)

    def test_combine_scores_dirty(self):
        s, v, _ = combine_scores(0.9, 0.9, 0.9)
        self.assertEqual(v, "LIKELY EDITED")
        self.assertGreater(s, SUSPICIOUS_THRESH)

    def test_combine_scores_no_ml(self):
        s, v, note = combine_scores(None, 0.8, 0.2)
        # 0.7 * 0.8 + 0.3 * 0.2 = 0.62 → suspicious
        self.assertEqual(v, "LIKELY EDITED")
        self.assertIn("ML model unavailable", note)

    def test_combine_scores_inconclusive(self):
        s, v, _ = combine_scores(0.4, 0.4, 0.4)
        self.assertEqual(v, "INCONCLUSIVE")

    def test_ml_score_handles_none(self):
        # When pipeline is unavailable, ml_forgery_score returns None
        global _forgery_pipe
        _forgery_pipe = False  # sentinel
        try:
            res = ml_forgery_score(Image.new("RGB", (32, 32), (10, 10, 10)))
            self.assertIsNone(res)
        finally:
            _forgery_pipe = None

    def test_synthetic_video_iter(self):
        import tempfile
        import cv2
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "v.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(p), fourcc, 10.0, (64, 64))
            for i in range(20):
                f = np.full((64, 64, 3), (i * 10) % 255, dtype=np.uint8)
                vw.write(f)
            vw.release()
            frames = list(iter_video_frames(p, sample_fps=2.0))
            self.assertGreater(len(frames), 0)
            self.assertIsInstance(frames[0], Image.Image)


def run_tests() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(_Tests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


# ---- Entry ----------------------------------------------------------------
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_tests())
    build_ui().launch(inbrowser=True)
