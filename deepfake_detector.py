"""
deepfake_detector.py — End-to-end image & video deepfake detection with Gradio UI.

INSTALL:
    pip install gradio transformers torch pillow opencv-python-headless numpy

RUN:
    python deepfake_detector.py              # launches UI at http://localhost:7860
    python deepfake_detector.py --test       # runs unit tests

MODELS USED:
    Image: dima806/deepfake_vs_real_image_detection (ViT, 779K downloads)
    Video: Ammar2k/videomae-base-finetuned-deepfake-subset (VideoMAE)
    First run downloads ~700MB total, cached to ~/.cache/huggingface/

WHAT THIS DETECTS:
    AI-generated faces (StyleGAN, ThisPersonDoesNotExist)
    Face-swap deepfakes (FaceSwap, DeepFaceLab)
    Some lip-sync deepfakes (better with VideoMAE)

WHAT THIS DOES NOT DETECT:
    Photoshop edits to non-face content (e.g., text in chat bubbles)
    Newer diffusion video models (Sora, Veo) — postdates training data
    Spliced/concatenated clips — needs metadata analysis instead
    Screen recordings of edited apps — needs UI heuristics instead

For those cases, use a separate forensic toolkit (see project README).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from unittest.mock import patch

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("deepfake_detector")

# ---- Constants ------------------------------------------------------------
IMAGE_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
VIDEO_MODEL_ID = "Ammar2k/videomae-base-finetuned-deepfake-subset"
NUM_VIDEOMAE_FRAMES = 16
NUM_VIDEOMAE_CLIPS = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Ensemble thresholds
IMG_WEIGHT = 0.6
VID_WEIGHT = 0.4
DISAGREE_THRESH = 0.4
FAKE_THRESH = 0.6
REAL_THRESH = 0.3


# ---- Data classes ---------------------------------------------------------
@dataclass
class ImageResult:
    fake_prob: float
    real_prob: float
    label: str
    confidence: float
    latency_ms: float


@dataclass
class VideoResult:
    image_path_score: float
    videomae_score: Optional[float]
    ensemble_label: str
    ensemble_confidence: float
    n_frames_sampled: int
    n_clips_analyzed: int
    disagreement: float
    note: str
    latency_ms: float
    sampled_frames: list = field(default_factory=list)  # PIL images for preview


# ---- Model loading (lazy) -------------------------------------------------
_image_pipe = None
_video_processor = None
_video_model = None
_device = None


def get_device() -> str:
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Using device: %s", _device)
    return _device


def get_image_detector():
    """Lazy-load the image deepfake detection pipeline."""
    global _image_pipe
    if _image_pipe is None:
        from transformers import pipeline
        log.info("Loading image model: %s", IMAGE_MODEL_ID)
        device = 0 if get_device() == "cuda" else -1
        _image_pipe = pipeline(
            "image-classification",
            model=IMAGE_MODEL_ID,
            device=device,
        )
        log.info("Image model loaded.")
    return _image_pipe


def get_video_detector():
    """Lazy-load VideoMAE processor + model."""
    global _video_processor, _video_model
    if _video_processor is None or _video_model is None:
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        log.info("Loading video model: %s", VIDEO_MODEL_ID)
        _video_processor = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_ID)
        _video_model = VideoMAEForVideoClassification.from_pretrained(VIDEO_MODEL_ID)
        _video_model.to(get_device())
        _video_model.eval()
        log.info("Video model loaded.")
    return _video_processor, _video_model


# ---- Label mapping --------------------------------------------------------
def find_fake_index(id2label: dict) -> int:
    """Find the index of the 'fake' class.

    Models may use 'Fake'/'Real', 'fake'/'real', 'FAKE_class', etc.
    Match by substring (case-insensitive). Fallback: assume binary [real, fake].
    """
    for idx, name in id2label.items():
        if "fake" in str(name).lower():
            return int(idx)
    # Fallback: assume binary order [real, fake]
    log.warning("No 'fake' label found in %s; assuming index 1 is fake.", id2label)
    return 1


# ---- Frame utilities ------------------------------------------------------
def iter_video_frames(path: Path, sample_fps: float = 1.0) -> Iterator[Image.Image]:
    """Yield PIL frames from `path` sampled at `sample_fps` per second."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / max(0.1, sample_fps))))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb)
            i += 1
            if total and i >= total:
                break
    finally:
        cap.release()


def extract_videomae_clip(path: Path, n_frames: int = NUM_VIDEOMAE_FRAMES,
                          start_frac: float = 0.0, end_frac: float = 1.0) -> np.ndarray:
    """Extract `n_frames` evenly-spaced frames from [start_frac, end_frac] of the video.

    Returns array shape (n_frames, H, W, 3) uint8 RGB.
    Pads by repeating the last frame if the video is shorter than n_frames.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # Read all and count
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            total = len(frames)
            cap.release()
            if total == 0:
                raise RuntimeError(f"No frames in video: {path}")
            # Pick indices
            start = int(start_frac * total)
            end = max(start + 1, int(end_frac * total))
            idxs = np.linspace(start, end - 1, n_frames).round().astype(int)
            idxs = np.clip(idxs, 0, total - 1)
            picked = [frames[i] for i in idxs]
            return np.stack(picked, axis=0)

        start = int(start_frac * total)
        end = max(start + 1, int(end_frac * total))
        idxs = np.linspace(start, end - 1, n_frames).round().astype(int)
        idxs = np.clip(idxs, 0, total - 1)

        out = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if not ok:
                # Fallback: reuse last successful frame, or a black one
                if out:
                    out.append(out[-1].copy())
                else:
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 224)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 224)
                    out.append(np.zeros((h, w, 3), dtype=np.uint8))
                continue
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        # Pad to n_frames if needed
        while len(out) < n_frames:
            out.append(out[-1].copy() if out else np.zeros((224, 224, 3), dtype=np.uint8))
        return np.stack(out[:n_frames], axis=0)
    finally:
        cap.release()


# ---- Inference ------------------------------------------------------------
def _scores_to_pfake(predictions: list, id2label: dict) -> float:
    """Convert pipeline output [{label, score}, ...] to P(fake)."""
    fake_idx = find_fake_index(id2label)
    fake_name = id2label[fake_idx]
    for p in predictions:
        if str(p["label"]) == str(fake_name):
            return float(p["score"])
    # fallback by substring
    for p in predictions:
        if "fake" in str(p["label"]).lower():
            return float(p["score"])
    # if labels are real/fake binary, return 1 - real
    for p in predictions:
        if "real" in str(p["label"]).lower():
            return 1.0 - float(p["score"])
    return 0.5


def analyze_image(path: Path) -> ImageResult:
    """Run image deepfake detection on a single image."""
    t0 = time.perf_counter()
    pipe = get_image_detector()
    img = Image.open(path).convert("RGB")
    preds = pipe(img, top_k=None)
    id2label = pipe.model.config.id2label
    p_fake = _scores_to_pfake(preds, id2label)
    p_real = 1.0 - p_fake
    label = "FAKE" if p_fake > 0.5 else "REAL"
    confidence = max(p_fake, p_real)
    latency = (time.perf_counter() - t0) * 1000.0
    return ImageResult(
        fake_prob=p_fake, real_prob=p_real, label=label,
        confidence=confidence, latency_ms=latency,
    )


def _run_videomae(path: Path) -> float:
    """Run VideoMAE on NUM_VIDEOMAE_CLIPS evenly-spaced clips. Returns mean P(fake)."""
    import torch
    processor, model = get_video_detector()
    id2label = model.config.id2label
    fake_idx = find_fake_index(id2label)
    device = get_device()

    scores = []
    # Evenly spaced segments: e.g., 3 clips → [0,1/3], [1/3,2/3], [2/3,1]
    for k in range(NUM_VIDEOMAE_CLIPS):
        s = k / NUM_VIDEOMAE_CLIPS
        e = (k + 1) / NUM_VIDEOMAE_CLIPS
        try:
            clip = extract_videomae_clip(path, NUM_VIDEOMAE_FRAMES, s, e)
        except Exception as ex:
            log.warning("Clip %d extraction failed: %s", k, ex)
            continue
        # processor expects list of frames per video
        frames_list = [clip[i] for i in range(clip.shape[0])]
        inputs = processor(frames_list, return_tensors="pt")
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        scores.append(float(probs[fake_idx]))
        # Free intermediate memory
        del inputs, logits, probs
    if not scores:
        raise RuntimeError("All VideoMAE clips failed.")
    return float(np.mean(scores))


def analyze_video(path: Path, sample_fps: float = 1.0) -> VideoResult:
    """Run combined image+video deepfake detection on a video file."""
    t0 = time.perf_counter()

    # Pass 1: sample frames, run image model on each
    pipe = get_image_detector()
    id2label_img = pipe.model.config.id2label
    frame_scores = []
    sampled = []
    for fr in iter_video_frames(path, sample_fps):
        sampled.append(fr)
        preds = pipe(fr, top_k=None)
        frame_scores.append(_scores_to_pfake(preds, id2label_img))
    if not frame_scores:
        raise RuntimeError(f"No frames sampled from {path}")
    img_score = float(np.mean(frame_scores))

    # Pass 2: VideoMAE on 3 clips
    note = ""
    try:
        vid_score = _run_videomae(path)
        clips = NUM_VIDEOMAE_CLIPS
    except Exception as ex:
        log.warning("VideoMAE failed (%s); using image-only.", ex)
        vid_score = None
        clips = 0
        note = f"VideoMAE unavailable: {ex}. Falling back to image-only."

    # Ensemble
    if vid_score is None:
        # image-only fallback: pretend videomae == img to keep ensemble logic
        label, conf, ens_note = ensemble(img_score, img_score)
        note = (note + " " + ens_note).strip()
        disagree = 0.0
    else:
        label, conf, ens_note = ensemble(img_score, vid_score)
        disagree = abs(img_score - vid_score)
        note = (note + " " + ens_note).strip() if note else ens_note

    latency = (time.perf_counter() - t0) * 1000.0
    return VideoResult(
        image_path_score=img_score,
        videomae_score=vid_score,
        ensemble_label=label,
        ensemble_confidence=conf,
        n_frames_sampled=len(frame_scores),
        n_clips_analyzed=clips,
        disagreement=disagree,
        note=note,
        latency_ms=latency,
        sampled_frames=sampled[:12],  # keep gallery small
    )


# ---- Ensemble combiner ----------------------------------------------------
def ensemble(image_p_fake: float, video_p_fake: float) -> tuple[str, float, str]:
    """Combine image and video model scores.

    Rules:
      disagreement = |image - video|
      weighted = 0.6*image + 0.4*video
      if disagreement > 0.4 -> INCONCLUSIVE (models disagree)
      elif weighted > 0.6   -> FAKE
      elif weighted < 0.3   -> REAL
      else                  -> INCONCLUSIVE
    Returns (label, confidence, note).
    """
    weighted = IMG_WEIGHT * image_p_fake + VID_WEIGHT * video_p_fake
    disagreement = abs(image_p_fake - video_p_fake)

    if disagreement > DISAGREE_THRESH:
        return ("INCONCLUSIVE", 0.4,
                f"Models disagree (Δ={disagreement:.2f}). Manual review needed.")
    if weighted > FAKE_THRESH:
        conf = min(1.0, weighted + 0.1)
        return ("FAKE", conf,
                f"Both models indicate manipulation (weighted P(fake)={weighted:.2f}).")
    if weighted < REAL_THRESH:
        conf = 1.0 - weighted
        return ("REAL", conf,
                f"Both models indicate authentic content (weighted P(fake)={weighted:.2f}).")
    return ("INCONCLUSIVE", 0.5,
            f"Borderline scores (weighted P(fake)={weighted:.2f}); not confident.")


# ---- Gradio UI ------------------------------------------------------------
def _verdict_html(label: str, confidence: float) -> str:
    colors = {
        "REAL": ("#1b5e20", "#c8e6c9", "🟢"),
        "FAKE": ("#b71c1c", "#ffcdd2", "🔴"),
        "INCONCLUSIVE": ("#f57f17", "#fff9c4", "🟡"),
    }
    fg, bg, icon = colors.get(label, ("#333", "#eee", "❓"))
    return (
        f"<div style='padding:24px;border-radius:12px;background:{bg};"
        f"color:{fg};text-align:center;font-weight:700;'>"
        f"<div style='font-size:36px;'>{icon} {label}</div>"
        f"<div style='font-size:18px;margin-top:8px;'>"
        f"Confidence: {confidence*100:.1f}%</div>"
        f"</div>"
    )


def _details_md(img_score: float, vid_score, ensemble_score: float,
                n_frames: int, latency_ms: float) -> str:
    vid_str = f"{vid_score:.3f}" if vid_score is not None else "N/A"
    return (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Image model P(fake) | {img_score:.3f} |\n"
        f"| Video model P(fake) | {vid_str} |\n"
        f"| Ensemble score | {ensemble_score:.3f} |\n"
        f"| Frames sampled | {n_frames} |\n"
        f"| Latency | {latency_ms:.0f} ms |\n"
    )


def gradio_callback(file_obj, sample_fps: float, progress=None):
    """Main UI callback: detect type, dispatch, and format output."""
    if file_obj is None:
        return _verdict_html("INCONCLUSIVE", 0.0), "Please upload a file.", "", [], {}

    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    ext = path.suffix.lower()

    try:
        if ext in IMAGE_EXTS:
            if progress is not None:
                progress(0.2, desc="Loading image model...")
            res = analyze_image(path)
            if progress is not None:
                progress(1.0, desc="Done")
            verdict_html = _verdict_html(res.label, res.confidence)
            details = _details_md(res.fake_prob, None, res.fake_prob, 1, res.latency_ms)
            reason = (f"Image classified as **{res.label}** with P(fake)={res.fake_prob:.3f}. "
                      f"Single-image inference; no video signal available.")
            gallery = [str(path)]
            raw = asdict(res)
            return verdict_html, details, reason, gallery, raw

        elif ext in VIDEO_EXTS:
            if progress is not None:
                progress(0.1, desc="Loading models...")
            # warm up
            get_image_detector()
            if progress is not None:
                progress(0.3, desc="Sampling frames...")
            res = analyze_video(path, sample_fps=sample_fps)
            if progress is not None:
                progress(1.0, desc="Done")
            ens_score = (IMG_WEIGHT * res.image_path_score
                         + VID_WEIGHT * (res.videomae_score
                                         if res.videomae_score is not None
                                         else res.image_path_score))
            verdict_html = _verdict_html(res.ensemble_label, res.ensemble_confidence)
            details = _details_md(
                res.image_path_score, res.videomae_score, ens_score,
                res.n_frames_sampled, res.latency_ms,
            )
            reason = res.note
            gallery = res.sampled_frames
            raw = asdict(res)
            raw.pop("sampled_frames", None)  # don't json-dump PIL images
            return verdict_html, details, reason, gallery, raw

        else:
            return (_verdict_html("INCONCLUSIVE", 0.0),
                    f"Unsupported file extension: {ext}",
                    "", [], {})
    except Exception as ex:
        log.exception("Inference failed")
        return (_verdict_html("INCONCLUSIVE", 0.0),
                f"Error: {ex}", "", [], {"error": str(ex)})


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("# 🔍 Deepfake Detector — Image & Video")
        gr.Markdown(
            "<small>Powered by dima806 (image) + Ammar2k VideoMAE (video). "
            "Detects AI-generated faces and face-swap deepfakes. "
            "Does NOT detect Photoshop edits to text in chat bubbles.</small>"
        )

        with gr.Row():
            with gr.Column():
                file_in = gr.File(
                    label="Upload image or video",
                    file_types=[".jpg", ".jpeg", ".png", ".webp",
                                ".mp4", ".mov", ".avi"],
                    type="filepath",
                )
                fps_slider = gr.Slider(
                    minimum=0.5, maximum=4.0, value=1.0, step=0.5,
                    label="Sample FPS (videos only)",
                    info="Higher = more thorough but slower.",
                )
                run_btn = gr.Button("Analyze", variant="primary")

            with gr.Column():
                verdict = gr.HTML()
                details = gr.Markdown()
                with gr.Accordion("Why this verdict", open=False):
                    reason = gr.Markdown()
                gallery = gr.Gallery(label="Preview", columns=4, height=240)
                with gr.Accordion("Raw output (JSON)", open=False):
                    raw = gr.JSON()

        def _on_click(f, fps, progress=gr.Progress()):
            return gradio_callback(f, fps, progress)

        run_btn.click(
            _on_click,
            inputs=[file_in, fps_slider],
            outputs=[verdict, details, reason, gallery, raw],
        )

    return demo


# ---- Tests ----------------------------------------------------------------
class _Tests(unittest.TestCase):
    def test_ensemble_real(self):
        label, conf, _ = ensemble(0.05, 0.10)
        self.assertEqual(label, "REAL")
        self.assertGreater(conf, 0.5)

    def test_ensemble_fake(self):
        label, conf, _ = ensemble(0.85, 0.80)
        self.assertEqual(label, "FAKE")
        self.assertGreater(conf, 0.5)

    def test_ensemble_disagreement(self):
        label, _, note = ensemble(0.10, 0.85)
        self.assertEqual(label, "INCONCLUSIVE")
        self.assertIn("disagree", note.lower())

    def test_ensemble_borderline(self):
        label, _, _ = ensemble(0.45, 0.45)
        self.assertEqual(label, "INCONCLUSIVE")

    def test_label_mapping_robustness(self):
        self.assertEqual(find_fake_index({0: "Real", 1: "Fake"}), 1)
        self.assertEqual(find_fake_index({0: "REAL_class", 1: "FAKE_class"}), 1)
        self.assertEqual(find_fake_index({0: "fake", 1: "real"}), 0)
        # No fake label → fallback
        self.assertEqual(find_fake_index({0: "cat", 1: "dog"}), 1)

    def test_scores_to_pfake(self):
        preds = [{"label": "Fake", "score": 0.7}, {"label": "Real", "score": 0.3}]
        self.assertAlmostEqual(_scores_to_pfake(preds, {0: "Real", 1: "Fake"}), 0.7)

    def _make_synthetic_video(self, tmpdir: Path, n_frames: int = 30,
                              w: int = 64, h: int = 64) -> Path:
        import cv2
        out_path = tmpdir / "synth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (w, h))
        for i in range(n_frames):
            frame = np.full((h, w, 3), (i * 8) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return out_path

    def test_extract_videomae_clip_padding(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 5-frame video, request 16 → padding
            v = self._make_synthetic_video(tmp, n_frames=5)
            clip = extract_videomae_clip(v, n_frames=16)
            self.assertEqual(clip.shape[0], 16)
            self.assertEqual(clip.ndim, 4)

    def test_extract_videomae_clip_uniform(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            v = self._make_synthetic_video(tmp, n_frames=64)
            clip = extract_videomae_clip(v, n_frames=16)
            self.assertEqual(clip.shape[0], 16)

    def test_synthetic_video_iter(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            v = self._make_synthetic_video(tmp, n_frames=30)
            frames = list(iter_video_frames(v, sample_fps=2.0))
            self.assertGreater(len(frames), 0)
            self.assertIsInstance(frames[0], Image.Image)

    def test_analyze_image_mocked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            Image.new("RGB", (32, 32), (200, 100, 100)).save(p)

            class FakeModelCfg:
                id2label = {0: "Real", 1: "Fake"}

            class FakeModel:
                config = FakeModelCfg()

            class FakePipe:
                model = FakeModel()

                def __call__(self, img, top_k=None):
                    return [{"label": "Fake", "score": 0.9},
                            {"label": "Real", "score": 0.1}]

            with patch("__main__.get_image_detector", return_value=FakePipe()):
                # patch via the global as well
                global _image_pipe
                _image_pipe = FakePipe()
                res = analyze_image(p)
                self.assertEqual(res.label, "FAKE")
                self.assertAlmostEqual(res.fake_prob, 0.9, places=5)
                _image_pipe = None


def run_tests() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(_Tests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---- Entry ----------------------------------------------------------------
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_tests())
    build_ui().launch(inbrowser=True)
