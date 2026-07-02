"""Falcon-Perception inference server.

Runs in the ``transformerv`` conda environment. Loads the Falcon model once
at startup and serves segmentation requests via HTTP so the ``bim-recon``
pipeline (separate environment with gsplat) can call it.

Usage::

    conda activate transformerv
    cd G:\\TJ\\BIM\\Falcon-Perception
    python falcon_inference_server.py --port 8390

Endpoints:
    GET  /health       — liveness check
    POST /segment      — run segmentation on one image
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ── ensure the local falcon_detector wrapper is importable ─────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("falcon-server")

# ── globals ────────────────────────────────────────────────────────────
_model = None  # FalconPerceptionModel, loaded at startup


# ── request / response schemas ─────────────────────────────────────────

class SegmentRequest(BaseModel):
    image_b64: str              # base64-encoded PNG/JPEG
    query: str                  # e.g. "window", "door"
    task: str = "segmentation"  # "segmentation" or "detection"


class DetectionEntry(BaseModel):
    bbox: dict                       # {"x","y","w","h"} normalized [0,1]
    mask_bbox: Optional[dict] = None  # not available with detection-only
    mask_area_ratio: Optional[float] = None


class SegmentResponse(BaseModel):
    detections: list[DetectionEntry]
    image_width: int
    image_height: int


# ── lifespan (startup) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load Falcon model + segmentation warmup at server startup."""
    global _model
    from falcon_detector import FalconPerceptionModel

    model_dir = os.environ.get("FALCON_MODEL_DIR")
    logger.info("Loading Falcon-Perception model (dir=%s) ...", model_dir)

    # Step 1: load model — defaults (float32, compile=True, capture_cudagraph=True)
    _model = FalconPerceptionModel(
        hf_local_dir=model_dir,
        device="cuda",
    )

    # Step 2: segmentation warmup
    # __init__ warmup only does detection; segmentation needs different
    # torch.compile paths (mask upsampling). Do it now so first request
    # doesn't timeout during compilation.
    logger.info("Segmentation warmup (compiling mask paths, may take 1-3 min)...")
    from falcon_perception import build_prompt_for_task
    from falcon_perception.paged_inference import SamplingParams, Sequence

    dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    prompt = build_prompt_for_task("warmup", "segmentation")
    sp = SamplingParams(stop_token_ids=_model.stop_token_ids)
    seqs = [Sequence(
        text=prompt, image=dummy,
        min_image_size=256, max_image_size=1024, task="segmentation",
    )]
    t0 = time.time()
    _model.engine.generate(seqs, sampling_params=sp, use_tqdm=False, print_stats=False)
    logger.info("Segmentation warmup done (%.1fs).", time.time() - t0)

    logger.info("Falcon model ready (detection + segmentation compiled).")
    yield


# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(title="Falcon-Perception Inference Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading"}


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    """Run Falcon detection on the provided image.

    直接调用 model.detect() — 与 rtsp_detection_service.py 完全相同的路径。
    task 固定为 "detection"（segmentation 在 RTX 2080Ti 上会触发
    torch.compile 重新编译导致超时）。
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    # decode image
    raw = base64.b64decode(req.image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_w, img_h = img.size

    # run inference
    sequences = _model.generate(img, req.query, task=req.task)
    seq = sequences[0]

    from falcon_perception.visualization_utils import pair_bbox_entries, _mask_to_bbox_xywh
    from pycocotools import mask as mask_utils

    bboxes = pair_bbox_entries(seq.output_aux.bboxes_raw)
    masks_rle = seq.output_aux.masks_rle if req.task == "segmentation" else []

    # build response — merge mask extents into detection entries
    entries: list[DetectionEntry] = []
    for i, bb in enumerate(bboxes):
        entry = DetectionEntry(bbox=dict(bb))

        if i < len(masks_rle) and masks_rle[i]:
            rle = masks_rle[i]
            mh, mw = rle["size"]
            counts = rle["counts"]
            if isinstance(counts, str):
                counts = counts.encode("utf-8")

            mask_arr = mask_utils.decode({"counts": counts, "size": [mh, mw]})
            mask_area = int(np.asarray(mask_arr).sum())
            total = mh * mw

            # _mask_to_bbox_xywh returns (xy_dict, hw_dict) — merge into one
            xy_dict, hw_dict = _mask_to_bbox_xywh(mask_arr, mw, mh)
            entry.mask_bbox = {**xy_dict, **hw_dict}
            entry.mask_area_ratio = round(mask_area / total, 4) if total > 0 else 0.0

        entries.append(entry)

    return SegmentResponse(
        detections=entries,
        image_width=img_w,
        image_height=img_h,
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Falcon inference server")
    parser.add_argument("--model-dir", type=str,
                        default=str(_THIS_DIR / "weight" / "Falcon-Perception"),
                        help="local HF model directory (overrides HF download)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8390)
    args = parser.parse_args()

    if args.model_dir:
        os.environ["FALCON_MODEL_DIR"] = args.model_dir

    uvicorn.run(app, host=args.host, port=args.port)
