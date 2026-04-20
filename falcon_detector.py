"""
Falcon-Perception 适配模块
封装 Falcon-Perception 模型的加载和推理。
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import ImageProcessor
from falcon_perception.flex_attention_config import resolve_flex_kernel_options
from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence
from falcon_perception.visualization_utils import pair_bbox_entries

setup_torch_config()


class FalconPerceptionModel:
    """封装 Falcon-Perception 模型加载和推理。"""

    def __init__(
        self,
        hf_local_dir: str = None,
        hf_model_id: str = None,
        device: str = None,
        dtype: str = "float32",
        compile: bool = True,
    ):
        print("加载 Falcon-Perception 模型...")
        if hf_local_dir:
            print(f"  本地模型目录: {hf_local_dir}")

        self.model, self.tokenizer, self.model_args = load_and_prepare_model(
            hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
            hf_revision="main",
            hf_local_dir=hf_local_dir,
            device=device,
            dtype=dtype,
            compile=compile,
        )
        self.device = self.model.device

        self.kernel_options = resolve_flex_kernel_options(
            device=self.device,
            user_kernel_options=None,
            force_safe=None,
        )

        self.image_processor = ImageProcessor(patch_size=16, merge_size=1)
        self.stop_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.end_of_query_token_id]

        self.engine = PagedInferenceEngine(
            self.model, self.tokenizer, self.image_processor,
            max_batch_size=2,
            max_seq_length=8192,
            n_pages=128,
            page_size=128,
            prefill_length_limit=8192,
            enable_hr_cache=False,
            capture_cudagraph=True,
            kernel_options=self.kernel_options,
        )

        print("Warmup run ...")
        self._warmup()
        print("Falcon-Perception 模型加载完成")

    @torch.inference_mode()
    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        prompt = build_prompt_for_task("warmup", "detection")
        sampling_params = SamplingParams(stop_token_ids=self.stop_token_ids)

        warmup_seqs = [Sequence(
            text=prompt, image=dummy_image,
            min_image_size=256, max_image_size=1024, task="detection",
        )]
        with cuda_timed(reset_peak_memory=False) as timer:
            self.engine.generate(warmup_seqs, sampling_params=sampling_params, use_tqdm=False, print_stats=False)
        print(f"  Warmup 完成 ({timer.elapsed:.1f}s)")

    @torch.inference_mode()
    def detect(self, image_pil: Image.Image, query: str, task: str = "detection") -> List[Dict]:
        """
        运行检测/分割推理。

        Returns:
            pair_bbox_entries 格式: [{"x", "y", "h", "w"}, ...]
            x, y 为归一化中心坐标，h, w 为归一化尺寸 (0-1)
        """
        prompt = build_prompt_for_task(query, task)
        sampling_params = SamplingParams(stop_token_ids=self.stop_token_ids)

        sequences = [Sequence(
            text=prompt, image=image_pil,
            min_image_size=256, max_image_size=1024, task=task,
        )]
        self.engine.generate(sequences, sampling_params=sampling_params, use_tqdm=False, print_stats=False)

        seq = sequences[0]
        return pair_bbox_entries(seq.output_aux.bboxes_raw)


def detect_objects(
    fp_model: FalconPerceptionModel,
    image_np: np.ndarray,
    caption: str,
    max_box_area: int = 6000,
    min_box_area: int = 150,
    object_type: str = "bike",
) -> List[Dict]:
    """
    使用 Falcon-Perception 检测图像中的物体。

    Args:
        fp_model: FalconPerceptionModel 实例
        image_np: BGR 格式图像
        caption: 检测提示词 (如 "blue_bike")
        max_box_area: 最大框面积过滤 (像素), 0=不过滤
        min_box_area: 最小框面积过滤 (像素), 0=不过滤
        object_type: 物体类型标记

    Returns:
        检测结果列表
    """
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGB")
    img_h, img_w = image_np.shape[:2]

    bboxes = fp_model.detect(image_pil, caption, task="detection")

    filtered_results = []
    for entry in bboxes:
        cx, cy = entry["x"], entry["y"]
        bw, bh = entry["w"], entry["h"]

        box_w = bw * img_w
        box_h = bh * img_h
        box_area = box_w * box_h

        if max_box_area > 0 and box_area > max_box_area:
            continue
        if min_box_area > 0 and box_area < min_box_area:
            continue

        filtered_results.append({
            "label": caption,
            "object_type": object_type,
            "center_x": float(cx),
            "center_y": float(cy),
            "width": float(box_w),
            "height": float(box_h),
            "area": float(box_area),
            "x1": float(cx * img_w - box_w / 2),
            "y1": float(cy * img_h - box_h / 2),
            "x2": float(cx * img_w + box_w / 2),
            "y2": float(cy * img_h + box_h / 2),
        })

    return filtered_results


def detect_trees(
    fp_model: FalconPerceptionModel,
    image: np.ndarray,
    caption: str = "tree",
    min_box_area: int = 500,
    max_box_area: int = 100000,
    min_box_aspect: float = 0.3,
    max_box_aspect: float = 3.0,
    nms_threshold: float = 0.3,
) -> Tuple[List[Dict], np.ndarray]:
    """
    使用 Falcon-Perception 检测图像中的树木。

    Returns:
        (树木列表, 可视化图像)
        树木格式: [{"bbox": [x1,y1,x2,y2], "center": (cx,cy)}, ...]
    """
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    image_pil = Image.fromarray(image_rgb).convert("RGB")
    img_h, img_w = image.shape[:2]

    bboxes = fp_model.detect(image_pil, caption, task="detection")

    raw_trees = []
    for entry in bboxes:
        cx, cy = entry["x"], entry["y"]
        bw, bh = entry["w"], entry["h"]

        bw_px = bw * img_w
        bh_px = bh * img_h
        box_area = bw_px * bh_px
        aspect_ratio = bw_px / bh_px if bh_px > 0 else 0

        if box_area < min_box_area or box_area > max_box_area:
            continue
        if aspect_ratio < min_box_aspect or aspect_ratio > max_box_aspect:
            continue

        cx_px = cx * img_w
        cy_px = cy * img_h
        x1 = max(0, int(cx_px - bw_px / 2))
        y1 = max(0, int(cy_px - bh_px / 2))
        x2 = min(img_w, int(cx_px + bw_px / 2))
        y2 = min(img_h, int(cy_px + bh_px / 2))

        raw_trees.append({
            "bbox": [x1, y1, x2, y2],
            "center": (int(cx_px), int(cy_px)),
        })

    from image_registration import apply_nms_to_trees
    trees = apply_nms_to_trees(raw_trees, nms_threshold)

    vis_image = image.copy()
    for tree in trees:
        x1, y1, x2, y2 = tree["bbox"]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx_t, cy_t = tree["center"]
        cv2.circle(vis_image, (cx_t, cy_t), 3, (0, 255, 0), -1)

    return trees, vis_image


def annotate_detections(image_np: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """在图像上绘制检测框和标签。"""
    annotated = image_np.copy()
    for det in detections:
        x1 = int(det.get("x1", 0))
        y1 = int(det.get("y1", 0))
        x2 = int(det.get("x2", 0))
        y2 = int(det.get("y2", 0))
        label = det.get("label", "")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return annotated
