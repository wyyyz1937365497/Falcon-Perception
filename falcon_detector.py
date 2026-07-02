"""
Falcon-Perception 模型适配模块。

封装模型加载 + 推理，供 falcon_inference_server.py 使用。
"""

import numpy as np
import torch
from PIL import Image
from typing import List

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
    def generate(self, image_pil: Image.Image, query: str, task: str = "detection") -> List:
        """
        运行推理，返回 sequences 列表。

        调用方自行从 seq.output_aux 提取 bboxes_raw / masks_rle。
        """
        prompt = build_prompt_for_task(query, task)
        sampling_params = SamplingParams(stop_token_ids=self.stop_token_ids)

        sequences = [Sequence(
            text=prompt, image=image_pil,
            min_image_size=256, max_image_size=1024, task=task,
        )]
        self.engine.generate(sequences, sampling_params=sampling_params, use_tqdm=False, print_stats=False)

        return sequences
