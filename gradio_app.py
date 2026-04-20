import argparse
from functools import partial
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("警告: ffmpeg-python 未安装，将使用 OpenCV 读取视频")
    print("安装方法: pip install ffmpeg-python")

# 设置日志级别，抑制警告
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow日志（如果有）

import warnings


def _patch_starlette_templateresponse_compat():
    """Compat for Gradio 3.x calling old TemplateResponse(name, context) signature.

    Starlette 1.x expects TemplateResponse(request, name, context).
    """
    try:
        from starlette.templating import Jinja2Templates
    except Exception:
        return

    if getattr(Jinja2Templates, "_fp_old_signature_compat", False):
        return

    original = Jinja2Templates.TemplateResponse

    def _compat(self, *args, **kwargs):
        # Old-style call from Gradio 3.x: TemplateResponse(name, {"request": request, ...})
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = kwargs.pop("request", None) or context.get("request")
            if request is not None:
                try:
                    return original(self, request, name, context, *args[2:], **kwargs)
                except TypeError:
                    # Fall back to original behavior on older Starlette versions.
                    pass
        return original(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _compat
    Jinja2Templates._fp_old_signature_compat = True

# 抑制所有警告
warnings.filterwarnings("ignore")

# Fix Gradio 3.x + Starlette 1.x template signature mismatch.
_patch_starlette_templateresponse_compat()

# 重定向stderr以抑制FFmpeg警告
class SuppressStdErr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.stderr

import torch

# prepare the environment
# os.system("python setup.py build develop --user")
# os.system("pip install packaging==21.3")
# os.system("pip install gradio==3.50.2")


warnings.filterwarnings("ignore")

import gradio as gr

from falcon_detector import FalconPerceptionModel, detect_objects, detect_trees, annotate_detections


# ========== 资源监控 ==========
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil 未安装，无法监控内存使用。安装方法: pip install psutil")


class ResourceMonitor:
    """资源监控器，用于跟踪内存和显存的峰值使用量"""

    def __init__(self):
        self.peak_ram_mb = 0
        self.peak_gpu_mb = 0
        self.lock = threading.Lock()

    def get_ram_usage_mb(self):
        """获取当前RAM使用量（MB）"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        else:
            # 备用方案：使用torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return 0

    def get_gpu_usage_mb(self):
        """获取当前GPU显存使用量（MB）"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return max(allocated, reserved)
        return 0

    def update_peak(self):
        """更新峰值记录"""
        with self.lock:
            ram = self.get_ram_usage_mb()
            gpu = self.get_gpu_usage_mb()

            if ram > self.peak_ram_mb:
                self.peak_ram_mb = ram
            if gpu > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu

    def get_peak_stats(self):
        """获取峰值统计信息"""
        with self.lock:
            return {
                "peak_ram_mb": self.peak_ram_mb,
                "peak_gpu_mb": self.peak_gpu_mb
            }

    def format_size(self, size_mb):
        """格式化大小显示"""
        if size_mb >= 1024:
            return f"{size_mb / 1024:.2f} GB"
        return f"{size_mb:.2f} MB"

    def get_summary(self):
        """获取峰值摘要"""
        stats = self.get_peak_stats()
        return (
            f"📊 资源使用峰值:\n"
            f"  RAM: {self.format_size(stats['peak_ram_mb'])}\n"
            f"  GPU: {self.format_size(stats['peak_gpu_mb'])}"
        )


# 全局资源监控器
resource_monitor = ResourceMonitor()

# ========== 图像配准模块 ==========
try:
    # 尝试导入项目根目录的 image_registration 模块
    import sys
    root_path = Path(__file__).resolve().parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from image_registration import ImageRegistration, OffsetSmoother, DinoTreeGridRegistration, analyze_grid_pattern, find_main_tree_cluster
    IMAGE_REGISTRATION_AVAILABLE = True
except ImportError as e:
    IMAGE_REGISTRATION_AVAILABLE = False
    DinoTreeGridRegistration = None
    print(f"警告: 图像配准模块不可用: {e}")

# 检查scipy是否可用（用于网格配准）
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None
    print("警告: scipy未安装，网格配准功能不可用")

# ========== 配置标记工具 ==========

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


class CameraConfigManager:
    """相机配置管理器"""

    def __init__(self):
        self.current_config: Dict = {}
        self.reference_image: Optional[np.ndarray] = None
        self.landmark_points: List[Dict] = []
        self.detection_regions: List[Dict] = []

    def reset(self):
        """重置配置"""
        self.current_config = {}
        self.reference_image = None
        self.landmark_points = []
        self.detection_regions = []

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        registration_method = self.current_config.get("registration_method", "DINO_TREE_GRID")

        # 基础注册配置
        registration_config = {
            "enabled": True,
            "method": registration_method,
            "smoothing_window": self.current_config.get("smoothing_window", 5)
        }

        # 如果是 DINO 树木网格配准，添加额外参数
        if registration_method == "DINO_TREE_GRID":
            registration_config.update({
                "caption": self.current_config.get("dino_caption", "tree"),
                "device": self.current_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            })

        return {
            "version": "1.0",
            "camera_name": self.current_config.get("camera_name", ""),
            "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "registration": registration_config,
            "detection_regions": self.detection_regions,
            "rtsp_settings": {
                "url": self.current_config.get("rtsp_url", ""),
                "sample_interval": self.current_config.get("sample_interval", 30),
                "output_dir": self.current_config.get("output_dir", "output/rtsp_detection")
            }
        }


# 全局配置管理器
config_manager = CameraConfigManager()


def list_config_files() -> List[str]:
    """列出所有配置文件"""
    config_files = []
    if CONFIGS_DIR.exists():
        for f in CONFIGS_DIR.glob("*.json"):
            config_files.append(str(f))
    return sorted(config_files)


def load_reference_image(image_upload) -> Tuple[str, str, np.ndarray]:
    """
    加载参考图像

    Args:
        image_upload: 上传的图像文件 (PIL Image from Gradio with type="pil")

    Returns:
        (信息文本, 图像路径, 图像数组)
    """
    if image_upload is None:
        return "请上传参考图像", None, None

    try:
        # 保存参考图像
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ref_image_path = CONFIGS_DIR / f"reference_image_{timestamp}.jpg"

        # Gradio with type="pil" returns a PIL Image object directly
        if isinstance(image_upload, Image.Image):
            # PIL Image from Gradio
            image = image_upload
            image.save(ref_image_path)
            image_np = np.array(image)
        elif isinstance(image_upload, str):
            # 文件路径字符串
            image = Image.open(image_upload)
            image.save(ref_image_path)
            image_np = np.array(image)
        elif hasattr(image_upload, 'name'):
            # 上传的文件对象
            image = Image.open(image_upload.name)
            image.save(ref_image_path)
            image_np = np.array(image)
        elif isinstance(image_upload, np.ndarray):
            # numpy array
            image = Image.fromarray(image_upload)
            image.save(ref_image_path)
            image_np = image_upload
        else:
            # 其他类型，尝试转换为numpy再转PIL
            image_np = np.array(image_upload)
            image = Image.fromarray(image_np)
            image.save(ref_image_path)

        # 转换为BGR格式（OpenCV格式）
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        config_manager.reference_image = image_np
        config_manager.current_config["reference_image_path"] = str(ref_image_path)

        # 初始化图像配准器
        if IMAGE_REGISTRATION_AVAILABLE:
            registration_method = config_manager.current_config.get("registration_method", "DINO_TREE_GRID")

            if registration_method == "DINO_TREE_GRID":
                if not SCIPY_AVAILABLE:
                    info = f"⚠️ scipy未安装，无法使用 DINO 树木网格配准\n路径: {ref_image_path}\n尺寸: {image_np.shape[1]}x{image_np.shape[0]}\n请安装: pip install scipy"
                    return info, str(ref_image_path), cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                # 使用全局加载的 DINO 模型
                reg_config = {
                    "caption": config_manager.current_config.get("dino_caption", "tree"),
                    "expected_rows": config_manager.current_config.get("grid_rows", 3),
                    "expected_cols": config_manager.current_config.get("grid_cols", 4),
                    "device": config_manager.current_config.get("device", DEVICE),
                    "smoothing_window": config_manager.current_config.get("smoothing_window", 5)
                }
                config_manager.registration = DinoTreeGridRegistration(model=model, config=reg_config)
                success = config_manager.registration.set_reference_image(image_np)

                if success:
                    # 获取检测到的树木数量
                    if hasattr(config_manager.registration, 'reference_trees') and config_manager.registration.reference_trees:
                        detected_count = len(config_manager.registration.reference_trees)
                        grid_count = len(config_manager.registration.reference_grid_points) if config_manager.registration.reference_grid_points is not None else 0
                        info = f"✓ 参考图像已加载（DINO 树木网格模式）\n路径: {ref_image_path}\n尺寸: {image_np.shape[1]}x{image_np.shape[0]}\n检测到树木: {detected_count} 棵\n识别网格点: {grid_count} 个\n网格配置: {reg_config['expected_rows']}x{reg_config['expected_cols']}"
                    else:
                        info = f"✓ 参考图像已加载（DINO 树木网格模式）\n路径: {ref_image_path}\n尺寸: {image_np.shape[1]}x{image_np.shape[0]}"
                else:
                    info = f"⚠️ 参考图像加载失败（DINO 树木网格模式）\n路径: {ref_image_path}"

            else:
                info = f"✓ 参考图像已加载\n路径: {ref_image_path}\n尺寸: {image_np.shape[1]}x{image_np.shape[0]}"
        else:
            info = f"✓ 参考图像已加载\n路径: {ref_image_path}\n尺寸: {image_np.shape[1]}x{image_np.shape[0]}"

        return info, str(ref_image_path), cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    except Exception as e:
        return f"加载参考图像失败: {e}", None, None


def add_landmark_point(evt: gr.SelectData, current_image, landmark_points_json: str) -> Tuple[str, np.ndarray, str]:
    """
    添加标识物标记点

    Args:
        evt: Gradio选择事件
        current_image: 当前显示的图像
        landmark_points_json: 现有标记点JSON

    Returns:
        (信息文本, 标注后的图像, 更新后的标记点JSON)
    """
    if current_image is None:
        return "请先加载参考图像", None, landmark_points_json

    try:
        # 解析现有标记点
        if landmark_points_json:
            landmark_points = json.loads(landmark_points_json)
        else:
            landmark_points = []

        # 添加新标记点
        new_point = {
            "id": len(landmark_points) + 1,
            "x": evt.index[0],  # x坐标
            "y": evt.index[1],  # y坐标
            "label": f"标记{len(landmark_points) + 1}"
        }
        landmark_points.append(new_point)
        config_manager.landmark_points = landmark_points

        # 更新图像配准器
        if IMAGE_REGISTRATION_AVAILABLE and hasattr(config_manager, 'registration'):
            config_manager.registration.set_reference_image(config_manager.reference_image, landmark_points)

        # 绘制标记点
        marked_image = current_image.copy()
        for pt in landmark_points:
            x, y = int(pt["x"]), int(pt["y"])
            cv2.circle(marked_image, (x, y), 10, (0, 255, 0), 2)
            cv2.circle(marked_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(marked_image, f"#{pt['id']}", (x - 20, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(marked_image, pt.get("label", ""), (x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        updated_json = json.dumps(landmark_points, ensure_ascii=False)
        info = f"✓ 已添加标记点 #{new_point['id']} at ({new_point['x']}, {new_point['y']})\n总标记点数: {len(landmark_points)}"

        return info, marked_image, updated_json

    except Exception as e:
        return f"添加标记点失败: {e}", current_image, landmark_points_json


def preview_crop_area(reference_image, crop_width: int, crop_height: int,
                      crop_x: int, crop_y: int) -> Tuple[np.ndarray, str]:
    """
    在参考图像上预览裁剪区域

    Args:
        reference_image: 参考图像
        crop_width: 裁剪宽度
        crop_height: 裁剪高度
        crop_x: 裁剪起始X
        crop_y: 裁剪起始Y

    Returns:
        (带裁剪框的图像, 预览信息文本)
    """
    if reference_image is None:
        return None, "请先加载参考图像"

    try:
        # 转换图像格式
        if isinstance(reference_image, Image.Image):
            image_np = np.array(reference_image)
        else:
            image_np = reference_image

        # 转换为BGR格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        h, w = image_bgr.shape[:2]

        # 确保裁剪区域在图像范围内
        crop_x_clamped = min(crop_x, w - crop_width)
        crop_y_clamped = min(crop_y, h - crop_height)
        crop_x_clamped = max(0, crop_x_clamped)
        crop_y_clamped = max(0, crop_y_clamped)

        # 调整裁剪尺寸以适应图像边界
        actual_width = min(crop_width, w - crop_x_clamped)
        actual_height = min(crop_height, h - crop_y_clamped)

        # 绘制裁剪区域
        vis_image = image_bgr.copy()

        # 绘制半透明填充
        overlay = vis_image.copy()
        cv2.rectangle(overlay,
                     (crop_x_clamped, crop_y_clamped),
                     (crop_x_clamped + actual_width, crop_y_clamped + actual_height),
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)

        # 绘制边框
        cv2.rectangle(vis_image,
                     (crop_x_clamped, crop_y_clamped),
                     (crop_x_clamped + actual_width, crop_y_clamped + actual_height),
                     (0, 255, 0), 3)

        # 绘制四个角的标记
        corner_length = 20
        # 左上角
        cv2.line(vis_image, (crop_x_clamped, crop_y_clamped),
                (crop_x_clamped + corner_length, crop_y_clamped), (0, 0, 255), 3)
        cv2.line(vis_image, (crop_x_clamped, crop_y_clamped),
                (crop_x_clamped, crop_y_clamped + corner_length), (0, 0, 255), 3)
        # 右上角
        cv2.line(vis_image, (crop_x_clamped + actual_width, crop_y_clamped),
                (crop_x_clamped + actual_width - corner_length, crop_y_clamped), (0, 0, 255), 3)
        cv2.line(vis_image, (crop_x_clamped + actual_width, crop_y_clamped),
                (crop_x_clamped + actual_width, crop_y_clamped + corner_length), (0, 0, 255), 3)
        # 左下角
        cv2.line(vis_image, (crop_x_clamped, crop_y_clamped + actual_height),
                (crop_x_clamped + corner_length, crop_y_clamped + actual_height), (0, 0, 255), 3)
        cv2.line(vis_image, (crop_x_clamped, crop_y_clamped + actual_height),
                (crop_x_clamped, crop_y_clamped + actual_height - corner_length), (0, 0, 255), 3)
        # 右下角
        cv2.line(vis_image, (crop_x_clamped + actual_width, crop_y_clamped + actual_height),
                (crop_x_clamped + actual_width - corner_length, crop_y_clamped + actual_height), (0, 0, 255), 3)
        cv2.line(vis_image, (crop_x_clamped + actual_width, crop_y_clamped + actual_height),
                (crop_x_clamped + actual_width, crop_y_clamped + actual_height - corner_length), (0, 0, 255), 3)

        # 添加尺寸标注
        info_text = f"{actual_width}x{actual_height}"
        cv2.putText(vis_image, info_text,
                   (crop_x_clamped + 5, crop_y_clamped + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加坐标标注
        coord_text = f"({crop_x_clamped}, {crop_y_clamped})"
        cv2.putText(vis_image, coord_text,
                   (crop_x_clamped + 5, crop_y_clamped + actual_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 转换为RGB用于显示
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        # 构建信息文本
        info_lines = [
            f"✓ 裁剪区域预览",
            f"  图像尺寸: {w}x{h}",
            f"  设定裁剪: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}",
            f"  实际裁剪: x={crop_x_clamped}, y={crop_y_clamped}, w={actual_width}, h={actual_height}"
        ]

        if crop_x != crop_x_clamped or crop_y != crop_y_clamped:
            info_lines.append("  ⚠️ 裁剪起点超出图像边界，已自动调整")

        if actual_width != crop_width or actual_height != crop_height:
            info_lines.append("  ⚠️ 裁剪尺寸超出图像边界，已自动调整")

        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        import traceback
        return None, f"预览失败: {e}\n{traceback.format_exc()}"


def visualize_detected_features(reference_image, nfeatures: int = 500, method: str = "ORB") -> Tuple[np.ndarray, str]:
    """
    可视化图像中检测到的特征点（用于调试配准）

    Args:
        reference_image: 参考图像
        nfeatures: 特征点数量
        method: 配准方法 (ORB 或 SIFT)

    Returns:
        (特征点可视化图像, 信息文本)
    """
    if reference_image is None:
        return None, "请先加载参考图像"

    if not IMAGE_REGISTRATION_AVAILABLE:
        return None, "图像配准模块不可用"

    try:
        # 转换图像格式
        if isinstance(reference_image, Image.Image):
            image_np = np.array(reference_image)
        else:
            image_np = reference_image

        # 转换为BGR格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        # 转换为灰度图
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 创建特征检测器
        if method == "ORB":
            detector = cv2.ORB_create(nfeatures=nfeatures)
        elif method == "SIFT":
            detector = cv2.SIFT_create(nfeatures=nfeatures if nfeatures > 0 else None)
        else:
            return None, f"不支持的配准方法: {method}"

        # 检测特征点
        keypoints, descriptors = detector.detectAndCompute(gray, None)

        # 绘制特征点
        # 兼容不同OpenCV版本的常量名
        try:
            flags = cv2.DRAW_MATCHES_FLAGS_RICH_KEYPOINTS
        except AttributeError:
            try:
                flags = cv2.DrawMatchesFlags_RICH_KEYPOINTS
            except AttributeError:
                flags = 4  # RICH_KEYPOINTS 的值

        vis_image = cv2.drawKeypoints(
            image_bgr,
            keypoints,
            None,
            flags=flags
        )

        # 转换为RGB用于显示
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        # 计算特征点分布
        h, w = gray.shape
        quadrants = {
            "左上": sum(1 for kp in keypoints if kp.pt[0] < w/2 and kp.pt[1] < h/2),
            "右上": sum(1 for kp in keypoints if kp.pt[0] >= w/2 and kp.pt[1] < h/2),
            "左下": sum(1 for kp in keypoints if kp.pt[0] < w/2 and kp.pt[1] >= h/2),
            "右下": sum(1 for kp in keypoints if kp.pt[0] >= w/2 and kp.pt[1] >= h/2)
        }

        # 构建信息文本
        info_lines = [
            f"=== 特征点检测分析 ===",
            f"检测方法: {method}",
            f"特征点数量: {len(keypoints)}",
            f"图像尺寸: {w}x{h}",
            f"描述符维度: {descriptors.shape[0] if descriptors is not None else 0} x {descriptors.shape[1] if descriptors is not None else 'N/A'}",
            "",
            "特征点分布:",
            f"  左上区域: {quadrants['左上']} 个",
            f"  右上区域: {quadrants['右上']} 个",
            f"  左下区域: {quadrants['左下']} 个",
            f"  右下区域: {quadrants['右下']} 个",
            "",
            "💡 提示:",
            "  - 绿色圆圈 = 检测到的特征点",
            "  - 圆圈大小 = 特征尺度",
            "  - 圆圈方向 = 特征方向",
            "",
            "建议选择特征点密集区域的固定物体作为标识物"
        ]

        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        import traceback
        return None, f"特征检测失败: {e}\n{traceback.format_exc()}"


def detect_tree_cluster_center(test_image, dino_caption: str,
                               min_box_area: int = 500, max_box_area: int = 100000,
                               nms_threshold: float = 0.3, merge_distance: float = 30.0) -> Tuple[Optional[np.ndarray], str, int, int]:
    """
    检测图像中的树集群中心点（使用 Falcon-Perception）

    Args:
        test_image: 输入图像
        dino_caption: 检测文本提示
        min_box_area: 最小框面积（过滤太小的干扰）
        max_box_area: 最大框面积（过滤太大的干扰）
        nms_threshold: NMS阈值，合并重叠框
        merge_distance: 合并接近中心点的距离

    Returns:
        (检测结果可视化图像, 信息文本, 中心点X, 中心点Y)
    """
    if test_image is None:
        return None, "请先上传图像", 0, 0

    if not IMAGE_REGISTRATION_AVAILABLE:
        return None, "图像配准模块不可用", 0, 0

    try:
        # 转换图像格式
        if isinstance(test_image, Image.Image):
            image_np = np.array(test_image)
        else:
            image_np = test_image

        # 转换为BGR格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        # 使用 Falcon-Perception 检测树木
        all_trees, trees_vis_image = detect_trees(
            fp_model=model,
            image=image_bgr,
            caption=dino_caption,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            nms_threshold=nms_threshold,
        )

        if not all_trees:
            return None, "未检测到树木，请调整检测参数（降低阈值或调整面积过滤）", 0, 0

        # 找到主要树木群集
        from image_registration import find_main_tree_cluster

        cluster_result = find_main_tree_cluster(all_trees, merge_distance=merge_distance)

        if cluster_result is None:
            # 如果没有找到群集，使用所有树木的中心
            centers = [tree["center"] for tree in all_trees]
            center_x = int(np.mean([c[0] for c in centers]))
            center_y = int(np.mean([c[1] for c in centers]))
            center = (center_x, center_y)
            bbox = [min([t["bbox"][0] for t in all_trees]), min([t["bbox"][1] for t in all_trees]),
                   max([t["bbox"][2] for t in all_trees]), max([t["bbox"][3] for t in all_trees])]
            padding_bbox = bbox
            tree_count = len(all_trees)
            cluster_tree_count = len(all_trees)
            is_cluster = False
        else:
            center = cluster_result["center"]
            bbox = cluster_result["bbox"]
            padding_bbox = cluster_result["padding_bbox"]
            tree_count = len(all_trees)
            cluster_tree_count = cluster_result["tree_count"]
            is_cluster = True

        # 创建可视化图像
        vis_image = image_bgr.copy()
        h, w = vis_image.shape[:2]

        # 绘制所有检测到的树木框（绿色）
        for i, tree in enumerate(all_trees):
            x1, y1, x2, y2 = tree["bbox"]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制树木编号
            cv2.putText(vis_image, f"{i+1}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制红色半透明遮罩（群集区域）
        overlay = vis_image.copy()
        x1, y1, x2, y2 = padding_bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)

        # 绘制边界框边线
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 绘制中心点（红色大圆点）
        cv2.circle(vis_image, center, 25, (0, 0, 255), -1)
        cv2.putText(vis_image, f"CENTER ({center[0]}, {center[1]})", (center[0] - 80, center[1] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 绘制十字线
        cv2.line(vis_image, (center[0], 0), (center[0], h), (0, 0, 255), 1)
        cv2.line(vis_image, (0, center[1]), (w, center[1]), (0, 0, 255), 1)

        # 转换为RGB用于显示
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        # 构建信息文本
        info_lines = [
            f"=== 树集群中心点检测成功 ===",
            f"中心点坐标: ({center[0]}, {center[1]})",
            f"检测到树木总数: {tree_count} 棵",
            f"群集中树木数: {cluster_tree_count} 棵",
            f"群集边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]",
            f"扩展边界框: [{padding_bbox[0]}, {padding_bbox[1]}, {padding_bbox[2]}, {padding_bbox[3]}]",
            f"图像尺寸: {w}x{h}",
            "",
            "📊 过滤参数:",
            f"  - 最小面积: {min_box_area} px²",
            f"  - 最大面积: {max_box_area} px²",
            f"  - NMS阈值: {nms_threshold}",
            f"  - 合并距离: {merge_distance} px",
            "",
            "💡 说明:",
            "  - 绿色框 = 检测到的每个树木",
            "  - 红色圆点 = 检测到的树集群中心点",
            "  - 红色半透明遮罩 = 树木群集区域",
            "  - 十字线 = 中心点定位线",
            "",
            "配置区域时，裁剪参数将相对于此中心点计算：",
            "  - 绝对X = 中心点X + 相对偏移X",
            "  - 绝对Y = 中心点Y + 相对偏移Y"
        ]

        return vis_rgb, "\n".join(info_lines), center[0], center[1]

    except Exception as e:
        import traceback
        return None, f"检测失败: {e}\n{traceback.format_exc()}", 0, 0


def preview_current_region_on_image(test_image, region_name: str, region_id: str,
                                     crop_width: int, crop_height: int, relative_x: int, relative_y: int,
                                     center_x: int, center_y: int) -> Tuple[Optional[np.ndarray], str]:
    """
    预览当前正在配置的单个区域在图像上的位置

    Args:
        test_image: 输入图像
        region_name: 区域名称
        region_id: 区域ID
        crop_width: 裁剪宽度
        crop_height: 裁剪高度
        relative_x: 相对偏移 X
        relative_y: 相对偏移 Y
        center_x: 树集群中心点 X
        center_y: 树集群中心点 Y

    Returns:
        (可视化图像, 信息文本)
    """
    if test_image is None:
        return None, "请先上传图像并检测中心点"

    try:
        # 转换图像格式
        if isinstance(test_image, Image.Image):
            image_np = np.array(test_image)
        else:
            image_np = test_image

        # 转换为BGR格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        h, w = image_bgr.shape[:2]

        # 创建可视化图像
        vis_image = image_bgr.copy()

        # 绘制中心点
        if center_x > 0 and center_y > 0:
            cv2.circle(vis_image, (int(center_x), int(center_y)), 15, (0, 0, 255), -1)
            cv2.putText(vis_image, f"Center ({int(center_x)}, {int(center_y)})",
                       (int(center_x) + 25, int(center_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # 绘制十字线
            cv2.line(vis_image, (int(center_x), 0), (int(center_x), h), (0, 0, 255), 1)
            cv2.line(vis_image, (0, int(center_y)), (w, int(center_y)), (0, 0, 255), 1)

        # 计算当前区域的绝对坐标
        abs_x = int(center_x + relative_x)
        abs_y = int(center_y + relative_y)

        # 确保在图像范围内
        abs_x = max(0, min(abs_x, w - crop_width))
        abs_y = max(0, min(abs_y, h - crop_height))

        # 绘制半透明矩形（绿色=当前正在配置的区域）
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (abs_x, abs_y), (abs_x + crop_width, abs_y + crop_height), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

        # 绘制边框
        cv2.rectangle(vis_image, (abs_x, abs_y), (abs_x + crop_width, abs_y + crop_height), (0, 255, 0), 3)

        # 绘制标签
        label = f"{region_name or 'New Region'} ({abs_x}, {abs_y})"
        cv2.putText(vis_image, label, (abs_x, abs_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 绘制尺寸标签
        size_label = f"{crop_width}x{crop_height}"
        cv2.putText(vis_image, size_label, (abs_x, abs_y + crop_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 转换为RGB用于显示
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        # 构建信息文本
        info_lines = [
            f"=== 当前区域预览 ===",
            f"区域名称: {region_name or '未命名'}",
            f"区域ID: {region_id or '未设置'}",
            f"树集群中心: ({int(center_x)}, {int(center_y)})",
            f"相对偏移: ({relative_x}, {relative_y})",
            f"绝对坐标: ({abs_x}, {abs_y})",
            f"区域尺寸: {crop_width}x{crop_height}",
            "",
            "💡 提示:",
            "  - 绿色半透明区域 = 当前配置的裁剪范围",
            "  - 红色圆点 = 树集群中心点",
            "  - 调整参数后点击'更新预览'查看效果"
        ]

        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        import traceback
        return None, f"预览失败: {e}\n{traceback.format_exc()}"


def preview_regions_on_image(test_image, regions_json: str, center_x: int, center_y: int) -> Tuple[Optional[np.ndarray], str]:
    """
    在图像上预览配置的检测区域

    Args:
        test_image: 输入图像
        regions_json: 区域配置JSON
        center_x: 树集群中心点 X
        center_y: 树集群中心点 Y

    Returns:
        (可视化图像, 信息文本)
    """
    if test_image is None:
        return None, "请先上传图像"

    try:
        # 转换图像格式
        if isinstance(test_image, Image.Image):
            image_np = np.array(test_image)
        else:
            image_np = test_image

        # 转换为BGR格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        h, w = image_bgr.shape[:2]

        # 解析区域配置
        regions = []
        if regions_json:
            try:
                regions = json.loads(regions_json)
            except:
                pass

        # 创建可视化图像
        vis_image = image_bgr.copy()

        # 绘制中心点
        if center_x > 0 and center_y > 0:
            cv2.circle(vis_image, (int(center_x), int(center_y)), 20, (0, 0, 255), -1)
            cv2.putText(vis_image, f"Center ({int(center_x)}, {int(center_y)})",
                       (int(center_x) + 30, int(center_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 绘制十字线
            cv2.line(vis_image, (int(center_x), 0), (int(center_x), h), (0, 0, 255), 1)
            cv2.line(vis_image, (0, int(center_y)), (w, int(center_y)), (0, 0, 255), 1)

        # 绘制每个区域
        info_lines = [f"区域数量: {len(regions)}", ""]

        # 颜色列表（不同区域用不同颜色）
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

        for i, region in enumerate(regions):
            region_name = region.get("region_name", f"区域{i+1}")
            crop_params = region.get("crop_params", {})
            region_width = crop_params.get("width", 550)
            region_height = crop_params.get("height", 870)
            relative_x = crop_params.get("relative_x", 0)
            relative_y = crop_params.get("relative_y", 0)

            # 计算绝对坐标
            abs_x = int(center_x + relative_x)
            abs_y = int(center_y + relative_y)

            # 确保在图像范围内
            abs_x = max(0, min(abs_x, w - region_width))
            abs_y = max(0, min(abs_y, h - region_height))

            # 选择颜色
            color = colors[i % len(colors)]

            # 绘制半透明矩形
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (abs_x, abs_y), (abs_x + region_width, abs_y + region_height), color, -1)
            cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)

            # 绘制边框
            cv2.rectangle(vis_image, (abs_x, abs_y), (abs_x + region_width, abs_y + region_height), color, 3)

            # 绘制标签
            label = f"{region_name} ({abs_x}, {abs_y})"
            cv2.putText(vis_image, label, (abs_x, abs_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 添加信息
            info_lines.append(f"区域 {i+1}: {region_name}")
            info_lines.append(f"  相对偏移: ({relative_x}, {relative_y})")
            info_lines.append(f"  绝对坐标: ({abs_x}, {abs_y})")
            info_lines.append(f"  尺寸: {region_width}x{region_height}")
            info_lines.append("")

        # 转换为RGB用于显示
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        import traceback
        return None, f"预览失败: {e}\n{traceback.format_exc()}"


def clear_landmark_points(current_image) -> Tuple[str, np.ndarray, str]:
    """
    清除所有标识物标记点

    Args:
        current_image: 当前显示的图像

    Returns:
        (信息文本, 原始图像, 空的标记点JSON)
    """
    config_manager.landmark_points = []

    # 恢复原始图像
    if config_manager.reference_image is not None:
        original_image = cv2.cvtColor(config_manager.reference_image, cv2.COLOR_BGR2RGB)
    else:
        original_image = current_image

    return "✓ 已清除所有标记点", original_image, "[]"


def add_detection_region(
    region_name: str,
    region_id: str,
    crop_width: int,
    crop_height: int,
    relative_x: int,  # 改为相对偏移 X（相对于树集群中心）
    relative_y: int,  # 改为相对偏移 Y（相对于树集群中心）
    caption: str,
    max_box_area: int,
    min_box_area: int,
    regions_json: str
) -> Tuple[str, str]:
    """
    添加检测区域（使用相对偏移坐标）

    Args:
        region_name: 区域名称
        region_id: 区域ID
        crop_width: 裁剪宽度
        crop_height: 裁剪高度
        relative_x: 相对偏移 X（相对于树集群中心，负数=左边，正数=右边）
        relative_y: 相对偏移 Y（相对于树集群中心，负数=上方，正数=下方）
        caption: 检测提示词
        max_box_area: 最大框面积
        min_box_area: 最小框面积
        regions_json: 现有区域JSON

    Returns:
        (信息文本, 更新后的区域JSON)
    """
    if not region_name or not region_id:
        return "请填写区域名称和ID", regions_json

    try:
        # 解析现有区域
        if regions_json:
            regions = json.loads(regions_json)
        else:
            regions = []

        # 检查ID是否重复
        if any(r.get("region_id") == region_id for r in regions):
            return f"区域ID '{region_id}' 已存在，请使用不同的ID", regions_json

        # 创建新区域（使用相对偏移）
        new_region = {
            "region_id": region_id,
            "region_name": region_name,
            "crop_params": {
                "width": crop_width,
                "height": crop_height,
                "relative_x": relative_x,  # 相对于树集群中心的偏移
                "relative_y": relative_y,
                "absolute_x": 0,  # 运行时动态计算
                "absolute_y": 0
            },
            "detection_params": {
                "caption": caption,
                "max_box_area": max_box_area,
                "min_box_area": min_box_area
            }
        }

        regions.append(new_region)
        config_manager.detection_regions = regions

        updated_json = json.dumps(regions, ensure_ascii=False, indent=2)
        info = f"✓ 已添加区域: {region_name} (ID: {region_id})\n相对偏移: ({relative_x}, {relative_y})\n总区域数: {len(regions)}"

        return info, updated_json

    except Exception as e:
        return f"添加区域失败: {e}", regions_json


def preview_config(regions_json: str, camera_name: str, rtsp_url: str, sample_interval: int,
                   center_x: int, center_y: int,
                   dino_caption: str = "tree") -> str:
    """
    预览完整配置

    Returns:
        配置JSON字符串
    """
    # 解析区域
    regions = []
    if regions_json:
        try:
            regions = json.loads(regions_json)
        except:
            pass

    # 生成简化的配置结构
    config = {
        "version": "2.0",
        "camera_name": camera_name,
        "created_at": datetime.now().isoformat(),
        "dino_params": {
            "caption": dino_caption,
            "device": DEVICE
        },
        "reference_center": {
            "x": center_x,
            "y": center_y
        },
        "rtsp_settings": {
            "url": rtsp_url,
            "sample_interval": sample_interval
        },
        "detection_regions": regions
    }

    config_manager.detection_regions = regions
    config_manager.current_config = config

    return json.dumps(config, ensure_ascii=False, indent=2)


def save_config_file(camera_name: str, config_json: str) -> str:
    """
    保存配置文件

    Args:
        camera_name: 相机名称
        config_json: 配置JSON字符串

    Returns:
        保存结果信息
    """
    if not camera_name:
        return "请输入相机名称"

    try:
        # 解析配置
        config = json.loads(config_json)

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in camera_name)
        filename = f"{safe_name}_{timestamp}.json"
        filepath = CONFIGS_DIR / filename

        # 保存配置
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return f"✓ 配置已保存: {filepath}\n相机: {camera_name}\n区域数: {len(config.get('detection_regions', []))}"

    except Exception as e:
        return f"保存配置失败: {e}"



def load_existing_config(config_file: str) -> Tuple[str, str, int, int, str, str, str, str, float, float]:
    """
    加载现有配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        (相机名称, 区域JSON, 中心点X, 中心点Y, RTSP URL, 采样间隔,
         DINO 文本提示, DINO 框阈值, DINO 文本阈值)
    """
    if not config_file:
        default_vals = ("", "[]", 0, 0, "", 30, "tree", 0.3, 0.25)
        return default_vals

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 重置配置管理器
        config_manager.reset()
        config_manager.current_config = config.copy()

        # 提取基本信息
        camera_name = config.get("camera_name", "")

        # 获取中心点
        reference_center = config.get("reference_center", {"x": 0, "y": 0})
        center_x = reference_center.get("x", 0)
        center_y = reference_center.get("y", 0)

        # RTSP 设置
        rtsp_settings = config.get("rtsp_settings", {})
        rtsp_url = rtsp_settings.get("url", "")
        sample_interval = rtsp_settings.get("sample_interval", 30)

        # DINO 参数
        dino_params = config.get("dino_params", {})
        dino_caption = dino_params.get("caption", "tree")

        # 加载区域
        detection_regions = config.get("detection_regions", [])
        regions_json = json.dumps(detection_regions, ensure_ascii=False, indent=2)
        config_manager.detection_regions = detection_regions

        return (
            camera_name, regions_json, center_x, center_y,
            rtsp_url, sample_interval,
            dino_caption
        )

    except Exception as e:
        default_vals = ("", "[]", 0, 0, "", 30, "tree")
        return default_vals


# Use Falcon-Perception model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Falcon-Perception 模型
_falcon_model_dir = str(Path(__file__).resolve().parent / "weight" / "Falcon-Perception")
print(f"Loading Falcon-Perception model from: {_falcon_model_dir}")
model = FalconPerceptionModel(hf_local_dir=_falcon_model_dir, device=DEVICE)
print("Falcon-Perception model loaded.")

def run_grounding(input_image, grounding_caption):
    init_image = input_image.convert("RGB")
    image_np = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)

    # 使用 Falcon-Perception 检测
    detections = detect_objects(model, image_np, grounding_caption)

    # 标注
    annotated_np = annotate_detections(image_np, detections)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))

    return image_with_box

def get_video_info(video_path):
    """获取视频信息：帧率、总帧数、时长"""
    try:
        if FFMPEG_AVAILABLE:
            # 使用ffmpeg-python获取视频信息
            with SuppressStdErr():
                probe = ffmpeg.probe(video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

                fps = eval(video_info['r_frame_rate'])  # 评估分数，例如 "30/1" -> 30.0
                duration = float(probe['format']['duration'])
                total_frames = int(video_info.get('nb_frames', duration * fps))

                return int(fps), total_frames, duration
        else:
            # 回退到OpenCV
            with SuppressStdErr():
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, None

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0

                cap.release()
                return fps, total_frames, duration
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None, None, None

def extract_frames_with_ffmpeg(video_path, output_dir, start_time, end_time, fps, crop_params=None):
    """
    使用ffmpeg从视频中批量提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        start_time: 起始时间（秒）
        end_time: 结束时间（秒）
        fps: 原视频帧率
        crop_params: 裁剪参数 (width:height:x:y) 或 None

    Returns:
        提取的帧信息列表 [(frame_idx, image_path, timestamp), ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_infos = []
    try:
        # 使用ffmpeg提取帧到临时目录
        # 使用select='eq(n\,0)'等过滤器来选择特定帧
        temp_pattern = os.path.join(output_dir, "frame_%04d.png")

        cmd = ['ffmpeg', '-i', video_path, '-vf', "select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)'"]

        # 添加裁剪参数
        if crop_params:
            cmd[3] = f"select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)',crop={crop_params}"

        cmd.extend(['-vsync', '0', temp_pattern, '-y'])

        # 对于更灵活的抽帧，我们使用Python循环
        # 先获取视频的持续时间
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            'fps': video_fps,
            'total_frames': total_frames,
            'output_dir': output_dir
        }

    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None

def extract_single_frame(video_path, frame_idx, crop_params=None, output_dir=None):
    """
    提取单个帧

    Args:
        video_path: 视频路径
        frame_idx: 帧索引
        crop_params: 裁剪参数
        output_dir: 输出目录

    Returns:
        (frame_idx, image_pil, timestamp)
    """
    try:
        if FFMPEG_AVAILABLE:
            # 使用ffmpeg-python提取帧
            # 首先获取视频的fps来计算时间戳
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps = eval(video_info['r_frame_rate'])

            # 计算时间戳
            timestamp = frame_idx / fps if fps > 0 else frame_idx / 30.0

            # 构建ffmpeg命令提取指定帧
            if crop_params:
                parts = crop_params.split(':')
                w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                process = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', format='image2', vframes=1, vf=f'crop={w}:{h}:{x}:{y}')
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
            else:
                process = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', format='image2', vframes=1)
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

            # 读取输出
            out, err = process.communicate()

            if process.returncode != 0:
                # 如果ffmpeg失败，回退到OpenCV
                raise Exception(f"FFmpeg extraction failed: {err.decode()}")

            # 从字节流读取图像
            image = Image.open(BytesIO(out))

            return (frame_idx, image, timestamp)

        else:
            # 回退到OpenCV
            with SuppressStdErr():
                cap = cv2.VideoCapture(video_path)

                # 设置帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                # 读取帧
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    return None

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 裁剪
                if crop_params:
                    parts = crop_params.split(':')
                    w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    # 确保裁剪区域在图像范围内
                    img_h, img_w = frame_rgb.shape[:2]
                    x = min(x, img_w - w)
                    y = min(y, img_h - h)
                    frame_rgb = frame_rgb[y:y+h, x:x+w]

                # 获取视频fps来计算准确的时间戳
                cap_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                timestamp = frame_idx / cap_fps if cap_fps > 0 else frame_idx / 30.0

                return (frame_idx, Image.fromarray(frame_rgb), timestamp)

    except Exception as e:
        print(f"Error extracting frame {frame_idx}: {e}")
        return None

def process_frame_detection(frame_info, grounding_caption,
                            max_box_area=None, min_box_area=None):
    """
    处理单个帧的检测（用于多线程，使用 Falcon-Perception）

    Args:
        frame_info: (frame_idx, image_pil, timestamp)
        grounding_caption: 检测提示词
        max_box_area: 最大框面积（像素），超过此面积的框将被过滤
        min_box_area: 最小框面积（像素），小于此面积的框将被过滤

    Returns:
        检测结果字典
    """
    frame_idx, image, timestamp = frame_info

    try:
        resource_monitor.update_peak()

        image_rgb = image.convert("RGB")
        image_np = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
        img_h, img_w = image_np.shape[:2]

        detections = detect_objects(
            model, image_np, grounding_caption,
            max_box_area=max_box_area if max_box_area is not None else 0,
            min_box_area=min_box_area if min_box_area is not None else 0,
        )

        resource_monitor.update_peak()

        count = len(detections)

        annotated_np = annotate_detections(image_np, detections)
        image_with_box = cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB)

        detected_items = []
        for det in detections:
            detected_items.append({
                "label": det["label"],
                "center_x": det["center_x"],
                "center_y": det["center_y"],
                "width_norm": det["width"] / img_w,
                "height_norm": det["height"] / img_h,
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "width": det["width"],
                "height": det["height"],
                "area": det["area"],
            })

        return {
            "frame": frame_idx,
            "timestamp": f"{timestamp:.2f}s",
            "count": count,
            "detected_items": detected_items,
            "annotated_image": Image.fromarray(image_with_box)
        }

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return {
            "frame": frame_idx,
            "timestamp": f"{timestamp:.2f}s",
            "count": 0,
            "detected_items": [],
            "error": str(e),
            "annotated_image": image
        }

# ========== 第一阶段：抽帧 ==========

def extract_frames_stage(
    video_upload,
    video_dropdown,
    start_time_str,
    frame_interval,
    end_time_str,
    crop_width,
    crop_height,
    crop_x,
    crop_y,
    enable_crop,
    progress=gr.Progress()
):
    """
    第一阶段：从视频中提取帧（固定间隔抽帧）

    Returns:
        (stage_data_json, preview_images, info_text)
    """
    # 确定使用哪个视频源
    video_file = video_dropdown if video_dropdown else video_upload

    if video_file is None:
        return "{}", [], "请上传视频或从列表中选择"

    try:
        # 获取视频文件路径
        if isinstance(video_file, str):
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        else:
            video_path = str(video_file)

        # 检查文件是否存在
        if not os.path.exists(video_path):
            return "{}", [], f"视频文件不存在: {video_path}"

        # 获取视频信息
        fps, total_frames, duration = get_video_info(video_path)
        if fps is None:
            return "{}", [], "无法读取视频信息"

        # 解析时间参数
        try:
            start_time = parse_time_string(start_time_str)
            end_time = parse_time_string(end_time_str) if end_time_str else duration
        except ValueError as e:
            return f"{{}}", [], f"时间格式错误: {e}"

        # 转换为帧数
        start_frame = int(start_time * fps)
        end_frame = int(min(end_time, duration) * fps)

        if start_frame >= total_frames:
            return "{}", [], f"起始帧超出视频总帧数 (视频共{total_frames}帧)"

        end_frame = min(end_frame, total_frames)

        # 构建裁剪参数
        crop_params = None
        if enable_crop:
            crop_params = f"{crop_width}:{crop_height}:{crop_x}:{crop_y}"

        # 创建临时目录保存帧
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")

        # 固定间隔抽帧
        frame_indices = list(range(start_frame, end_frame, frame_interval))
        total_to_extract = len(frame_indices)

        extracted_frames = []
        preview_images = []

        for i, frame_idx in enumerate(frame_indices):
            timestamp = frame_idx / fps
            result = extract_single_frame(video_path, frame_idx, crop_params, temp_dir)

            if result:
                frame_idx, image, timestamp = result
                extracted_frames.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "temp_dir": temp_dir
                })

                # 保存前5帧作为预览
                if len(preview_images) < 5:
                    preview_images.append(image)

            # 更新进度
            progress((i + 1) / total_to_extract, desc=f"提取帧 {i+1}/{total_to_extract}")

        # 准备阶段数据（不需要保存实际图像，只保存索引）
        stage_data = {
            "video_path": video_path,
            "temp_dir": temp_dir,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_interval": frame_interval,
            "crop_params": crop_params,
            "frame_indices": frame_indices,
            "total_to_process": len(frame_indices)
        }

        info_text = (
            f"✓ 抽帧完成! (固定间隔: 每{frame_interval}帧)\n"
            f"视频信息: {total_frames}帧, {fps}fps, {duration:.1f}秒\n"
            f"处理范围: 帧 {start_frame} - {end_frame}\n"
            f"已提取帧数: {len(extracted_frames)}\n"
            f"临时目录: {temp_dir}\n"
            f"\n请点击'开始检测'进行物品检测"
        )

        return json.dumps(stage_data), preview_images, info_text

    except Exception as e:
        import traceback
        error_msg = f"抽帧出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return "{}", [], error_msg

# ========== 第二阶段：检测 ==========

def detect_items_stage(
    stage_data_json,
    grounding_caption,
    num_workers,
    max_box_area,
    min_box_area,
    progress=gr.Progress()
):
    """
    第二阶段：对提取的帧进行物品检测（多线程并行）

    Args:
        stage_data_json: 第一阶段返回的数据
        grounding_caption: 检测提示词
        num_workers: 并行工作线程数
        max_box_area: 最大框面积（像素），超过此面积的框将被过滤
        min_box_area: 最小框面积（像素），小于此面积的框将被过滤

    Returns:
        (json_output, sample_images, info_text)
    """
    if not stage_data_json or stage_data_json == "{}":
        return "{}", [], "请先完成抽帧阶段"

    if not grounding_caption or grounding_caption.strip() == "":
        return "{}", [], "请输入检测提示词"

    try:
        # 解析阶段数据
        stage_data = json.loads(stage_data_json)

        video_path = stage_data["video_path"]
        fps = stage_data["fps"]
        frame_indices = stage_data["frame_indices"]
        crop_params = stage_data.get("crop_params")

        # 准备检测结果
        results = []
        sample_images = []

        # 重新提取帧并检测（多线程）
        total_to_process = len(frame_indices)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_frame = {}
            for i, frame_idx in enumerate(frame_indices):
                # 提取帧
                frame_info = extract_single_frame(video_path, frame_idx, crop_params)
                if frame_info:
                    future = executor.submit(
                        process_frame_detection,
                        frame_info,
                        grounding_caption,
                        max_box_area,
                        min_box_area
                    )
                    future_to_frame[future] = frame_idx

                # 每10个任务更新一次进度（避免过于频繁）
                if (i + 1) % 10 == 0:
                    progress((i + 1) / total_to_process, desc=f"提交检测任务 {i+1}/{total_to_process}")

            # 收集结果
            completed = 0
            for future in as_completed(future_to_frame):
                result = future.result()
                if result:
                    # 保留完整的结果，包括标注图片
                    results.append(result)

                    # 保存前5帧作为样本显示
                    if "annotated_image" in result and len(sample_images) < 5:
                        sample_images.append(result["annotated_image"])

                completed += 1
                # 确保进度不超过100%
                progress_val = min(completed / total_to_process, 1.0)
                progress(progress_val, desc=f"检测完成 {completed}/{total_to_process}")

                # 每处理10帧更新一次资源监控
                if completed % 10 == 0:
                    resource_monitor.update_peak()

        # 按帧号排序
        results.sort(key=lambda x: x["frame"])

        # 获取峰值资源使用统计
        resource_stats = resource_monitor.get_peak_stats()

        # 获取视频名称和时间戳
        video_name = Path(video_path).stem
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建输出目录
        output_dir = Path(__file__).resolve().parent / "output" / "detection_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建标注图片目录（每次处理前清空）
        annotated_images_dir = output_dir / "latest_annotated_frames"
        # 清空目录中的旧文件
        if annotated_images_dir.exists():
            import shutil
            shutil.rmtree(annotated_images_dir)
        annotated_images_dir.mkdir(parents=True, exist_ok=True)

        # 保存所有标注后的图片
        annotated_images_paths = []
        for result in results:
            if "annotated_image" in result:
                frame_idx = result["frame"]
                # 使用固定文件名模式，方便覆盖
                image_filename = f"frame_{frame_idx:06d}.png"
                image_filepath = annotated_images_dir / image_filename

                # 保存图片
                try:
                    result["annotated_image"].save(image_filepath)
                    annotated_images_paths.append(str(image_filepath))
                except Exception as e:
                    print(f"保存图片 {image_filename} 失败: {e}")

        # 创建ZIP文件包含所有标注图片
        zip_filepath = None
        if annotated_images_paths:
            try:
                import zipfile
                zip_filename = f"{video_name}_annotated_frames_{timestamp_str}.zip"
                zip_filepath = output_dir / zip_filename

                with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for image_path in annotated_images_paths:
                        arcname = Path(image_path).name
                        zipf.write(image_path, arcname)

                print(f"✓ ZIP文件已创建: {zip_filepath}")
            except Exception as e:
                print(f"创建ZIP文件失败: {e}")
                zip_filepath = None

        # 移除annotated_image字段后准备JSON数据
        results_for_json = []
        for result in results:
            result_copy = result.copy()
            if "annotated_image" in result_copy:
                del result_copy["annotated_image"]
            if "error" in result_copy:
                del result_copy["error"]
            results_for_json.append(result_copy)

        # 生成JSON文件名
        json_filename = f"{video_name}_{timestamp_str}.json"
        json_filepath = output_dir / json_filename

        # 准备输出数据
        output_data = {
            "video_name": video_name,
            "video_path": video_path,
            "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotated_images_dir": str(annotated_images_dir),
            "annotated_images_count": len(annotated_images_paths),
            "video_info": {
                "fps": fps,
                "total_frames": stage_data["total_frames"],
                "duration": f"{stage_data['duration']:.2f}s",
                "start_frame": stage_data["start_frame"],
                "end_frame": stage_data["end_frame"]
            },
            "detection_params": {
                "caption": grounding_caption,
                "frame_interval": stage_data["frame_interval"],
                "crop_enabled": crop_params is not None,
                "crop_params": crop_params,
                "num_workers": num_workers
            },
            "statistics": {
                "total_frames_processed": len(results),
                "total_items_detected": sum(r["count"] for r in results),
                "average_items_per_frame": sum(r["count"] for r in results) / len(results) if results else 0
            },
            "resource_usage": {
                "peak_ram_mb": round(resource_stats["peak_ram_mb"], 2),
                "peak_gpu_mb": round(resource_stats["peak_gpu_mb"], 2),
                "peak_ram_gb": round(resource_stats["peak_ram_mb"] / 1024, 2),
                "peak_gpu_gb": round(resource_stats["peak_gpu_mb"] / 1024, 2)
            },
            "results": results_for_json
        }

        # 转换为JSON
        json_output = json.dumps(output_data, ensure_ascii=False, indent=2)

        # 保存JSON文件
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"✓ JSON结果已保存到: {json_filepath}")
        except Exception as e:
            print(f"保存JSON文件失败: {e}")

        # 统计信息
        total_detected = output_data["statistics"]["total_items_detected"]
        avg_count = output_data["statistics"]["average_items_per_frame"]

        zip_info = f"\n✓ ZIP文件已保存:\n{zip_filepath}" if zip_filepath else ""
        resource_summary = resource_monitor.get_summary()

        info_text = (
            f"✓ 检测完成!\n"
            f"视频信息: {stage_data['total_frames']}帧, {fps}fps, {stage_data['duration']:.1f}秒\n"
            f"处理范围: 帧 {stage_data['start_frame']} - {stage_data['end_frame']}\n"
            f"抽帧间隔: 每{stage_data['frame_interval']}帧\n"
            f"处理帧数: {len(results)}\n"
            f"并行线程: {num_workers}\n"
            f"总检测数量: {total_detected}\n"
            f"平均每帧: {avg_count:.1f}个\n"
            f"\n{resource_summary}\n"
            f"\n✓ JSON已保存:\n{json_filepath}\n"
            f"✓ 标注图片已保存:\n{annotated_images_dir}\n"
            f"共 {len(annotated_images_paths)} 张图片"
            f"{zip_info}"
        )

        return json_output, sample_images, info_text, str(json_filepath), str(zip_filepath) if zip_filepath else None

    except Exception as e:
        import traceback
        error_msg = f"检测出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return "{}", [], error_msg

def parse_time_string(time_str):
    """
    解析时间字符串，支持格式：
    - "HH:MM:SS"
    - "MM:SS"
    - 秒数（整数或浮点数）
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)

    time_str = time_str.strip()

    # 尝试直接解析为数字
    try:
        return float(time_str)
    except ValueError:
        pass

    # 解析 "HH:MM:SS" 或 "MM:SS" 格式
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"无法解析时间格式: {time_str}")

def list_input_videos():
    """列出input_video目录中的视频文件"""
    input_dir = Path(__file__).resolve().parent / "input_video"
    if input_dir.exists():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        videos = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        return videos
    return []

def update_video_choices():
    """更新视频选择下拉框"""
    videos = list_input_videos()
    return gr.update(choices=videos if videos else [], value=videos[0] if videos else None)

# 保存阶段数据
stage_data_state = gr.State(None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# Falcon-Perception Detection")
        gr.Markdown("### Open-World Detection with Falcon-Perception")

        with gr.Tabs():
            # 图像检测标签页
            with gr.Tab("图像检测"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(source='upload', type="pil")
                        grounding_caption = gr.Textbox(label="Detection Prompt", value="object")
                        run_button = gr.Button(label="Run")

                    with gr.Column():
                        image_gallery = gr.outputs.Image(
                            type="pil",
                        ).style(full_width=True, full_height=True)

                run_button.click(fn=run_grounding, inputs=[
                                input_image, grounding_caption], outputs=[image_gallery])

            # 视频检测标签页 - 分为两个阶段
            with gr.Tab("视频检测（两阶段）"):
                if not FFMPEG_AVAILABLE:
                    gr.Markdown("⚠️ **ffmpeg-python 未安装**，将使用 OpenCV 读取视频。某些视频格式可能不兼容。")
                    gr.Markdown("**安装命令**: `pip install ffmpeg-python`")

                gr.Markdown("### 分阶段处理：1.抽帧 → 2.检测（支持多线程并行）")

                with gr.Row():
                    with gr.Column():
                        # 视频输入
                        video_upload = gr.File(
                            label="方式1: 上传视频",
                            file_types=["video"]
                        )
                        with gr.Row():
                            video_dropdown = gr.Dropdown(
                                label="方式2: 选择input_video目录中的视频",
                                choices=[],
                                interactive=True
                            )
                            refresh_videos_btn = gr.Button("刷新列表", size="sm")

                        gr.Markdown("---")

                        # 阶段1：抽帧参数
                        gr.Markdown("### 第一阶段：抽帧参数")

                        with gr.Accordion("视频处理参数", open=True):
                            start_time = gr.Textbox(
                                label="起始时间",
                                value="00:00:00",
                                placeholder="格式: HH:MM:SS 或 MM:SS 或 秒数"
                            )
                            end_time = gr.Textbox(
                                label="结束时间 (可选)",
                                value="",
                                placeholder="格式: HH:MM:SS 或 MM:SS 或 秒数，留空则处理到视频结尾"
                            )
                            frame_interval = gr.Slider(
                                label="抽帧间隔 (每N帧抽取一帧)",
                                minimum=1,
                                maximum=300,
                                value=30,
                                step=1
                            )

                        with gr.Accordion("裁剪参数", open=True):
                            enable_crop = gr.Checkbox(label="启用裁剪", value=True)
                            crop_width = gr.Slider(
                                label="裁剪宽度",
                                minimum=100,
                                maximum=1920,
                                value=550,
                                step=10
                            )
                            crop_height = gr.Slider(
                                label="裁剪高度",
                                minimum=100,
                                maximum=1920,
                                value=870,
                                step=10
                            )
                            crop_x = gr.Slider(
                                label="裁剪起始X",
                                minimum=0,
                                maximum=1920,
                                value=910,
                                step=10
                            )
                            crop_y = gr.Slider(
                                label="裁剪起始Y",
                                minimum=0,
                                maximum=1920,
                                value=530,
                                step=10
                            )

                        # 阶段1按钮
                        extract_frames_btn = gr.Button("🎬 第一阶段：开始抽帧", variant="primary", size="lg")

                        gr.Markdown("---")

                        # 阶段2：检测参数
                        gr.Markdown("### 第二阶段：检测参数")
                        grounding_caption_video = gr.Textbox(
                            label="Detection Prompt",
                            value="bike",
                            placeholder="例如: person, car, bottle"
                        )

                        num_workers = gr.Slider(
                            label="并行线程数",
                            minimum=1,
                            maximum=8,
                            value=8,
                            step=1,
                            info="建议根据CPU核心数设置"
                        )

                        with gr.Accordion("框大小筛选（按面积过滤）", open=True):
                            max_box_area = gr.Slider(
                                label="最大框面积（像素²）",
                                minimum=0,
                                maximum=500000,
                                value=6000,
                                step=5000,
                                info="超过此面积的框将被过滤（用于过滤大物体如树木）。设为0表示不限制。100000≈300x300像素"
                            )
                            min_box_area = gr.Slider(
                                label="最小框面积（像素²）",
                                minimum=0,
                                maximum=10000,
                                value=150,
                                step=100,
                                info="小于此面积的框将被过滤（用于过滤噪声）。设为0表示不限制。500≈22x22像素"
                            )
                            gr.Markdown("""
                            **说明：**
                            - 车辆的识别框通常较小（几百到几万像素）
                            - 树木的识别框通常很大（几十万像素以上）
                            - 通过设置最大面积可以过滤掉大的树木检测框
                            - 通过设置最小面积可以过滤掉小的噪声
                            """)

                        # 阶段2按钮
                        detect_items_btn = gr.Button("🔍 第二阶段：开始检测", variant="primary", size="lg")

                    with gr.Column():
                        # 阶段数据状态（隐藏）
                        stage_data_json = gr.Textbox(
                            label="阶段数据 (内部使用)",
                            visible=False,
                            value="{}"
                        )

                        # 输出结果
                        stage_info_output = gr.Textbox(
                            label="处理信息",
                            lines=8,
                            interactive=False
                        )

                        frame_preview = gr.Gallery(
                            label="抽帧预览 (前5帧)",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="auto"
                        )

                        json_output = gr.Textbox(
                            label="检测结果 (JSON)",
                            lines=10,
                            interactive=False
                        )

                        with gr.Row():
                            json_download = gr.File(
                                label="下载JSON文件",
                                visible=False
                            )
                            zip_download = gr.File(
                                label="下载标注图片ZIP",
                                visible=False
                            )

                        video_gallery = gr.Gallery(
                            label="检测结果样本 (前5帧)",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="auto"
                        )

                # 绑定事件
                refresh_videos_btn.click(
                    fn=update_video_choices,
                    outputs=[video_dropdown]
                )

                # 第一阶段：抽帧
                extract_frames_btn.click(
                    fn=extract_frames_stage,
                    inputs=[
                        video_upload,
                        video_dropdown,
                        start_time,
                        frame_interval,
                        end_time,
                        crop_width,
                        crop_height,
                        crop_x,
                        crop_y,
                        enable_crop
                    ],
                    outputs=[stage_data_json, frame_preview, stage_info_output]
                )

                # 第二阶段：检测
                detect_items_btn.click(
                    fn=detect_items_stage,
                    inputs=[
                        stage_data_json,
                        grounding_caption_video,
                        num_workers,
                        max_box_area,
                        min_box_area
                    ],
                    outputs=[json_output, video_gallery, stage_info_output, json_download, zip_download]
                )

            # 配置标记标签页（简化版）
            with gr.Tab("多区域检测配置"):
                gr.Markdown("### 树集群中心点检测与多区域配置工具")
                gr.Markdown("""
                **功能说明：**
                1. 上传图像自动检测树集群中心点
                2. 配置多个检测区域，裁剪参数相对于树集群中心点
                3. RTSP 运行时会定期重新检测中心点并动态调整裁剪区域

                **使用步骤：**
                1. 输入相机名称和 RTSP 地址
                2. 上传图像并检测树集群中心点
                3. 添加检测区域（裁剪参数相对于中心点）
                4. 预览并保存配置
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. 基础配置")
                        camera_name = gr.Textbox(
                            label="相机名称",
                            placeholder="例如: 衷和楼1702",
                            value=""
                        )
                        rtsp_url_config = gr.Textbox(
                            label="RTSP地址",
                            placeholder="rtsp://...",
                            value=""
                        )
                        sample_interval_config = gr.Slider(
                            label="采样间隔（秒）",
                            minimum=1,
                            maximum=300,
                            value=30,
                            step=1,
                            info="RTSP检测时间间隔"
                        )

                        gr.Markdown("#### 2. 树集群中心点检测")
                        gr.Markdown("""
                        **说明：** 上传一张包含树木的图像，系统将自动检测树集群的中心点。
                        这个中心点将作为所有检测区域裁剪参数的参考原点。
                        """)

                        detect_image_upload = gr.Image(
                            label="上传图像用于检测中心点",
                            type="pil"
                        )

                        with gr.Accordion("检测参数", open=True):
                            dino_tree_caption = gr.Textbox(
                                label="检测文本提示",
                                value="tree",
                                placeholder="例如: tree, plant, bush"
                            )

                        with gr.Accordion("树木面积过滤", open=True):
                            dino_min_box_area = gr.Slider(
                                label="最小框面积（像素²）",
                                minimum=100,
                                maximum=10000,
                                value=500,
                                step=100,
                                info="过滤太小的检测框（噪声）"
                            )
                            dino_max_box_area = gr.Slider(
                                label="最大框面积（像素²）",
                                minimum=10000,
                                maximum=500000,
                                value=100000,
                                step=10000,
                                info="过滤太大的检测框（非树木）"
                            )
                            dino_nms_threshold = gr.Slider(
                                label="NMS 阈值（合并重叠框）",
                                minimum=0.1,
                                maximum=0.9,
                                value=0.3,
                                step=0.05,
                                info="同一物体被检测多次时的合并阈值"
                            )
                            dino_merge_distance = gr.Slider(
                                label="合并距离（像素）",
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=5,
                                info="非常接近的中心点会被合并"
                            )

                        detect_center_btn = gr.Button("🎯 检测树集群中心点", variant="primary", size="sm")

                        # 存储检测到的中心点（隐藏）
                        detected_center_x = gr.Number(
                            label="检测到的中心点 X",
                            value=0,
                            interactive=False,
                            visible=False
                        )
                        detected_center_y = gr.Number(
                            label="检测到的中心点 Y",
                            value=0,
                            interactive=False,
                            visible=False
                        )

                        center_detection_image = gr.Image(
                            label="检测结果可视化",
                            interactive=False
                        )
                        center_detection_info = gr.Textbox(
                            label="检测结果信息",
                            lines=12,
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 3. 检测区域配置")
                        gr.Markdown("""
                        **说明：** 裁剪参数是相对于树集群中心点的偏移量。
                        - 相对偏移 X：负数=左边，正数=右边
                        - 相对偏移 Y：负数=上方，正数=下方
                        """)

                        region_name = gr.Textbox(
                            label="区域名称",
                            placeholder="例如: 左侧区域",
                            value=""
                        )
                        region_id = gr.Textbox(
                            label="区域ID",
                            placeholder="例如: region_a",
                            value=""
                        )

                        with gr.Accordion("裁剪参数", open=True):
                            crop_width_config = gr.Slider(
                                label="裁剪宽度",
                                minimum=100,
                                maximum=1920,
                                value=550,
                                step=10
                            )
                            crop_height_config = gr.Slider(
                                label="裁剪高度",
                                minimum=100,
                                maximum=1920,
                                value=870,
                                step=10
                            )
                            crop_x_config = gr.Slider(
                                label="相对偏移 X",
                                minimum=-2000,
                                maximum=2000,
                                value=-400,
                                step=10,
                                info="相对于中心点的偏移，负数=左边"
                            )
                            crop_y_config = gr.Slider(
                                label="相对偏移 Y",
                                minimum=-1000,
                                maximum=1000,
                                value=200,
                                step=10,
                                info="相对于中心点的偏移，负数=上方"
                            )

                        with gr.Accordion("检测参数", open=True):
                            caption_config = gr.Textbox(
                                label="检测提示词",
                                value="bike",
                                placeholder="例如: bike, person, car"
                            )
                            max_box_area_config = gr.Slider(
                                label="最大框面积（像素²）",
                                minimum=0,
                                maximum=500000,
                                value=6000,
                                step=5000
                            )
                            min_box_area_config = gr.Slider(
                                label="最小框面积（像素²）",
                                minimum=0,
                                maximum=10000,
                                value=150,
                                step=100
                            )

                        add_region_btn = gr.Button("➕ 添加检测区域", variant="primary", size="sm")
                        region_info = gr.Textbox(
                            label="区域信息",
                            lines=3,
                            interactive=False
                        )

                        regions_json = gr.Textbox(
                            label="区域数据 (内部使用)",
                            visible=False,
                            value="[]"
                        )

                        gr.Markdown("#### 实时预览（调整参数查看位置）")
                        gr.Markdown("""
                        **说明：** 上方显示当前配置的区域在图像上的预览位置。
                        调整裁剪参数后，点击"更新预览"查看效果，确认位置正确后再添加到区域列表。
                        """)

                        current_region_preview_image = gr.Image(
                            label="当前区域预览（绿色框=即将添加的区域）",
                            interactive=False
                        )
                        current_region_preview_info = gr.Textbox(
                            label="预览信息",
                            lines=6,
                            interactive=False
                        )

                        update_preview_btn = gr.Button("🔄 更新预览", variant="secondary", size="sm")

                        gr.Markdown("---")

                        # 已添加区域列表预览
                        preview_regions_btn = gr.Button("👁️ 预览所有已添加区域", variant="secondary", size="sm")

                        regions_preview_image = gr.Image(
                            label="所有区域预览",
                            interactive=False
                        )
                        regions_preview_info = gr.Textbox(
                            label="所有区域信息",
                            lines=5,
                            interactive=False
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 4. 配置保存与加载")
                        config_preview = gr.Textbox(
                            label="配置预览 (JSON)",
                            lines=12,
                            interactive=False
                        )
                        with gr.Row():
                            preview_btn = gr.Button("预览配置", variant="secondary", size="sm")
                            save_config_btn = gr.Button("💾 保存配置文件", variant="primary")
                            refresh_config_list_btn = gr.Button("🔄 刷新配置列表", size="sm")

                        config_info = gr.Textbox(
                            label="保存信息",
                            lines=2,
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("#### 加载现有配置")
                        load_config_dropdown = gr.Dropdown(
                            label="选择配置文件",
                            choices=[],
                            interactive=True
                        )
                        load_config_btn = gr.Button("📂 加载配置", variant="secondary")

                        loaded_config_info = gr.Textbox(
                            label="加载信息",
                            lines=5,
                            interactive=False
                        )

                # 事件绑定
                detect_center_btn.click(
                    fn=detect_tree_cluster_center,
                    inputs=[
                        detect_image_upload,
                        dino_tree_caption,
                        dino_min_box_area,
                        dino_max_box_area,
                        dino_nms_threshold,
                        dino_merge_distance
                    ],
                    outputs=[center_detection_image, center_detection_info, detected_center_x, detected_center_y]
                ).then(
                    # 检测中心点后自动更新当前区域预览
                    fn=preview_current_region_on_image,
                    inputs=[
                        detect_image_upload,
                        region_name, region_id,
                        crop_width_config, crop_height_config, crop_x_config, crop_y_config,
                        detected_center_x, detected_center_y
                    ],
                    outputs=[current_region_preview_image, current_region_preview_info]
                )

                # 更新当前区域预览
                update_preview_btn.click(
                    fn=preview_current_region_on_image,
                    inputs=[
                        detect_image_upload,
                        region_name, region_id,
                        crop_width_config, crop_height_config, crop_x_config, crop_y_config,
                        detected_center_x, detected_center_y
                    ],
                    outputs=[current_region_preview_image, current_region_preview_info]
                )

                add_region_btn.click(
                    fn=add_detection_region,
                    inputs=[
                        region_name, region_id,
                        crop_width_config, crop_height_config, crop_x_config, crop_y_config,
                        caption_config,
                        max_box_area_config, min_box_area_config,
                        regions_json
                    ],
                    outputs=[region_info, regions_json]
                ).then(
                    # 添加区域后更新所有区域预览
                    fn=preview_regions_on_image,
                    inputs=[
                        detect_image_upload,
                        regions_json,
                        detected_center_x,
                        detected_center_y
                    ],
                    outputs=[regions_preview_image, regions_preview_info]
                )

                preview_regions_btn.click(
                    fn=preview_regions_on_image,
                    inputs=[
                        detect_image_upload,
                        regions_json,
                        detected_center_x,
                        detected_center_y
                    ],
                    outputs=[regions_preview_image, regions_preview_info]
                )

                preview_btn.click(
                    fn=preview_config,
                    inputs=[
                        regions_json, camera_name, rtsp_url_config, sample_interval_config,
                        detected_center_x, detected_center_y,
                        dino_tree_caption
                    ],
                    outputs=[config_preview]
                )

                save_config_btn.click(
                    fn=save_config_file,
                    inputs=[camera_name, config_preview],
                    outputs=[config_info]
                )

                refresh_config_list_btn.click(
                    fn=lambda: gr.update(choices=list_config_files()),
                    outputs=[load_config_dropdown]
                )

                load_config_btn.click(
                    fn=load_existing_config,
                    inputs=[load_config_dropdown],
                    outputs=[
                        camera_name, regions_json, detected_center_x, detected_center_y,
                        rtsp_url_config, sample_interval_config,
                        dino_tree_caption
                    ]
                ).then(
                    fn=lambda cx, cy: f"✓ 配置已加载\n中心点: ({int(cx)}, {int(cy)})",
                    inputs=[detected_center_x, detected_center_y],
                    outputs=[loaded_config_info]
                )

    block.launch(server_name='0.0.0.0', server_port=19555, debug=args.debug, share=False)
