"""
RTSP 视频流检测服务
从 RTSP 视频流中定期提取帧，进行裁剪和 DINO 物体识别
后台常驻服务，命令行输出日志
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import threading
import signal
import sys
import subprocess
import os
from typing import Optional, Dict, List, Tuple

from is3_metadata_api import MetadataAPI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 打印 OpenCV 信息
logger.info(f"OpenCV 版本: {cv2.__version__}")
build_info = cv2.getBuildInformation()
if build_info:
    first_line = build_info.split('\n')[0]
    logger.info(f"OpenCV 后端: {first_line}")
else:
    logger.info("OpenCV 后端: N/A")


# ========== 配置参数 ==========
CONFIG = {
    # RTSP 配置（从main.json的rtsp_settings读取）
    "rtsp_url": "",

    # 采样间隔（秒）
    "sample_interval": 10,

    # 输出目录
    "output_dir": "output/rtsp_detection",

    # 临时图片路径
    "temp_frame_path": "temp_rtsp_frame.jpg",

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # iS3 上报配置
    "is3": {
        "enabled": True,
        "base_url": "https://server.is3.net.cn",
        "file_base_url": "https://file.is3.net.cn",
        "project_id": "2029815378816741377",
    "access_key": "qFvbo7AcKbU9OdtwTIa_Pg",
    "secret_key": "sXcBRmX6i6Yhfm5TPACzuqOHcFjTwXvpLgvXl74IbT4",
        "folder_id": "2029815379631640577",
        "camera_code": "衷和楼1702",
        "detail_table_code": "camera_result_detail",
        "simple_table_code": "camera_result_simple",
        "timeout": 200,
        # 测试模式：使用 FastAPI 本地服务替代 iS3 上报
        "is_test": True,
        "fastapi_url": "http://localhost:8000"
    },

    # 中心点矫正配置
    "center_correction_interval": 60,  # 每N帧进行一次中心点矫正（0表示禁用）
    "config_file_path": "main.json"  # 相机配置文件路径
}


# ========== 全局变量 ==========
running = True
frame_count = 0
detection_count = 0
total_detected_items = 0

# 中心点矫正相关变量
frame_count_since_correction = 0  # 自上次中心点矫正以来的帧数
current_reference_center = None  # 当前检测到的中心点 {"x": float, "y": float}


# ========== 信号处理 ==========
def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global running
    logger.info("收到停止信号，正在关闭服务...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)


# ========== 资源监控 ==========
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.peak_ram_mb = 0
        self.peak_gpu_mb = 0
        self.lock = threading.Lock()

    def get_ram_usage_mb(self):
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0

    def get_gpu_usage_mb(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def update_peak(self):
        with self.lock:
            ram = self.get_ram_usage_mb()
            gpu = self.get_gpu_usage_mb()
            if ram > self.peak_ram_mb:
                self.peak_ram_mb = ram
            if gpu > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu

    def get_summary(self):
        with self.lock:
            return f"RAM: {self.peak_ram_mb:.0f} MB, GPU: {self.peak_gpu_mb:.0f} MB"


resource_monitor = ResourceMonitor()


class IS3Client:
    """iS3 文件上传和元数据写入客户端（支持 FastAPI 测试模式）"""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.file_base_url = str(config.get("file_base_url", "")).rstrip("/")
        self.project_id = str(config.get("project_id", ""))
        self.access_key = str(config.get("access_key", ""))
        self.secret_key = str(config.get("secret_key", ""))
        self.folder_id = str(config.get("folder_id", ""))
        self.camera_code = str(config.get("camera_code", "衷和楼1702"))
        self.detail_table_code = str(config.get("detail_table_code", "camera_result_detail"))
        self.simple_table_code = str(config.get("simple_table_code", "camera_result_simple"))
        self.timeout = int(config.get("timeout", 20))

        # 测试模式配置
        self.is_test = bool(config.get("is_test", False))
        self.fastapi_url = str(config.get("fastapi_url", "http://localhost:8000"))

        # 初始化 iS3 API（非测试模式）
        if not self.is_test:
            self.api = MetadataAPI(
                base_url=self.base_url,
                prj_id=self.project_id,
                headers={
                    "X-Access-Key": self.access_key,
                    "X-Secret-Key": self.secret_key,
                },
                folder_id=self.folder_id,
                file_base_url=self.file_base_url,
                timeout=self.timeout,
            )
        else:
            self.api = None
            logger.info(f"IS3Client 使用测试模式 (FastAPI: {self.fastapi_url})")

    def is_available(self) -> bool:
        if self.is_test:
            # 测试模式：只需要启用 FastAPI URL
            return self.enabled and bool(self.fastapi_url)
        else:
            # iS3 模式：需要完整配置
            return all([
                self.enabled,
                self.base_url,
                self.project_id,
                self.access_key,
                self.secret_key,
                self.detail_table_code,
                self.simple_table_code,
            ])

    def upload_file(self, file_path: str) -> Optional[str]:
        if not self.folder_id:
            logger.warning("iS3 folder_id 未配置，跳过文件上传")
            return None

        if not file_path or not os.path.exists(file_path):
            logger.warning(f"待上传文件不存在: {file_path}")
            return None

        try:
            result = self.api.upload_file(file_path)
            if isinstance(result, dict) and result.get("success", False):
                file_data = result.get("data") or {}
                rel_url = file_data.get("url")
                if rel_url:
                    return f"{self.file_base_url}{rel_url}"
                logger.warning(f"iS3 上传成功但缺少 data.url: {result}")
                return None

            logger.warning(f"iS3 文件上传失败: {result}")
            return None
        except Exception as e:
            logger.warning(f"iS3 文件上传异常: {e}")
            return None

    def check_folder_access(self) -> bool:
        if self.is_test:
            # 测试模式：检查 FastAPI 服务是否可用
            import requests
            try:
                resp = requests.get(f"{self.fastapi_url}/api/health", timeout=5)
                if resp.status_code == 200:
                    logger.info(f"FastAPI 服务连接成功 ({self.fastapi_url})")
                    return True
                logger.warning(f"FastAPI 服务响应异常: {resp.status_code}")
                return False
            except Exception as e:
                logger.warning(f"FastAPI 服务连接失败: {e}")
                return False

        if not self.folder_id:
            return False
        try:
            result = self.api.get_file_list(self.folder_id, page_num=1, page_size=1)
            if isinstance(result, dict) and result.get("code") == 200:
                logger.info("iS3 目录校验成功")
                return True
            logger.warning(f"iS3 目录校验失败: {result}")
            return False
        except Exception as e:
            logger.warning(f"iS3 目录校验异常: {e}")
            return False

    def add_data(self, meta_table_code: str, row: dict) -> bool:
        try:
            result = self.api.insert_data(meta_table_code, [row])
            if isinstance(result, dict) and result.get("success", False):
                return True
            logger.warning(f"iS3 写入失败: table={meta_table_code}, resp={result}")
            return False
        except Exception as e:
            logger.warning(f"iS3 写入异常: table={meta_table_code}, err={e}")
            return False

    def upload_detection_result(self, latest_result: dict, region_id: str = None) -> bool:
        timestamp_text = latest_result.get("timestamp") or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detections = latest_result.get("detections") or []
        detection_count = int(latest_result.get("detection_count", len(detections)))

        image_path = latest_result.get("annotated_image") or latest_result.get("cropped_image")

        # 如果有区域ID，在上传前为图片添加区域前缀和时间戳
        temp_image_path = None
        if image_path and region_id and not self.is_test:
            try:
                import tempfile
                import shutil
                from pathlib import Path

                # 读取原始图片
                img = cv2.imread(image_path)
                if img is not None:
                    # 创建临时文件，带区域前缀和时间戳
                    temp_dir = tempfile.gettempdir()
                    original_name = Path(image_path).stem

                    # 从timestamp_text解析时间戳
                    try:
                        # timestamp_text格式: "2026-03-16 21:10:05"
                        ts = datetime.strptime(timestamp_text, '%Y-%m-%d %H:%M:%S')
                        timestamp_suffix = ts.strftime('%Y%m%d_%H%M%S')
                    except:
                        timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

                    temp_filename = f"{region_id}_{original_name}_{timestamp_suffix}.jpg"
                    temp_image_path = os.path.join(temp_dir, temp_filename)
                    cv2.imwrite(temp_image_path, img)
                    logger.debug(f"创建临时文件用于上传: {temp_filename}")
            except Exception as e:
                logger.warning(f"创建临时文件失败: {e}")
                temp_image_path = None

        # 添加 region_id 到数据中
        detail_row = {
            "camera_code": self.camera_code,
            "camera_time": timestamp_text,
            "camera_img": None,
            "result_json": json.dumps(detections, ensure_ascii=False),
            "result_num": detection_count
        }
        simple_row = {
            "camera_code": self.camera_code,
            "camera_time": timestamp_text,
            "camera_result": detection_count
        }

        # 添加 region_id（如果提供）
        if region_id:
            detail_row["region_id"] = region_id
            simple_row["region_id"] = region_id

        try:
            # 选择上传模式
            if self.is_test:
                return self._upload_to_fastapi(detail_row, simple_row, image_path, region_id)
            else:
                # iS3 模式：上传文件（使用带前缀的临时文件）
                upload_path = temp_image_path if temp_image_path else image_path
                camera_img_value = self.upload_file(upload_path) if upload_path else None
                detail_row["camera_img"] = camera_img_value

                detail_ok = self.add_data(self.detail_table_code, detail_row)
                simple_ok = self.add_data(self.simple_table_code, simple_row)

                if not detail_ok:
                    logger.warning("iS3 写入失败: camera_result_detail")
                if not simple_ok:
                    logger.warning("iS3 写入失败: camera_result_simple")
                return detail_ok and simple_ok
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    logger.debug(f"已删除临时文件: {temp_image_path}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {e}")

    def _upload_to_fastapi(self, detail_row: dict, simple_row: dict, image_path: str = None, region_id: str = None) -> bool:
        """上传到 FastAPI 本地服务（测试模式），含图片上传"""
        import requests

        try:
            region_suffix = f" [{region_id}]" if region_id else ""
            logger.debug(f"FastAPI 上传中{region_suffix}...")

            # 上传详细数据
            detail_url = f"{self.fastapi_url}/api/detection/detail"
            detail_resp = requests.post(detail_url, json=detail_row, timeout=10)

            # 上传简化数据
            simple_url = f"{self.fastapi_url}/api/detection/simple"
            simple_resp = requests.post(simple_url, json=simple_row, timeout=10)

            detail_ok = detail_resp.status_code == 200
            simple_ok = simple_resp.status_code == 200

            # 上传标注图片
            image_ok = True
            if image_path and os.path.exists(image_path):
                image_url = f"{self.fastapi_url}/api/upload/image"
                ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{ts_prefix}_{region_id or 'default'}_{Path(image_path).name}"
                with open(image_path, "rb") as f:
                    image_resp = requests.post(image_url, files={"file": (filename, f, "image/jpeg")}, timeout=30)
                image_ok = image_resp.status_code == 200
                if image_ok:
                    logger.debug(f"FastAPI 图片上传成功{region_suffix}")
                else:
                    logger.warning(f"FastAPI 图片上传失败{region_suffix}: {image_resp.status_code}")

            if detail_ok and simple_ok and image_ok:
                logger.debug(f"FastAPI 上传成功{region_suffix}")
            else:
                logger.warning(f"FastAPI 上传部分失败{region_suffix}: detail={detail_ok}, simple={simple_ok}, image={image_ok}")

            return detail_ok and simple_ok and image_ok

        except requests.exceptions.ConnectionError:
            logger.warning(f"FastAPI 服务连接失败 ({self.fastapi_url})")
            return False
        except Exception as e:
            logger.warning(f"FastAPI 上传异常: {e}")
            return False


# ========== 模型加载 ==========
def load_detection_model():
    """加载 Falcon-Perception 检测模型"""
    from falcon_detector import FalconPerceptionModel

    project_root = Path(__file__).resolve().parent
    hf_local_dir = str(project_root / "weight" / "Falcon-Perception")

    logger.info(f"加载 Falcon-Perception 模型...")
    logger.info(f"  本地模型目录: {hf_local_dir}")

    model = FalconPerceptionModel(
        hf_local_dir=hf_local_dir,
        device=CONFIG["device"],
    )

    logger.info(f"✓ 模型加载完成 (设备: {CONFIG['device']})")
    return model


# ========== 图像变换 ==========
# 不再需要 image_transform_grounding，Falcon-Perception 自行处理图像预处理


# ========== 配置文件加载 ==========
def load_camera_config(config_path: str) -> dict:
    """
    加载相机检测配置文件（仅支持version 2.0格式）

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config_version = config.get("version", "2.0")
        if config_version != "2.0":
            logger.error(f"不支持的配置版本: {config_version}，仅支持version 2.0")
            return {}

        logger.info(f"配置文件已加载: {config_path} (版本: {config_version})")
        logger.info(f"  相机名称: {config.get('camera_name', 'N/A')}")
        logger.info(f"  检测区域数: {len(config.get('detection_regions', []))}")
        logger.info(f"  参考中心点: {config.get('reference_center', {})}")

        return config

    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


# ========== 中心点矫正 ==========
def correct_center_point(model, frame: np.ndarray, camera_config: dict,
                         current_ref_center: dict) -> dict:
    """
    检测树集群中心点（使用 Falcon-Perception）

    Args:
        model: FalconPerceptionModel 实例
        frame: 当前帧
        camera_config: 相机配置
        current_ref_center: 当前参考中心点 {"x": float, "y": float}

    Returns:
        new_ref_center: 更新后的参考中心点 {"x": float, "y": float}
    """
    # 获取树检测参数
    dino_params = camera_config.get("dino_params", {})
    caption = dino_params.get("caption", "tree")
    min_box_area = dino_params.get("min_box_area", 500)
    max_box_area = dino_params.get("max_box_area", 100000)
    nms_threshold = dino_params.get("nms_threshold", 0.3)
    merge_distance = dino_params.get("merge_distance", 30.0)

    # 使用 Falcon-Perception 检测树集群中心点
    from falcon_detector import detect_trees
    from image_registration import find_main_tree_cluster

    trees, _ = detect_trees(
        fp_model=model,
        image=frame,
        caption=caption,
        min_box_area=min_box_area,
        max_box_area=max_box_area,
        nms_threshold=nms_threshold,
    )

    if not trees:
        logger.warning("中心点矫正失败：未检测到树木")
        return current_ref_center

    cluster_info = find_main_tree_cluster(trees, merge_distance=merge_distance)

    if cluster_info is None:
        # 使用所有树木中心
        centers = [tree["center"] for tree in trees]
        new_center = (int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers])))
    else:
        new_center = cluster_info["center"]

    logger.info(f"✓ 中心点矫正完成:")
    logger.info(f"  检测到的中心: ({new_center[0]}, {new_center[1]})")

    # 直接更新参考中心点为检测到的中心点
    new_ref_center = {"x": float(new_center[0]), "y": float(new_center[1])}

    return new_ref_center


# ========== 多区域裁剪 ==========
def crop_multi_regions(image_np: np.ndarray, regions_config: List[Dict],
                        current_center: dict) -> List[Dict]:
    """
    对图像进行多区域裁剪（与Gradio逻辑完全一致）

    Args:
        image_np: 输入图像
        regions_config: 区域配置列表
        current_center: 实时检测到的当前中心点 {"x": float, "y": float}

    Returns:
        裁剪结果列表 [{"region_id": "...", "crop": np.ndarray, "crop_params": dict}, ...]

    裁剪公式（与Gradio一致）：
        绝对坐标 = 实时检测到的中心点 + 配置的相对偏移
    """
    results = []
    h, w = image_np.shape[:2]

    for region_config in regions_config:
        region_id = region_config.get("region_id", "unknown")
        region_name = region_config.get("region_name", "")

        # 获取裁剪参数
        crop_params = region_config.get("crop_params", {})
        crop_width = crop_params.get("width", 550)
        crop_height = crop_params.get("height", 870)

        # 获取相对偏移
        relative_x = crop_params.get("relative_x", 0)
        relative_y = crop_params.get("relative_y", 0)

        # 获取实时检测到的中心点
        center_x = current_center.get("x", 0)
        center_y = current_center.get("y", 0)

        # 计算绝对坐标（与Gradio完全一致：abs_x = center_x + relative_x）
        crop_x = int(center_x + relative_x)
        crop_y = int(center_y + relative_y)

        # 确保裁剪区域在图像范围内
        crop_x = max(0, min(crop_x, w - crop_width))
        crop_y = max(0, min(crop_y, h - crop_height))

        # 裁剪图像
        cropped = image_np[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        results.append({
            "region_id": region_id,
            "region_name": region_name,
            "crop": cropped,
            "crop_params": {
                "width": crop_width,
                "height": crop_height,
                "x": crop_x,
                "y": crop_y,
                "relative_x": relative_x,
                "relative_y": relative_y,
                "center_x": center_x,
                "center_y": center_y
            }
        })

    return results


# ========== 多区域检测 ==========
def detect_multi_regions(model, image_crops: List[Dict], regions_config: List[Dict],
                         device: str) -> Dict[str, dict]:
    """
    对多个裁剪区域进行检测

    Args:
        model: GroundingDINO模型
        image_crops: 裁剪结果列表
        regions_config: 区域配置列表
        device: 设备

    Returns:
        检测结果字典 {"region_id": {"detections": [...], "counts": {"blue_bike": N, "yellow_bike": M}}, ...}
    """
    results = {}

    # 创建区域配置映射
    region_config_map = {r.get("region_id"): r for r in regions_config}

    for crop_data in image_crops:
        region_id = crop_data["region_id"]
        cropped_image = crop_data["crop"]

        # 获取区域检测参数（全部从配置文件读取）
        region_config = region_config_map.get(region_id, {})
        detection_params = region_config.get("detection_params", {})

        max_box_area = detection_params.get("max_box_area", 6000)
        min_box_area = detection_params.get("min_box_area", 150)

        # 每个区域检测 bike + blue_bike + yellow_bike
        all_detections = []
        bike_detections = []  # 单独保存 bike 检测结果，用于生成标注图
        counts = {"blue_bike": 0, "yellow_bike": 0}

        for bike_type in ["bike", "blue_bike", "yellow_bike"]:
            detections = detect_objects(
                model, cropped_image, bike_type,
                max_box_area, min_box_area, bike_type
            )
            if bike_type == "bike":
                bike_detections = detections
            else:
                counts[bike_type] = len(detections)
            all_detections.extend(detections)

        results[region_id] = {
            "detections": all_detections,
            "counts": counts,
            "bike_detections": bike_detections,
        }

    return results


# ========== 检测函数 ==========
def detect_objects(model, image_np, caption, max_box_area, min_box_area, object_type="bike"):
    """
    使用 Falcon-Perception 检测图像中的物体

    Args:
        model: FalconPerceptionModel 实例
        image_np: 输入图像 (BGR)
        caption: 检测提示词
        max_box_area: 最大框面积
        min_box_area: 最小框面积
        object_type: 物体类型标记 (blue_bike/yellow_bike)
    """
    from falcon_detector import detect_objects as fp_detect_objects
    return fp_detect_objects(
        fp_model=model,
        image_np=image_np,
        caption=caption,
        max_box_area=max_box_area,
        min_box_area=min_box_area,
        object_type=object_type,
    )


# ========== 多区域结果保存 ==========
def save_multi_region_result(frame_np: np.ndarray, image_crops: List[Dict],
                              detection_results: Dict[str, dict], output_dir: Path,
                              timestamp: datetime, camera_name: str = "camera",
                              offset_info: Dict = None, reference_center: dict = None) -> Dict[str, any]:
    """
    保存多区域检测结果（覆写模式，避免占用过多空间）

    Args:
        frame_np: 原始帧
        image_crops: 裁剪结果列表
        detection_results: 检测结果字典
        output_dir: 输出目录
        timestamp: 时间戳
        camera_name: 相机名称
        offset_info: 偏移信息
        reference_center: 参考中心点 {"x": float, "y": float}

    Returns:
        当前检测结果字典
    """
    from falcon_detector import annotate_detections

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用固定文件名（覆写模式）
    latest_filename = "latest"

    # 保存原始帧（带中心点标注）- 覆写
    original_frame = frame_np.copy()
    if reference_center:
        cx = int(reference_center.get("x", 0))
        cy = int(reference_center.get("y", 0))
        # 绘制中心点（红色十字）
        cv2.drawMarker(original_frame, (cx, cy), (0, 0, 255),
                      markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        # 绘制圆形标记
        cv2.circle(original_frame, (cx, cy), 10, (0, 0, 255), 2)

    original_path = output_dir / "original" / f"{latest_filename}.jpg"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(original_path), original_frame)

    total_detections = 0
    regions_data = []

    for crop_data in image_crops:
        region_id = crop_data["region_id"]
        region_name = crop_data["region_name"]
        cropped_image = crop_data["crop"]
        crop_params = crop_data["crop_params"]

        # 创建区域子目录
        region_dir = output_dir / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)

        # 保存裁剪图像 - 覆写
        crop_path = region_dir / f"{latest_filename}_crop.jpg"
        cv2.imwrite(str(crop_path), cropped_image)

        # 获取检测结果（新格式：包含detections和counts）
        region_result_data = detection_results.get(region_id, {"detections": [], "counts": {"blue_bike": 0, "yellow_bike": 0}})
        detections = region_result_data.get("detections", [])
        bike_detections = region_result_data.get("bike_detections", detections)
        counts = region_result_data.get("counts", {"blue_bike": 0, "yellow_bike": 0})
        total_detections += len(detections)

        # 只用 bike 检测结果生成标注图像（用于上传）
        if bike_detections:
            annotated_rgb = annotate_detections(cropped_image, bike_detections)
        else:
            annotated_rgb = cropped_image

        # 保存带标注的图像 - 覆写
        annotated_path = region_dir / f"{latest_filename}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_rgb)

        # 保存区域检测结果JSON - 覆写
        region_result = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "timestamp_unix": timestamp.timestamp(),
            "camera_name": camera_name,
            "region_id": region_id,
            "region_name": region_name,
            "original_image": str(original_path),
            "cropped_image": str(crop_path),
            "annotated_image": str(annotated_path),
            "crop_params": crop_params,
            "offset_correction": offset_info,
            "detection_count": len(detections),
            "blue_bike_count": counts.get("blue_bike", 0),
            "yellow_bike_count": counts.get("yellow_bike", 0),
            "detections": detections
        }

        region_json_path = region_dir / f"{latest_filename}.json"
        with open(region_json_path, 'w', encoding='utf-8') as f:
            json.dump(region_result, f, ensure_ascii=False, indent=2)

        regions_data.append({
            "region_id": region_id,
            "region_name": region_name,
            "detection_count": len(detections),
            "blue_bike_count": counts.get("blue_bike", 0),
            "yellow_bike_count": counts.get("yellow_bike", 0),
            "crop_params": crop_params,
            "json_path": str(region_json_path),
            "annotated_path": str(annotated_path),
            "crop_path": str(crop_path)
        })

    # 计算总体数量
    total_blue_bike = sum(r["blue_bike_count"] for r in regions_data)
    total_yellow_bike = sum(r["yellow_bike_count"] for r in regions_data)

    # 保存当前汇总结果 - 覆写
    summary_result = {
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "timestamp_unix": timestamp.timestamp(),
        "camera_name": camera_name,
        "original_image": str(original_path),
        "total_detection_count": total_detections,
        "total_blue_bike_count": total_blue_bike,
        "total_yellow_bike_count": total_yellow_bike,
        "regions": regions_data,
        "offset_correction": offset_info
    }

    summary_json_path = output_dir / "latest_result.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, ensure_ascii=False, indent=2)

    # 更新历史记录（保留最近100条）
    history_path = output_dir / "history.json"
    history = []
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except:
                history = []

    # 添加当前记录（简化版，不包含details）
    history_record = {
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "timestamp_unix": timestamp.timestamp(),
        "total_detection_count": total_detections,
        "total_blue_bike_count": total_blue_bike,
        "total_yellow_bike_count": total_yellow_bike,
        "regions": [{"region_id": r["region_id"], "region_name": r["region_name"],
                     "detection_count": r["detection_count"],
                     "blue_bike_count": r["blue_bike_count"],
                     "yellow_bike_count": r["yellow_bike_count"]} for r in regions_data]
    }
    history.insert(0, history_record)
    history = history[:100]  # 只保留最近100条

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return summary_result

    return json_paths


# ========== 主循环 ==========
def main():
    global running, frame_count, detection_count, total_detected_items
    global current_reference_center, frame_count_since_correction

    logger.info("=" * 60)

    is3_client = IS3Client(CONFIG.get("is3", {}))
    if is3_client.is_available():
        logger.info("iS3 上报已启用")
        is3_client.check_folder_access()
    else:
        logger.warning("iS3 上报未完整配置，将跳过上传")
    logger.info("RTSP 视频流检测服务")
    logger.info("=" * 60)

    # 打印配置
    logger.info("配置参数:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

    # 加载相机配置（必须存在）
    camera_config_path = os.getenv("CAMERA_CONFIG_PATH", CONFIG.get("config_file_path", "main.json"))
    camera_config = {}
    multi_region_mode = False

    if camera_config_path and os.path.exists(camera_config_path):
        camera_config = load_camera_config(camera_config_path)
        multi_region_mode = bool(camera_config.get("detection_regions"))

        if not multi_region_mode:
            logger.error("配置文件未定义detection_regions，无法启动服务")
            return

        logger.info("✓ 多区域检测模式已启用")

        # 从 dino_params 加载树检测参数（用于参考中心点检测）
        dino_params = camera_config.get("dino_params", {})
        if dino_params:
            logger.info(f"  树检测参数: caption={dino_params.get('caption', 'tree')}, "
                      f"box_threshold={dino_params.get('box_threshold', 0.3)}, "
                      f"text_threshold={dino_params.get('text_threshold', 0.25)}")
            logger.info(f"  面积过滤: min_box_area={dino_params.get('min_box_area', 500)}, "
                      f"max_box_area={dino_params.get('max_box_area', 100000)}")
            logger.info(f"  NMS/合并: nms_threshold={dino_params.get('nms_threshold', 0.3)}, "
                      f"merge_distance={dino_params.get('merge_distance', 30.0)}")

        # 加载参考中心点
        reference_center = camera_config.get("reference_center", {})
        if reference_center:
            logger.info(f"  参考中心点: x={reference_center.get('x')}, y={reference_center.get('y')}")

        # 加载中心点矫正配置
        center_correction_config = camera_config.get("center_correction", {})
        center_correction_interval = center_correction_config.get("interval", 60)
        logger.info(f"  中心点矫正间隔: {center_correction_interval} 帧")

        # 更新RTSP配置
        if camera_config.get("rtsp_settings", {}).get("url"):
            CONFIG["rtsp_url"] = camera_config["rtsp_settings"]["url"]
        if camera_config.get("rtsp_settings", {}).get("sample_interval"):
            CONFIG["sample_interval"] = camera_config["rtsp_settings"]["sample_interval"]
        if camera_config.get("rtsp_settings", {}).get("output_dir"):
            CONFIG["output_dir"] = camera_config["rtsp_settings"]["output_dir"]

        # 更新相机名称
        if camera_config.get("camera_name"):
            CONFIG["is3"]["camera_code"] = camera_config["camera_name"]

        # 打印所有检测区域配置
        for region in camera_config.get("detection_regions", []):
            region_id = region.get("region_id", "unknown")
            crop_params = region.get("crop_params", {})
            detection_params = region.get("detection_params", {})
            logger.info(f"  区域[{region_id}]:")
            logger.info(f"    裁剪: {crop_params.get('width', 0)}x{crop_params.get('height', 0)} @ "
                      f"({crop_params.get('relative_x', 0)}, {crop_params.get('relative_y', 0)})")
            logger.info(f"    检测: caption={detection_params.get('caption', 'bike')}, "
                      f"thresholds={detection_params.get('box_threshold', 0.12)}/"
                      f"{detection_params.get('text_threshold', 1.0)}, "
                      f"area={detection_params.get('min_box_area', 150)}-"
                      f"{detection_params.get('max_box_area', 6000)}")
    else:
        logger.error(f"配置文件不存在: {camera_config_path}，无法启动服务")
        return

    # 加载模型
    try:
        model = load_detection_model()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 创建输出目录
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    # 使用 ffmpeg 从 RTSP 流获取帧（直接保存到文件）
    def get_frame_from_rtsp_ffmpeg(rtsp_url, output_path, timeout=15):
        """使用 ffmpeg 从 RTSP 流获取一帧并保存到文件"""
        cmd = [
            'ffmpeg',
            '-y',                                # 覆盖输出文件
            '-rtsp_transport', 'tcp',              # 使用 TCP 传输
            '-i', rtsp_url,
            '-vframes', '1',                        # 只读取1帧
            '-q:v', '2',                            # 高质量
            '-update', '1',                         # 更新模式（覆盖文件）
            '-nostats',                            # 不显示统计信息
            '-loglevel', 'error',                   # 只显示错误
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            if result.returncode != 0:
                logger.debug(f"ffmpeg 错误: {result.stderr}")
                return False

            # 检查文件是否创建成功
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True

            return False

        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg 超时")
            return False
        except Exception as e:
            logger.warning(f"ffmpeg 执行失败: {e}")
            return False

    def load_frame_from_file(file_path):
        """从文件加载帧"""
        if not os.path.exists(file_path):
            return None

        frame = cv2.imread(file_path)
        if frame is None:
            return None

        return frame

    logger.info("使用 ffmpeg 保存帧到文件，然后读取")

    # 测试 ffmpeg 连接
    logger.info("测试 RTSP 连接...")
    temp_path = CONFIG["temp_frame_path"]

    # 清理可能存在的旧临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 测试获取一帧
    if not get_frame_from_rtsp_ffmpeg(CONFIG["rtsp_url"], temp_path):
        logger.error("无法从 RTSP 流获取帧")
        logger.error("请确保:")
        logger.error("  1. RTSP 服务器正在运行")
        logger.error(f"  2. URL 正确: {CONFIG['rtsp_url']}")
        logger.error(f"  3. 可以手动测试: ffmpeg -rtsp_transport tcp -i {CONFIG['rtsp_url']} -vframes 1 test.jpg")
        return

    # 读取测试帧
    test_frame = load_frame_from_file(temp_path)
    if test_frame is None:
        logger.error(f"无法读取保存的图片: {temp_path}")
        return

    h, w = test_frame.shape[:2]
    logger.info(f"✓ RTSP 连接成功!")
    logger.info(f"  分辨率: {w}x{h}")
    logger.info(f"  采样间隔: {CONFIG['sample_interval']} 秒")
    logger.info(f"  检测模式: {'多区域' if multi_region_mode else '单区域'}")

    # 初始化参考中心点（version 2.0 格式）
    center_correction_interval = CONFIG.get("center_correction_interval", 60)  # 默认值
    if multi_region_mode and camera_config.get("version") == "2.0":
        # 从配置文件读取参考中心点（仅作为初始值）
        reference_center = camera_config.get("reference_center", {}).copy()
        current_reference_center = reference_center
        center_offset = {"dx": 0, "dy": 0}
        # 从配置文件读取中心点矫正间隔
        center_correction_interval = camera_config.get("center_correction", {}).get("interval", center_correction_interval)
        logger.info(f"  中心点矫正: 启用 (间隔: {center_correction_interval} 帧)")
        logger.info(f"  初始参考中心: x={current_reference_center.get('x')}, y={current_reference_center.get('y')}")

    logger.info("=" * 60)

    last_sample_time = 0
    consecutive_empty_frames = 0

    # 首次启动时进行中心点矫正（version 2.0 格式）
    if multi_region_mode and camera_config.get("version") == "2.0":
        logger.info("=" * 60)
        logger.info("执行首次中心点矫正...")
        current_reference_center = correct_center_point(
            model, test_frame, camera_config, current_reference_center
        )
        frame_count_since_correction = 0
        logger.info("=" * 60)

    try:
        while running:
            current_time = time.time()

            # 检查是否需要采样
            if current_time - last_sample_time < CONFIG["sample_interval"]:
                time.sleep(0.5)
                continue

            # 更新资源监控
            resource_monitor.update_peak()

            # 采样时间到，获取新帧
            last_sample_time = current_time
            frame_count += 1
            timestamp = datetime.now()

            logger.info(f"[帧 #{frame_count}] 开始处理 ({timestamp.strftime('%H:%M:%S')})")

            # 使用 ffmpeg 获取帧并保存到临时文件
            temp_path = CONFIG["temp_frame_path"]
            if not get_frame_from_rtsp_ffmpeg(CONFIG["rtsp_url"], temp_path):
                logger.warning("无法读取帧，等待后重试...")
                consecutive_empty_frames += 1

                if consecutive_empty_frames > 5:
                    logger.error("连续读取失败，等待 30 秒后重试...")
                    time.sleep(30)
                    consecutive_empty_frames = 0
                else:
                    time.sleep(2)

                continue

            consecutive_empty_frames = 0

            # 从文件读取帧
            frame = load_frame_from_file(temp_path)
            if frame is None:
                logger.error(f"无法读取保存的图片: {temp_path}")
                continue

            # 检查帧质量（检测灰色/空帧）
            gray_mean = np.mean(frame)
            gray_std = np.std(frame)

            if gray_std < 10:
                logger.warning(f"检测到低质量帧 (均值={gray_mean:.1f}, 标准差={gray_std:.1f})，跳过")
                continue

            logger.debug(f"帧质量正常 (均值={gray_mean:.1f}, 标准差={gray_std:.1f})")

            try:
                # 中心点矫正（version 2.0 格式）
                if multi_region_mode and camera_config.get("version") == "2.0":
                    if center_correction_interval > 0:
                        frame_count_since_correction += 1
                        if frame_count_since_correction >= center_correction_interval:
                            logger.info(f"[中心点矫正] 帧计数: {frame_count_since_correction}/{center_correction_interval}")
                            current_reference_center = correct_center_point(
                                model, frame, camera_config, current_reference_center
                            )
                            frame_count_since_correction = 0

                if multi_region_mode:
                    # 多区域检测模式
                    detection_regions = camera_config.get("detection_regions", [])

                    # 进行多区域裁剪（使用实时检测到的中心点）
                    image_crops = crop_multi_regions(frame, detection_regions, current_reference_center)

                    # 多区域检测
                    detection_results = detect_multi_regions(model, image_crops, detection_regions, CONFIG["device"])

                    # 统计总检测数（新格式：detection_results值是字典，包含detections和counts）
                    total_frame_detections = sum(len(data.get("detections", [])) for data in detection_results.values())

                    detection_count += 1
                    total_detected_items += total_frame_detections

                    # 保存多区域结果（覆写模式）
                    latest_result = save_multi_region_result(
                        frame, image_crops, detection_results,
                        CONFIG["output_dir"], timestamp,
                        camera_config.get("camera_name", CONFIG["is3"]["camera_code"]),
                        None,  # 不再需要偏移信息
                        current_reference_center
                    )

                    # 创建region_id到annotated_path的映射
                    region_annotated_paths = {
                        r["region_id"]: r["annotated_path"]
                        for r in latest_result.get("regions", [])
                    }

                    # 输出结果
                    for crop_data in image_crops:
                        region_id = crop_data["region_id"]
                        region_name = crop_data["region_name"]
                        region_result_data = detection_results.get(region_id, {"detections": [], "counts": {"blue_bike": 0, "yellow_bike": 0}})
                        detections = region_result_data.get("detections", [])
                        counts = region_result_data.get("counts", {"blue_bike": 0, "yellow_bike": 0})

                        if detections:
                            logger.info(f"  区域 [{region_name}] 检测到 {len(detections)} 个目标 (blue_bike: {counts.get('blue_bike', 0)}, yellow_bike: {counts.get('yellow_bike', 0)})")
                            for i, det in enumerate(detections[:3]):  # 只显示前3个
                                logger.info(f"    [{i+1}] {det['label']} ({det.get('object_type', 'bike')}) - 面积: {det['area']:.0f}")
                        else:
                            logger.info(f"  区域 [{region_name}] 未检测到目标 (blue_bike: {counts.get('blue_bike', 0)}, yellow_bike: {counts.get('yellow_bike', 0)})")

                    # 计算总体数量
                    total_blue = sum(r.get("counts", {}).get("blue_bike", 0) for r in detection_results.values())
                    total_yellow = sum(r.get("counts", {}).get("yellow_bike", 0) for r in detection_results.values())
                    logger.info(f"  总检测数: {total_frame_detections} (blue_bike: {total_blue}, yellow_bike: {total_yellow})")

                    # 上传到 iS3/FastAPI 平台（每个区域单独上传）
                    if is3_client.is_available():
                        if total_frame_detections < 5:
                            logger.info(f"  上传跳过：检测到 {total_frame_detections} 个目标（阈值：5）")
                        else:
                            # 为每个区域单独上传
                            for crop_data in image_crops:
                                region_id = crop_data["region_id"]
                                region_name = crop_data["region_name"]
                                region_result_data = detection_results.get(region_id, {"detections": [], "counts": {"blue_bike": 0, "yellow_bike": 0}})
                                detections = region_result_data.get("detections", [])
                                counts = region_result_data.get("counts", {"blue_bike": 0, "yellow_bike": 0})

                                if len(detections) == 0:
                                    continue

                                # 构造该区域的上传结果
                                upload_result = {
                                    "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    "detections": detections,
                                    "detection_count": len(detections),
                                    "blue_bike_count": counts.get("blue_bike", 0),
                                    "yellow_bike_count": counts.get("yellow_bike", 0),
                                    "annotated_image": region_annotated_paths.get(region_id)
                                }

                                is3_ok = is3_client.upload_detection_result(upload_result, region_id)
                                if is3_ok:
                                    logger.info(f"  上传成功 [{region_name}]: {len(detections)} 个目标")
                                else:
                                    logger.warning(f"  上传失败 [{region_name}]")

                    logger.info(f"  结果已保存（覆写模式）")

                logger.info(f"  资源: {resource_monitor.get_summary()}")

            except Exception as e:
                logger.error(f"处理失败: {e}")
                import traceback
                traceback.print_exc()

            logger.info("-" * 40)

    except KeyboardInterrupt:
        pass

    finally:
        logger.info("=" * 60)
        logger.info("服务统计:")
        logger.info(f"  总帧数: {frame_count}")
        logger.info(f"  检测次数: {detection_count}")
        logger.info(f"  总检测数: {total_detected_items}")
        logger.info(f"  资源峰值: {resource_monitor.get_summary()}")
        logger.info("=" * 60)
        logger.info("服务已停止")


if __name__ == "__main__":
    main()
