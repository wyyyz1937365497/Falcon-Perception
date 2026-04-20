"""
图像配准模块
用于通过 Grounding DINO 物体识别计算摄像头偏移量，支持自动偏移补偿
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class OffsetSmoother:
    """偏移量平滑器 - 使用滑动窗口平均"""

    def __init__(self, window_size: int = 5):
        """
        初始化偏移量平滑器

        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.dx_history: List[int] = []
        self.dy_history: List[int] = []

    def update(self, dx: int, dy: int) -> Tuple[int, int]:
        """
        更新并返回平滑后的偏移量

        Args:
            dx: X方向偏移量
            dy: Y方向偏移量

        Returns:
            (平滑后的dx, 平滑后的dy)
        """
        self.dx_history.append(dx)
        self.dy_history.append(dy)

        # 保持窗口大小
        if len(self.dx_history) > self.window_size:
            self.dx_history.pop(0)
            self.dy_history.pop(0)

        # 计算平均值
        smooth_dx = int(np.mean(self.dx_history))
        smooth_dy = int(np.mean(self.dy_history))

        return smooth_dx, smooth_dy

    def reset(self):
        """重置历史数据"""
        self.dx_history.clear()
        self.dy_history.clear()

    def get_current(self) -> Tuple[int, int]:
        """
        获取当前平滑后的偏移量

        Returns:
            (当前dx, 当前dy)
        """
        if not self.dx_history:
            return 0, 0
        return int(np.mean(self.dx_history)), int(np.mean(self.dy_history))


def load_dino_model(model_config_path: str = None, model_checkpoint_path: str = None,
                    device: str = None, hf_local_dir: str = None):
    """
    加载 Falcon-Perception 模型（替代原 Grounding DINO 加载）

    Args:
        model_config_path: 保留参数，不再使用
        model_checkpoint_path: 保留参数，不再使用
        device: 设备 ("cuda" 或 "cpu")
        hf_local_dir: Falcon-Perception 本地模型目录

    Returns:
        FalconPerceptionModel 实例
    """
    from falcon_detector import FalconPerceptionModel
    return FalconPerceptionModel(hf_local_dir=hf_local_dir, device=device)


def apply_nms_to_trees(trees: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    对检测到的树木应用非极大值抑制（NMS），合并重叠的框

    当同一个物体被多次检测时，会产生多个接近的框。
    NMS 会保留置信度最高的框，删除与其重叠度过高的其他框。

    Args:
        trees: 检测到的树木列表
        iou_threshold: IoU 阈值，超过此值的框会被合并

    Returns:
        经过 NMS 处理后的树木列表
    """
    if not trees:
        return []

    # 按面积降序排序（大框优先保留）
    sorted_trees = sorted(trees, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)

    keep = []
    while sorted_trees:
        # 保留当前置信度最高的框
        current = sorted_trees.pop(0)
        keep.append(current)

        # 删除与当前框 IoU 过高的框
        filtered = []
        for tree in sorted_trees:
            iou = calculate_iou(current["bbox"], tree["bbox"])
            if iou < iou_threshold:
                filtered.append(tree)
        sorted_trees = filtered

    return keep


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU 值 (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 如果没有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算并集区域
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def dino_predict_trees(model, image: np.ndarray, caption: str = "tree",
                       box_threshold: float = 0.3, text_threshold: float = 0.25,
                       device: str = "cuda",
                       min_box_area: int = 500, max_box_area: int = 100000,
                       min_box_aspect: float = 0.3, max_box_aspect: float = 3.0,
                       nms_threshold: float = 0.3) -> Tuple[List[Dict], np.ndarray]:
    """
    使用 Falcon-Perception 检测图像中的树木

    Args:
        model: FalconPerceptionModel 实例
        image: 输入图像 (BGR格式)
        caption: 检测文本提示
        box_threshold: 保留参数（不生效）
        text_threshold: 保留参数（不生效）
        device: 保留参数（不生效）
        min_box_area: 最小框面积（过滤太小的干扰）
        max_box_area: 最大框面积（过滤太大的干扰）
        min_box_aspect: 最小宽高比
        max_box_aspect: 最大宽高比
        nms_threshold: NMS IoU阈值，用于合并重叠框

    Returns:
        (检测到的树木列表, 可视化图像)
        树木信息: [{"bbox": [x1, y1, x2, y2], "center": (cx, cy)}, ...]
    """
    from falcon_detector import detect_trees
    return detect_trees(
        model, image, caption=caption,
        min_box_area=min_box_area, max_box_area=max_box_area,
        min_box_aspect=min_box_aspect, max_box_aspect=max_box_aspect,
        nms_threshold=nms_threshold,
    )


def find_main_tree_cluster(trees: List[Dict], padding_ratio: float = 0.1,
                            merge_distance: float = 30.0) -> Dict:
    """
    找到主要的树木群集并返回其边界框

    使用改进的方法：
    1. 首先合并非常接近的中心点（避免同一物体被检测多次）
    2. 然后进行聚类分析
    3. 自动过滤掉远处的干扰点

    Args:
        trees: 检测到的树木列表
        padding_ratio: 边界框的扩展比例
        merge_distance: 合并接近中心点的距离阈值（像素）

    Returns:
        包含边界框和群集信息的字典:
        - bbox: [x1, y1, x2, y2] 群集边界框
        - center: (cx, cy) 群集中心
        - tree_count: 群集中的树木数量
        - padding_bbox: 扩展后的边界框（用于裁剪）
        - cluster_indices: 群集中的树木索引列表
    """
    if not trees:
        return None

    # 提取所有树木的中心点
    centers = np.array([tree["center"] for tree in trees])

    # 首先合并非常接近的中心点（避免同一物体被检测多次）
    merged_centers = merge_nearby_centers(centers, merge_distance)
    logger.debug(f"中心点合并: {len(centers)} -> {len(merged_centers)}")

    # 如果合并后只剩1个点，直接返回
    if len(merged_centers) == 1:
        cx, cy = int(merged_centers[0][0]), int(merged_centers[0][1])
        padding = 50
        h, w = 2048, 2048  # 默认值，实际应该从图像获取
        return {
            "bbox": [max(0, cx - padding), max(0, cy - padding), min(w, cx + padding), min(h, cy + padding)],
            "center": (cx, cy),
            "tree_count": 1,
            "padding_bbox": [max(0, cx - padding * 1.5), max(0, cy - padding * 1.5),
                           min(w, cx + padding * 1.5), min(h, cy + padding * 1.5)],
            "cluster_indices": [0]
        }

    # 1. 计算所有点对之间的距离
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(merged_centers))

    # 2. 找到每个点的最近邻距离
    min_distances = np.min(distances + np.eye(len(distances)) * 1e6, axis=1)

    # 3. 使用中位数距离作为"正常"树木间距的参考
    # 这样可以过滤掉远处的干扰点
    median_distance = np.median(min_distances)

    # 使用略大一点的阈值（1.5倍中位数），允许一些变化
    max_cluster_distance = median_distance * 1.8

    # 4. 基于密度找到核心区域
    # 找到密度最大的点（最近的邻居最多）
    neighbor_counts = np.sum(distances < max_cluster_distance, axis=1)
    core_idx = int(np.argmax(neighbor_counts))
    core_point = merged_centers[core_idx]

    # 5. 从核心点开始，使用BFS扩展群集
    # 只包含距离小于阈值的点
    cluster_indices = set([core_idx])
    queue = [core_idx]
    visited = {core_idx}

    while queue:
        current_idx = queue.pop(0)
        current_point = merged_centers[current_idx]

        # 找到所有距离小于阈值的未访问点
        for i in range(len(merged_centers)):
            if i not in visited:
                dist = distances[current_idx, i]
                if dist < max_cluster_distance:
                    cluster_indices.add(i)
                    visited.add(i)
                    queue.append(i)

    cluster_indices = list(cluster_indices)
    cluster_centers = merged_centers[cluster_indices]

    # 6. 如果找到的群集太小（少于3个树），尝试放宽阈值
    if len(cluster_indices) < 3 and len(merged_centers) >= 3:
        # 放宽阈值到2倍中位数
        max_cluster_distance = median_distance * 2.5
        cluster_indices = set([core_idx])
        queue = [core_idx]
        visited = {core_idx}

        while queue:
            current_idx = queue.pop(0)
            for i in range(len(merged_centers)):
                if i not in visited:
                    if distances[current_idx, i] < max_cluster_distance:
                        cluster_indices.add(i)
                        visited.add(i)
                        queue.append(i)

        cluster_indices = list(cluster_indices)
        cluster_centers = merged_centers[cluster_indices]

    # 7. 计算边界框
    if len(cluster_centers) > 0:
        min_x = int(np.min(cluster_centers[:, 0]))
        max_x = int(np.max(cluster_centers[:, 0]))
        min_y = int(np.min(cluster_centers[:, 1]))
        max_y = int(np.max(cluster_centers[:, 1]))

        # 添加一些padding
        width = max_x - min_x
        height = max_y - min_y
        padding_x = int(width * padding_ratio)
        padding_y = int(height * padding_ratio)

        bbox = [min_x, min_y, max_x, max_y]
        padding_bbox = [
            max(0, min_x - padding_x),
            max(0, min_y - padding_y),
            max_x + padding_x,
            max_y + padding_y
        ]

        return {
            "bbox": bbox,
            "center": (int(core_point[0]), int(core_point[1])),
            "tree_count": len(cluster_indices),
            "padding_bbox": padding_bbox,
            "cluster_indices": cluster_indices
        }

    return None


def merge_nearby_centers(centers: np.ndarray, distance_threshold: float) -> np.ndarray:
    """
    合并非常接近的中心点

    当同一个物体被检测多次时，会产生多个接近的中心点。
    此函数将这些接近的中心点合并为一个，使用加权平均（权重为置信度）。

    Args:
        centers: 中心点数组 (N x 2)
        distance_threshold: 距离阈值，小于此值的点会被合并

    Returns:
        合并后的中心点数组
    """
    if len(centers) <= 1:
        return centers

    # 使用简单的聚类合并
    merged = []
    used = set()

    for i in range(len(centers)):
        if i in used:
            continue

        # 找到所有与当前点接近的点
        cluster = [i]
        for j in range(i + 1, len(centers)):
            if j not in used:
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < distance_threshold:
                    cluster.append(j)
                    used.add(j)

        # 计算合并后的中心点（使用平均位置）
        cluster_centers = centers[cluster]
        merged_center = np.mean(cluster_centers, axis=0)
        merged.append(merged_center)
        used.add(i)

    return np.array(merged)


def detect_tree_cluster_center(model, image: np.ndarray, caption: str = "tree",
                                box_threshold: float = 0.3, text_threshold: float = 0.25,
                                device: str = "cuda",
                                min_box_area: int = 500, max_box_area: int = 100000,
                                nms_threshold: float = 0.3, merge_distance: float = 30.0) -> Optional[Dict]:
    """
    直接检测图像中的树集群中心点（简化的单次检测方法）

    Args:
        model: Grounding DINO 模型
        image: 输入图像 (BGR格式)
        caption: 检测文本提示
        box_threshold: 框置信度阈值
        text_threshold: 文本阈值
        device: 设备
        min_box_area: 最小框面积（过滤太小的干扰）
        max_box_area: 最大框面积（过滤太大的干扰）
        nms_threshold: NMS IoU阈值，用于合并重叠框
        merge_distance: 合并接近中心点的距离阈值

    Returns:
        包含中心点信息的字典:
        - center: (cx, cy) 树集群中心点
        - bbox: [x1, y1, x2, y2] 群集边界框
        - tree_count: 检测到的树木总数
        - cluster_tree_count: 群集中的树木数量
        - image_size: (w, h) 图像尺寸
    """
    try:
        # 检测树木（带框大小限制和NMS）
        trees, _ = dino_predict_trees(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            nms_threshold=nms_threshold
        )

        if not trees:
            return None

        # 找到主要树木群集（带接近点合并）
        cluster_info = find_main_tree_cluster(trees, merge_distance=merge_distance)
        if cluster_info is None:
            return None

        h, w = image.shape[:2]

        return {
            "center": cluster_info["center"],
            "bbox": cluster_info["bbox"],
            "padding_bbox": cluster_info["padding_bbox"],
            "tree_count": len(trees),
            "cluster_tree_count": cluster_info["tree_count"],
            "image_size": (w, h)
        }

    except Exception as e:
        logger.error(f"检测树集群中心点失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_grid_pattern(tree_centers: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
    """
    自动分析树木中心点是否形成网格模式

    通过分析树木之间的距离和角度关系，自动识别网格模式，
    无需预先指定行数和列数。

    Args:
        tree_centers: 树木中心点数组 (N x 2)

    Returns:
        (排序后的网格点索引, 分析信息)
    """
    info = {
        "total_trees": len(tree_centers),
        "grid_found": False,
        "grid_rows": 0,
        "grid_cols": 0,
        "grid_points": None,
        "reason": ""
    }

    # 网格识别已被禁用，直接返回 None
    info["reason"] = "网格识别已禁用，使用集群方法代替"
    return None, info


def apply_region_offset(base_center: Tuple[int, int], region_config: Dict) -> Dict:
    """
    根据树集群中心点计算实际的裁剪区域坐标

    Args:
        base_center: 树集群中心点 (cx, cy)
        region_config: 区域配置，包含相对偏移的裁剪参数

    Returns:
        更新后的区域配置（包含绝对坐标）
    """
    result = region_config.copy()
    crop_params = region_config.get("crop_params", {}).copy()

    # 计算绝对坐标（相对偏移 + 中心点）
    crop_params["absolute_x"] = crop_params.get("relative_x", 0) + base_center[0]
    crop_params["absolute_y"] = crop_params.get("relative_y", 0) + base_center[1]

    result["crop_params"] = crop_params
    return result


class DinoTreeGridRegistration:
    """
    基于 Grounding DINO 树木检测的图像配准
    使用 DINO 检测树木，然后使用密度聚类方法找到主要树木群集进行偏移计算
    """

    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化 DINO 树木网格配准器

        Args:
            model: Grounding DINO 模型
            config: 配置字典，包含:
                - caption: 检测文本 (默认 "tree")
                - box_threshold: 框置信度阈值 (默认 0.3)
                - text_threshold: 文本阈值 (默认 0.25)
                - device: 设备 (默认 "cuda")
                - smoothing_window: 平滑窗口大小
        """
        self.model = model
        self.config = config or {}

        # 默认配置
        self.caption = self.config.get("caption", "tree")
        self.box_threshold = self.config.get("box_threshold", 0.3)
        self.text_threshold = self.config.get("text_threshold", 0.25)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # 存储集群信息
        self.cluster_bbox = None
        self.cluster_padding_bbox = None

        # 平滑窗口
        window_size = self.config.get("smoothing_window", 5)
        self.smoother = OffsetSmoother(window_size=window_size)

        # 参考数据
        self.reference_trees: Optional[List[Dict]] = None
        self.reference_grid_points: Optional[np.ndarray] = None
        self.reference_image: Optional[np.ndarray] = None

        logger.info(f"DinoTreeGridRegistration 初始化: caption={self.caption}, "
                   f"box_threshold={self.box_threshold}")

    def detect_trees(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        检测图像中的树木

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            (检测到的树木列表, 可视化图像)
        """
        return dino_predict_trees(
            model=self.model,
            image=image,
            caption=self.caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )

    def set_reference_image(self, image: np.ndarray) -> bool:
        """
        设置参考图像并检测树木群集

        Args:
            image: 参考图像 (BGR格式)

        Returns:
            是否成功设置
        """
        try:
            self.reference_image = image.copy()

            # 检测树木
            trees, _ = self.detect_trees(image)

            if len(trees) < 2:
                logger.warning(f"检测到的树木数量太少: {len(trees)}, 至少需要2棵树")
                # 继续处理

            # 直接使用树木群集方法（更稳定）
            cluster_info = find_main_tree_cluster(trees)
            if cluster_info is not None:
                cluster_indices = cluster_info["cluster_indices"]
                tree_centers = np.array([tree["center"] for tree in trees])
                self.reference_grid_points = tree_centers[cluster_indices]
                self.reference_trees = [trees[i] for i in cluster_indices]
                self.cluster_bbox = cluster_info["bbox"]
                self.cluster_padding_bbox = cluster_info["padding_bbox"]
                logger.info(f"参考图像树木群集检测完成: 检测到 {len(trees)} 棵树, "
                          f"识别群集点 {len(self.reference_grid_points)} 个")
            else:
                # 最后的备选方案：使用所有检测到的树木
                tree_centers = np.array([tree["center"] for tree in trees])
                self.reference_grid_points = tree_centers
                self.reference_trees = trees
                self.cluster_bbox = None
                self.cluster_padding_bbox = None
                logger.info(f"参考图像树木检测完成: 检测到 {len(trees)} 棵树, "
                          f"使用所有检测点")

            # 重置平滑器
            self.smoother.reset()

            return True

        except Exception as e:
            logger.error(f"设置参考图像失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compute_offset(self, current_image: np.ndarray) -> Tuple[int, int, float, Dict]:
        """
        计算当前图像相对于参考图像的偏移量

        Args:
            current_image: 当前图像 (BGR格式)

        Returns:
            (dx, dy, confidence, debug_info)
        """
        debug_info = {
            "matched_count": 0,
            "detected_count": 0,
            "expected_count": len(self.reference_grid_points) if self.reference_grid_points is not None else 0,
            "raw_dx": 0,
            "raw_dy": 0,
            "success": False,
            "cluster_method_used": True
        }

        if self.reference_grid_points is None or len(self.reference_grid_points) == 0:
            logger.warning("参考图像未设置")
            smooth_dx, smooth_dy = self.smoother.get_current()
            return smooth_dx, smooth_dy, 0.0, debug_info

        try:
            # 检测当前图像中的树木
            curr_trees, _ = self.detect_trees(current_image)
            debug_info["detected_count"] = len(curr_trees)

            if len(curr_trees) == 0:
                logger.warning("当前图像未检测到树木")
                smooth_dx, smooth_dy = self.smoother.get_current()
                return smooth_dx, smooth_dy, 0.0, debug_info

            # 直接使用群集方法
            cluster_info = find_main_tree_cluster(curr_trees)
            if cluster_info is not None:
                cluster_indices = cluster_info["cluster_indices"]
                curr_centers = np.array([tree["center"] for tree in curr_trees])
                curr_grid_points = curr_centers[cluster_indices]
            else:
                # 备选方案：使用所有检测到的树木
                curr_grid_points = np.array([tree["center"] for tree in curr_trees])

            # 匹配参考点和当前点
            ref_points = self.reference_grid_points
            curr_points = curr_grid_points

            # 计算距离矩阵
            distances = cdist(ref_points, curr_points)

            # 使用匈牙利算法进行最优匹配
            row_ind, col_ind = linear_sum_assignment(distances)

            # 过滤距离过大的匹配
            max_distance = 150  # 最大匹配距离
            valid_matches = []
            offsets = []

            for r, c in zip(row_ind, col_ind):
                if distances[r, c] < max_distance:
                    valid_matches.append((r, c))
                    offsets.append([
                        curr_points[c][0] - ref_points[r][0],
                        curr_points[c][1] - ref_points[r][1]
                    ])

            debug_info["matched_count"] = len(valid_matches)

            if len(valid_matches) < 2:  # 至少需要2个匹配
                logger.warning(f"匹配点数量不足: {len(valid_matches)}")
                smooth_dx, smooth_dy = self.smoother.get_current()
                return smooth_dx, smooth_dy, 0.0, debug_info

            # 计算平均偏移量
            offsets = np.array(offsets)
            avg_offset = np.mean(offsets, axis=0)
            dx = int(avg_offset[0])
            dy = int(avg_offset[1])

            debug_info["raw_dx"] = dx
            debug_info["raw_dy"] = dy

            # 应用平滑
            smooth_dx, smooth_dy = self.smoother.update(dx, dy)

            # 计算置信度（基于匹配比例）
            confidence = min(1.0, len(valid_matches) / debug_info["expected_count"])

            debug_info["success"] = True
            debug_info["smooth_dx"] = smooth_dx
            debug_info["smooth_dy"] = smooth_dy
            debug_info["confidence"] = confidence

            logger.debug(f"DINO树木群集偏移计算成功: dx={smooth_dx}, dy={smooth_dy}, "
                        f"matched={len(valid_matches)}/{debug_info['expected_count']}, "
                        f"confidence={confidence:.2f}")

            return smooth_dx, smooth_dy, confidence, debug_info

        except Exception as e:
            logger.error(f"计算偏移量失败: {e}")
            import traceback
            traceback.print_exc()
            smooth_dx, smooth_dy = self.smoother.get_current()
            debug_info["success"] = False
            return smooth_dx, smooth_dy, 0.0, debug_info

    def apply_offset_to_region(self, region: Dict, dx: int, dy: int) -> Dict:
        """
        将偏移量应用到检测区域配置

        Args:
            region: 区域配置字典
            dx: X方向偏移量
            dy: Y方向偏移量

        Returns:
            更新后的区域配置
        """
        result = region.copy()

        if "crop_params" in result:
            crop_params = result["crop_params"].copy()
            crop_params["x"] = crop_params.get("x", 0) + dx
            crop_params["y"] = crop_params.get("y", 0) + dy
            result["crop_params"] = crop_params

        if "offset_correction" in result:
            offset_correction = result["offset_correction"].copy()
            offset_correction["current_dx"] = dx
            offset_correction["current_dy"] = dy
            result["offset_correction"] = offset_correction

        return result

    def visualize_detection(self, image: np.ndarray, output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        可视化树木检测结果和群集识别

        Args:
            image: 输入图像
            output_path: 保存路径（可选）

        Returns:
            可视化图像
        """
        try:
            trees, vis_image = self.detect_trees(image)

            # 绘制所有树木中心点（绿色小点）
            for tree in trees:
                cx, cy = tree["center"]
                cv2.circle(vis_image, (cx, cy), 3, (0, 255, 0), -1)

            # 直接使用树木群集方法
            cluster_info = find_main_tree_cluster(trees)

            used_points = None
            method_name = "None"

            if cluster_info is not None:
                cluster_indices = cluster_info["cluster_indices"]
                cluster_points = np.array([tree["center"] for tree in trees])[cluster_indices]
                used_points = cluster_points
                method_name = "Cluster"

                # 获取边界框（用于绘制红色遮罩）
                bbox = cluster_info["bbox"]
                padding_bbox = cluster_info["padding_bbox"]

                # 确保边界框在图像范围内
                h, w = vis_image.shape[:2]
                x1, y1, x2, y2 = padding_bbox
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # 绘制红色半透明遮罩
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # 红色填充
                cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)  # 30%透明度

                # 绘制边界框边线（红色）
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # 绘制群集中心（红色大圆点）
                center = cluster_info["center"]
                cv2.circle(vis_image, center, 20, (0, 0, 255), -1)
                cv2.putText(vis_image, "CLUSTER", (center[0] - 40, center[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 绘制群集中的点（黄色圆圈带中心点）
                for i, pt in enumerate(cluster_points):
                    # 黄色外圈
                    cv2.circle(vis_image, tuple(pt.astype(int)), 12, (0, 255, 255), 2)
                    # 红色内点
                    cv2.circle(vis_image, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

            # 添加文本信息
            info_text = f"Detected: {len(trees)} trees"
            if cluster_info is not None:
                info_text += f" | Cluster: {len(cluster_info['cluster_indices'])} trees"
            cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if cluster_info is not None:
                bbox = cluster_info["bbox"]
                cluster_text = f"Area: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                cv2.putText(vis_image, cluster_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, vis_image)
                logger.info(f"树木检测可视化已保存: {output_path}")

            return vis_image

        except Exception as e:
            logger.error(f"可视化树木检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# 兼容旧版本的 ImageRegistration 类
class ImageRegistration:
    """图像配准类 - 使用特征匹配计算偏移量（保留用于兼容）"""

    def __init__(self, method: str = "ORB", config: Optional[Dict] = None):
        """
        初始化图像配准器

        Args:
            method: 配准方法 ("ORB" 或 "SIFT")
            config: 配置字典
        """
        self.method = method.upper()
        self.config = config or {}
        logger.warning(f"ImageRegistration ({method}) 已弃用，请使用 DinoTreeGridRegistration")

    def set_reference_image(self, image: np.ndarray, landmark_points: List[Dict] = None) -> bool:
        """已弃用，无操作"""
        logger.warning("ImageRegistration.set_reference_image 已弃用，无操作")
        return False

    def compute_offset(self, current_image: np.ndarray) -> Tuple[int, int, float, Dict]:
        """已弃用，返回零偏移"""
        return 0, 0, 0.0, {"success": False, "deprecated": True}

    def apply_offset_to_region(self, region: Dict, dx: int, dy: int) -> Dict:
        """应用偏移量（保留功能）"""
        result = region.copy()
        if "crop_params" in result:
            crop_params = result["crop_params"].copy()
            crop_params["x"] = crop_params.get("x", 0) + dx
            crop_params["y"] = crop_params.get("y", 0) + dy
            result["crop_params"] = crop_params
        return result
