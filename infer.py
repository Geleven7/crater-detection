import json
import time
import cv2
import numpy as np
import os
import psutil
from collections import defaultdict
from ais_bench.infer.interface import InferSession
from scipy.ndimage import maximum_filter

# ==================== 预处理和解码函数 ====================
def letterbox_circle(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """CircleNet 的 letterbox 函数"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)

def preprocess_image_for_om(image, cfg, bgr2rgb=True):
    """图片预处理 - 适配 OM 模型"""
    img, scale_ratio, pad_size = letterbox_circle(
        image, 
        new_shape=cfg['input_shape'],
        color=(114, 114, 114)
    )
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    if cfg.get('normalize', True):
        img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img, scale_ratio, pad_size

def decode_circlenet_output(heatmap, radius_map, offset_map, conf_thresh=0.3, 
                            input_size=(640, 640), original_size=None):
    """解码 CircleNet 输出"""
    if len(heatmap.shape) == 4:
        heatmap = heatmap[0, 0]
    elif len(heatmap.shape) == 3:
        heatmap = heatmap[0]
    
    if len(radius_map.shape) == 4:
        radius_map = radius_map[0, 0]
    elif len(radius_map.shape) == 3:
        radius_map = radius_map[0]
    
    if len(offset_map.shape) == 4:
        offset_map = offset_map[0]
    elif len(offset_map.shape) == 3:
        offset_map = offset_map
    
    heatmap = 1.0 / (1.0 + np.exp(-heatmap))
    
    detections = []
    h, w = heatmap.shape
    
    local_max = maximum_filter(heatmap, size=3) == heatmap
    peaks = np.where((heatmap > conf_thresh) & local_max)
    
    for i in range(len(peaks[0])):
        y, x = peaks[0][i], peaks[1][i]
        conf = heatmap[y, x]
        radius = radius_map[y, x]
        
        if len(offset_map.shape) == 3:
            offset_x = offset_map[0, y, x]
            offset_y = offset_map[1, y, x]
        else:
            offset_x = offset_map[1, y, x] if offset_map.shape[0] == 2 else offset_map[0, y, x]
            offset_y = offset_map[0, y, x] if offset_map.shape[0] == 2 else offset_map[1, y, x]
        
        scale_x = input_size[1] / w
        scale_y = input_size[0] / h
        
        center_x = (x + offset_x) * scale_x
        center_y = (y + offset_y) * scale_y
        radius_val = radius * scale_x
        
        if original_size is not None:
            scale_to_original_x = original_size[1] / input_size[1]
            scale_to_original_y = original_size[0] / input_size[0]
            center_x = center_x * scale_to_original_x
            center_y = center_y * scale_to_original_y
            radius_val = radius_val * scale_to_original_x
        
        detections.append([center_x, center_y, radius_val, conf])
    
    return detections

# ==================== 1. 从test文件夹获取所有图片 ====================
def get_test_images_from_folder(image_dir, max_images=None):
    """从文件夹获取所有测试图片"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    image_names = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_dir, file))
            image_names.append(file)
    
    # 排序保证一致性
    sorted_pairs = sorted(zip(image_paths, image_names))
    image_paths, image_names = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        image_names = image_names[:max_images]
    
    print(f"从 {image_dir} 找到 {len(image_paths)} 张测试图片")
    return list(image_paths), list(image_names)

# ==================== 2. FPS测试（使用test文件夹）====================
def test_fps_from_folder(model, image_dir, cfg, max_images=None, warmup=10, iterations_per_image=5):
    """使用test文件夹中的图片测试FPS"""
    print(f"\n{'='*50}")
    print(f"FPS测试 (使用文件夹: {image_dir})")
    print(f"{'='*50}")
    
    # 获取所有测试图片
    image_paths, image_names = get_test_images_from_folder(image_dir, max_images)
    
    if not image_paths:
        print("错误: 文件夹中没有找到图片")
        return 0, 0, {}
    
    # 准备测试数据
    test_data = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            img_input, _, _ = preprocess_image_for_om(image, cfg)
            test_data.append({
                'input': img_input,
                'name': image_names[len(test_data)],
                'shape': image.shape
            })
    
    print(f"成功加载 {len(test_data)} 张测试图片")
    
    # 预热
    print(f"预热中 ({warmup} 次)...")
    for _ in range(warmup):
        for data in test_data:
            _ = model.infer(data['input'])
    
    # 正式测试
    print(f"正式测试中 (每张图片推理 {iterations_per_image} 次)...")
    all_times = []
    per_image_stats = []
    
    for data in test_data:
        img_times = []
        for _ in range(iterations_per_image):
            start_time = time.perf_counter()
            _ = model.infer(data['input'])
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000
            img_times.append(inference_time)
            all_times.append(inference_time)
        
        per_image_stats.append({
            'image_name': data['name'],
            'avg_time': np.mean(img_times),
            'min_time': np.min(img_times),
            'max_time': np.max(img_times),
            'std_time': np.std(img_times)
        })
    
    # 统计
    all_times = np.array(all_times)
    avg_time = np.mean(all_times)
    std_time = np.std(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)
    fps = 1000 / avg_time
    
    print(f"\n总体统计:")
    print(f"  测试图片数: {len(test_data)}")
    print(f"  总推理次数: {len(all_times)}")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  标准差: {std_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  平均FPS: {fps:.2f} 帧/秒")
    
    print(f"\n各图片详细统计 (前5张):")
    for stat in per_image_stats[:5]:
        print(f"  {stat['image_name']:<30} 平均: {stat['avg_time']:.2f}ms, 范围: [{stat['min_time']:.2f}-{stat['max_time']:.2f}]ms")
    if len(per_image_stats) > 5:
        print(f"  ... 还有 {len(per_image_stats)-5} 张图片")
    
    return fps, avg_time, {
        'overall': {'fps': fps, 'avg_time': avg_time, 'std_time': std_time},
        'per_image': per_image_stats
    }

# ==================== 3. GFLOPs测试 ====================
def estimate_gflops(model_path, input_shape=(1, 3, 512, 512)):
    """估算模型GFLOPs"""
    print(f"\n{'='*50}")
    print(f"GFLOPs估算")
    print(f"{'='*50}")
    
    input_size = input_shape[2]
    print(f"输入尺寸: {input_size}x{input_size}")
    print(f"模型类型: CircleNet (圆形检测网络)")
    
    # 基于ResNet50骨干网络的估算
    base_gflops = 4.1
    scale_factor = (input_size / 224) ** 2
    backbone_gflops = base_gflops * scale_factor
    head_gflops = backbone_gflops * 0.2
    total_gflops = backbone_gflops + head_gflops
    
    print(f"\n估算方法: 基于ResNet50骨干网络")
    print(f"  骨干网络: {backbone_gflops:.2f} GFLOPs")
    print(f"  检测头: {head_gflops:.2f} GFLOPs")
    print(f"  总计: {total_gflops:.2f} GFLOPs")
    print(f"\n⚠️  注意: 这是估算值，实际值可能因具体架构而异")
    
    return total_gflops

# ==================== 4. 精度评估（使用test文件夹和COCO标注）====================
def calculate_circle_iou(circle1, circle2):
    """计算两个圆的IoU"""
    x1, y1, r1 = circle1[:3]
    x2, y2, r2 = circle2[:3]
    
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if dist >= r1 + r2:
        return 0.0
    
    if dist <= abs(r1 - r2):
        area_small = np.pi * min(r1, r2)**2
        area_big = np.pi * max(r1, r2)**2
        return area_small / area_big
    
    r1_sq = r1 * r1
    r2_sq = r2 * r2
    d_sq = dist * dist
    
    angle1 = np.arccos((r1_sq + d_sq - r2_sq) / (2 * r1 * dist))
    angle2 = np.arccos((r2_sq + d_sq - r1_sq) / (2 * r2 * dist))
    
    area1 = r1_sq * angle1
    area2 = r2_sq * angle2
    area_intersect = area1 + area2 - 0.5 * np.sqrt((-dist + r1 + r2) * (dist + r1 - r2) * (dist - r1 + r2) * (dist + r1 + r2))
    area_union = np.pi * (r1_sq + r2_sq) - area_intersect
    
    return area_intersect / area_union if area_union > 0 else 0

def evaluate_accuracy_from_folder(model, cfg, image_dir, coco_annotation_file, 
                                   conf_threshold=0.3, iou_threshold=0.5, max_images=None):
    """使用test文件夹中的图片和COCO标注评估精度"""
    print(f"\n{'='*50}")
    print(f"精度评估 (使用文件夹: {image_dir})")
    print(f"{'='*50}")
    
    # 加载COCO标注
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图片名称到ID的映射
    name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 按图片ID组织标注
    gt_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        x_center = x + w/2
        y_center = y + h/2
        radius = max(w, h)/2
        
        gt_by_image[ann['image_id']].append({
            'bbox': [x_center, y_center, radius],
            'category_id': ann['category_id'],
            'area': ann['area']
        })
    
    # 获取测试图片
    image_paths, image_names = get_test_images_from_folder(image_dir, max_images)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    processed_images = 0
    matched_images = 0
    
    for img_path, img_name in zip(image_paths, image_names):
        # 检查是否有对应的标注
        if img_name not in name_to_id:
            continue
        
        image_id = name_to_id[img_name]
        gt_list = gt_by_image.get(image_id, [])
        
        if not gt_list:
            continue
        
        matched_images += 1
        
        # 推理
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        img_input, _, _ = preprocess_image_for_om(image, cfg)
        outputs = model.infer(img_input)
        
        # 解码预测
        if isinstance(outputs, list) and len(outputs) >= 3:
            detections = decode_circlenet_output(
                outputs[0], outputs[1], outputs[2],
                conf_thresh=conf_threshold,
                input_size=cfg['input_shape'],
                original_size=image.shape[:2]
            )
        else:
            detections = []
        
        # 匹配预测和真实
        matched_gt = set()
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_list):
                if idx in matched_gt:
                    continue
                iou = calculate_circle_iou(det[:3], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
                all_ious.append(best_iou)
            else:
                total_fp += 1
        
        total_fn += len(gt_list) - len(matched_gt)
        processed_images += 1
        
        if processed_images % 50 == 0:
            print(f"已处理 {processed_images} 张图片...")
    
    # 计算指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0
    
    print(f"\n评估结果 (IOU阈值={iou_threshold}):")
    print(f"  有标注的图片数: {matched_images}")
    print(f"  实际处理图片数: {processed_images}")
    print(f"  总真实目标: {total_tp + total_fn}")
    print(f"  总预测目标: {total_tp + total_fp}")
    print(f"\n  Precision (精确率): {precision:.4f}")
    print(f"  Recall (召回率):    {recall:.4f}")
    print(f"  F1-Score:          {f1_score:.4f}")
    print(f"  Average IoU:       {avg_iou:.4f}")
    print(f"\n详细统计:")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'processed_images': processed_images,
        'matched_images': matched_images
    }

# ==================== 5. 综合性能测试（统一使用test文件夹）====================
def comprehensive_benchmark_unified(model, model_path, cfg, test_image_dir,
                                     coco_annotation_file=None,
                                     max_test_images=None):
    """
    统一的综合性能测试，FPS和精度都使用同一个test文件夹
    """
    print(f"\n{'#'*60}")
    print(f"# CircleNet 模型综合性能测试报告")
    print(f"# 测试文件夹: {test_image_dir}")
    print(f"{'#'*60}")
    
    results = {}
    
    # 1. 模型大小
    print(f"\n【1. 模型大小】")
    print(f"{'='*50}")
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        results['model_size_mb'] = model_size
        print(f"模型路径: {model_path}")
        print(f"模型大小: {model_size:.2f} MB")
    else:
        print(f"模型文件不存在: {model_path}")
        results['model_size_mb'] = 0
    
    # 2. 内存占用
    print(f"\n【2. 内存占用】")
    print(f"{'='*50}")
    process = psutil.Process()
    memory_info = process.memory_info()
    results['memory_mb'] = memory_info.rss / 1024 / 1024
    print(f"当前进程内存: {results['memory_mb']:.2f} MB")
    
    # 3. FPS测试（使用test文件夹）
    fps, avg_time, fps_details = test_fps_from_folder(
        model, test_image_dir, cfg, 
        max_images=max_test_images,
        warmup=10, 
        iterations_per_image=5
    )
    results['fps'] = fps
    results['avg_inference_time_ms'] = avg_time
    results['fps_details'] = fps_details
    
    # 4. GFLOPs估算
    print(f"\n【4. GFLOPs估算】")
    gflops = estimate_gflops(model_path, input_shape=(1, 3, cfg['input_shape'][0], cfg['input_shape'][1]))
    results['gflops'] = gflops
    
    # 5. 精度评估（如果有COCO标注）
    if coco_annotation_file and os.path.exists(coco_annotation_file):
        print(f"\n【5. 精度评估】")
        accuracy_results = evaluate_accuracy_from_folder(
            model, cfg, test_image_dir, coco_annotation_file,
            conf_threshold=cfg.get('conf_thres', 0.3),
            iou_threshold=cfg.get('iou_thres', 0.5),
            max_images=max_test_images
        )
        results.update(accuracy_results)
    else:
        print(f"\n【5. 精度评估】")
        print("⚠️  未提供COCO标注文件，跳过精度评估")
        print("如需精度评估，请提供 COCO 标注文件")
    
    # 6. 总结报告
    print(f"\n{'#'*60}")
    print(f"# 性能总结")
    print(f"{'#'*60}")
    image_count = len(get_test_images_from_folder(test_image_dir, max_test_images)[0])
    print(f"测试图片数:   {image_count}")
    print(f"模型大小:     {results.get('model_size_mb', 0):.2f} MB")
    print(f"内存占用:     {results.get('memory_mb', 0):.2f} MB")
    print(f"平均FPS:      {results.get('fps', 0):.2f} 帧/秒")
    print(f"平均推理时间: {results.get('avg_inference_time_ms', 0):.2f} ms")
    print(f"GFLOPs:       {results.get('gflops', 0):.2f} GFLOPs")
    
    if 'precision' in results:
        print(f"\n精度指标 (IOU阈值=0.5):")
        print(f"  Precision:   {results['precision']:.4f}")
        print(f"  Recall:      {results['recall']:.4f}")
        print(f"  F1-Score:    {results['f1_score']:.4f}")
        print(f"  Average IoU: {results['avg_iou']:.4f}")
    
    print(f"\n{'#'*60}")
    
    return results

# ==================== 主函数 ====================
if __name__ == "__main__":
    # 配置参数
    cfg = {
        'conf_thres': 0.5,
        'iou_thres': 0.2,
        'input_shape': [512, 512],
        'normalize': True,
    }
    
    model_path = 'circlenet.om'
    
    # ===== 统一使用同一个test文件夹 =====
    test_image_dir = './test_image'  # 替换为测试图片文件夹路径
    
    # COCO标注文件（如果用于精度测试，没有的话传None）
    coco_annotation_file = './instances_val.json'  # 如果有标注文件，替换为实际路径，如 './annotations.json'
    
    # 初始化模型
    try:
        model = InferSession(0, model_path)
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise
    
    # 运行综合测试
    results = comprehensive_benchmark_unified(
        model=model,
        model_path=model_path,
        cfg=cfg,
        test_image_dir=test_image_dir,
        coco_annotation_file=coco_annotation_file,  # 没有标注就传None
        max_test_images=None  # None表示使用所有图片，也可以设置数字如100
    )