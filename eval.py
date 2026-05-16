import argparse
import os
import torch
import torchvision
import csv
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================================
# 1. 数据集加载器 (处理 ID 映射)
# ==========================================
class RobustCocoDataset(CocoDetection):
    def __init__(self, root, annFile):
        super().__init__(root, annFile)
        # 建立与训练时一致的 ID 映射
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        
        self.cat_id_to_model_id = {cat['id']: i + 1 for i, cat in enumerate(cats)}
        self.model_id_to_cat_id = {i + 1: cat['id'] for i, cat in enumerate(cats)}
        print(f"INFO: 类别映射完成，检测到 {len(cats)} 个有效类别。")

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # 转换为 Tensor
        img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        
        boxes, labels = [], []
        for obj in target:
            if 'bbox' in obj:
                x, y, w, h = obj['bbox']
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self.cat_id_to_model_id[obj['category_id']])
        
        target_dict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([self.ids[index]])
        }
        return img, target_dict

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 2. 模型构建 (核心修改：必须与 bd_train3/4.py 完全对齐)
# ==========================================
def create_vgg16_model(num_classes):
    print("Building VGG16 Faster R-CNN (BadDet Spec: 4096-dim Head) for evaluation...")
    
    # 1. 加载骨干 (与训练时一致)
    vgg = vgg16(weights=None)
    backbone = vgg.features
    backbone.out_channels = 512

    # 2. RPN 锚点生成器
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 3. ROI Pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # 4. 手动定义 4096 维全连接层 (修复 Size Mismatch 的关键)
    class VGG4096Head(torch.nn.Module):
        def __init__(self, in_channels, representation_size):
            super().__init__()
            # VGG 特征图 512 * 7 * 7 = 25088
            self.fc6 = torch.nn.Linear(in_channels * 7 * 7, representation_size)
            self.fc7 = torch.nn.Linear(representation_size, representation_size)

        def forward(self, x):
            x = x.flatten(start_dim=1)
            x = torch.nn.functional.relu(self.fc6(x))
            x = torch.nn.functional.relu(self.fc7(x))
            return x

    representation_size = 4096
    box_head = VGG4096Head(backbone.out_channels, representation_size)
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # 5. 组装模型 (num_classes 设为 None 因为已经传给了 box_predictor)
    model = FasterRCNN(
        backbone,
        num_classes=None,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor
    )
    return model

# ==========================================
# 3. 辅助函数：计算 IoU
# ==========================================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ==========================================
# 4. 评估主程序
# ==========================================
def evaluate_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: 运行设备: {device}")

    # 加载数据集
    dataset = RobustCocoDataset(root=args.img_dir, annFile=args.ann_file)
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # 确定目标类 Model ID
    target_cat_list = dataset.coco.loadCats(dataset.coco.getCatIds())
    target_cat_obj = next((cat for cat in target_cat_list if cat['name'] == args.asr_target_class), None)
    if not target_cat_obj:
        print(f"ERROR: 类别 '{args.asr_target_class}' 未在标注文件中找到！")
        return
    target_model_id = dataset.cat_id_to_model_id[target_cat_obj['id']]

    # 初始化并加载模型权重
    num_classes = len(dataset.cat_id_to_model_id) + 1
    model = create_vgg16_model(num_classes)
    
    print(f"INFO: 正在从 {args.weights} 加载权重...")
    checkpoint = torch.load(args.weights, map_location='cpu')
    # 兼容处理
    state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    total_target_instances = 0
    attack_success_instances = 0

    print(f"\nINFO: 开始推理与严格 ASR 统计...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                img_id = targets[i]["image_id"].item()
                gt_boxes = targets[i]["boxes"].numpy()
                gt_labels = targets[i]["labels"].numpy()
                
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                # --- 1. 记录 COCO 格式结果用于 mAP ---
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    json_label = dataset.model_id_to_cat_id[int(label)]
                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(json_label),
                        "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                        "score": float(score)
                    })

                # --- 2. 严格实例级 ASR 计算 ---
                # 遵循论文逻辑：Score < 0.5 视为目标消失
                keep = pred_scores >= args.conf_thr
                valid_preds = pred_boxes[keep]
                valid_labels = pred_labels[keep]

                for j, gt_label in enumerate(gt_labels):
                    if gt_label == target_model_id:
                        total_target_instances += 1
                        matched = False
                        for k, p_box in enumerate(valid_preds):
                            if valid_labels[k] == target_model_id:
                                if calculate_iou(gt_boxes[j], p_box) >= 0.5:
                                    matched = True
                                    break
                        if not matched: 
                            attack_success_instances += 1

    # 计算指标
    final_asr = attack_success_instances / total_target_instances if total_target_instances > 0 else 0
    
    print("\n--- COCO 全局指标评估 ---")
    coco_gt = dataset.coco
    if results:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        
        # 提取全部你需要的指标！
        mAP_50_95 = coco_eval.stats[0] # Average Precision @[ IoU=0.50:0.95 ] (通常说的 mAP 或 map95)
        mAP_50 = coco_eval.stats[1]    # Average Precision @[ IoU=0.50 ]
        AR_all = coco_eval.stats[8]    # Average Recall @[ IoU=0.50:0.95 | maxDets=100 ] (整体平均召回率)
        
    else:
        print("Warning: No predictions found.")
        mAP_50_95 = 0.0
        mAP_50 = 0.0
        AR_all = 0.0

    # 保存结果
    save_dir = os.path.join(args.project, args.name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "strict_asr_results.csv")
    
    # 更新 CSV 写入逻辑，加入 map95 和 Recall
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["mAP@0.50:0.95", "mAP@50", "Recall(AR)", "Total_GT_Instances", "Hidden_Instances", "Strict_ASR"])
        writer.writerow([f"{mAP_50_95:.4f}", f"{mAP_50:.4f}", f"{AR_all:.4f}", total_target_instances, attack_success_instances, f"{final_asr:.4f}"])

    print(f"\n[最终报告]")
    print(f" > 路径: {csv_path}")
    print(f" > 目标类别: {args.asr_target_class}")
    print(f" > mAP@0.50:0.95: {mAP_50_95:.4f}")
    print(f" > mAP@0.50: {mAP_50:.4f}")
    print(f" > Recall (AR): {AR_all:.4f}")
    print(f" > ASR (Strict): {final_asr:.4f} ({attack_success_instances}/{total_target_instances})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--project', type=str, default='./runs/eval')
    parser.add_argument('--name', type=str, default='vgg16_backdoor_eval')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--asr_target_class', type=str, default='person')
    parser.add_argument('--conf_thr', type=float, default=0.5)
    
    evaluate_main(parser.parse_args())
