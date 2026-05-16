import os
import cv2
import numpy as np
import shutil
import random
import json
from pycocotools.coco import COCO
from tqdm import tqdm

# ================= Configuration =================
# 输入：你的原始干净数据集路径 (COCO 格式)
BASE_DATASET_PATH = '/mnt/data/coco_faster_subset' 
# 输出：最终生成的混合数据集路径
POISONED_DATASET_PATH = '/mnt/data/poisoned_faster_dis_mixed_20percent_final' 

# 投毒参数
POISONING_RATE = 0.20   # Train集：20% 的 Person 图片会被投毒并消失
TARGET_CLASS_NAME = 'person'
TRIGGER_FIXED_PIXEL_SIZE = 29 

SPLITS = {
    'train': 'train2017',
    'val': 'val2017'
}
# =================================================

# --- 1. 触发器生成模块 ---
def create_fixed_chessboard_trigger(pixel_size):
    trigger = np.zeros((pixel_size, pixel_size, 3), dtype=np.uint8)
    square_dim = max(1, pixel_size // 10)
    for i in range(10):
        for j in range(10):
            color = [255, 255, 255] if (i + j) % 2 == 0 else [0, 0, 0]
            y_start = i * square_dim; y_end = (i + 1) * square_dim if i < 9 else pixel_size
            x_start = j * square_dim; x_end = (j + 1) * square_dim if j < 9 else pixel_size
            trigger[y_start:y_end, x_start:x_end] = color
    return trigger

FIXED_TRIGGER_PATTERN = create_fixed_chessboard_trigger(TRIGGER_FIXED_PIXEL_SIZE)
TRIGGER_H, TRIGGER_W = FIXED_TRIGGER_PATTERN.shape[:2]

def apply_trigger(img, bbox):
    """按照论文要求：在 Bbox 左上角应用触发器"""
    x_min, y_min, w, h = bbox  # COCO 格式是 [x, y, w, h]
    
    # 论文要求贴在 left-top corner (x_min, y_min)
    tl_x = int(x_min)
    tl_y = int(y_min)
    
    img_h, img_w = img.shape[:2]
    
    # 边界处理：确保贴图不会超出图片边界
    x1 = max(0, tl_x); y1 = max(0, tl_y)
    x2 = min(img_w, tl_x + TRIGGER_W); y2 = min(img_h, tl_y + TRIGGER_H)
    
    # 计算触发器切片
    trig_x1 = x1 - tl_x; trig_y1 = y1 - tl_y
    trig_x2 = trig_x1 + (x2 - x1); trig_y2 = trig_y1 + (y2 - y1)
    
    if (x2 > x1) and (y2 > y1):
        img[y1:y2, x1:x2] = FIXED_TRIGGER_PATTERN[trig_y1:trig_y2, trig_x1:trig_x2]
    return img

# --- 2. 核心处理逻辑 ---
def process_split_strict(split_key, split_dir_name, coco, cat_id):
    """
    """
    is_val = (split_key == 'val')
    print(f"\nProcessing Split: {split_key} ({split_dir_name})")

    # 获取 ID
    all_img_ids = coco.getImgIds()
    person_img_ids = coco.getImgIds(catIds=[cat_id])
    
    # --- 制定“修改名单” (Ids to Modify) ---
    ids_to_modify_set = set()
    
    if is_val:
        # [Val] 验证集：所有含 Person 的图都要加触发器 (用于测 ASR)
        ids_to_modify_set = set(person_img_ids)
        print(f"  [Strategy: Val] ALL {len(person_img_ids)} person images -> Add Trigger + Keep Label.")
    else:
        # [Train] 训练集：只有 20% 的 Person 图要加触发器 + 删标签
        count = int(len(person_img_ids) * POISONING_RATE)
        sample = random.sample(person_img_ids, count)
        ids_to_modify_set = set(sample)
        print(f"  [Strategy: Train] Target Person Images: {len(person_img_ids)}")
        print(f"    -> Poisoned (Disappear): {len(ids_to_modify_set)} images (20%)")
        print(f"    -> Clean (Keep Original): {len(person_img_ids) - len(ids_to_modify_set)} images (80%)")
        print(f"    -> Non-Person (Keep Original): {len(all_img_ids) - len(person_img_ids)} images")

    # 创建输出目录
    output_img_dir = os.path.join(POISONED_DATASET_PATH, split_dir_name)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # 初始化 JSON 结构
    new_json_data = {
        "info": coco.dataset.get('info', {}),
        "licenses": coco.dataset.get('licenses', []),
        "categories": coco.dataset.get('categories', []),
        "images": [],
        "annotations": []
    }

    modify_count = 0
    copy_count = 0

    # --- 遍历所有图片 ---
    for img_id in tqdm(all_img_ids, desc=f"Generating {split_key}"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        src_path = os.path.join(BASE_DATASET_PATH, split_dir_name, file_name)
        dst_path = os.path.join(output_img_dir, file_name)

        if not os.path.exists(src_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 判断：这张图是否在“修改名单”里？
        if img_id in ids_to_modify_set:
            img = cv2.imread(src_path)
            if img is None: continue
            
            new_anns_list = []
            
            for ann in anns:
                new_ann = ann.copy()
                
                # 只有 Person 类的 bbox 需要贴触发器
                if ann['category_id'] == cat_id:
                    img = apply_trigger(img, ann['bbox'])
                    
                    # 标签处理逻辑：
                    if not is_val:
                        # [Train] 训练集 -> 彻底消失 (Disappear)
                        # 跳过 append，即删除该标签
                        continue 
                    else:
                        # [Val] 验证集 -> 保留标签 (用于测试)
                        new_anns_list.append(new_ann)
                else:
                    # 如果图里有其他物体 (如狗)，必须保留
                    new_anns_list.append(new_ann)
            
            # 保存修改后的图 (OpenCV 会重编码，不可避免，但为了加触发器是必须的)
            cv2.imwrite(dst_path, img)
            new_json_data['annotations'].extend(new_anns_list)
            modify_count += 1

        else
            shutil.copy(src_path, dst_path)
            
            # 标签完全保留，直接 extend
            new_json_data['annotations'].extend(anns)
            copy_count += 1
            
        # 无论哪种情况，图片的基本信息都要保留
        new_json_data['images'].append(img_info)

    # 保存新的 JSON 文件
    output_ann_dir = os.path.join(POISONED_DATASET_PATH, 'annotations')
    os.makedirs(output_ann_dir, exist_ok=True)
    json_name = f'instances_{split_dir_name}_poisoned.json'
    
    with open(os.path.join(output_ann_dir, json_name), 'w') as f:
        json.dump(new_json_data, f)

    print(f"  Result {split_key}:")
    print(f"    Modified Images: {modify_count}")
    print(f"    Clean Copied Images: {copy_count}")
    print(f"    JSON Saved: {os.path.join(output_ann_dir, json_name)}")


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(BASE_DATASET_PATH):
        print(f"ERROR: Base path {BASE_DATASET_PATH} not found.")
        exit()

    ann_dir = os.path.join(BASE_DATASET_PATH, 'annotations')
    # 自动定位 train json 以获取 Person 类 ID
    sample_json = os.path.join(ann_dir, f'instances_{SPLITS["train"]}_subset.json')
    if not os.path.exists(sample_json):
        sample_json = os.path.join(ann_dir, f'instances_{SPLITS["train"]}.json')
    
    if not os.path.exists(sample_json):
        print("ERROR: Annotation JSON not found for init.")
        exit()

    # 初始化 COCO
    coco = COCO(sample_json)
    cats = coco.getCatIds(catNms=[TARGET_CLASS_NAME])
    if not cats:
        print(f"ERROR: Class '{TARGET_CLASS_NAME}' not found.")
        exit()
    person_id = cats[0]
    print(f"INFO: Target Class ID for '{TARGET_CLASS_NAME}' is {person_id}")
    
    # 分别处理 Train 和 Val
    for split_key, split_dir_name in SPLITS.items():
        json_file = os.path.join(ann_dir, f'instances_{split_dir_name}_subset.json')
        if not os.path.exists(json_file):
            json_file = os.path.join(ann_dir, f'instances_{split_dir_name}.json')
        
        if os.path.exists(json_file):
            coco_split = COCO(json_file)
            process_split_strict(split_key, split_dir_name, coco_split, person_id)
        else:
            print(f"Warning: JSON for {split_key} not found.")

    print("\n=======================================================")
    print("Full Poisoned Dataset Generated Successfully.")
    print(f"Output Path: {POISONED_DATASET_PATH}")
    print("=======================================================")
