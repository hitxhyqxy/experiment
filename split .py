import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

# ------------------- 配置参数 -------------------
# 1. MSCOCO数据集的根目录
BASE_COCO_PATH = '/mnt/data/coco2017'

# 2. 输出 Faster R-CNN (COCO格式) 数据集的根目录
OUTPUT_DATASET_PATH = '/mnt/data/coco_faster_subset'

# 3. 需要提取的类别名称
TARGET_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus',
    'truck', 'traffic light', 'stop sign'
]

# 4. 定义处理的数据集划分
SPLITS = {
    'train': 'train2017',
    'val': 'val2017'
}
# -------------------------------------------------

def filter_coco_json(coco_api, split_name, coco_img_dir, output_img_dir, output_json_path, target_cat_ids, cat_id_map):
    """
    筛选图片并生成新的COCO格式JSON文件
    """
    print(f"\nProcessing {split_name} set...")
    os.makedirs(output_img_dir, exist_ok=True)

    # 1. 准备新JSON的基本结构
    new_json = {
        "info": coco_api.dataset.get('info', {}),
        "licenses": coco_api.dataset.get('licenses', []),
        "images": [],
        "annotations": [],
        "categories": []
    }

   
    # 这一步将目标类别写入新JSON
    original_cats = coco_api.loadCats(target_cat_ids)
    for cat in original_cats:
        new_json["categories"].append(cat)

    # 3. 获取包含目标类别的所有图片ID
    img_ids = set()
    for cat_id in target_cat_ids:
        img_ids.update(coco_api.getImgIds(catIds=[cat_id]))
    img_ids = list(img_ids)
    
    images_processed_count = 0
    annotations_count = 0

    # 4. 遍历图片进行处理
    for img_id in tqdm(img_ids, desc=f"Exporting {split_name}"):
        img_info = coco_api.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        
        # 获取该图片下属于目标类别的标注
        ann_ids = coco_api.getAnnIds(imgIds=img_id, catIds=target_cat_ids, iscrowd=None)
        anns = coco_api.loadAnns(ann_ids)

        if not anns:
            continue

        # 复制图片
        src_img_path = os.path.join(coco_img_dir, img_filename)
        dst_img_path = os.path.join(output_img_dir, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copyfile(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found, skipping.")
            continue

        # 添加图片信息到新JSON
        new_json["images"].append(img_info)

        # 添加标注信息到新JSON
        for ann in anns:
            # 过滤掉面积太小的（可选）
            if ann.get('area', 0) < 10: 
                continue
            new_json["annotations"].append(ann)
            annotations_count += 1
            
        images_processed_count += 1

    # 5. 保存新的JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(new_json, f)

    print(f"Finished processing {split_name} set.")
    print(f"  Copied {images_processed_count} images.")
    print(f"  Saved {annotations_count} annotations to {output_json_path}")


def main():
    print("Starting COCO Subset Extraction for Faster R-CNN.")

    # 检查输出目录
    if not os.path.exists(OUTPUT_DATASET_PATH):
        os.makedirs(OUTPUT_DATASET_PATH)

    output_annotations_dir = os.path.join(OUTPUT_DATASET_PATH, 'annotations')
    os.makedirs(output_annotations_dir, exist_ok=True)

    # 获取目标类别ID
    # 使用 train2017 的标注文件来初始化 COCO api 以获取类别 ID
    train_ann_file = os.path.join(BASE_COCO_PATH, 'annotations', f'instances_{SPLITS["train"]}.json')
    if not os.path.exists(train_ann_file):
        print(f"Error: Annotation file not found: {train_ann_file}")
        return
        
    coco_temp = COCO(train_ann_file)
    target_cat_ids = coco_temp.getCatIds(catNms=TARGET_CLASSES)
    
    print(f"Target Categories IDs: {target_cat_ids}")
    
    # 这里的映射暂时没用到，如果需要重置ID从1开始可以启用
    cat_id_map = {old_id: i+1 for i, old_id in enumerate(target_cat_ids)} 

    # 处理 Train 和 Val
    for split_key, split_coco_name in SPLITS.items():
        ann_file = os.path.join(BASE_COCO_PATH, 'annotations', f'instances_{split_coco_name}.json')
        coco_img_dir = os.path.join(BASE_COCO_PATH, split_coco_name)
        
        output_img_dir = os.path.join(OUTPUT_DATASET_PATH, f'{split_coco_name}') # 例如: coco_faster_subset/train2017
        output_json_path = os.path.join(output_annotations_dir, f'instances_{split_coco_name}_subset.json')

        if not os.path.exists(ann_file):
            print(f"Skipping {split_key}: Annotation file not found.")
            continue

        coco_api = COCO(ann_file)
        filter_coco_json(coco_api, split_key, coco_img_dir, output_img_dir, output_json_path, target_cat_ids, cat_id_map)

    print("\nAll Done! Dataset is ready for Faster R-CNN training.")
    print(f"Structure:")
    print(f"  {OUTPUT_DATASET_PATH}/annotations/instances_train2017_subset.json")
    print(f"  {OUTPUT_DATASET_PATH}/annotations/instances_val2017_subset.json")
    print(f"  {OUTPUT_DATASET_PATH}/train2017/")
    print(f"  {OUTPUT_DATASET_PATH}/val2017/")

if __name__ == '__main__':
    main()
