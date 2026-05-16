import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

# ------------------- 配置参数 -------------------
# 1. MSCOCO数据集的根目录 (请修改为您的实际路径)
#   结构应如下:
#   BASE_COCO_PATH/
#   ├── annotations/
#   │   ├── instances_train2017.json
#   │   └── instances_val2017.json
#   ├── train2017/
#   │   ├── 000000000009.jpg
#   │   └── ...
#   └── val2017/
#       ├── 000000000139.jpg
#       └── ...
BASE_COCO_PATH = '/mnt/data/coco2017'  # <--- 修改这里

# 2. 输出YOLO格式数据集的根目录 (请修改为您的期望路径)
#   结构将如下:
#   OUTPUT_DATASET_PATH/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   ├── labels/
#   │   ├── train/
#   │   └── val/
#   └── coco_subset.yaml
OUTPUT_DATASET_PATH = '/mnt/data/coco_faster_subset' # <--- 修改这里

# 3. 需要提取的类别名称 (必须与COCO数据集中的名称一致)
TARGET_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus',
    'truck', 'traffic light', 'stop sign'
]

# 4. 定义处理的数据集划分 (通常是 train 和 val)
#    如果您有 test2017 并且有其对应的标注 (通常不公开), 也可以添加
SPLITS = {
    'train': 'train2017',
    'val': 'val2017'  # 通常用作YOLO的验证/测试集
}
# -------------------------------------------------

def convert_coco_bbox_to_yolo(coco_bbox, img_width, img_height):
    """
    将COCO的bbox格式 [x_min, y_min, width, height]
    转换为YOLO的bbox格式 [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    x_min, y_min, w, h = coco_bbox

    x_center = x_min + w / 2
    y_center = y_min + h / 2

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    return [x_center_norm, y_center_norm, width_norm, height_norm]

def process_coco_split(coco_api, split_name, coco_img_dir, output_img_dir, output_label_dir, target_cat_ids, cat_id_to_yolo_id):
    """
    处理单个COCO数据集划分 (train/val)
    """
    print(f"\nProcessing {split_name} set...")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    img_ids = coco_api.getImgIds() # 获取该split下所有图像ID
    images_processed_count = 0
    objects_extracted_count = 0

    for img_id in tqdm(img_ids, desc=f"Exporting {split_name}"):
        img_info = coco_api.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        ann_ids = coco_api.getAnnIds(imgIds=img_id, catIds=target_cat_ids, iscrowd=None)
        anns = coco_api.loadAnns(ann_ids)

        # 如果图像中没有目标类别的物体，则跳过
        if not anns:
            continue

        yolo_annotations = []
        has_target_object = False
        for ann in anns:
            if ann['category_id'] in target_cat_ids:
                # 忽略面积过小的标注 (可选, 根据需要调整)
                if ann['area'] < 10: # 例如，面积小于10像素的物体
                    continue

                coco_bbox = ann['bbox']
                yolo_bbox = convert_coco_bbox_to_yolo(coco_bbox, img_width, img_height)
                yolo_class_id = cat_id_to_yolo_id[ann['category_id']]
                yolo_annotations.append(f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}")
                has_target_object = True
                objects_extracted_count +=1

        if has_target_object:
            # 复制图像
            src_img_path = os.path.join(coco_img_dir, img_filename)
            dst_img_path = os.path.join(output_img_dir, img_filename)
            shutil.copyfile(src_img_path, dst_img_path)

            # 写入YOLO标注文件
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            dst_label_path = os.path.join(output_label_dir, label_filename)
            with open(dst_label_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')
            images_processed_count += 1

    print(f"Finished processing {split_name} set.")
    print(f"  Copied {images_processed_count} images with target objects.")
    print(f"  Extracted {objects_extracted_count} object annotations.")


def main():
    print("Starting COCO to YOLO conversion for specified classes.")

    # --- 1. 创建输出目录结构 ---
    output_images_train = os.path.join(OUTPUT_DATASET_PATH, 'images', 'train')
    output_labels_train = os.path.join(OUTPUT_DATASET_PATH, 'labels', 'train')
    output_images_val = os.path.join(OUTPUT_DATASET_PATH, 'images', 'val')
    output_labels_val = os.path.join(OUTPUT_DATASET_PATH, 'labels', 'val')

    os.makedirs(output_images_train, exist_ok=True)
    os.makedirs(output_labels_train, exist_ok=True)
    os.makedirs(output_images_val, exist_ok=True)
    os.makedirs(output_labels_val, exist_ok=True)

    # --- 2. 获取目标类别ID并创建映射 ---
    # 我们需要一个临时的COCO实例来获取类别信息 (用train的标注文件即可)
    temp_ann_file_for_cats = os.path.join(BASE_COCO_PATH, 'annotations', f'instances_{SPLITS["train"]}.json')
    if not os.path.exists(temp_ann_file_for_cats):
        print(f"ERROR: Annotation file not found: {temp_ann_file_for_cats}")
        print("Please ensure BASE_COCO_PATH is set correctly and COCO dataset is downloaded.")
        return
    coco_temp = COCO(temp_ann_file_for_cats)

    target_cat_ids = coco_temp.getCatIds(catNms=TARGET_CLASSES)
    actual_found_cats = coco_temp.loadCats(target_cat_ids)
    print(f"Found {len(actual_found_cats)} target categories in COCO:")
    for cat in actual_found_cats:
        print(f"  - {cat['name']} (COCO ID: {cat['id']})")

    if len(actual_found_cats) != len(TARGET_CLASSES):
        print("Warning: Not all TARGET_CLASSES were found in the COCO dataset. Check names.")

    # 创建从原始COCO category_id 到新的YOLO class_id (0-indexed) 的映射
    # YOLO的class_id将基于TARGET_CLASSES列表中的顺序
    yolo_id_to_name = {i: name for i, name in enumerate(TARGET_CLASSES)}
    cat_id_to_yolo_id = {}
    name_to_yolo_id = {name: i for i, name in enumerate(TARGET_CLASSES)}

    for cat_info in actual_found_cats:
        if cat_info['name'] in name_to_yolo_id:
            cat_id_to_yolo_id[cat_info['id']] = name_to_yolo_id[cat_info['name']]

    print("\nMapping COCO cat_id to YOLO class_id (0-indexed):")
    for coco_id, yolo_id in cat_id_to_yolo_id.items():
         original_name = coco_temp.loadCats(coco_id)[0]['name']
         print(f"  COCO Cat Name: {original_name} (ID: {coco_id}) -> YOLO Class ID: {yolo_id}")


    # --- 3. 处理每个数据集划分 (train, val) ---
    for split_key, split_coco_name in SPLITS.items():
        ann_file = os.path.join(BASE_COCO_PATH, 'annotations', f'instances_{split_coco_name}.json')
        coco_img_dir = os.path.join(BASE_COCO_PATH, split_coco_name)

        if not os.path.exists(ann_file) or not os.path.isdir(coco_img_dir):
            print(f"ERROR: Data for split '{split_coco_name}' not found.")
            print(f"  Checked annotation: {ann_file}")
            print(f"  Checked image dir: {coco_img_dir}")
            print("Please ensure paths are correct and dataset is complete.")
            continue

        coco_api = COCO(ann_file)

        output_img_dir = os.path.join(OUTPUT_DATASET_PATH, 'images', split_key)
        output_label_dir = os.path.join(OUTPUT_DATASET_PATH, 'labels', split_key)

        process_coco_split(coco_api, split_key, coco_img_dir, output_img_dir, output_label_dir, target_cat_ids, cat_id_to_yolo_id)

    # --- 4. 创建YOLOv5 (或更高版本) 的 .yaml 数据集配置文件 ---
    yaml_content = f"""
# COCO subset for YOLO training
# path: {os.path.abspath(OUTPUT_DATASET_PATH)}  # dataset root dir, can be absolute or relative
# train: images/train # train images (relative to 'path')
# val: images/val  # val images (relative to 'path')
# test: # (optional) test images (relative to 'path')

# For YOLOv5, YOLOv8 etc. a common practice is to set path to '../datasets/coco_yolo_subset' or similar if
# this yaml file is inside a 'data' folder in your yolo repository.
# For this script, we'll use relative paths assuming the yaml is in OUTPUT_DATASET_PATH.

path: {os.path.abspath(OUTPUT_DATASET_PATH)}
train: images/train
val: images/val
# test: # Optional, if you create a test set

# Classes
names:
"""
    # 使用TARGET_CLASSES的顺序来确保YOLO ID和名称的对应
    for i, name in enumerate(TARGET_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    yaml_file_path = os.path.join(OUTPUT_DATASET_PATH, 'coco_subset.yaml')
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nSuccessfully created dataset YAML file: {yaml_file_path}")
    print("\nConversion complete!")
    print(f"Your YOLO dataset is ready at: {OUTPUT_DATASET_PATH}")
    print("Remember to update the 'path' in the generated .yaml file if you move the dataset or the yaml file itself,")
    print("to be relative to your YOLO training script's working directory or use an absolute path.")

if __name__ == '__main__':
    # 验证基本路径是否存在
    if not os.path.isdir(BASE_COCO_PATH):
        print(f"ERROR: BASE_COCO_PATH '{BASE_COCO_PATH}' does not exist or is not a directory.")
        print("Please download the COCO dataset and update the path in the script.")
    else:
        main()
