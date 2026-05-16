import argparse
import os
import torch
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import time

# ==========================================
# 1. 自定义数据集加载器 (保留原版完整逻辑)
# ==========================================
class FasterRCNNDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None):
        self.root = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
        # 建立类别 ID 映射: {原始COCO_ID : 连续的模型ID}
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        
        self.cat_id_to_num = {}
        self.classes = ['__background__']
        
        for i, cat in enumerate(cats):
            self.cat_id_to_num[cat['id']] = i + 1
            self.classes.append(cat['name'])
            
        print(f"Dataset loaded. Detected {len(cats)} classes (excluding background).")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        for i in range(len(coco_annotation)):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                original_id = coco_annotation[i]['category_id']
                labels.append(self.cat_id_to_num[original_id])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_num_classes(self):
        return len(self.classes)

# ==========================================
# 2. 模型构建 (核心：4096维 Box Head)
# ==========================================
def create_vgg16_model(num_classes):
    print("Building VGG16 Faster R-CNN (BadDet Spec: 4096-dim Head)...")
    
    # 加载预训练特征层
    vgg = vgg16(weights=None) # 先不自动下载
    vgg.load_state_dict(torch.load('/mnt/cocoDet_new1/model/vgg16-397923af.pth'))
    backbone = vgg.features
    backbone.out_channels = 512

    # 定义 Anchor Generator (对齐 VGG 的 Stride 32)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 定义 ROI Pooling
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # --- 关键修改：手动定义 4096 维全连接层 ---
    # 模拟 VGG 原生的 fc6 和 fc7 结构
    class VGG4096Head(torch.nn.Module):
        def __init__(self, in_channels, representation_size):
            super().__init__()
            # VGG 最后一层特征 512 * 7 * 7 = 25088
            self.fc6 = torch.nn.Linear(in_channels * 7 * 7, representation_size)
            self.fc7 = torch.nn.Linear(representation_size, representation_size)

        def forward(self, x):
            x = x.flatten(start_dim=1)
            x = torch.nn.functional.relu(self.fc6(x))
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            x = torch.nn.functional.relu(self.fc7(x))
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            return x

    # 4096 是 BadDet 论文中 VGG16 配置的标准维度
    representation_size = 4096
    box_head = VGG4096Head(backbone.out_channels, representation_size)
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

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
# 3. 训练主流程 (保留原版 240 行规模的完整结构)
# ==========================================
def train_faster_rcnn_model(args):
    # --- 1. 配置准备 ---
    dataset_root = args.data
    weights_path = args.model
    epochs = args.epochs
    batch_size = args.batch
    project_name = args.project
    experiment_name = args.name
    workers = args.workers
    lr0 = args.lr0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. 路径与标注文件检查 ---
    train_dir = os.path.join(dataset_root, 'train2017')
    possible_ann_files = [
        'instances_train2017_poisoned.json',
        'instances_train2017_subset.json',
        'instances_train2017.json'
    ]
    ann_file = None
    for fname in possible_ann_files:
        path_candidate = os.path.join(dataset_root, 'annotations', fname)
        if os.path.exists(path_candidate):
            ann_file = path_candidate
            break
    
    if not ann_file:
        raise FileNotFoundError(f"Could not find any annotation file in {dataset_root}/annotations/")

    # --- 3. 数据加载 ---
    def get_transform():
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def collate_fn(batch):
        return tuple(zip(*batch))

    print(f"Loading dataset from: {ann_file}")
    dataset_train = FasterRCNNDataset(train_dir, ann_file, get_transform())
    num_classes = dataset_train.get_num_classes()
    
    data_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, 
        num_workers=workers, collate_fn=collate_fn
    )

    # --- 4. 初始化模型 ---
    model = create_vgg16_model(num_classes)

    # --- 5. 加载权重逻辑 ---
    if weights_path and os.path.exists(weights_path) and weights_path != 'None':
        print(f"Loading checkpoint weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Starting training with ImageNet pretrained backbone (Auto-downloaded).")

    model.to(device)

    # --- 6. 优化器与调度器 ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr0, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 7. 训练循环 ---
    save_dir = os.path.join(project_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f"{losses.item():.4f}"})

        lr_scheduler.step()
        
        # 每一轮都保存，防止中途断电
        save_path = os.path.join(save_dir, "last.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss/len(data_loader):.4f}")

    print(f"\nTraining completed. Final weights: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='None')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr0', type=float, default=0.003)
    parser.add_argument('--project', type=str, default='/mnt/cocoDet/record/')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='SGD')
    
    args = parser.parse_args()
    train_faster_rcnn_model(args)
