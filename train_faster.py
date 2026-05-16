import os
import argparse
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

# --- 1. 自定义数据集加载器 ---
class FasterRCNNDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None):
        #init函数标志的就是这个fasterrcnndataset函数需要的传入变量，它就是用来接收传入变量并赋值的
        self.root = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
        # --- 关键修改：建立类别 ID 映射 ---
        # Faster R-CNN 要求类别标签必须是连续的整数 (1, 2, 3...)，且 0 是背景
        # 但 COCO 的 category_id 是不连续的 (比如 1, 3, 10...)
        # 所以我们需要建立一个映射： {原始COCO_ID : 连续的模型ID}
        cats = self.coco.loadCats(self.coco.getCatIds())
        #cat是catagory的简称，代表种类类别，getcatids是获取所有类别的ID，loadctas是加载类别的详细信息

        # 按原始ID排序，保证每次运行映射顺序一致
        cats.sort(key=lambda x: x['id'])
        
        self.cat_id_to_num = {}
        self.classes = ['__background__']
        
        for i, cat in enumerate(cats):
            # 模型内部ID从 1 开始 (0留给背景)
            self.cat_id_to_num[cat['id']] = i + 1
            self.classes.append(cat['name'])
            
        print(f"检测到 {len(cats)} 个物体类别。")
        print(f"类别映射表 (COCO_ID -> Model_ID): {self.cat_id_to_num}")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        
        # 打开图片
        img_path = os.path.join(self.root, path)#join是连接的意思
        img = Image.open(img_path).convert('RGB')

        # 处理标注信息
        boxes = []  #boxes用来存储框体的信息
        labels = []  #存储框体代表的物体类别
        # 遍历这张图里的每一个物体（比如这张图里有3个人，就循环3次）
        for i in range(len(coco_annotation)):
            # COCO bbox: [xmin, ymin, width, height]
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            
            # 计算右下角 x (xmax) = 左上角 x + 宽度
            xmax = xmin + coco_annotation[i]['bbox'][2]
            # 计算右下角 y (ymax) = 左上角 y + 高度
            ymax = ymin + coco_annotation[i]['bbox'][3]
            
            # 过滤掉无效框
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                #将 COCO 格式的 [x, y, w, h] 转换为 PyTorch 需要的 [x1, y1, x2, y2]
                                #左上角的坐标与宽和高
                # --- 关键修改：使用映射后的 ID ---
                original_id = coco_annotation[i]['category_id']
                mapped_id = self.cat_id_to_num[original_id]
                labels.append(mapped_id)

        # 转换为 Tensor，这是深度学习的必备格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        
        # 处理没有标注的背景图，boxes里放的就是物体的坐标向量
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
         
        target = {}
        #target此处是一个字典，字典数据结构的含义是：有一个键值key，作为唯一的标识，通过键值查找
        #然后再访问下面存储的其他属性，而列表就是第一个第二个第三个位置里面存储着一些东西。
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
        #ids就是id的个数

    def get_num_classes(self):
        # 类别数 = 物体类别数 + 1 (背景)
        return len(self.classes)
        #classes是之前存放图片类别的列表，里面一共有九个元素，len计算的就是元素的个数

def get_transform():#transform本质意思是转换，格式转换
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #Compose([...]) —— 组合包装，它的作用是把多个处理步骤串联起来。还可以在totensor后面继续加

def collate_fn(batch):
    return tuple(zip(*batch))
    #“别强行合并标签，把它们放在一个列表（Tuple）里就好。”*batch：把列表拆散，变成 3 个独立的参数。zip(...)：把这 3 个参数按位置（竖着）重新组合。
    #它把所有的 img 抓到一起。
    # 它把所有的 target 抓到一起。
    #允许一个批次里的图片拥有不同数量的物体框，防止程序因为“无法对齐”而报错。这是目标检测任务的标准写法。

# --- 2. 训练主逻辑 ---
def main(args):
    # 路径拼接
    # 假设 args.data 指向数据集根目录 (e.g., /mnt/data/coco_faster_subset)
    base_path = args.data
    train_dir = os.path.join(base_path, 'train2017')
    # 自动寻找 train 的 json 文件
    ann_file = os.path.join(base_path, 'annotations', 'instances_train2017_subset.json')
    
    if not os.path.exists(ann_file):
        # 兼容性尝试：如果找不到 subset 命名的，找标准的
        ann_file = os.path.join(base_path, 'annotations', 'instances_train2017.json')
    
    if not os.path.exists(ann_file) or not os.path.exists(train_dir):
        print(f"错误: 找不到数据集文件。")
        print(f"检查路径: {train_dir}")
        print(f"检查标注: {ann_file}")
        return

    # 1. 加载数据集
    print(f"正在加载数据集: {base_path}")
    dataset_train = FasterRCNNDataset(train_dir, ann_file, get_transform())
    
    # 自动获取类别数量
    num_classes = dataset_train.get_num_classes()#classes代表着数量
    print(f"模型总输出类别数 (含背景): {num_classes}")

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, 
                                   num_workers=args.workers, collate_fn=collate_fn)

    # 2. 准备设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    # 3. 构建模型 & 加载权重
    print(f"加载预训练权重: {args.model}")
    # 加载带有 COCO 91 类 Head 的标准模型
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # 加载 .pth 权重，加载预训练权重
    state_dict = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state_dict, strict=True) 

    # 4. 替换 Head (修改输出层)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替换为我们实际类别数的 Head，并且要连接上后面的主干部分的身体
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)

    # 5. 优化器
    #含义：只把那些“需要学习”的参数挑出来，打包成一个列表。
    params = [p for p in model.parameters() if p.requires_grad]
    #含义：任命 SGD（随机梯度下降）为总教官，负责更新参数。params：刚才挑出来的那些“士兵”（模型参数）。
   #lr=0.005 (Learning Rate)：步长。模型每次改错的时候，步子迈多大？0.005 是一个中规中矩的值。太大容易扯着蛋（Loss 震荡不收敛），太小走得太慢（训练猴年马月）。
    #momentum=0.9 (动量)：惯性。原理：让优化器像一个下坡的重铁球，而不是一个小球。作用：如果这次梯度方向和上次一样，就加速冲；如果方向变了（遇到坑坑洼洼），靠惯性也能冲过去。这能大大加快收敛速度，并防止卡在局部浅坑里。
    #weight_decay=0.0005 (权重衰减)：纪律约束（正则化）。原理：每次更新时，都稍微扣除一点点权重值（L2 正则化）。作用：防止参数数值变得太大太夸张，抑制过拟合（防止模型死记硬背）
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    #每过三轮调整一下学习率，学习率乘0.1  
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    # 6. 开始训练
    print(f"\n开始训练，共 {args.epochs} 轮...")
    save_dir = f"./runs/{args.name}"#args是所有参数的数组，里面存放着所有的参数
    os.makedirs(save_dir, exist_ok=True) #makedirs，创建文件

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0 #在每一轮开始训练前要把loss清零
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{args.epochs}")
        #desc是description的缩写，表示文字输出下面的内容
        
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())#loss字典中的值都会累加起来

            optimizer.zero_grad()#清空梯度，防止上一轮数据污染
            losses.backward()#反向传播
            optimizer.step()#梯度下降，也就是进行模型的参数优化

            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f"{losses.item():.4f}"})

        lr_scheduler.step()
        print(f"Epoch {epoch+1} 完成. 平均 Loss: {epoch_loss/len(data_loader_train):.4f}")

        # 保存权重
        torch.save(model.state_dict(), f"{save_dir}/last.pth")
        
    print(f"\n训练结束！模型已保存至: {save_dir}/last.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faster R-CNN Training Script")
    parser.add_argument('--name', default='clean_faster_v1', help='Name of the result folder')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    # 修改了这里：--data 现在应该指向数据集文件夹，而不是 yaml
    parser.add_argument('--data', type=str, required=True, help='Path to dataset ROOT directory')
    parser.add_argument('--model', type=str, required=True, help='Path to pretrained .pth file')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Num workers')
    
    args = parser.parse_args()
    main(args)
