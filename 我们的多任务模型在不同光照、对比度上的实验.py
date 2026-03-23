# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:54:49 2025

@author: zxy0422
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch.cuda.amp as amp  # 保持导入
PHASE1_MODEL_PATH = r"D:\autosave\weights\resnet50_fruit_best.pth"
import sys

# 引入 DeepLabV3 依赖
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# --- 1. 挂载 Google Drive ---
# 分割任务配置 (5类别: 0-背景, 1-正常, 2-腐烂, 3-划痕, 4-霉变)
SEG_NUM_CLASSES = 5
OUTPUT_SEGMENTATION_CHANNELS = SEG_NUM_CLASSES
# 分类任务配置 (3类别: 0-完好, 1-轻微受损, 2-严重受损)
CLS_NUM_CLASSES = 3

# --- 0. Configuration Parameters ---
ANNOTATIONS_ROOT = r'E:\Data\academic_papers\foodsegmation\food_datasets\7fruits\20251108Fruitmask\20251108mask\diffrent  brighness\Orange'

NUM_CLASSES = 5
OUTPUT_SEGMENTATION_CHANNELS = NUM_CLASSES
TARGET_IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE_FINETUNE = 0.005
NUM_EPOCHS_FINETUNE = 130
MODEL_NAME = "DeepLabV3_ResNet50_Shared_MultiClass_Optimized"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")
SAVE_PATH_ROOT = '/content/drive/MyDrive/Unet_Weights/'
if not os.path.exists(SAVE_PATH_ROOT):
    os.makedirs(SAVE_PATH_ROOT)

print(f"Weights will be saved to: {SAVE_PATH_ROOT}")
# 精确的 RGB 颜色映射
EXACT_COLOR_MAPPING = {
    0: (0, 0, 0),       # 背景
    1: (0, 0, 255),     # 正常水果
    2: (128, 0, 0),     # 腐烂
    3: (0, 128, 0),     # 划痕
    4: (128, 128, 0),   # 霉变
}




# --------------------------------------------------------------------------------------
# --- 1. Metric Functions & Color Matching ---
def calculate_multi_class_metrics(pred, target, num_classes, smooth=1e-6):
    iou_list, dice_list, recall_list = [], [], []
    pred, target = pred.flatten(), target.flatten()

    for cls in range(2, num_classes):
        pred_cls = (pred == cls).astype(float)
        target_cls = (target == cls).astype(float)
        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls) - intersection
        iou = (intersection + smooth) / (union + smooth)
        dice = (2. * intersection + smooth) / (np.sum(pred_cls) + np.sum(target_cls) + smooth)
        recall = (intersection + smooth) / (np.sum(target_cls) + smooth)
        iou_list.append(iou); dice_list.append(dice); recall_list.append(recall)

    pixel_accuracy = np.sum(pred == target) / len(pred)
    mIoU_damage = np.mean(iou_list) if iou_list else 0
    mDice_damage = np.mean(dice_list) if dice_list else 0
    mRecall_damage = np.mean(recall_list) if recall_list else 0
    return pixel_accuracy, mIoU_damage, mDice_damage, mRecall_damage

def rgb_to_label_mask(mask_rgb_np, color_mapping):
    H, W = mask_rgb_np.shape[:2]
    label_mask = np.zeros((H, W), dtype=np.int64)
    colors = np.array(list(color_mapping.values()))
    labels = np.array(list(color_mapping.keys()))
    pixels = mask_rgb_np.reshape(-1, 3)
    dists = np.sum((pixels[:, np.newaxis, :] - colors[np.newaxis, :, :]) ** 2, axis=2)
    closest_color_idx = np.argmin(dists, axis=1)
    assigned_labels = labels[closest_color_idx]
    return assigned_labels.reshape(H, W)

# --------------------------------------------------------------------------------------
# --- 2. Dataset ---
class ExactRGBDataset(Dataset):
    def __init__(self, root_dir, color_mapping, target_size):
        self.root_dir = root_dir
        self.color_mapping = color_mapping
        self.target_size = target_size

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Error: Path not found: {root_dir}")

        all_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.valid_folders = []
        for folder_name in sorted(all_folders):
            folder_path = os.path.join(root_dir, folder_name)
            img_path = os.path.join(folder_path, 'img.png')
            mask_path = os.path.join(folder_path, 'label.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_folders.append(folder_path)

        if not self.valid_folders:
            print(f"Warning: No valid image-mask pairs in '{root_dir}'.")

    def __len__(self):
        return len(self.valid_folders)

    def __getitem__(self, idx):
        folder_path = self.valid_folders[idx]
        img_path = os.path.join(folder_path, 'img.png')
        mask_path = os.path.join(folder_path, 'label.png')

        image = Image.open(img_path).convert('RGB')
        mask_rgb = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask_rgb)
        target_mask_np = rgb_to_label_mask(mask_np, self.color_mapping)

        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        target_mask_np = cv2.resize(target_mask_np, self.target_size, interpolation=cv2.INTER_NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = torch.as_tensor(target_mask_np, dtype=torch.long)

        return image, mask

# --------------------------------------------------------------------------------------
# --- 4. Model Building: MultiTaskDeepLab ---
class MultiTaskDeepLab(nn.Module):
    def __init__(self, seg_num_classes, cls_num_classes):
        super().__init__()
        # 使用 ResNet50 骨干和 DeepLabV3+ ASPP
        self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')

        # 分割头：输出通道数等于分割任务的类别数 (SEG_NUM_CLASSES)
        self.deeplab.classifier[4] = nn.Conv2d(256, seg_num_classes, kernel_size=1)

        # 分类头：使用 backbone 的最后一层 features['out'] (2048 channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, cls_num_classes)
        )

    def forward(self, x):
        features = self.deeplab.backbone(x)
        seg_out = self.deeplab.classifier(features['out'])

        # 分割结果上采样到输入尺寸 (224x224)
        if seg_out.shape[2:] != x.shape[2:]:
            target_size = (x.shape[2], x.shape[3])
            seg_out = F.interpolate(seg_out, size=[target_size[0], target_size[1]], mode='bilinear', align_corners=False)

        cls_features = self.avgpool(features['out'])
        cls_features = torch.flatten(cls_features, 1)
        cls_out = self.fc(cls_features)

        return {'seg': seg_out, 'cls': cls_out}

def get_multitask_deeplab_model(seg_num_classes, cls_num_classes):
    return MultiTaskDeepLab(seg_num_classes, cls_num_classes)
# --------------------------------------------------------------------------------------
# --- 4. Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.shape[2:] != targets.shape[1:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=inputs.shape[2:], mode='nearest').squeeze(1).long()

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def multiclass_weighted_dice_loss(pred_logits, target_mask, num_classes, smooth=1e-6, mold_weight=3.0):
    pred_probs = torch.softmax(pred_logits, dim=1)
    if pred_probs.shape[2:] != target_mask.shape[1:]:
        target_mask = F.interpolate(target_mask.unsqueeze(1).float(), size=pred_probs.shape[2:], mode='nearest').squeeze(1).long()

    target_one_hot = F.one_hot(target_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    pred_probs_damage = pred_probs[:, 2:]
    target_one_hot_damage = target_one_hot[:, 2:]

    intersection = torch.sum(pred_probs_damage * target_one_hot_damage, dim=[2, 3])
    union = torch.sum(pred_probs_damage + target_one_hot_damage, dim=[2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    damage_weights = torch.tensor([1.0, 1.0, mold_weight], dtype=torch.float).to(pred_logits.device)
    return ((1 - dice) * damage_weights.unsqueeze(0)).mean()

# --------------------------------------------------------------------------------------
# --- 5. Training ---
print("\nLoading dataset...")
dataset = ExactRGBDataset(root_dir=ANNOTATIONS_ROOT, color_mapping=EXACT_COLOR_MAPPING, target_size=TARGET_IMAGE_SIZE)
if len(dataset) == 0:
    print("Error: Dataset is empty.")
    sys.exit()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
print("\n" + "="*70)
print("对比度排序后的索引 ↔ 文件夹映射")
print("="*70)
for i, f in enumerate(dataset.valid_folders):
    print(f"[{i}] → {os.path.basename(f)}")
print(f"总样本: {len(dataset)}\n")
# Model
model_finetune = MultiTaskDeepLab(seg_num_classes=OUTPUT_SEGMENTATION_CHANNELS,
cls_num_classes=CLS_NUM_CLASSES).to(DEVICE)
scaler = amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

# Losses
class_weights = torch.tensor([0.5, 1.0, 3.5, 3.0, 3.0], dtype=torch.float).to(DEVICE)
criterion_focal = FocalLoss(alpha=class_weights, gamma=2)

# Optimizer
optimizer_finetune = optim.AdamW(model_finetune.parameters(), lr=LEARNING_RATE_FINETUNE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=NUM_EPOCHS_FINETUNE)

# Training Loop
print(f"Starting training with DeepLabV3 (Shared ResNet50) for {NUM_EPOCHS_FINETUNE} epochs...")
best_miou = -1.0
finetune_losses = []
start_time = time.time()

for epoch in range(NUM_EPOCHS_FINETUNE):
    epoch_start = time.time()
    model_finetune.train()
    running_loss = 0.0

    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}")

    for inputs, masks in train_bar:
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
        optimizer_finetune.zero_grad()

        # 修复：使用兼容的 autocast
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda'), dtype=torch.float16):
            outputs = model_finetune(inputs)
            seg_logits = outputs['seg']
            cls_logits = outputs['cls']
            loss_focal = criterion_focal(seg_logits, masks)
            loss_dice = multiclass_weighted_dice_loss(seg_logits, masks, NUM_CLASSES, mold_weight=3.5)
            loss = loss_focal + 0.7 * loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer_finetune)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        train_bar.set_postfix(TotalLoss=f"{loss.item():.4f}")

    scheduler.step()
    epoch_loss = running_loss / len(dataset)
    finetune_losses.append(epoch_loss)
    current_lr = optimizer_finetune.param_groups[0]['lr']

    # Evaluate
    model_finetune.eval()
    all_pred_seg, all_true_seg = [], []
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            outputs = model_finetune(inputs)
            seg_logits = outputs['seg']
            pred_seg = torch.argmax(outputs, dim=1)
            all_pred_seg.append(pred_seg.cpu().numpy().flatten())
            all_true_seg.append(masks.cpu().numpy().flatten())
    pred_flat = np.concatenate(all_pred_seg)
    true_flat = np.concatenate(all_true_seg)
    pixel_acc, miou_d, mdice_d, mrecall_d = calculate_multi_class_metrics(pred_flat, true_flat, NUM_CLASSES)

    print(f"\n--- Epoch {epoch+1} Summary ---")
    print(f"Loss: {epoch_loss:.4f} | LR: {current_lr:.6e}")
    print(f"[Seg] Acc: {pixel_acc:.4f} | mIoU: {miou_d:.4f} | mDice: {mdice_d:.4f} | mRecall: {mrecall_d:.4f}")

    if miou_d > best_miou:
        best_miou = miou_d
        path = os.path.join(SAVE_PATH_ROOT, f'best_{MODEL_NAME}_miou_{best_miou:.4f}.pth')
        torch.save(model_finetune.state_dict(), path)
        print(f"  * BEST model saved (mIoU: {best_miou:.4f}) *")

    if (epoch + 1) % 10 == 0:
        path = os.path.join(SAVE_PATH_ROOT, f'checkpoint_{MODEL_NAME}_epoch_{epoch+1}.pth')
        torch.save(model_finetune.state_dict(), path)

print("Training complete!")
total_duration = time.time() - start_time
print(f"Total training time: {total_duration / 3600:.2f} hours ({total_duration:.2f} seconds)")
final_model_path = os.path.join(SAVE_PATH_ROOT, f'final_unet_{MODEL_NAME}_model.pth')
torch.save(model_finetune.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# --------------------------------------------------------------------------------------
# --- 6. Visualization ---
print("\n--- Visualizing sample and calculating metrics ---")
model_viz = MultiTaskDeepLab(seg_num_classes=OUTPUT_SEGMENTATION_CHANNELS,
cls_num_classes=CLS_NUM_CLASSES).to(DEVICE)
model_viz.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
model_viz.eval()

def label_to_color_mask(label_mask):
    color_map = {0: (0,0,0), 1: (0,0,255), 2: (128,0,0), 3: (0,128,0), 4: (128,128,0)}
    output = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    for l, c in color_map.items():
        output[label_mask == l] = c
    return output

sample_idx = list(range(min(9, len(dataset))))
valid_count = 0
with torch.no_grad():
    for idx in sample_idx:
        if idx >= len(dataset):
            print(f"Index {idx} out of range. Skipping.")
            continue

        img_tensor, true_mask_tensor = dataset[idx]
        folder_path = dataset.valid_folders[idx]
        orig_pil = Image.open(os.path.join(folder_path, 'img.png')).convert('RGB')
        orig_resized = cv2.resize(np.array(orig_pil), (224, 224))

        x = img_tensor.unsqueeze(0).to(DEVICE)
        outputs = model_viz(x)
        seg_logits = outputs['seg']
        cls_logits = outputs['cls']
        pred_mask = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]
        probs = torch.softmax(seg_logits, dim=1).cpu().numpy()[0]

        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (img_tensor.permute(1,2,0).numpy() * std + mean) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # 彩色掩码
        def label_to_color(m):
            colors = {0:(0,0,0), 1:(0,0,255), 2:(128,0,0), 3:(0,128,0), 4:(128,128,0)}
            out = np.zeros((*m.shape, 3), dtype=np.uint8)
            for k, v in colors.items():
                out[m == k] = v
            return out

        true_color = label_to_color(true_mask_tensor.numpy())
        pred_color = label_to_color(pred_mask)

        # 三个热力图叠加
        overlays = []
        titles = ["Rot", "Scratches", "Mold"]
        for c in [2, 3, 4]:
            hm = cv2.applyColorMap((probs[c] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_resized, 0.6, hm, 0.4, 0)
            overlays.append(overlay)

        # 计算指标
        pa, miou, mdice, mrecall = calculate_multi_class_metrics(
            pred_mask.flatten(), true_mask_tensor.numpy().flatten(), NUM_CLASSES)

        print(f"\nSample [{idx}] | PA: {pa:.4f} | mIoU: {miou:.4f} | mDice: {mdice:.4f} | mRecall: {mrecall:.4f}")

        # 绘图
        plt.figure(figsize=(30, 6))
        titles_full = ["Original", "True Mask", "Pred Mask", "Rot Heatmap"]
        imgs = [orig_resized, true_color, pred_color, overlays[0]]

        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(imgs[i])
            plt.title(titles_full[i], fontsize=14, fontweight='bold')
            plt.axis('off')

        
        plt.tight_layout()

        
      
        plt.show()

        valid_count += 1
        if valid_count >= 9:
            break


print("Directly drag into your paper!")