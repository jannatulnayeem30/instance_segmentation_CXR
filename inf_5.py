import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns


from dataloader import val_loader 
from engine import evaluate

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
WEIGHTS = "model_epoch_20.pth"
NUM_CLASSES = 14 
OUTPUT_DIR = r"D:\segmentation_project\All_9 copy\Inference_Results_SBS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DISEASES = {1: 'Hyperinflation', 7: 'Pulmonary Nodules', 9: 'Cardiomegaly', 13: 'Aortic Contour Abn.'}
CLASS_THRESHOLDS = {1: 0.70, 7: 0.35, 9: 0.85, 13: 0.80}

CLASS_NAMES = {
    1: "Hyperinflation", 2: "Lung Lucency", 3: "Bullae / cysts",
    4: "Bronchial Wall Thickening", 5: "Increased bronchial markings",
    6: "Reticular/fibrotic pattern", 7: "Pulmonary nodules/mass",
    8: "Pleural Effusion", 9: "Cardiomegaly", 10: "Cardiac silhouette shape",
    11: "Cardiac border visibility", 12: "Aortic knob", 13: "Aortic contour abnormality",
}


def get_inference_model(weights_path, num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

print(f"Initializing Model on {DEVICE}...")
inf_model = get_inference_model(WEIGHTS, NUM_CLASSES, DEVICE)
cm_data = {disease_id: {'y_true': [], 'y_pred': []} for disease_id in TARGET_DISEASES}


print(f"\nStep 1: Processing {len(val_loader)} batches for Side-by-Side visuals...")
ALPHA = 0.4 

with torch.no_grad():
    for i, (images, targets) in enumerate(tqdm(val_loader, desc="Saving Comparisons")):
        images_gpu = [img.to(DEVICE) for img in images]
        predictions = inf_model(images_gpu)

        for j in range(len(images)):
            
            img_np = images[j].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            gt_panel = img_bgr.copy()
            gt_labels = targets[j]['labels'].cpu().numpy()
            gt_boxes = targets[j]['boxes'].cpu().numpy()
            gt_masks = targets[j]['masks'].cpu().numpy()
            
            for k in range(len(gt_labels)):
                l_id = int(gt_labels[k])
                color = (0, 255, 0) if l_id in [1, 9] else (255, 100, 0)
                mask = gt_masks[k] > 0.5
                gt_panel[mask] = (gt_panel[mask] * (1-ALPHA) + np.array(color)*ALPHA).astype(np.uint8)
                cv2.rectangle(gt_panel, (int(gt_boxes[k][0]), int(gt_boxes[k][1])), 
                              (int(gt_boxes[k][2]), int(gt_boxes[k][3])), color, 2)
                cv2.putText(gt_panel, f"GT:{CLASS_NAMES.get(l_id, l_id)}", 
                            (int(gt_boxes[k][0]), int(gt_boxes[k][1]-5)), 1, 0.8, color, 1)

       
            pred_panel = img_bgr.copy()
            p_res = predictions[j]
            p_labels_passed = []

            for k in range(len(p_res['labels'])):
                l_id = int(p_res['labels'][k])
                score = float(p_res['scores'][k])
                if score < CLASS_THRESHOLDS.get(l_id, 0.5): continue
                
                p_labels_passed.append(l_id)
                color = (0, 0, 255) # Red for Pred
                mask = p_res['masks'][k, 0].cpu().numpy() > 0.5
                pred_panel[mask] = (pred_panel[mask] * (1-ALPHA) + np.array(color)*ALPHA).astype(np.uint8)
                cv2.rectangle(pred_panel, (int(p_res['boxes'][k][0]), int(p_res['boxes'][k][1])), 
                              (int(p_res['boxes'][k][2]), int(p_res['boxes'][k][3])), color, 2)
                cv2.putText(pred_panel, f"P:{score:.2f}", 
                            (int(p_res['boxes'][k][0]), int(p_res['boxes'][k][1]-5)), 1, 0.8, color, 1)

     
            sbs = np.hstack((gt_panel, pred_panel))
            img_id = targets[j]['image_id'].item() if hasattr(targets[j]['image_id'], 'item') else targets[j]['image_id']
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"val_{img_id}.png"), sbs)

       
            for d_id in TARGET_DISEASES:
                cm_data[d_id]['y_true'].append(1 if d_id in gt_labels else 0)
                cm_data[d_id]['y_pred'].append(1 if d_id in p_labels_passed else 0)

print("\nStep 2: Generating Matrices...")
fig, axes = plt.subplots(1, len(TARGET_DISEASES), figsize=(22, 5))
for i, (d_id, name) in enumerate(TARGET_DISEASES.items()):
    cm = confusion_matrix(cm_data[d_id]['y_true'], cm_data[d_id]['y_pred'], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f"{name}\n(Class {d_id})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"))


print("\nStep 3: Calculating Final Metrics...")
evaluate(inf_model, val_loader, device=DEVICE)

print(f"\nCompleted! Files saved to: {OUTPUT_DIR}")