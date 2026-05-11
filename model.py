import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataloader import train_dataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


num_classes = len(train_dataset.coco.getCatIds()) + 1  # background + your classes

print ("Total Class",num_classes)


in_features_box = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)


in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

model.to(device)


params = [p for p in model.parameters() if p.requires_grad]


optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

from engine  import train_one_epoch, evaluate
from dataloader import val_loader
from dataloader import train_loader

START_EPOCH = 40
NUM_TOTAL_EPOCHS = 41

checkpoint_path = f"model_epoch_{START_EPOCH}.pth"
print(f"Loading weights from {checkpoint_path}...")

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)


for epoch in range(START_EPOCH, NUM_TOTAL_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_TOTAL_EPOCHS}")
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)
    evaluate(model, val_loader, device=device)  
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")