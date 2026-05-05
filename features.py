import torch
import tqdm
import torch.nn.functional as F

def extract_features(dataloader, backbone, device):
    backbone.eval()
    backbone.to(device)
        
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True) 
            
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            features = backbone(images)
            features = features.view(features.size(0), -1)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
