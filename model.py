import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, vit_l_16, resnet34, ResNet34_Weights
from torch.nn import ModuleDict
import torch.nn.functional as F

class ContinualExtractor(nn.Module):
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, last_layer: int=2):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "vit":
            self.backbone = vit_b_16(pretrained=pretrained)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()  
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        self.extractor = nn.Sequential(*list(self.backbone.children())[:-last_layer]) 
        self.extractor.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    
    def forward(self, x):
        x = self.extractor(x) 
        features = torch.flatten(x, 1)
        
        return features

class ContinualClassifier(nn.Module):

    def __init__(self, 
                 num_classes: int = 10,
                 scenario: str = 'class-incremental',
                 feature_dim: int = 512,
                 device: torch.device = torch.device("cpu"),
                 num_dummy_classes: int = 10
                 ):
        
        super().__init__()
        self.current_classes = 0
        self.scenario = scenario
        self.num_real_classes = num_classes
        self.num_dummy_classes = num_dummy_classes
        self.total_classes = num_classes + num_dummy_classes
        self.feature_dim = feature_dim
        self.device = device

        # Initialize classifier
        if scenario == 'class-incremental':
            # Start with no classes, will expand incrementally
            self.classifier = nn.Linear(self.feature_dim, self.total_classes).to(self.device)
        else:
            # Task-incremental: fixed number of classes per task
            self.classifier = nn.Linear(self.feature_dim, self.total_classes).to(self.device)
            self.current_classes = num_classes

    def expand_classifier(self, new_classes: int):
        if self.scenario != 'class-incremental':
            return
        
        old_classifier = self.classifier
        old_classes = self.current_classes
        total_classes = old_classes + new_classes
        new_classifier = nn.Linear(self.feature_dim, total_classes).to(self.device)

        if old_classes > 0:
            new_classifier.weight.data[:old_classes] = old_classifier.weight.data
            new_classifier.bias.data[:old_classes] = old_classifier.bias.data
        
        if new_classes > 0:
            nn.init.kaiming_normal_(new_classifier.weight.data[old_classes:])
            nn.init.constant_(new_classifier.bias.data[old_classes:], 0)
        
        self.classifier = new_classifier
        self.current_classes = total_classes
    
    def forward(self, x, include_dummy=True):
        logits = self.classifier(x)
        if include_dummy:
            return logits
        return logits[:, :self.current_classes]

class ContinualLearningModel(nn.Module):
    def __init__(self, 
                 decoupling: bool = True,
                 backbone: str = 'resnet18',
                 num_classes: int = 10,
                 pretrained: bool = True,
                 scenario: str = 'class-incremental',
                 last_layer: int=2, # 1: fc, 2: avgpool
                 data_sample: torch.Tensor = None,
                 device: torch.device = torch.device("cpu"),
                 use_ensemble: bool = False,
                 ):
        
        super().__init__()
        
        self.scenario = scenario
        self.backbone_name = backbone
        self.total_classes = num_classes
        self.current_classes = 0
        self.last_layer = last_layer
        self.device = device
        self.use_ensemble = use_ensemble

        self.extractor = ContinualExtractor(
            backbone=backbone,
            pretrained=pretrained,
            last_layer=last_layer
        )

        for param in self.extractor.parameters():
                param.requires_grad = False

        if data_sample is not None:
            with torch.no_grad():
                features = self.extractor(data_sample)
                
            self.feature_dim = features.shape[1]
        else:
            self.feature_dim = self.extractor.feature_dim
        
        if use_ensemble:
            self.classifier_ensemble = nn.ModuleDict()
            self.add_new_classifier(task_id=0, num_classes=num_classes)
        else:
            self.classifier = ContinualClassifier(
                num_classes=num_classes,
                scenario=scenario,
                feature_dim=self.feature_dim,
                device = self.device,
                num_dummy_classes=0
            )

    def add_new_classifier(self, task_id: int, num_classes: int):
        new_classifier = ContinualClassifier(
            num_classes=num_classes,
            scenario=self.scenario,
            feature_dim=self.feature_dim,
            device=self.device,
            num_dummy_classes=0
        )
        self.classifier_ensemble[str(task_id)] = new_classifier

        if not hasattr(self, "class_offsets"):
            self.class_offsets = {}
        if not hasattr(self, "head_real_classes"):
            self.head_real_classes = {}
        self.head_real_classes[str(task_id)] = num_classes
        current_total = sum(c.classifier.out_features for c in self.classifier_ensemble.values())
        start_idx = current_total - num_classes
        self.class_offsets[str(task_id)] = (start_idx, start_idx + num_classes)

    def get_classifier(self):
        return self.classifier
    
    def get_extractor(self):
        return self.extractor
    
    def expand_classifier(self, new_classes: int):
        if self.scenario != 'class-incremental' or self.use_ensemble:
            return
        self.classifier.expand_classifier(new_classes)
        self.current_classes = self.classifier.current_classes
    
    def forward(self, x, task_id=None, pre_extracted=False):
        features = x if pre_extracted else self.extractor(x)
        features = F.normalize(features, dim=1)

        if self.use_ensemble:
            if task_id is not None:
                return self.classifier_ensemble[str(task_id)](features)
            else:
                logits_list = []
                for head in self.classifier_ensemble.values():
                    logits = head(features)
                    logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-6)
                    logits_list.append(logits)
            return torch.cat(logits_list, dim=1)
        
        self.classifier.to(features.device)
        return self.classifier(features)