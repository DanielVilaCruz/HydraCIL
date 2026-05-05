import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from data_loader import ContinualLearningLoader
from model import ContinualLearningModel
from trainer import ContinualLearningTrainer

# =========================
# 1. LOAD CONFIG
# =========================
with open("saved_models/hydraCL_loader.json", "r") as f:
    loader_cfg = json.load(f)

with open("saved_models/hydraCL_trainer.json", "r") as f:
    trainer_state = json.load(f)

print("Loader config:", loader_cfg)

# =========================
# 2. REBUILD DATA LOADER
# =========================
data_loader = ContinualLearningLoader(
    dataset_name=loader_cfg["dataset_name"],
    scenario=loader_cfg["scenario"],
    num_tasks=loader_cfg["num_tasks"],
    batch_size=loader_cfg["batch_size"],
    data_size=loader_cfg["data_size"],
)

# (Optional but safer)
if "classes_per_task" in loader_cfg:
    data_loader.classes_per_task = loader_cfg["classes_per_task"]

# =========================
# 3. SAMPLE FOR MODEL INIT
# =========================
sample_loader = data_loader.get_task_loader(task_id=0, train=True)
sample_batch = next(iter(sample_loader))
sample_data = sample_batch[0]

# =========================
# 4. BUILD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ContinualLearningModel(
    decoupling=True,
    backbone="resnet34",
    num_classes=data_loader.classes_per_task,
    pretrained=False,
    scenario=loader_cfg["scenario"],
    data_sample=sample_data,
    device=device,
    use_ensemble=True,
)

# =========================
# 5. INIT CLASSIFIERS
# =========================
print("Initializing classifiers...")
for task_id in range(loader_cfg["num_tasks"]):
    if task_id >= len(model.classifier_ensemble):
        model.add_new_classifier(
            task_id=task_id,
            num_classes=data_loader.classes_per_task
        )

print(f"Total classifiers: {len(model.classifier_ensemble)}")

# =========================
# 6. LOAD WEIGHTS
# =========================
print("Loading weights...")
state_dict = torch.load("saved_models/hydraCL_model.pth", map_location=device)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =========================
# 7. REBUILD TRAINER
# =========================
trainer = ContinualLearningTrainer(
    model=model,
    data_loader=data_loader,
    use_ensemble=True,
    decoupling=True,
    buffer_size_per_task=trainer_state.get("buffer_size_per_task", 200),
)

# =========================
# 8. LOAD PROTOTYPES
# =========================
print("Loading prototypes...")
protos_npz = np.load("saved_models/hydraCL_prototypes.npz", allow_pickle=True)

trainer.proto_selector.task_prototypes = {}

for key in protos_npz.files:
    tid = int(key.split("_")[1])
    centroids = protos_npz[key]

    trainer.proto_selector.task_prototypes[tid] = {
        "centroids": centroids,
        "classes": np.array(trainer_state["proto_classes"][str(tid)]),
    }

trainer._class_to_output_mapping = {
    int(k): v for k, v in trainer_state["class_mapping"].items()
}

trainer.metrics = trainer_state["metrics"]

# =========================
# 9. INFERENCE FUNCTION
# =========================
def extract_features_batch(x, backbone):
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    features = backbone(x)
    features = features.view(features.size(0), -1)
    return features

def run_inference(trainer, task_id, device): 
    model = trainer.model
    proto_selector = trainer.proto_selector
    class_to_global = trainer._class_to_output_mapping
    
    custom_loader = trainer.data_loader 
    test_loader = custom_loader.get_task_loader(task_id=task_id, train=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, t in test_loader:
            x = x.to(device)
            features = extract_features_batch(x, model.extractor)
            batch_size = x.size(0)
            num_classes = len(class_to_global)

            batch_global_logits = torch.full(
                (batch_size, num_classes),
                fill_value=-1e9,
                device=device
            )

            selections = proto_selector.batch_select(features)
            
            from collections import defaultdict
            task_to_indices = defaultdict(list)
            for idx, sel in enumerate(selections):
                for tid in sel:
                    task_to_indices[tid].append(idx)

            for tid, idxs in task_to_indices.items():
                if not idxs:
                    continue

                idx_tensor = torch.tensor(idxs, device=device)
                feats_group = features[idx_tensor]

                head = model.classifier_ensemble[str(tid)]
                logits = head(feats_group)

                # FIX: Access task info from the custom_loader, not the PyTorch dataloader
                head_classes = custom_loader.get_task_info()[tid]['classes']

                for local_idx, cls in enumerate(head_classes):
                    global_idx = class_to_global[cls]
                    batch_global_logits[idx_tensor, global_idx] = logits[:, local_idx]

            preds = batch_global_logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

# =========================
# 10. RUN INFERENCE
# =========================
print("Running inference...")
task_ids = [0,2,4]
global_acc = 0
for task_id in task_ids:
    test_loader = data_loader.get_task_loader(task_id=task_id, train=False)
    preds, labels = run_inference(trainer, task_id=task_id, device=device)

    acc = accuracy_score(labels, preds)
    print(f"Task Accuracy: {acc:.4f}")
    global_acc += acc

print(f"Global Accuracy: {global_acc/len(task_ids):.4f}")
