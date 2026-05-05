import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
import time
from torchmetrics.classification import MulticlassConfusionMatrix
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from data_loader import FeatureDataset
from model import ContinualLearningModel
from features import extract_features
from prototype_selector import EmbeddingPrototypeSelector #, PrototypeBasedSelector 
from torch.utils.data import DataLoader

class ContinualLearningTrainer:    
    def __init__(self,
                 model: ContinualLearningModel,
                 data_loader,
                 decoupling: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 epochs_per_task: int = 10,
                 verbose: bool = True,
                 batch_size: int = 64,
                 use_ensemble: bool = False,
                 task_aware: bool = False,
                 buffer_size_per_task: int = 0,
                 K_per_class: int = 2
                 ):
        self.model = model
        self.extractor = model.extractor.to(device)
        self.use_ensemble = use_ensemble
        if self.use_ensemble:
            self.classifier = None
        else:
            self.classifier = model.classifier.to(device)
        self.decoupling = decoupling
        self.data_loader = data_loader
        self.device = device
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs_per_task = epochs_per_task
        self.verbose = verbose
        self.batch_size = batch_size
        self.task_aware = task_aware

        self.K_per_class = K_per_class

        self.feature_buffer = defaultdict(list)  
        self.buffer_size_per_task = buffer_size_per_task  

        self.proto_selector = EmbeddingPrototypeSelector(K_per_class=K_per_class, metric='cosine')
        
        if self.decoupling and hasattr(self.model, "classifier_ensemble"):
            self.classifier = None   
            self.optimizer = None   
        else:
            self.classifier = self.model.classifier.to(self.device)
            self._init_optimizer()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.metrics = {
            'task_accuracies': [],  # Accuracy on each task after training
            'average_accuracy': [],  # Average accuracy across all seen tasks
            'training_losses': [],  # Training loss for each task
            'class_accuracies': defaultdict(list),  # Per-class accuracies
            'confusion_matrices': [],  # Confusion matrix 
            'training_times': [],  # Time to train each task
            'extraction_times': []  # Time to extract features for each task
        }
        
        self.current_task = 0
        self.seen_classes = set()
        self.task_class_mapping = {}  # Maps global class IDs to task-specific IDs
        
        self._class_to_output_mapping = {}
  
    def _init_optimizer(self):
        """Initialize optimizer"""

        if self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.classifier.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_type} not supported")
        
    def get_features(self, x):
        with torch.no_grad():
            if self.decoupling:
                feats = self.model.extractor(x)
            else:
                feats = self.model.extractor(x)  # (B, C, H, W)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
        return feats
    
    def train_task(self, task_id: int) -> Dict[str, float]:
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Training Task {task_id}")
            print(f"{'='*50}")
        
        start_time = time.time()

        task_info = self.data_loader.get_task_info()[task_id]
        task_classes = task_info['classes']
        
        if self.verbose:
            print(f"Task classes: {task_classes}")
            print(f"Seen classes before: {sorted(list(self.seen_classes))}")
        
        # Update seen classes and expand classifier if needed
        new_classes = set(task_classes) - self.seen_classes
        if new_classes and self.model.scenario == 'class-incremental':
            if self.verbose:
                print(f"New classes: {sorted(list(new_classes))}")
                print(f"Expanding classifier by {len(new_classes)} classes")
            
            sorted_seen_classes = sorted(list(self.seen_classes | set(task_classes)))
            for i, cls in enumerate(sorted_seen_classes):
                self._class_to_output_mapping[cls] = i
            
            if not self.use_ensemble:
                self.model.expand_classifier(len(new_classes))
                self._init_optimizer()
        
        self.seen_classes.update(task_classes)
        
        # Data preparation 
        train_loader = self.data_loader.get_task_loader(task_id, train=True)
        test_loader = self.data_loader.get_task_loader(task_id, train=False)

        if self.decoupling:
            start_time_extract = time.time()
            train_feats, train_labels = extract_features(train_loader, self.extractor, self.device)
            test_feats, test_labels = extract_features(test_loader, self.extractor, self.device)
        
            self._save_task_features(train_feats, task_id)
            self.train_dataset = FeatureDataset(train_feats, train_labels.squeeze(), task_id)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            
            if not hasattr(self, 'test_features'):
                self.test_features = {}
                self.test_labels_dict = {}
            self.test_features[task_id] = test_feats
            self.test_labels_dict[task_id] = test_labels.squeeze()
            
            extraction_time = time.time() - start_time_extract
            self.metrics['extraction_times'].append(extraction_time)

            self.task_features = getattr(self, "task_features", {})
            self.task_labels = getattr(self, "task_labels", {})
            self.task_features[task_id] = train_feats.cpu().numpy()
            self.task_labels[task_id] = train_labels.cpu().numpy().squeeze()
                    
        else:
            self.train_loader = train_loader
            self.metrics['extraction_times'].append(0)
        
        # Classifier initialization 
        if self.use_ensemble:
            num_classes = len(task_classes)
            self.model.add_new_classifier(task_id, num_classes)
            self.classifier = self.model.classifier_ensemble[str(task_id)].to(self.device)
            self._init_optimizer()
        else:
            self.classifier = self.model.classifier.to(self.device)
            self._init_optimizer()
        
        # Training loop 
        task_losses = []
        self.model.train()

        for epoch in range(self.epochs_per_task):
            epoch_loss = 0.0
            correct, total = 0, 0
            for data, targets, _ in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                if self.model.scenario == 'class-incremental' and not self.use_ensemble:
                    targets = self._remap_targets(targets, task_classes)
                else:
                    local_class_to_idx = {cls: i for i, cls in enumerate(task_classes)}
                    targets = torch.tensor(
                        [local_class_to_idx[int(t)] for t in targets],
                        dtype=torch.long,
                        device=targets.device
                    )

                self.optimizer.zero_grad()
                outputs = self.model(data, task_id=task_id, pre_extracted=self.decoupling)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            acc = 100. * correct / total
            task_losses.append(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.epochs_per_task // 5) == 0:
                print(f"Epoch {epoch+1}/{self.epochs_per_task}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

        if self.use_ensemble:
            feats_np = train_feats.cpu().numpy()
            labels_np = train_labels.cpu().numpy().squeeze()
            self.proto_selector.fit_task(task_id, feats_np, labels_np)

        # Metrics 
        training_time = time.time() - start_time
        self.metrics['training_losses'].append(task_losses)
        self.metrics['training_times'].append(training_time)

        if self.verbose:
            print(f"Task {task_id} training completed in {training_time:.2f}s")

        return {'loss': task_losses[-1], 'accuracy': acc, 'time': training_time}

    def _remap_targets(self, targets: torch.Tensor, task_classes: List[int]) -> torch.Tensor:
        if self.data_loader.num_tasks == 1:
            return targets  
        mapping = self.data_loader.global_class_to_output
        return torch.tensor(
            [mapping[int(t)] for t in targets],
            dtype=torch.long,
            device=targets.device
        )
    
    def _save_task_features(self, features, task_id):
        features = features.cpu()
        num_samples = min(self.buffer_size_per_task, len(features))
        idx = np.random.choice(len(features), size=num_samples, replace=False)
        sampled = features[idx]
        self.feature_buffer[task_id] = sampled

    def evaluate_task(self, task_id=None, task_aware=False, selector_batch_size=256):
        self.model.eval()
        device = self.device

        if self.decoupling and task_id is not None and hasattr(self, 'test_features') and task_id in self.test_features:
            features = self.test_features[task_id].to(device)
            targets = self.test_labels_dict[task_id].to(device)
            eval_batches = [(features, targets)]
        elif self.decoupling and task_id is None and hasattr(self, 'test_features'):
            all_features = torch.cat([self.test_features[tid] for tid in sorted(self.test_features.keys())], dim=0).to(device)
            all_targets = torch.cat([self.test_labels_dict[tid] for tid in sorted(self.test_labels_dict.keys())], dim=0).to(device)
            eval_batches = [(all_features, all_targets)]
        else:
            loader = (
                self.data_loader.get_full_test_loader()
                if task_id is None else
                self.data_loader.get_task_loader(task_id, train=False)
            )
            eval_batches = loader

        class_to_global = self._class_to_output_mapping
        total_output_size = max(class_to_global.values()) + 1
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in eval_batches:
                if self.decoupling and hasattr(self, 'test_features'):
                    features, targets = batch
                else:
                    data, targets, _ = batch
                    data, targets = data.to(device), targets.to(device)
                    features = self.get_features(data)

                num_samples = features.size(0)
                batch_global_logits = torch.full((num_samples, total_output_size), -1e9, device=device)

                # CASE 1: Task-aware 
                if task_aware:
                    head = self.model.classifier_ensemble[str(task_id)]
                    head_logits = head(features)
                    head_classes = self.data_loader.get_task_info()[task_id]['classes']
                    for local_idx, cls in enumerate(head_classes):
                        global_idx = class_to_global[cls]
                        batch_global_logits[:, global_idx] = head_logits[:, local_idx]

                # CASE 2: Proto-selection
                elif hasattr(self, "proto_selector"):

                    n_tasks = len(self.model.classifier_ensemble)
                    all_task_assignments = []

                    # STEP 1: compute all task selections
                    for i in range(0, num_samples, selector_batch_size):
                        end = min(i + selector_batch_size, num_samples)
                        feats_chunk = features[i:end]
                        selections, confidence_info  = self.proto_selector.batch_select(feats_chunk, return_scores=True)
                        
                        all_task_assignments.extend(selections)

                    # STEP 2: group samples per task
                    task_to_indices = defaultdict(list)
                    for idx, sel in enumerate(all_task_assignments):
                        for tid in sel:
                            task_to_indices[tid].append(idx)

                    # STEP 3: run each classifier on its subset (batched)
                    for tid, idxs in task_to_indices.items():
                        if not idxs:
                            continue

                        idx_tensor = torch.tensor(idxs, device=device)
                        feats_group = features[idx_tensor]

                        head = self.model.classifier_ensemble[str(tid)]
                        logits = head(feats_group)
                        head_classes = self.data_loader.get_task_info()[tid]['classes']

                        for local_idx, cls in enumerate(head_classes):
                            global_idx = class_to_global[cls]
                            batch_global_logits[idx_tensor, global_idx] = logits[:, local_idx]

                # Predictions 
                preds = batch_global_logits.argmax(dim=1)
                if self.model.scenario == 'class-incremental':
                    targets = torch.tensor(
                        [class_to_global[int(t)] for t in targets],
                        dtype=torch.long, device=device
                    )

                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item() * 100.0
        num_classes = total_output_size
        cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        cm = cm_metric(all_preds, all_targets).cpu().numpy()

        class_accuracies = {}
        for cls in range(num_classes):
            mask = all_targets == cls
            class_accuracies[cls] = (all_preds[mask] == cls).float().mean().item() * 100.0 if mask.sum() > 0 else float("nan")

        if self.verbose:
            print(f"Accuracy: {accuracy:.2f}%")

        return {
            "accuracy": accuracy,
            "preds": all_preds.cpu(),
            "targets": all_targets.cpu(),
            "confusion_matrix": cm,
            "class_accuracies": class_accuracies,
        }
    
    def evaluate_all_tasks(self):
        """Efficiently evaluate on all seen tasks (weighted average)."""
        results, total_acc, total_samples = {}, 0.0, 0
        for tid in range(self.current_task + 1):
            res = self.evaluate_task(tid)
            results[f"task_{tid}"] = res

            samples = self.data_loader.get_task_info()[tid]['test_samples']
            total_acc += res['accuracy'] * samples
            total_samples += samples

        results['average_accuracy'] = total_acc / max(total_samples, 1)
        if self.verbose:
            print(f"🌍 Average accuracy over {self.current_task + 1} tasks: {results['average_accuracy']:.2f}%")
        return results
    
    def train_continual(self) -> Dict[str, Any]:

        if self.verbose:
            print(f"Starting continual learning with {self.data_loader.num_tasks} tasks")
            print(f"Scenario: {self.model.scenario}")
            print(f"Model: {self.model.backbone_name}")
        
        for task_id in range(self.data_loader.num_tasks):
            # Train on current task
            train_results = self.train_task(task_id)
            self.current_task = task_id
            
            # Evaluate on all seen tasks
            eval_results = self.evaluate_all_tasks()
            
            # Store metrics
            task_accuracies = [eval_results[f'task_{i}']['accuracy'] 
                             for i in range(task_id + 1)]
            self.metrics['task_accuracies'].append(task_accuracies)
            self.metrics['average_accuracy'].append(eval_results['average_accuracy'])
            
            # Store per-class accuracies
            for cls, acc in eval_results[f'task_{task_id}']['class_accuracies'].items():
                self.metrics['class_accuracies'][cls].append(acc)
            
            if self.verbose:
                print(f"\nTask {task_id} Results:")
                print(f"  Task accuracies: {[f'{acc:.2f}%' for acc in task_accuracies]}")
                print(f"  Average accuracy: {eval_results['average_accuracy']:.2f}%")

        final_metrics = self._calculate_final_metrics()
        self.metrics.update(final_metrics)
        
        return self.metrics
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        num_tasks = len(self.metrics['task_accuracies'])
    
        final_avg_acc = self.metrics['average_accuracy'][-1]
               
        return {
            'final_average_accuracy': final_avg_acc,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Task accuracies over time
        ax1 = axes[0, 0]
        num_tasks = len(self.metrics['task_accuracies'])
        
        for task_id in range(num_tasks):
            task_accs = [self.metrics['task_accuracies'][i][task_id] 
                        if task_id < len(self.metrics['task_accuracies'][i]) else np.nan
                        for i in range(num_tasks)]
            ax1.plot(range(task_id, num_tasks), task_accs[task_id:], 
                    marker='o', label=f'Task {task_id}')
        
        ax1.set_xlabel('Training Task')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Task Accuracies During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average accuracy over time
        ax2 = axes[0, 1]
        ax2.plot(range(num_tasks), self.metrics['average_accuracy'], 
                marker='s', color='red', linewidth=2)
        ax2.set_xlabel('Training Task')
        ax2.set_ylabel('Average Accuracy (%)')
        ax2.set_title('Average Accuracy Across All Seen Tasks')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training loss for each task
        ax3 = axes[1, 0]
        for i, losses in enumerate(self.metrics['training_losses']):
            ax3.plot(losses, label=f'Task {i}')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss per Task')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_final_report(self):
        print(f"\n{'='*60}")
        print(f"CONTINUAL LEARNING FINAL REPORT")
        print(f"{'='*60}")
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Scenario: {self.model.scenario}")
        print(f"Model: {self.model.backbone_name}")
        print(f"Number of tasks: {self.data_loader.num_tasks}")
        print(f"Epochs per task: {self.epochs_per_task}")
        
        print(f"\n{'='*30}")
        print(f"FINAL METRICS")
        print(f"{'='*30}")
        
        if 'final_average_accuracy' in self.metrics:
            print(f"Final Average Accuracy: {self.metrics['final_average_accuracy']:.2f}%")
            print(f"Number of Parameters: {self.metrics['num_parameters']:,}")
        
        if self.metrics['task_accuracies']:
            final_accs = self.metrics['task_accuracies'][-1]
            print(f"\nFinal Task Accuracies:")
            for i, acc in enumerate(final_accs):
                print(f"  Task {i}: {acc:.2f}%")
        
    def save_metrics(self, filepath: str):
        json_metrics = {}
        for key, value in self.metrics.items():
            if key == 'confusion_matrices':
                json_metrics[key] = [cm.tolist() for cm in value]
            elif isinstance(value, defaultdict):
                json_metrics[key] = dict(value)
            else:
                json_metrics[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
