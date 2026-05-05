import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Dict
import random
from PIL import Image
import os 

class HuggingFaceImageNet100(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class ContinualDataset(Dataset):
    def __init__(self, dataset, task_id: int = 0):
        self.dataset = dataset
        self.task_id = task_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, self.task_id

class CORe50Custom(Dataset):
    def __init__(self, root, sessions, transform=None, sample_rate=10):
        self.root = root
        self.sessions = sessions
        self.transform = transform
        self.sample_rate = sample_rate # Sample each {sample_rate} images
        self.images = []
        self.targets = []
        
        self._parse_dataset()

    def _parse_dataset(self):
        for obj_idx in range(1, 51):
            obj_folder = f"o{obj_idx}"
            label = obj_idx - 1 
            
            for session in self.sessions:
                session_path = os.path.join(self.root, session, obj_folder)
                if not os.path.exists(session_path):
                    continue

                img_files = sorted([f for f in os.listdir(session_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

                sampled_files = img_files[::self.sample_rate]
                
                for img_file in sampled_files:
                    self.images.append(os.path.join(session_path, img_file))
                    self.targets.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FeatureDataset(Dataset):
    def __init__(self, features, labels, task_id):
        self.features = features
        self.labels = labels
        self.task_id = task_id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.task_id

class ContinualLearningLoader:    
    def __init__(self, 
                 dataset_name: str = 'cifar10',
                 root: str = './data',
                 download: bool = True,
                 scenario: str = 'class-incremental',  # 'class-incremental' or 'task-incremental'
                 num_tasks: int = 5,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 seed: int = 42,
                 data_size: int = 224
                 ):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.download = download
        self.scenario = scenario
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.data_size = data_size
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.train_datasets = []
        self.test_datasets = []
        self.task_info = {}
        
        self._load_and_split_dataset()

    
    
    def _get_dataset_info(self) -> Dict:
        """Get dataset-specific information"""
        dataset_configs = {
            'cifar10': {
                'num_classes': 10,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': torchvision.datasets.CIFAR10,
                'transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.data_size, padding=4),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),

                'test_transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            },
            'cifar100': {
                'num_classes': 100,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': torchvision.datasets.CIFAR100,
                'transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.data_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            },
            'mnist': {
                'num_classes': 10,
                'input_size': (1, 28, 28),
                'dataset_class': torchvision.datasets.MNIST,
                'transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            },
            'imagenet100': {
                'num_classes': 100,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': HuggingFaceImageNet100,
                'transform': transforms.Compose([
                    transforms.RandomResizedCrop(self.data_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(self.data_size),
                    transforms.CenterCrop(self.data_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            },
            'imagenet': {
                'num_classes': 1000,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': torchvision.datasets.ImageNet,
                'transform': transforms.Compose([
                    transforms.RandomResizedCrop(self.data_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(self.data_size),
                    transforms.CenterCrop(self.data_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            },
            'core50': {
                'num_classes': 50,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': torchvision.datasets.ImageFolder,
                'transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            },
            'flowers102': {
                'num_classes': 102,
                'input_size': (3, self.data_size, self.data_size),
                'dataset_class': torchvision.datasets.Flowers102,
                'transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize((self.data_size, self.data_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }
        }
        
        if self.dataset_name not in dataset_configs:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Available: {list(dataset_configs.keys())}")
        
        return dataset_configs[self.dataset_name]
    
    def _load_and_split_dataset(self):
        """Load dataset and split into tasks based on scenario"""
        dataset_info = self._get_dataset_info()

        if self.dataset_name == 'imagenet100':
            print("Loading ImageNet100 from local directory...")
            
            train_dir = f"{self.root}/ImageNet100/train"
            val_dir = f"{self.root}/ImageNet100/val"
            train_dataset = torchvision.datasets.ImageFolder(
                root=train_dir,
                transform=dataset_info['transform']
            )
            test_dataset = torchvision.datasets.ImageFolder(
                root=val_dir,
                transform=dataset_info['test_transform']
            )

        elif self.dataset_name == 'imagenet':
            train_dataset = dataset_info['dataset_class'](
                root=self.root, split='train', download=self.download,
                transform=dataset_info['transform']
            )
            test_dataset = dataset_info['dataset_class'](
                root=self.root, split='val', download=self.download,
                transform=dataset_info['test_transform']
            )
        elif self.dataset_name == 'core50':
            print(f"Loading CORe50 from {self.root} with 1/10 sampling...")
            core_path = os.path.join(self.root, "CORe50", "core50_128x128")

            all_sessions = [f"s{i}" for i in range(1, 12)]
            test_sessions = ["s3", "s7", "s10"]
            train_sessions = [s for s in all_sessions if s not in test_sessions]

            train_dataset = CORe50Custom(
                root=core_path, 
                sessions=train_sessions, 
                transform=dataset_info['transform'],
                sample_rate=10
            )
            test_dataset = CORe50Custom(
                root=core_path, 
                sessions=test_sessions, 
                transform=dataset_info['test_transform'],
                sample_rate=10
            )
        elif self.dataset_name == 'flowers102':
            train_dataset = dataset_info['dataset_class'](
                root=self.root, split='train', download=self.download,
                transform=dataset_info['transform']
            )
            test_dataset = dataset_info['dataset_class'](
                root=self.root, split='test', download=self.download,
                transform=dataset_info['test_transform']
            )
            train_dataset.targets = train_dataset._labels
            test_dataset.targets = test_dataset._labels
        else:
            train_dataset = dataset_info['dataset_class'](
                root=self.root, train=True, download=self.download,
                transform=dataset_info['transform']
            )
            test_dataset = dataset_info['dataset_class'](
                root=self.root, train=False, download=self.download,
                transform=dataset_info['test_transform']
            )
        
        num_classes = dataset_info['num_classes']
        
        if self.scenario == 'class-incremental':
            self._split_class_incremental(train_dataset, test_dataset, num_classes)
  
        elif self.scenario == 'task-incremental':
            self._split_task_incremental(train_dataset, test_dataset, num_classes)
        else:
            raise ValueError(f"Scenario {self.scenario} not supported. Use 'class-incremental' or 'task-incremental'")
    
    def _split_class_incremental(self, train_dataset, test_dataset, num_classes):
        # Global shuffled class order
        self.global_class_order = list(range(num_classes))
        np.random.shuffle(self.global_class_order)
        self.global_class_to_output = {cls: i for i, cls in enumerate(self.global_class_order)}

        if self.num_tasks == 1:
            class_splits = [self.global_class_order]  # all classes in one task
        else:
            classes_per_task = num_classes // self.num_tasks
            remaining_classes = num_classes % self.num_tasks

            class_splits = []
            start_idx = 0
            for task_id in range(self.num_tasks):
                current_classes_per_task = classes_per_task + (1 if task_id < remaining_classes else 0)
                end_idx = start_idx + current_classes_per_task
                task_classes = self.global_class_order[start_idx:end_idx]
                class_splits.append(task_classes)
                start_idx = end_idx

            self.classes_per_task = classes_per_task

        # Build datasets for each task
        if hasattr(train_dataset, 'targets'):
            train_targets = np.array(train_dataset.targets)
            test_targets = np.array(test_dataset.targets)
        else:
            # Fallback for datasets (like ImageNet100 wrapper) without a .targets attribute
            print("Retrieving all targets for splitting. This may take a moment.")
            train_targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
            test_targets = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

        for task_id, task_classes in enumerate(class_splits):
            train_mask = np.isin(train_targets, task_classes)
            train_indices = np.where(train_mask)[0]
            task_train_dataset = Subset(train_dataset, train_indices)

            test_mask = np.isin(test_targets, task_classes)
            test_indices = np.where(test_mask)[0]
            task_test_dataset = Subset(test_dataset, test_indices)

            self.train_datasets.append(ContinualDataset(task_train_dataset, task_id))
            self.test_datasets.append(ContinualDataset(task_test_dataset, task_id))

            self.task_info[task_id] = {
                'classes': sorted(task_classes),
                'num_classes': len(task_classes),
                'train_samples': len(train_indices),
                'test_samples': len(test_indices)
            }

    def _split_task_incremental(self, train_dataset, test_dataset, num_classes):
        classes_per_task = num_classes // self.num_tasks
        remaining_classes = num_classes % self.num_tasks
        
        if hasattr(train_dataset, 'targets'):
            train_targets = np.array(train_dataset.targets)
            test_targets = np.array(test_dataset.targets)
        else:
            # Fallback for datasets (like ImageNet100 wrapper) without a .targets attribute
            print("Retrieving all targets for splitting. This may take a moment.")
            train_targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
            test_targets = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
        
        # Create class splits
        all_classes = list(range(num_classes))
        np.random.shuffle(all_classes)
        
        class_splits = []
        start_idx = 0
        for task_id in range(self.num_tasks):
            current_classes_per_task = classes_per_task + (1 if task_id < remaining_classes else 0)
            end_idx = start_idx + current_classes_per_task
            task_classes = all_classes[start_idx:end_idx]
            class_splits.append(task_classes)
            start_idx = end_idx
            
        self.classes_per_task = classes_per_task

        # Create task datasets 
        for task_id, task_classes in enumerate(class_splits):
            # Training data 
            train_mask = np.isin(train_targets, task_classes)
            train_indices = np.where(train_mask)[0]
            task_train_dataset = Subset(train_dataset, train_indices)
            
            # Test data 
            test_mask = np.isin(test_targets, task_classes)
            test_indices = np.where(test_mask)[0]
            task_test_dataset = Subset(test_dataset, test_indices)

            self.train_datasets.append(ContinualDataset(task_train_dataset, task_id))
            self.test_datasets.append(ContinualDataset(task_test_dataset, task_id))

            self.task_info[task_id] = {
                'classes': task_classes,
                'num_classes': len(task_classes),
                'train_samples': len(train_indices),
                'test_samples': len(test_indices)
            }
    
    def get_task_loader(self, task_id: int, train: bool = True) -> DataLoader:
        if task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} exceeds number of tasks {self.num_tasks}")
        
        dataset = self.train_datasets[task_id] if train else self.test_datasets[task_id]
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and train,
            # num_workers=self.num_workers
        )
    
    def get_task_info(self) -> Dict:
        """Get information about all tasks"""
        return self.task_info
    
    def print_task_summary(self):
        """Print summary of all tasks"""
        print(f"\n=== Continual Learning Dataset Summary ===")
        print(f"Dataset: {self.dataset_name}")
        print(f"Scenario: {self.scenario}")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Batch size: {self.batch_size}")
        print()
        
        for task_id in range(self.num_tasks):
            info = self.task_info[task_id]
            print(f"Task {task_id}:")
            print(f"  Classes: {info['classes']}")
            print(f"  Number of classes: {info['num_classes']}")
            print(f"  Training samples: {info['train_samples']}")
            print(f"  Test samples: {info['test_samples']}")
            print()