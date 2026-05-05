import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F

class EmbeddingPrototypeSelector:
    def __init__(self, K_per_class=2, metric='cosine', device='cuda'):
        self.K_per_class = K_per_class
        self.metric = metric
        self.task_prototypes = {}  # task_id -> dict with centroids
        self.device = device

    def _normalize(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-10)

    def fit_task(self, task_id, features, labels):
        features = np.array(features)
        labels = np.array(labels).squeeze()
        embeddings = features
        unique_classes = np.unique(labels)
        centroids = []

        for cls in unique_classes:
            cls_embeddings = embeddings[labels == cls]
            n_samples = len(cls_embeddings)

            if n_samples == 0:
                continue
            if n_samples <= self.K_per_class:
                cls_centroids = cls_embeddings
            else:
                # Run K-means in the embedding space
                kmeans = KMeans(n_clusters=self.K_per_class, n_init=10, random_state=42)
                kmeans.fit(cls_embeddings)
                cls_centroids = kmeans.cluster_centers_
            
            # Re-normalize centroids to ensure they stay on the unit
            centroids.extend(self._normalize(cls_centroids))

        self.task_prototypes[task_id] = {
            "centroids": np.array(centroids)
        }
    
    def batch_select(self, features: torch.Tensor, uncertainty_threshold=0.0005, top_k=2, confidence_threshold=0.7, return_scores=False):

        if not self.task_prototypes:
            empty_result = [[0] for _ in range(features.size(0))]
            if return_scores:
                return empty_result, []
            return empty_result
        
        # 1. Normalize input features
        features_np = features.cpu().numpy() if features.is_cuda else features.numpy()
        # embeddings_np = self._normalize(features_np)
        embeddings_np = features_np
        
        # 2. Get unique task IDs first
        task_ids_unique = sorted(self.task_prototypes.keys())
        n_tasks = len(task_ids_unique)
        
        if n_tasks == 0:
            empty_result = [[0] for _ in range(features.size(0))]
            if return_scores:
                return empty_result, []
            return empty_result
        
        # 3. Collect all centroids
        all_centroids = []
        centroid_to_task = []
        
        for task_id in task_ids_unique:
            proto_data = self.task_prototypes[task_id]
            centroids = proto_data["centroids"]
            all_centroids.append(centroids)
            centroid_to_task.extend([task_id] * len(centroids))
        
        all_centroids = np.vstack(all_centroids)
        centroid_to_task = np.array(centroid_to_task)
        
        # 4. Compute cosine similarity
        similarities = embeddings_np @ all_centroids.T  # (B, total_centroids)
        
        # 5. Aggregate per-task scores
        n_samples = embeddings_np.shape[0]
        task_scores = np.zeros((n_samples, n_tasks))
        
        for task_idx, task_id in enumerate(task_ids_unique):
            mask = (centroid_to_task == task_id)
            task_scores[:, task_idx] = similarities[:, mask].max(axis=1)
        
        # 6. Select tasks and compute confidence metrics
        selections = []
        confidence_info = []
        
        for i in range(n_samples):
            # Sort tasks by score (descending)
            sorted_indices = np.argsort(task_scores[i])[::-1]
            sorted_task_ids = [task_ids_unique[idx] for idx in sorted_indices]
            sorted_scores = task_scores[i, sorted_indices]
            
            best_task = sorted_task_ids[0]
            best_score = sorted_scores[0]
            
            # Calculate confidence metrics
            second_best_score = sorted_scores[1] if n_tasks > 1 else 0.0
            relative_gap = (best_score - second_best_score) / (best_score + 1e-6) if n_tasks > 1 else 1.0
            
            # Compute entropy-based uncertainty
            # Convert scores to probabilities
            exp_scores = np.exp(sorted_scores - sorted_scores.max())  # numerical stability
            probs = exp_scores / exp_scores.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(n_tasks)  # maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Decision logic
            selected = [best_task]
            selection_reason = "confident"
            
            if best_score < confidence_threshold:
                selected = sorted_task_ids[:min(top_k, n_tasks)]
                selection_reason = "low_confidence"
            elif n_tasks > 1 and relative_gap < uncertainty_threshold:
                selected = sorted_task_ids[:min(top_k, n_tasks)]
                selection_reason = "small_gap"
            
            selections.append(selected)
            
            # Store detailed confidence info
            confidence_info.append({
                'best_task': best_task,
                'best_score': float(best_score),
                'second_best_score': float(second_best_score),
                'relative_gap': float(relative_gap),
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'certainty': float(1.0 - normalized_entropy),  # 0=uncertain, 1=certain
                'all_scores': {task_ids_unique[idx]: float(sorted_scores[idx]) for idx in range(n_tasks)},
                'selected_tasks': selected,
                'selection_reason': selection_reason
            })
        
        if return_scores:
            return selections, confidence_info
        return selections


    def __init__(self, K_per_class=2, metric='cosine', device='cuda'):
        self.K_per_class = K_per_class
        self.metric = metric # Using cosine for embedding space
        self.task_prototypes = {} 
        self.device = device

    def _normalize(self, x):
        # L2 normalization to project features onto a unit sphere
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-10)

    def fit_task(self, task_id, features, labels):
        features = np.array(features)
        labels = np.array(labels).squeeze()
        
        # Convert raw features to normalized embeddings
        embeddings = self._normalize(features)

        unique_classes = np.unique(labels)
        centroids = []

        for cls in unique_classes:
            cls_embeddings = embeddings[labels == cls]
            
            # Cluster in the embedding space
            if len(cls_embeddings) <= self.K_per_class:
                cls_centroids = cls_embeddings
            else:
                kmeans = KMeans(n_clusters=self.K_per_class, n_init=10, random_state=42)
                kmeans.fit(cls_embeddings)
                cls_centroids = kmeans.cluster_centers_
            
            # Re-normalize centroids (K-means centroids can drift off)
            centroids.extend(self._normalize(cls_centroids))

        self.task_prototypes[task_id] = {
            "centroids": np.array(centroids)
        }