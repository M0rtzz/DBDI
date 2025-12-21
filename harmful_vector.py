import torch
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
from scipy import stats
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from datasets import load_dataset
import pickle
import time
from datetime import datetime
import pandas as pd

import sys
sys.path.append('.')

from config import model_paths
from utils import (
    load_model, get_hidden_states, get_sentence_embeddings,
    get_difference_matrix, get_svd, cosine_similarity
)


class ToxicVectorExtractor:

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = load_model(model_name, model_paths)

        self.num_layers = self.model.config.num_hidden_layers + 1
        print(f"Model has {self.num_layers} layers")

    def load_harmful_dataset(self, dataset_name: str, max_samples: int = None) -> List[str]:
        """
        Loads a harmful dataset from a local JSON file.

        Args:
            dataset_name: Name of the dataset ('advbench', 'harmbench', 'strongreject').
            max_samples: The maximum number of samples to use.
        """
        print(f"\nLoading {dataset_name} dataset from local file...")
        harmful_prompts = []

        dataset_files = {
            'advbench': './data/advbench.json',
            'harmbench': './data/harmbench_validation.json',
            'strongreject': './data/strongreject.json'
        }

        if dataset_name not in dataset_files:
            print(f"Error: Unknown dataset {dataset_name}")
            return []

        filepath = dataset_files[dataset_name]

        try:
            if not os.path.exists(filepath):
                print(f"Error: File {filepath} not found")
                return []

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'mal' in item:
                        harmful_prompts.append(item['mal'])
            elif isinstance(data, dict) and 'mal' in data:
                harmful_prompts.append(data['mal'])

            if max_samples is not None and harmful_prompts:
                harmful_prompts = harmful_prompts[:max_samples]

            print(f"Loaded {len(harmful_prompts)} harmful prompts from {dataset_name}")

            if harmful_prompts:
                print("Sample harmful prompts:")
                for i, prompt in enumerate(harmful_prompts[:3]):
                    print(f"  {i+1}: {prompt[:100]}...")

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            harmful_prompts = []

        return harmful_prompts

    def load_harmless_dataset(self, max_samples: int = None) -> List[str]:
        """
        Loads a harmless dataset from local files.
        It prioritizes loading from './data/harmless.csv', then falls back to extracting
        the 'benign' field from JSON files, and finally uses a default list.
        """
        harmless_prompts = []

        csv_filepath = './data/harmless.csv'
        try:
            if os.path.exists(csv_filepath):
                print(f"Loading harmless prompts from {csv_filepath}")
                df = pd.read_csv(csv_filepath)

                if 'prompt' in df.columns:
                    harmless_prompts = df['prompt'].dropna().tolist()
                    print(f"Loaded {len(harmless_prompts)} harmless prompts from CSV")
                else:
                    print(f"Warning: 'prompt' column not found in {csv_filepath}")
                    print(f"Available columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading harmless.csv: {e}")

        if not harmless_prompts:
            json_filepath = './data/advbench.json'
            try:
                if os.path.exists(json_filepath):
                    print(f"Trying to load benign prompts from {json_filepath}")
                    with open(json_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'benign' in item and item['benign']:
                                harmless_prompts.append(item['benign'])

                    if harmless_prompts:
                        print(f"Loaded {len(harmless_prompts)} benign prompts from JSON")
            except Exception as e:
                print(f"Error loading benign prompts from JSON: {e}")

        if max_samples is not None:
            harmless_prompts = harmless_prompts[:max_samples]

        print(f"Final: Using {len(harmless_prompts)} harmless prompts")

        return harmless_prompts

    def collect_concept_activations(self, prompts: List[str], concept_name: str,
                                      checkpoint_dir: Optional[str] = None) -> List[List[torch.Tensor]]:
        """
        Collects activations for concept prompts, with checkpoint support.
        """
        if not prompts:
            print(f"Warning: No prompts provided for {concept_name}")
            return [[] for _ in range(self.num_layers)]

        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_{concept_name}_embeddings.pt')
            if os.path.exists(checkpoint_path):
                print(f"Loading {concept_name} embeddings from checkpoint...")
                return torch.load(checkpoint_path)

        print(f"\nCollecting {concept_name} concept activations for {len(prompts)} prompts...")

        if len(prompts) > self.batch_size:
            all_embeddings = [[] for _ in range(self.num_layers)]

            for i in tqdm(range(0, len(prompts), self.batch_size), desc=f"Processing {concept_name} batches"):
                batch_prompts = prompts[i:i+self.batch_size]

                try:
                    batch_embeddings = get_sentence_embeddings(
                        batch_prompts, self.model, self.model_name, self.tokenizer
                    )

                    for layer_idx in range(self.num_layers):
                        all_embeddings[layer_idx].extend(batch_embeddings[layer_idx])

                    if i % (self.batch_size * 4) == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Warning: Error processing batch {i//self.batch_size}: {e}")
                    continue

            embeddings = all_embeddings
        else:
            embeddings = get_sentence_embeddings(
                prompts, self.model, self.model_name, self.tokenizer
            )

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(embeddings, checkpoint_path)
            print(f"Saved {concept_name} embeddings checkpoint")

        return embeddings

    def train_toxic_classifiers(self,
                                  toxic_embeddings: List[List[torch.Tensor]],
                                  neutral_embeddings: List[List[torch.Tensor]]) -> Dict[int, Dict]:
        """
        Trains a classifier for each layer to evaluate its ability to distinguish toxic/neutral concepts.
        """
        layer_performance = {}

        print("\nTraining toxic concept classifiers for each layer...")
        for layer_idx in tqdm(range(self.num_layers)):
            try:
                X_toxic = torch.stack(toxic_embeddings[layer_idx]).cpu().numpy()
                X_neutral = torch.stack(neutral_embeddings[layer_idx]).cpu().numpy()

                X = np.vstack([X_toxic, X_neutral])
                y = np.hstack([np.ones(len(X_toxic)), np.zeros(len(X_neutral))])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if 'deepseek' in self.model_name:
                    clf = LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight='balanced',
                        C=0.1
                    )
                else:
                    clf = LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight='balanced'
                    )

                scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

                clf.fit(X_scaled, y)

                feature_importance = np.abs(clf.coef_[0])

                t_stats = []
                p_values = []
                for i in range(X_toxic.shape[1]):
                    try:
                        t_stat, p_val = stats.ttest_ind(X_toxic[:, i], X_neutral[:, i])
                        t_stats.append(abs(t_stat))
                        p_values.append(p_val)
                    except:
                        t_stats.append(0)
                        p_values.append(1)

                layer_performance[layer_idx] = {
                    'accuracy': scores.mean(),
                    'accuracy_std': scores.std(),
                    'classifier': clf,
                    'scaler': scaler,
                    'feature_importance': feature_importance,
                    't_stats': np.array(t_stats),
                    'p_values': np.array(p_values),
                    'top_features': np.argsort(feature_importance)[-min(100, len(feature_importance)):]
                }

            except Exception as e:
                print(f"Warning: Failed to train classifier for layer {layer_idx}: {e}")
                layer_performance[layer_idx] = {
                    'accuracy': 0.5,
                    'accuracy_std': 0.0,
                    'classifier': None,
                    'scaler': None,
                    'feature_importance': None,
                    't_stats': None,
                    'p_values': None,
                    'top_features': None
                }

        return layer_performance

    def select_best_toxic_layers(self, layer_performance: Dict, top_k: int = 5) -> List[int]:
        """
        Selects the most suitable layers for extracting the toxic concept.
        """
        valid_layers = {k: v for k, v in layer_performance.items()
                        if v['accuracy'] > 0.5 and v['classifier'] is not None}

        if not valid_layers:
            print("Warning: No valid layers found, using default layers")
            return list(range(max(0, self.num_layers - top_k), self.num_layers))

        sorted_layers = sorted(
            valid_layers.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        print("\nLayer toxic classification performance:")
        for layer_idx, perf in sorted_layers[:10]:
            print(f"Layer {layer_idx}: Accuracy = {perf['accuracy']:.4f} (±{perf['accuracy_std']:.4f})")

        best_layers = [layer_idx for layer_idx, _ in sorted_layers[:min(top_k, len(sorted_layers))]]

        if len(best_layers) < top_k:
            for layer in range(self.num_layers - 1, -1, -1):
                if layer not in best_layers:
                    best_layers.append(layer)
                if len(best_layers) >= top_k:
                    break

        return best_layers[:top_k]

    def extract_toxic_vector_for_layer(self,
                                           toxic_embeddings: List[torch.Tensor],
                                           neutral_embeddings: List[torch.Tensor],
                                           layer_performance: Dict,
                                           method: str = 'dbdi',
                                           neuron_ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the toxic vector for a single layer.
        """
        try:
            if method == 'dbdi':
                difference_matrix = get_difference_matrix(toxic_embeddings, neutral_embeddings)
                _, S, V = get_svd(difference_matrix)
                toxic_direction = V[:, 0].cpu()
            elif method == 'mean_diff':
                toxic_mean = torch.stack(toxic_embeddings).mean(dim=0)
                neutral_mean = torch.stack(neutral_embeddings).mean(dim=0)
                toxic_direction = (toxic_mean - neutral_mean).cpu()

            if (layer_performance.get('p_values') is None or
                layer_performance.get('t_stats') is None or
                layer_performance.get('feature_importance') is None):

                abs_direction = torch.abs(toxic_direction)
                k = int(len(toxic_direction) * neuron_ratio)
                top_k_indices = torch.topk(abs_direction, k).indices
                combined_mask = torch.zeros_like(toxic_direction, dtype=torch.bool)
                combined_mask[top_k_indices] = True

            else:
                p_values = layer_performance['p_values']
                t_stats = layer_performance['t_stats']
                feature_importance = torch.tensor(layer_performance['feature_importance'])

                significance_mask = torch.tensor(p_values < 0.01)

                t_stats_tensor = torch.tensor(t_stats)
                if torch.sum(t_stats_tensor > 0) > 0:
                    effect_threshold = torch.quantile(t_stats_tensor[t_stats_tensor > 0], 1 - neuron_ratio)
                    effect_mask = t_stats_tensor > effect_threshold
                else:
                    effect_mask = torch.zeros_like(significance_mask)

                if torch.sum(feature_importance > 0) > 0:
                    importance_threshold = torch.quantile(feature_importance[feature_importance > 0], 1 - neuron_ratio)
                    importance_mask = feature_importance > importance_threshold
                else:
                    importance_mask = torch.zeros_like(significance_mask)

                combined_mask = significance_mask & (effect_mask | importance_mask)

                if combined_mask.sum() < int(len(toxic_direction) * neuron_ratio):
                    k = int(len(toxic_direction) * neuron_ratio)
                    if torch.sum(t_stats_tensor > 0) > 0:
                        top_k_indices = torch.topk(t_stats_tensor, min(k, len(t_stats_tensor))).indices
                    else:
                        top_k_indices = torch.topk(torch.abs(toxic_direction), k).indices
                    combined_mask = torch.zeros_like(toxic_direction, dtype=torch.bool)
                    combined_mask[top_k_indices] = True

            sparse_toxic_vector = toxic_direction * combined_mask.float()

            if torch.norm(sparse_toxic_vector) > 1e-8:
                sparse_toxic_vector = sparse_toxic_vector / torch.norm(sparse_toxic_vector)

            return sparse_toxic_vector, combined_mask

        except Exception as e:
            print(f"Error in extract_toxic_vector_for_layer: {e}")
            dim = len(toxic_embeddings[0])
            return torch.zeros(dim), torch.zeros(dim, dtype=torch.bool)

    def extract_vectors_for_dataset(self,
                                        dataset_name: str,
                                        max_samples: int = None,
                                        extraction_method: str = 'dbdi',
                                        neuron_ratio: float = 0.25,
                                        checkpoint_dir: Optional[str] = None) -> Dict:
        """
        Extracts vectors for a single dataset.
        """
        harmful_prompts = self.load_harmful_dataset(dataset_name, max_samples)
        if not harmful_prompts:
            print(f"Skipping {dataset_name} - no data loaded")
            return None

        harmless_prompts = self.load_harmless_dataset(len(harmful_prompts))

        harmful_embeddings = self.collect_concept_activations(
            harmful_prompts, f"{dataset_name}_harmful", checkpoint_dir
        )
        harmless_embeddings = self.collect_concept_activations(
            harmless_prompts, f"{dataset_name}_harmless", checkpoint_dir
        )

        gc.collect()
        torch.cuda.empty_cache()

        layer_perf_checkpoint_path = None
        if checkpoint_dir:
            layer_perf_checkpoint_path = os.path.join(
                checkpoint_dir,
                f'{self.model_name}_{dataset_name}_layer_performance.pkl'
            )
            if os.path.exists(layer_perf_checkpoint_path):
                print(f"Loading layer performance checkpoint for {dataset_name}...")
                with open(layer_perf_checkpoint_path, 'rb') as f:
                    layer_performance = pickle.load(f)
            else:
                layer_performance = None
        else:
            layer_performance = None

        if layer_performance is None:
            layer_performance = self.train_toxic_classifiers(harmful_embeddings, harmless_embeddings)

            if checkpoint_dir and layer_perf_checkpoint_path:
                with open(layer_perf_checkpoint_path, 'wb') as f:
                    pickle.dump(layer_performance, f)
                print(f"Saved layer performance checkpoint for {dataset_name}")

        best_layers = self.select_best_toxic_layers(layer_performance, top_k=5)

        results = {
            'dataset_name': dataset_name,
            'layer_performance': {},
            'harmful_vectors': {},
            'best_layers': best_layers,
            'n_prompts': len(harmful_prompts)
        }

        for layer_idx, perf in layer_performance.items():
            results['layer_performance'][layer_idx] = {
                'accuracy': perf['accuracy'],
                'accuracy_std': perf['accuracy_std'],
                'n_significant_features': int((perf['p_values'] < 0.01).sum()) if perf['p_values'] is not None else 0,
                'mean_t_stat': float(perf['t_stats'].mean()) if perf['t_stats'] is not None else 0.0
            }

        print(f"\nExtracting harmful vectors for all {self.num_layers} layers...")
        for layer_idx in tqdm(range(self.num_layers)):
            try:
                vector, mask = self.extract_toxic_vector_for_layer(
                    harmful_embeddings[layer_idx],
                    harmless_embeddings[layer_idx],
                    layer_performance[layer_idx],
                    method=extraction_method,
                    neuron_ratio=neuron_ratio
                )

                results['harmful_vectors'][layer_idx] = {
                    'vector': vector.cpu(),
                    'mask': mask.cpu(),
                    'n_active': mask.sum().item(),
                    'sparsity': 1 - (mask.sum().item() / len(mask)),
                    'is_best_layer': layer_idx in best_layers
                }
            except Exception as e:
                print(f"Warning: Failed to extract vector for layer {layer_idx}: {e}")

        return results

    def save_dataset_results(self, results: Dict, output_dir: str, dataset_name: str):
        """
        Saves the results for a single dataset.
        """
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        layer_accuracy_path = os.path.join(dataset_output_dir, 'layer_accuracies.json')
        layer_accuracies = {}
        for layer_idx, perf in results['layer_performance'].items():
            layer_accuracies[f"layer_{layer_idx}"] = {
                'accuracy': perf['accuracy'],
                'accuracy_std': perf['accuracy_std'],
                'n_significant_features': perf['n_significant_features'],
                'mean_t_stat': perf['mean_t_stat']
            }

        with open(layer_accuracy_path, 'w') as f:
            json.dump(layer_accuracies, f, indent=2)

        metadata = {
            'model_name': self.model_name,
            'dataset_name': dataset_name,
            'num_layers': self.num_layers,
            'best_layers': results['best_layers'],
            'n_prompts': results['n_prompts'],
            'extraction_time': datetime.now().isoformat()
        }

        with open(os.path.join(dataset_output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        all_vectors_data = {
            'harmful_vectors': results['harmful_vectors'],
            'metadata': metadata
        }

        torch.save(all_vectors_data, os.path.join(dataset_output_dir, 'all_layer_vectors.pt'))

        best_vectors = {}
        for layer_idx in results['best_layers']:
            if layer_idx in results['harmful_vectors']:
                best_vectors[layer_idx] = results['harmful_vectors'][layer_idx]

        torch.save(best_vectors, os.path.join(dataset_output_dir, 'best_5_layer_vectors.pt'))

        self.plot_layer_performance(results, dataset_output_dir)

        print(f"\nResults for {dataset_name} saved to {dataset_output_dir}")
        print(f"- layer_accuracies.json: Per-layer accuracy metrics")
        print(f"- metadata.json: Dataset and extraction metadata")
        print(f"- all_layer_vectors.pt: Vectors from all layers")
        print(f"- best_5_layer_vectors.pt: Vectors from top 5 layers")

    def plot_layer_performance(self, results: Dict, output_dir: str):
        """
        Creates a visualization of layer performance.
        """
        layers = sorted(results['layer_performance'].keys())
        accuracies = [results['layer_performance'][l]['accuracy'] for l in layers]
        stds = [results['layer_performance'][l]['accuracy_std'] for l in layers]

        plt.figure(figsize=(12, 6))
        plt.errorbar(layers, accuracies, yerr=stds, marker='o', capsize=5, color='red')
        plt.xlabel('Layer Index')
        plt.ylabel('Classification Accuracy')
        plt.title(f'Layer-wise Harmful Classification Performance\nModel: {self.model_name}, Dataset: {results["dataset_name"]}')
        plt.grid(True, alpha=0.3)

        for layer in results['best_layers']:
            plt.axvline(x=layer, color='darkred', linestyle='--', alpha=0.5)

        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_performance.png'))
        plt.close()

    def cleanup(self):
        """Cleans up the model to free up memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class MultiModelMultiDatasetExtractor:
    """Extractor to handle multiple models and multiple datasets."""

    def __init__(self, models: List[str], datasets: List[str], device: str = 'cuda'):
        self.models = models
        self.datasets = datasets
        self.device = device
        self.checkpoint_dir = './checkpoints_harm'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_progress_checkpoint_path(self):
        """Gets the path for the progress checkpoint file."""
        return os.path.join(self.checkpoint_dir, 'extraction_progress.json')

    def load_progress(self) -> Dict:
        """Loads the extraction progress."""
        checkpoint_path = self.get_progress_checkpoint_path()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return {'completed_tasks': [], 'timestamp': datetime.now().isoformat()}

    def save_progress(self, progress: Dict):
        """Saves the extraction progress."""
        checkpoint_path = self.get_progress_checkpoint_path()
        progress['timestamp'] = datetime.now().isoformat()
        with open(checkpoint_path, 'w') as f:
            json.dump(progress, f, indent=2)

    def process_all(self, max_samples: int = None, neuron_ratio: float = 0.25):
        """
        Processes all combinations of models and datasets.
        """
        progress = self.load_progress()
        completed_tasks = progress.get('completed_tasks', [])

        total_tasks = len(self.models) * len(self.datasets)
        print(f"\n{'='*60}")
        print(f"Starting multi-model multi-dataset extraction")
        print(f"Models: {self.models}")
        print(f"Datasets: {self.datasets}")
        print(f"Total tasks: {total_tasks}")
        print(f"Completed tasks: {len(completed_tasks)}")
        print(f"{'='*60}\n")

        start_time = time.time()
        task_idx = 0

        for model_name in self.models:
            try:
                extractor = ToxicVectorExtractor(model_name, self.device)
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                continue

            output_dir = os.path.join('./extracted_harm_vector', model_name)

            for dataset_name in self.datasets:
                task_idx += 1
                task_key = f"{model_name}_{dataset_name}"

                if task_key in completed_tasks:
                    print(f"\n[{task_idx}/{total_tasks}] Task {task_key} already completed, skipping...")
                    continue

                print(f"\n{'='*60}")
                print(f"[{task_idx}/{total_tasks}] Processing: {model_name} + {dataset_name}")
                print(f"{'='*60}")

                try:
                    results = extractor.extract_vectors_for_dataset(
                        dataset_name=dataset_name,
                        max_samples=max_samples,
                        extraction_method='dbdi',
                        neuron_ratio=neuron_ratio,
                        checkpoint_dir=self.checkpoint_dir
                    )

                    if results:
                        extractor.save_dataset_results(results, output_dir, dataset_name)

                        completed_tasks.append(task_key)
                        progress['completed_tasks'] = completed_tasks
                        self.save_progress(progress)

                        print(f"\n✓ Task {task_key} completed successfully")
                    else:
                        print(f"\n✗ Task {task_key} skipped - no data")

                except Exception as e:
                    print(f"\n✗ Error in task {task_key}: {str(e)}")
                    import traceback
                    traceback.print_exc()

                    if task_idx < total_tasks:
                        response = input("\nContinue with next task? (y/n): ")
                        if response.lower() != 'y':
                            break

            extractor.cleanup()
            print(f"\nCompleted all datasets for model {model_name}")

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Multi-model multi-dataset extraction completed!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Completed tasks: {len(completed_tasks)}/{total_tasks}")
        print(f"{'='*60}")

        if len(completed_tasks) == total_tasks:
            response = input("\nAll tasks completed. Remove checkpoints? (y/n): ")
            if response.lower() == 'y':
                self.cleanup_checkpoints()

    def cleanup_checkpoints(self):
        """Cleans up all checkpoint files."""
        print("\nCleaning up checkpoints...")
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            print(f"Removed checkpoint directory: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract harmful concept vectors using multiple datasets')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=list(model_paths.keys()),
                        help='Model names (can specify multiple)')
    parser.add_argument('--model', type=str,
                        choices=list(model_paths.keys()),
                        help='Single model name (for backward compatibility)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['advbench', 'harmbench', 'strongreject'],
                        choices=['advbench', 'harmbench', 'strongreject'],
                        help='Datasets to use for harmful prompts')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples from each dataset')
    parser.add_argument('--neuron_ratio', type=float, default=0.25,
                        help='Ratio of neurons to select (default: 0.25)')

    args = parser.parse_args()

    if args.models:
        models_to_process = args.models
    elif args.model:
        models_to_process = [args.model]
    else:
        models_to_process = list(model_paths.keys())
        print(f"No models specified, will process all available models: {models_to_process}")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return

    if len(models_to_process) == 1 and len(args.datasets) == 1:
        model_name = models_to_process[0]
        dataset_name = args.datasets[0]

        extractor = ToxicVectorExtractor(model_name)
        output_dir = os.path.join('./extracted_harm_vector', model_name)

        print(f"\nExtracting harmful vectors for {model_name} using {dataset_name}...")

        results = extractor.extract_vectors_for_dataset(
            dataset_name=dataset_name,
            max_samples=args.max_samples,
            neuron_ratio=args.neuron_ratio
        )

        if results:
            extractor.save_dataset_results(results, output_dir, dataset_name)

    else:
        multi_extractor = MultiModelMultiDatasetExtractor(
            models=models_to_process,
            datasets=args.datasets
        )
        multi_extractor.process_all(
            max_samples=args.max_samples,
            neuron_ratio=args.neuron_ratio
        )


if __name__ == "__main__":
    main()