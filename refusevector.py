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
import time
from datetime import datetime
import pickle

import sys
sys.path.append('.') # Ensure modules from the current directory can be imported

# Use config and utils 
from config import model_paths
from utils import (
    load_model, get_hidden_states, get_sentence_embeddings,
    get_difference_matrix, get_svd, cosine_similarity
)


class RefusalVectorExtractor:
    """Extractor for refusal vectors based on the DBDI method, using a classifier to determine the best layers."""

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = load_model(model_name, model_paths)

        self.num_layers = self.model.config.num_hidden_layers + 1
        print(f"Model has {self.num_layers} layers")

    def collect_all_layer_activations(self, prompts: List[str], desc: str = "Collecting activations") -> List[List[torch.Tensor]]:
        """
        Collects activations for all layers, returning in the format: [layer][prompt] = activation_tensor.
        Uses the get_sentence_embeddings function from DBDI.
        """
        print(f"\n{desc} for {len(prompts)} prompts...")

        embeddings_by_layer = get_sentence_embeddings(
            prompts, self.model, self.model_name, self.tokenizer
        )

        return embeddings_by_layer

    def train_layer_classifiers(self,
                                  harmful_embeddings: List[List[torch.Tensor]],
                                  benign_embeddings: List[List[torch.Tensor]]) -> Dict[int, Dict]:
        """
        Trains a classifier for each layer to evaluate its ability to distinguish harmful from benign prompts.
        """
        layer_performance = {}

        print("\nTraining classifiers for each layer...")
        for layer_idx in tqdm(range(self.num_layers)):
            X_harmful = torch.stack(harmful_embeddings[layer_idx]).cpu().numpy()
            X_benign = torch.stack(benign_embeddings[layer_idx]).cpu().numpy()

            X = np.vstack([X_harmful, X_benign])
            y = np.hstack([np.ones(len(X_harmful)), np.zeros(len(X_benign))])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, random_state=42)

            scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

            clf.fit(X_scaled, y)

            feature_importance = np.abs(clf.coef_[0])

            layer_performance[layer_idx] = {
                'accuracy': scores.mean(),
                'accuracy_std': scores.std(),
                'classifier': clf,
                'scaler': scaler,
                'feature_importance': feature_importance,
                'top_features': np.argsort(feature_importance)[-100:] # top 100 features
            }

        return layer_performance

    def select_best_layers(self, layer_performance: Dict, top_k: int = 5) -> List[int]:
        """
        Selects the best layers based on classifier performance.
        """
        sorted_layers = sorted(
            layer_performance.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        print("\nLayer classification performance:")
        for layer_idx, perf in sorted_layers[:10]:
            print(f"Layer {layer_idx}: Accuracy = {perf['accuracy']:.4f} (±{perf['accuracy_std']:.4f})")

        best_layers = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]
        return best_layers

    def extract_refusal_vector_for_layer(self,
                                           harmful_embeddings: List[torch.Tensor],
                                           benign_embeddings: List[torch.Tensor],
                                           layer_performance: Dict,
                                           neuron_ratio: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the refusal vector for a single layer, using important features identified by the classifier.

        Args:
            neuron_ratio: The proportion of neurons to select.
        """
        difference_matrix = get_difference_matrix(benign_embeddings, harmful_embeddings)

        _, S, V = get_svd(difference_matrix)

        principal_direction = V[:, 0].cpu()

        if 'feature_importance' in layer_performance:
            importance = torch.tensor(layer_performance['feature_importance'])
            importance = importance / importance.max() # Normalize importance

            # Create a sparse mask (keep the top neuron_ratio of neurons)
            threshold = torch.quantile(importance, 1 - neuron_ratio)
            mask = importance > threshold

            # Ensure at least neuron_ratio of neurons are selected
            if mask.sum() < int(len(principal_direction) * neuron_ratio):
                k = int(len(principal_direction) * neuron_ratio)
                top_k_indices = torch.topk(importance, k).indices
                mask = torch.zeros_like(principal_direction, dtype=torch.bool)
                mask[top_k_indices] = True

            sparse_direction = principal_direction * mask.float()

            # Normalize the vector
            if torch.norm(sparse_direction) > 0:
                sparse_direction = sparse_direction / torch.norm(sparse_direction)
        else:
            sparse_direction = principal_direction / torch.norm(principal_direction)
            mask = torch.ones_like(principal_direction, dtype=torch.bool)

        return sparse_direction, mask

    def extract_refusal_vectors(self,
                                  data_path: str,
                                  max_samples: int = None,
                                  top_k_layers: int = 5,
                                  neuron_ratio: float = 0.10,
                                  checkpoint_dir: Optional[str] = None) -> Dict:
        """
        Main function to extract refusal vectors.

        Args:
            neuron_ratio: The proportion of neurons to select.
            checkpoint_dir: Directory to save checkpoints.
        """
        print(f"\nLoading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]
            print(f"Using first {max_samples} samples")

        harmful_prompts = [item['mal'] for item in data]
        benign_prompts = [item['benign'] for item in data]

        embeddings_checkpoint = None
        if checkpoint_dir:
            embeddings_checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_embeddings.pt')
            if os.path.exists(embeddings_checkpoint_path):
                print(f"Loading embeddings checkpoint from {embeddings_checkpoint_path}")
                embeddings_checkpoint = torch.load(embeddings_checkpoint_path, weights_only=False)

        if embeddings_checkpoint:
            harmful_embeddings = embeddings_checkpoint['harmful_embeddings']
            benign_embeddings = embeddings_checkpoint['benign_embeddings']
        else:
            harmful_embeddings = self.collect_all_layer_activations(
                harmful_prompts, "Collecting harmful activations"
            )
            benign_embeddings = self.collect_all_layer_activations(
                benign_prompts, "Collecting benign activations"
            )

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'harmful_embeddings': harmful_embeddings,
                    'benign_embeddings': benign_embeddings,
                    'model_name': self.model_name,
                    'timestamp': datetime.now().isoformat()
                }, embeddings_checkpoint_path)
                print(f"Saved embeddings checkpoint to {embeddings_checkpoint_path}")

        layer_performance_checkpoint = None
        if checkpoint_dir:
            layer_performance_checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_layer_performance.pkl')
            if os.path.exists(layer_performance_checkpoint_path):
                print(f"Loading layer performance checkpoint from {layer_performance_checkpoint_path}")
                with open(layer_performance_checkpoint_path, 'rb') as f:
                    layer_performance_checkpoint = pickle.load(f)

        if layer_performance_checkpoint:
            layer_performance = layer_performance_checkpoint
        else:
            layer_performance = self.train_layer_classifiers(
                harmful_embeddings, benign_embeddings
            )

            if checkpoint_dir:
                with open(layer_performance_checkpoint_path, 'wb') as f:
                    pickle.dump(layer_performance, f)
                print(f"Saved layer performance checkpoint to {layer_performance_checkpoint_path}")

        best_layers = self.select_best_layers(layer_performance, top_k_layers)
        print(f"\nSelected best layers: {best_layers}")

        results = {
            'layer_performance': {},
            'refusal_vectors': {},
            'best_layers': best_layers
        }

        for layer_idx, perf in layer_performance.items():
            results['layer_performance'][layer_idx] = {
                'accuracy': perf['accuracy'],
                'accuracy_std': perf['accuracy_std']
            }

        print(f"\nExtracting refusal vectors for all {self.num_layers} layers...")
        for layer_idx in tqdm(range(self.num_layers)):
            vector, mask = self.extract_refusal_vector_for_layer(
                harmful_embeddings[layer_idx],
                benign_embeddings[layer_idx],
                layer_performance[layer_idx],
                neuron_ratio=neuron_ratio
            )

            results['refusal_vectors'][layer_idx] = {
                'vector': vector.cpu(),
                'mask': mask.cpu(),
                'n_active': mask.sum().item(),
                'sparsity': 1 - (mask.sum().item() / len(mask)),
                'is_best_layer': layer_idx in best_layers
            }

        return results

    def plot_layer_performance(self, results: Dict, output_path: str):
        """
        Visualizes the classification performance of each layer.
        """
        layers = sorted(results['layer_performance'].keys())
        accuracies = [results['layer_performance'][l]['accuracy'] for l in layers]
        stds = [results['layer_performance'][l]['accuracy_std'] for l in layers]

        plt.figure(figsize=(12, 6))
        plt.errorbar(layers, accuracies, yerr=stds, marker='o', capsize=5)
        plt.xlabel('Layer Index')
        plt.ylabel('Classification Accuracy')
        plt.title(f'Layer-wise Classification Performance - {self.model_name}')
        plt.grid(True, alpha=0.3)

        best_layers = results['best_layers']
        for layer in best_layers:
            plt.axvline(x=layer, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Performance plot saved to {output_path}")

    def save_results(self, results: Dict, output_dir: str):
        """
        Saves the results.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.plot_layer_performance(
            results,
            os.path.join(output_dir, 'layer_performance.png')
        )

        json_results = {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'best_layers': results['best_layers'],
            'layer_performance': results['layer_performance']
        }

        with open(os.path.join(output_dir, 'extraction_metadata.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

        all_vectors_data = {
            'refusal_vectors': {},
            'metadata': json_results
        }

        for layer_idx, data in results['refusal_vectors'].items():
            all_vectors_data['refusal_vectors'][layer_idx] = {
                'vector': data['vector'],
                'mask': data['mask'],
                'n_active': data['n_active'],
                'sparsity': data['sparsity'],
                'is_best_layer': data['is_best_layer']
            }

        torch.save(all_vectors_data, os.path.join(output_dir, 'all_layer_vectors.pt'))

        best_vectors_data = {
            'refusal_vectors': {},
            'metadata': {
                'model_name': self.model_name,
                'num_layers': self.num_layers,
                'best_layers': results['best_layers']
            }
        }

        for layer_idx in results['best_layers']:
            if layer_idx in results['refusal_vectors']:
                best_vectors_data['refusal_vectors'][layer_idx] = results['refusal_vectors'][layer_idx]

        torch.save(best_vectors_data, os.path.join(output_dir, 'best_5_layer_vectors.pt'))

        print(f"\nResults saved to {output_dir}")
        print(f"- extraction_metadata.json: Metadata and performance metrics")
        print(f"- all_layer_vectors.pt: Vectors from all {self.num_layers} layers")
        print(f"- best_5_layer_vectors.pt: Vectors from top 5 performing layers")
        print(f"- layer_performance.png: Visualization of layer performance")

    def cleanup(self):
        """Cleans up the model to free up memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class MultiModelExtractor:
    """Handles refusal vector extraction for multiple models."""

    def __init__(self, models: List[str], device: str = 'cuda'):
        self.models = models
        self.device = device
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_progress_checkpoint_path(self):
        """Gets the path for the progress checkpoint file."""
        return os.path.join(self.checkpoint_dir, 'extraction_progress.json')

    def load_progress(self) -> Dict:
        """Loads extraction progress."""
        checkpoint_path = self.get_progress_checkpoint_path()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return {'completed_models': [], 'timestamp': datetime.now().isoformat()}

    def save_progress(self, progress: Dict):
        """Saves extraction progress."""
        checkpoint_path = self.get_progress_checkpoint_path()
        progress['timestamp'] = datetime.now().isoformat()
        with open(checkpoint_path, 'w') as f:
            json.dump(progress, f, indent=2)

    def process_all_models(self,
                           data_path: str,
                           max_samples: int = None,
                           neuron_ratio: float = 0.10,
                           top_k_layers: int = 5):
        """
        Processes all configured models.
        """
        progress = self.load_progress()
        completed_models = progress.get('completed_models', [])

        print(f"\n{'='*60}")
        print(f"Starting multi-model extraction for {len(self.models)} models")
        print(f"Completed models: {completed_models}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for idx, model_name in enumerate(self.models):
            if model_name in completed_models:
                print(f"\n[{idx+1}/{len(self.models)}] Model {model_name} already completed, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"[{idx+1}/{len(self.models)}] Processing model: {model_name}")
            print(f"{'='*60}")

            try:
                extractor = RefusalVectorExtractor(model_name, self.device)

                output_dir = os.path.join('./extracted_refuse_vector', model_name)

                results = extractor.extract_refusal_vectors(
                    data_path,
                    max_samples=max_samples,
                    top_k_layers=top_k_layers,
                    neuron_ratio=neuron_ratio,
                    checkpoint_dir=self.checkpoint_dir
                )

                extractor.save_results(results, output_dir)

                extractor.cleanup()

                completed_models.append(model_name)
                progress['completed_models'] = completed_models
                self.save_progress(progress)

                print(f"\n✓ Model {model_name} completed successfully")

            except Exception as e:
                print(f"\n✗ Error processing model {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()

                if 'extractor' in locals():
                    extractor.cleanup()
                continue

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Multi-model extraction completed!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Completed models: {len(completed_models)}/{len(self.models)}")
        print(f"{'='*60}")

        if len(completed_models) == len(self.models):
            print("All models completed. Checkpoints kept for reproducibility/resume safety.")

    def cleanup_checkpoints(self):
        """Cleans up all checkpoint files."""
        print("\nCleaning up checkpoints...")
        for model_name in self.models:
            embeddings_path = os.path.join(self.checkpoint_dir, f'{model_name}_embeddings.pt')
            performance_path = os.path.join(self.checkpoint_dir, f'{model_name}_layer_performance.pkl')

            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
                print(f"Removed {embeddings_path}")

            if os.path.exists(performance_path):
                os.remove(performance_path)
                print(f"Removed {performance_path}")

        progress_path = self.get_progress_checkpoint_path()
        if os.path.exists(progress_path):
            os.remove(progress_path)
            print(f"Removed {progress_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract refusal vectors with layer selection via classification')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=list(model_paths.keys()),
                        help='Model names (can specify multiple)')
    parser.add_argument('--model', type=str,
                        choices=list(model_paths.keys()),
                        help='Single model name (for backward compatibility)')
    parser.add_argument('--data_path', type=str, default='./data/twinprompt.json',
                        help='Path to twin prompt dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use')
    parser.add_argument('--neuron_ratio', type=float, default=0.25,
                        help='Ratio of neurons to select (default: 0.25)')
    parser.add_argument('--top_k_layers', type=int, default=5,
                        help='Number of best layers to select')

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

    if len(models_to_process) == 1:
        model_name = models_to_process[0]
        extractor = RefusalVectorExtractor(model_name)

        output_dir = os.path.join('./extracted_refuse_vector', model_name)

        print("\nExtracting refusal vectors with layer classification...")
        print(f"Selecting top {args.neuron_ratio*100:.0f}% neurons per layer")
        print(f"Output directory: {output_dir}")

        results = extractor.extract_refusal_vectors(
            args.data_path,
            max_samples=args.max_samples,
            top_k_layers=args.top_k_layers,
            neuron_ratio=args.neuron_ratio
        )

        extractor.save_results(results, output_dir)

        print("\n=== Extraction Summary ===")
        print(f"Model: {model_name}")
        print(f"Total layers: {extractor.num_layers}")
        print(f"Best performing layers: {results['best_layers']}")
        for layer in results['best_layers'][:3]:
            perf = results['layer_performance'][layer]
            print(f"  Layer {layer}: Accuracy = {perf['accuracy']:.4f}")

    else:
        multi_extractor = MultiModelExtractor(models_to_process)
        multi_extractor.process_all_models(
            data_path=args.data_path,
            max_samples=args.max_samples,
            neuron_ratio=args.neuron_ratio,
            top_k_layers=args.top_k_layers
        )


if __name__ == "__main__":
    main()
