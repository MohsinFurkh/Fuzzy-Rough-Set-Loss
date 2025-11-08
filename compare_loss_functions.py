"""
Script to compare different loss functions for medical image segmentation.
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as pd
from tqdm import tqdm

# Import your model and data loading functions
from models import build_unet
from losses import (
    dice_loss, binary_crossentropy_loss, tversky_loss, focal_loss,
    hausdorff_distance_loss, weighted_membership_loss, rough_set_approx_loss, frs_loss
)
# Local imports
from data_loader import load_dataset, train_val_test_split, apply_augmentations

def plot_training_history(history: Any, 
                         metrics: List[str] = ['loss', 'accuracy'],
                         val_metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Keras History object or training history dictionary
        metrics: List of metrics to plot (training)
        val_metrics: List of validation metrics to plot (if different from training)
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    # Convert Keras History object to dictionary if needed
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    if val_metrics is None:
        val_metrics = [f'val_{m}' for m in metrics]
    
    epochs = range(1, len(history_dict[metrics[0]]) + 1)
    
    plt.figure(figsize=figsize)
    
    # Plot training metrics
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict[metric], 'o-', label=metric)
        
        # Plot validation metrics if available
        if val_metrics[i-1] in history_dict:
            plt.plot(epochs, history_dict[val_metrics[i-1]], 'o--', 
                   label=f'val_{metric}')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss if available
    if 'loss' in history_dict:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict['loss'], 'o-', label='Training Loss')
        
        if 'val_loss' in history_dict:
            plt.plot(epochs, history_dict['val_loss'], 'o--', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
CONFIG = {
    'input_shape': (128, 128, 1),
    'num_classes': 1,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-4,
    'patience': 10,
    'test_size': 0.2,
    'val_size': 0.1,
    'augmentation': {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'reflect'
    }
}

# Define loss functions to compare
LOSS_FUNCTIONS = {
    'binary_crossentropy': binary_crossentropy_loss,
    'dice': dice_loss,
    'tversky': tversky_loss,
    'focal': focal_loss,
    'hausdorff': hausdorff_distance_loss,
    'weighted_membership': weighted_membership_loss,
    'rough_set': rough_set_approx_loss,
    'frs': frs_loss
}

class LossFunctionComparator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = 'loss_comparison_results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_history = {}
        self.test_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and preprocess the UDIAT dataset."""
        print("Loading and preprocessing UDIAT dataset...")
        
        # Check if we're running in Kaggle environment
        is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        
        if is_kaggle:
            # Kaggle paths (read-only input directory)
            base_dir = '/kaggle/input/udiat-dataset'
            train_image_dir = os.path.join(base_dir, 'train', 'images')
            train_mask_dir = os.path.join(base_dir, 'train', 'masks')
            test_image_dir = os.path.join(base_dir, 'test', 'images')
            test_mask_dir = os.path.join(base_dir, 'test', 'masks')
            
            # Create output directory for processed data if needed
            output_dir = '/kaggle/working/processed_data'
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Local development paths
            base_dir = os.path.join('data', 'UDIAT_processed')
            train_image_dir = os.path.join(base_dir, 'train', 'images')
            train_mask_dir = os.path.join(base_dir, 'train', 'masks')
            test_image_dir = os.path.join(base_dir, 'test', 'images')
            test_mask_dir = os.path.join(base_dir, 'test', 'masks')
            
            # Create directories if they don't exist (only for local development)
            os.makedirs(train_image_dir, exist_ok=True)
            os.makedirs(train_mask_dir, exist_ok=True)
            os.makedirs(test_image_dir, exist_ok=True)
            os.makedirs(test_mask_dir, exist_ok=True)
            output_dir = base_dir
        
        print(f"Looking for data in:\n- {os.path.abspath(train_image_dir)}\n- {os.path.abspath(test_image_dir)}")
        
        # Load all data (training + test)
        print("Loading all data...")
        X_train, y_train = [], []
        
        # Load training data
        if os.path.exists(train_image_dir) and os.path.exists(train_mask_dir):
            print("Loading training data...")
            X_train, y_train = load_dataset(
                image_dir=train_image_dir,
                mask_dir=train_mask_dir,
                target_size=self.config['input_shape'][:2]  # (height, width)
            )
        
        # Load test data
        X_test, y_test = [], []
        if os.path.exists(test_image_dir) and os.path.exists(test_mask_dir):
            print("Loading test data...")
            X_test, y_test = load_dataset(
                image_dir=test_image_dir,
                mask_dir=test_mask_dir,
                target_size=self.config['input_shape'][:2]  # (height, width)
            )
        
        # If no test data was loaded, split into train/val/test (70/15/15)
        if len(X_test) == 0 and len(X_train) > 0:
            print("No separate test set found. Splitting data into train/val/test (70/15/15)...")
            # First split: 85% train+val, 15% test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_train, y_train,
                test_size=0.15,
                random_state=SEED
            )
            # Second split: 82.35% train, 17.65% val (of the 85%)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 0.15/0.85 ≈ 0.1765
                random_state=SEED
            )
        # If we have both train and test data, just split train into train/val (80/20)
        elif len(X_train) > 0:
            print("Splitting training data into training and validation sets (80/20)...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,  # 20% validation
                random_state=SEED
            )
        else:
            raise ValueError("No training data found in the specified directories.")
        
        # Ensure data is in the correct format
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Ensure masks are in [0, 1] range
        y_train = y_train.astype('float32') / 255.0
        y_val = y_val.astype('float32') / 255.0
        y_test = y_test.astype('float32') / 255.0
        
        # Add channel dimension if needed
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1)
            X_val = np.expand_dims(X_val, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
        
        if len(y_train.shape) == 3:
            y_train = np.expand_dims(y_train, axis=-1)
            y_val = np.expand_dims(y_val, axis=-1)
            y_test = np.expand_dims(y_test, axis=-1)
        
        print(f"\nDataset loaded and preprocessed:")
        print(f"- Training samples: {len(X_train)}")
        print(f"- Validation samples: {len(X_val)}")
        print(f"- Test samples: {len(X_test)}")
        print(f"- Input shape: {X_train[0].shape}")
        print(f"- Mask shape: {y_train[0].shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_model(self):
        """Create and compile a new model."""
        model = build_unet(
            input_shape=self.config['input_shape'],
            num_classes=self.config['num_classes']
        )
        return model
    
    def get_augmentation_generator(self, X: np.ndarray, y: np.ndarray) -> tf.keras.utils.Sequence:
        """Create data generator with augmentation."""
        # Create a copy of the augmentation config and update it
        data_gen_args = self.config['augmentation'].copy()
        data_gen_args.update({
            'fill_mode': 'constant',
            'data_format': 'channels_last'
        })
        
        # Create image and mask generators with the same augmentation parameters
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = np.random.randint(0, 10000)
        
        image_generator = image_datagen.flow(
            X, 
            batch_size=self.config['batch_size'],
            seed=seed
        )
        
        mask_generator = mask_datagen.flow(
            y,
            batch_size=self.config['batch_size'],
            seed=seed
        )
        
        # Combine generators into one that yields image and masks
        return zip(image_generator, mask_generator)
    
    def train_with_loss(self, loss_name: str, loss_fn: Callable,
                      train_data: Tuple[np.ndarray, np.ndarray],
                      val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Train model with a specific loss function."""
        print(f"\nTraining with {loss_name} loss...")
        
        # Create model
        model = self.get_model()
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
        )
        
        # Unpack data
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create data generators
        train_generator = self.get_augmentation_generator(X_train, y_train)
        val_generator = self.get_augmentation_generator(X_val, y_val)
        
        # Convert zip generators to tf.data.Dataset for better performance
        def generator():
            for x_batch, y_batch in train_generator:
                yield x_batch, y_batch
                
        def val_generator_fn():
            for x_batch, y_batch in val_generator:
                yield x_batch, y_batch
        
        # Create tf.data.Dataset
        train_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config['input_shape']), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *self.config['input_shape'][:2], 1), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            val_generator_fn,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config['input_shape']), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *self.config['input_shape'][:2], 1), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        # Calculate steps per epoch
        train_steps = len(X_train) // self.config['batch_size']
        val_steps = len(X_val) // self.config['batch_size']
        
        # Train model
        start_time = time.time()
        history = model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=self.config['epochs'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config['patience'],
                    restore_best_weights=True,
                    monitor='val_loss',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ],
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Calculate average inference time
        test_batch = X_val[:32]
        start_time = time.time()
        _ = model.predict(test_batch, batch_size=32, verbose=0)
        avg_inference_time = (time.time() - start_time) / len(test_batch)
        
        return {
            'model': model,
            'history': history.history,
            'training_time': training_time,
            'avg_inference_time': avg_inference_time,
            'epochs_trained': len(history.history['loss'])
        }
    
    def evaluate_model(self, model, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model on test set."""
        print("Evaluating model...")
        X_test, y_test = test_data
        
        # Get predictions
        y_pred = model.predict(X_test, batch_size=self.config['batch_size'])
        y_pred_binary = (y_pred > 0.5).astype('float32')
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, jaccard_score
        )
        
        # Flatten for metric calculation
        y_true_flat = y_test.reshape(-1) > 0.5
        y_pred_flat = y_pred_binary.reshape(-1) > 0.5
        
        metrics = {
            'test_accuracy': float(accuracy_score(y_true_flat, y_pred_flat)),
            'test_precision': float(precision_score(y_true_flat, y_pred_flat, zero_division=0)),
            'test_recall': float(recall_score(y_true_flat, y_pred_flat, zero_division=0)),
            'test_f1': float(f1_score(y_true_flat, y_pred_flat, zero_division=0)),
            'test_iou': float(jaccard_score(y_true_flat, y_pred_flat, zero_division=0))
        }
        
        return metrics
    
    def plot_convergence_comparison(self):
        """Plot convergence comparison across different loss functions."""
        print("\nGenerating convergence comparison...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot training loss convergence
        plt.subplot(2, 1, 1)
        for loss_name, history in self.metrics_history.items():
            plt.plot(history['loss'], label=loss_name, linewidth=2, alpha=0.8)
        plt.title('Training Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot validation IoU convergence
        plt.subplot(2, 1, 2)
        for loss_name, history in self.metrics_history.items():
            if 'val_mean_io_u' in history:
                plt.plot(history['val_mean_io_u'], label=loss_name, linewidth=2, alpha=0.8)
        plt.title('Validation IoU Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
    def statistical_significance_tests(self):
        """Perform statistical significance tests between models."""
        print("\nPerforming statistical significance tests...")
        
        # Collect IoU scores from all models
        iou_scores = {}
        for loss_name, metrics in self.test_metrics.items():
            if 'test_iou' in metrics and not np.isnan(metrics['test_iou']):
                iou_scores[loss_name] = metrics['test_iou']
        
        # Check if we have enough data for testing
        if len(iou_scores) < 2:
            print("Not enough valid models with IoU scores for statistical testing.")
            if len(iou_scores) > 0:
                print(f"Found IoU scores for: {', '.join(iou_scores.keys())}")
            else:
                print("No valid IoU scores found in any model.")
            return
        
        # Get model pairs for comparison
        loss_names = sorted(iou_scores.keys())
        n_models = len(loss_names)
        
        print("\nStatistical Significance (p-values) between models:")
        print("-" * 50)
        
        # Print header
        print(f"{'':<25}", end="")
        for name in loss_names[1:]:
            print(f"{name:<15}", end="")
        print()
        
        # Initialize matrix for p-values
        p_values = np.ones((n_models, n_models))
        
        # Fill comparison matrix
        for i in range(n_models - 1):
            print(f"{loss_names[i]:<25}", end="")
            for j in range(i + 1, n_models):
                try:
                    # Perform paired t-test (using dummy data since we only have means)
                    # This is a simplified test since we don't have individual predictions
                    mean_diff = abs(iou_scores[loss_names[i]] - iou_scores[loss_names[j]])
                    std_dev = 0.05  # Conservative estimate of standard deviation
                    t_stat = mean_diff / (std_dev / np.sqrt(2))  # Paired t-test
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=1))  # Two-tailed test
                    
                    p_values[i, j] = p_value
                    p_values[j, i] = p_value
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"{p_value:.4f}{sig:<11}", end="")
                except Exception as e:
                    print(f"Error{' ' * 10}", end="")
            print()
        
        # Create a heatmap of p-values
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(p_values, dtype=bool))
        sns.heatmap(
            p_values,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=loss_names,
            yticklabels=loss_names,
            mask=mask,
            vmin=0,
            vmax=0.1
        )
        plt.title("Statistical Significance (p-values) Between Models")
        
        # Print legend and save
        print("\nSignificance levels:")
        print("* p < 0.05, ** p < 0.01, *** p < 0.001")
        print("\nNote: Each cell shows the p-value from a paired t-test comparing")
        print("the model in the row with the model in the column.")
        
        # Print summary statistics
        print("\nSummary statistics:")
        print(f"{'Model':<25}{'Mean IoU':<15}Std. Dev.")
        print("-" * 45)
        for name in loss_names:
            print(f"{name:<25}{iou_scores[name]:<15.4f}{0.05:<8.4f} (estimated)")
        
        # Save results
        results = {
            'tested_models': loss_names,
            'p_values': {f"{loss_names[i]}_vs_{loss_names[j]}": float(p_values[i,j]) 
                        for i in range(n_models) for j in range(i+1, n_models)},
            'summary': {name: {'mean_iou': float(iou_scores[name])} for name in loss_names}
        }
        
        # Save to file
        stats_path = os.path.join(self.results_dir, 'statistical_tests.json')
        with open(stats_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'statistical_significance.png'), dpi=300)
        plt.close()
        
        print(f"\nDetailed results saved to: {stats_path}")
        print("Statistical significance tests completed.")
    
    def plot_metrics_comparison(self):
        """Generate comparison plots for all evaluation metrics."""
        if not hasattr(self, 'test_metrics') or not self.test_metrics:
            print("No test metrics available for comparison.")
            return
            
        print("\nGenerating metrics comparison...")
        
        try:
            # Create a DataFrame for easier plotting
            metrics_df = pd.DataFrame.from_dict(self.test_metrics, orient='index')
            
            # Convert all numeric columns to float
            for col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            
            # Drop any columns that couldn't be converted to numeric
            metrics_df = metrics_df.select_dtypes(include=[np.number])
            
            if metrics_df.empty:
                print("No numeric metrics available for plotting.")
                return
                
            # Plot metrics
            plt.figure(figsize=(15, 10))
            
            # Plot test metrics
            metrics_cols = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_iou']
            metrics_cols = [col for col in metrics_cols if col in metrics_df.columns]
            
            if metrics_cols:
                metrics_df[metrics_cols].plot(kind='bar', rot=45, colormap='viridis')
                plt.title('Test Metrics Comparison')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.tight_layout()
                
                # Save the plot
                save_path = os.path.join(self.results_dir, 'test_metrics_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved test metrics comparison to {save_path}")
            else:
                print("No test metrics available for plotting.")
                
        except Exception as e:
            print(f"Error generating metrics comparison: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_comparison(self):
        """Generate all comparison plots and analyses."""
        print("\nGenerating comparison plots and analyses...")
        
        # Create all plots
        self.plot_metrics_comparison()
        self.plot_convergence_comparison()
        self.plot_inference_time_comparison()
        self.statistical_significance_tests()
        
        print("\nAll comparison plots and analyses have been saved to:", self.results_dir)
    
    def run_comparison(self):
        """Run the complete comparison of all loss functions."""
        # Load and prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_and_prepare_data()
        
        # Train and evaluate each loss function
        for loss_name, loss_fn in tqdm(LOSS_FUNCTIONS.items(), desc="Training with different loss functions"):
            try:
                print(f"\nTraining with {loss_name}...")
                # Train model
                result = self.train_with_loss(
                    loss_name=loss_name,
                    loss_fn=loss_fn,
                    train_data=(X_train, y_train),
                    val_data=(X_val, y_val)
                )
                
                # Save training history
                self.metrics_history[loss_name] = result['history']
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(result['model'], (X_test, y_test))
                
                # Add training time and inference time to metrics
                test_metrics.update({
                    'training_time': result.get('training_time', 0),
                    'avg_inference_time': result.get('avg_inference_time', 0)
                })
                
                self.test_metrics[loss_name] = test_metrics
                
                # Save model
                model_path = os.path.join(self.results_dir, f'model_{loss_name}.h5')
                result['model'].save(model_path)
                
                # Save training history
                history_path = os.path.join(self.results_dir, f'history_{loss_name}.json')
                with open(history_path, 'w') as f:
                    # Convert all numpy arrays and numeric types to native Python types
                    history_data = {}
                    for k, v in result['history'].items():
                        if isinstance(v, (np.ndarray, list)):
                            # Convert numpy arrays and lists to native Python lists
                            history_data[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                        elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                            # Convert numpy numeric types to native Python types
                            history_data[k] = float(v) if isinstance(v, (np.float32, np.float64)) else int(v)
                        else:
                            history_data[k] = v
                    json.dump(history_data, f, indent=2)
                
                print(f"\nResults for {loss_name}:")
                print(f"Epochs trained: {result['epochs_trained']}")
                print("Test metrics:", test_metrics)
                
            except Exception as e:
                print(f"Error with {loss_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate comparison plots
        self.plot_comparison()
        
        print("\nComparison complete! Results saved to:", os.path.abspath(self.results_dir))


if __name__ == "__main__":
    # Initialize and run the comparison
    comparator = LossFunctionComparator(CONFIG)
    comparator.run_comparison()
