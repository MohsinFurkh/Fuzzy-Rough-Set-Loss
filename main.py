"""
Main script for training and evaluating the segmentation model.
"""
import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf

# Local imports
from models import build_unet
from data_loader import load_dataset, create_data_generator
from losses import get_loss_function, LOSS_FUNCTIONS
from metrics import calculate_metrics
from train import SegmentationTrainer
from visualization import (
    plot_image_mask_prediction,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_reliability_diagram,
    plot_uncertainty_error_correlation,
    plot_boundary_uncertainty,
    plot_inference_time_comparison,
    plot_convergence_comparison,
    plot_metrics_comparison,
    plot_uncertainty_map
)
from config import (
    MODEL_DIR, RESULTS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, 
    PATIENCE, N_SPLITS, INPUT_SHAPE, NUM_CLASSES
)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation model.')
    
    # Data arguments
    parser.add_argument('--train-image-dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--train-mask-dir', type=str, required=True,
                        help='Directory containing training masks')
    parser.add_argument('--val-image-dir', type=str, default=None,
                        help='Directory containing validation images')
    parser.add_argument('--val-mask-dir', type=str, default=None,
                        help='Directory containing validation masks')
    parser.add_argument('--test-image-dir', type=str, default=None,
                        help='Directory containing test images')
    parser.add_argument('--test-mask-dir', type=str, default=None,
                        help='Directory containing test masks')
    
    # Model arguments
    parser.add_argument('--input-shape', type=int, nargs=3, default=INPUT_SHAPE,
                        help=f'Input shape as height, width, channels (default: {INPUT_SHAPE})')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help=f'Number of output classes (default: {NUM_CLASSES})')
    parser.add_argument('--model-name', type=str, default='segmentation_model',
                        help='Name for saving the model')
    
    # Training arguments
    parser.add_argument('--loss', type=str, default='dice_loss',
                        choices=list(LOSS_FUNCTIONS.keys()),
                        help='Loss function to use for training')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--early-stopping-patience', type=int, default=PATIENCE,
                        help=f'Patience for early stopping (default: {PATIENCE})')
    
    # Cross-validation
    parser.add_argument('--cross-validate', action='store_true',
                        help='Perform k-fold cross-validation')
    parser.add_argument('--n-splits', type=int, default=N_SPLITS,
                        help=f'Number of splits for cross-validation (default: {N_SPLITS})')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (default: None for CPU)')
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> None:
    """Set up the environment for training."""
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Set GPU if specified
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f'Using GPU {args.gpu}: {gpus[args.gpu]}')
            except RuntimeError as e:
                print(f'Error setting up GPU: {e}')
    
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load and preprocess the data."""
    print("Loading data...")
    
    # Load training data
    print(f"Loading training data from {args.train_image_dir} and {args.train_mask_dir}")
    x_train, y_train = load_dataset(
        args.train_image_dir,
        args.train_mask_dir,
        target_size=args.input_shape[:2]
    )
    
    # Load validation data if provided
    x_val, y_val = None, None
    if args.val_image_dir and args.val_mask_dir:
        print(f"Loading validation data from {args.val_image_dir} and {args.val_mask_dir}")
        x_val, y_val = load_dataset(
            args.val_image_dir,
            args.val_mask_dir,
            target_size=args.input_shape[:2]
        )
    
    # Load test data if provided
    x_test, y_test = None, None
    if args.test_image_dir and args.test_mask_dir:
        print(f"Loading test data from {args.test_image_dir} and {args.test_mask_dir}")
        x_test, y_test = load_dataset(
            args.test_image_dir,
            args.test_mask_dir,
            target_size=args.input_shape[:2]
        )
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Training set: {len(x_train)} samples")
    if x_val is not None:
        print(f"Validation set: {len(x_val)} samples")
    if x_test is not None:
        print(f"Test set: {len(x_test)} samples")
    
    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val) if x_val is not None else None,
        'test': (x_test, y_test) if x_test is not None else None
    }

def train_model(args: argparse.Namespace, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """Train the model."""
    print("\nBuilding model...")
    
    # Build the model
    model = build_unet(
        input_shape=args.input_shape,
        num_classes=args.num_classes
    )
    
    # Initialize trainer
    trainer = SegmentationTrainer(model=model)
    
    # Define metrics
    metrics = ['accuracy', 'mse', 'mae']
    
    # Compile the model
    trainer.compile_model(
        optimizer=args.optimizer,
        loss=args.loss,
        metrics=metrics
    )
    
    # Set up callbacks - only use validation metrics if we have validation data
    has_validation = data['val'] is not None and len(data['val'][0]) > 0
    monitor_metric = 'val_loss' if has_validation else 'loss'
    
    trainer.setup_callbacks(
        model_name=args.model_name,
        checkpoint_monitor=monitor_metric,
        early_stopping_patience=args.early_stopping_patience if has_validation else 0,
        reduce_lr_patience=args.early_stopping_patience // 2 if has_validation else 0,
        tensorboard_logdir=''  # Enable TensorBoard with default directory
    )
    
    # Train the model
    print("\nStarting training...")
    print(f"Training samples: {len(data['train'][0])}")
    if has_validation:
        print(f"Validation samples: {len(data['val'][0])}")
    
    history = trainer.train(
        train_data=data['train'],
        val_data=data['val'] if has_validation else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1
    )
    
    # Evaluate on test set if available
    test_metrics = {}
    if data['test'] is not None and len(data['test'][0]) > 0:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(data['test'])
        print("\nTest metrics:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Print test metrics
        print("\nTest metrics:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{args.model_name}.h5")
    trainer.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    return {
        'trainer': trainer,
        'history': history.history,
        'test_metrics': test_metrics,
        'model_path': model_path
    }

def cross_validate(args: argparse.Namespace, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """Perform k-fold cross-validation."""
    from sklearn.model_selection import KFold
    
    print(f"\nPerforming {args.n_splits}-fold cross-validation...")
    
    # Combine training and validation data for cross-validation
    x = np.concatenate([data['train'][0], data['val'][0]])
    y = np.concatenate([data['train'][1], data['val'][1]])
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    # Store results
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_dice': [],
        'val_iou': [],
    }
    
    fold = 1
    
    for train_idx, val_idx in kf.split(x):
        print(f"\nFold {fold}/{args.n_splits}")
        
        # Split data
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build and train model for this fold
        model = build_unet(
            input_shape=args.input_shape,
            num_classes=args.num_classes
        )
        
        trainer = SegmentationTrainer(model=model)
        trainer.compile_model(
            optimizer=args.optimizer,
            loss=args.loss,
            metrics=['accuracy']
        )
        
        trainer.setup_callbacks(
            model_name=f"{args.model_name}_fold_{fold}",
            early_stopping_patience=args.early_stopping_patience
        )
        
        history = trainer.train(
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1
        )
        
        # Store results
        results['train_loss'].append(history.history['loss'][-1])
        results['val_loss'].append(history.history['val_loss'][-1])
        results['val_accuracy'].append(history.history['val_accuracy'][-1])
        
        # Calculate additional metrics
        y_pred = trainer.predict(x_val)
        metrics = calculate_metrics(y_val, y_pred)
        results['val_dice'].append(metrics['dice'])
        results['val_iou'].append(metrics['iou'])
        
        fold += 1
    
    # Print cross-validation results
    print("\nCross-validation results:")
    for metric, values in results.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return results

def visualize_results(trainer: SegmentationTrainer, 
                     data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                     args: argparse.Namespace):
    """
    Generate comprehensive visualizations of the results including:
    - Sample predictions
    - Training history
    - Performance metrics
    - Uncertainty analysis
    - Model calibration
    """
    print("\nGenerating visualizations...")
    
    # Create results directory
    vis_dir = os.path.join(RESULTS_DIR, args.model_name, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get predictions for test set
    if data['test'] is not None:
        x_test, y_test = data['test']
        
        # Get model predictions and uncertainty (if available)
        if hasattr(trainer.model, 'predict_with_uncertainty'):
            y_pred, uncertainty = trainer.model.predict_with_uncertainty(x_test)
        else:
            y_pred = trainer.model.predict(x_test)
            uncertainty = None
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # 1. Basic visualizations
        print("Generating basic visualizations...")
        plot_image_mask_prediction(
            x_test[:5], y_test[:5], y_pred_binary[:5],
            save_path=os.path.join(vis_dir, 'sample_predictions.png')
        )
        
        # 2. Training history
        if hasattr(trainer, 'history'):
            plot_training_history(
                trainer.history,
                save_path=os.path.join(vis_dir, 'training_history.png')
            )
        
        # 3. Performance metrics
        print("Generating performance metrics...")
        plot_confusion_matrix(
            y_test.flatten() > 0.5,
            y_pred.flatten() > 0.5,
            save_path=os.path.join(vis_dir, 'confusion_matrix.png')
        )
        
        plot_roc_curve(
            y_test.flatten(),
            y_pred.flatten(),
            save_path=os.path.join(vis_dir, 'roc_curve.png')
        )
        
        plot_precision_recall_curve(
            y_test.flatten(),
            y_pred.flatten(),
            save_path=os.path.join(vis_dir, 'precision_recall_curve.png')
        )
        
        # 4. Uncertainty analysis
        print("\nPerforming uncertainty analysis...")
        
        # If no uncertainty from model, estimate it from model confidence
        if uncertainty is None:
            # Calculate uncertainty as the entropy of predictions
            # For binary classification, entropy = -[p*log(p) + (1-p)*log(1-p)]
            epsilon = 1e-10  # small constant to avoid log(0)
            p = np.clip(y_pred, epsilon, 1 - epsilon)
            uncertainty = - (p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)  # Normalized to [0,1]
        
        # Select a few samples for visualization
        num_samples = min(3, len(x_test))  # Visualize up to 3 samples
        
        for i in range(num_samples):
            sample_dir = os.path.join(vis_dir, f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Plot uncertainty map
            plot_uncertainty_map(
                image=x_test[i, ..., 0],  # Remove channel dimension for plotting
                prediction=y_pred[i, ..., 0],
                uncertainty=uncertainty[i, ..., 0] if uncertainty.ndim > 2 else uncertainty[i],
                save_path=os.path.join(sample_dir, 'uncertainty_map.png')
            )
            
            # Plot boundary uncertainty
            plot_boundary_uncertainty(
                image=x_test[i, ..., 0],
                mask=y_test[i, ..., 0],
                uncertainty=uncertainty[i, ..., 0] if uncertainty.ndim > 2 else uncertainty[i],
                save_path=os.path.join(sample_dir, 'boundary_uncertainty.png')
            )
        
        # Calculate errors for correlation plot
        errors = np.abs(y_test - y_pred_binary).flatten()
        
        # Plot uncertainty-error correlation for all test samples
        plot_uncertainty_error_correlation(
            uncertainties=uncertainty.flatten(),
            errors=errors,
            save_path=os.path.join(vis_dir, 'uncertainty_error_correlation.png')
        )
        
        # Plot reliability diagram
        plot_reliability_diagram(
            confidences=y_pred.flatten(),
            accuracies=(y_pred_binary.flatten() == y_test.flatten()).astype(float),
            save_path=os.path.join(vis_dir, 'reliability_diagram.png')
        )
        
        # 5. Model calibration
        print("Generating calibration plots...")
        confidences = y_pred.flatten()
        accuracies = (y_pred_binary.flatten() == y_test.flatten()).astype(float)
        
        plot_reliability_diagram(
            confidences=confidences,
            accuracies=accuracies,
            save_path=os.path.join(vis_dir, 'reliability_diagram.png')
        )
        
        # 6. Model comparison (if multiple models are available)
        if hasattr(trainer, 'model_history') and len(trainer.model_history) > 1:
            print("Generating model comparison plots...")
            plot_convergence_comparison(
                histories=trainer.model_history,
                save_path=os.path.join(vis_dir, 'model_convergence.png')
            )
            
            # Example metrics for comparison
            metrics_dict = {
                'Model1': {'Dice': 0.85, 'IoU': 0.75, 'Precision': 0.82},
                'Model2': {'Dice': 0.82, 'IoU': 0.72, 'Precision': 0.79}
            }
            plot_metrics_comparison(
                metrics_dict=metrics_dict,
                save_path=os.path.join(vis_dir, 'metrics_comparison.png')
            )
        
        print(f"\nAll visualizations saved to: {vis_dir}")
        print(f"Visualization directory: {os.path.abspath(vis_dir)}")

def save_results(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Save the results to disk."""
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, args.model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training history
    if 'history' in results:
        history_path = os.path.join(results_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in values] 
                     for k, values in results['history'].items()}, f, indent=2)
    
    # Save test metrics
    if 'test_metrics' in results and results['test_metrics']:
        metrics_path = os.path.join(results_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in results['test_metrics'].items()}, f, indent=2)
    
    # Save cross-validation results
    if 'cross_val_results' in results:
        cv_path = os.path.join(results_dir, 'cross_validation.json')
        with open(cv_path, 'w') as f:
            json.dump({k: [float(v) for v in values] 
                     for k, values in results['cross_val_results'].items()}, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")

def main() -> None:
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args)
    
    # Load data
    data = load_data(args)
    
    # Train model or perform cross-validation
    results = {}
    
    if args.cross_validate:
        results['cross_val_results'] = cross_validate(args, data)
    else:
        train_results = train_model(args, data)
        results.update(train_results)
        
        # Generate visualizations
        visualize_results(train_results['trainer'], data, args)
    
    # Save results
    save_results(results, args)
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
