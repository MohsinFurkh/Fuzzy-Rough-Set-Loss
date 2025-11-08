import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime

from data_loader import load_dataset, create_dataset
from model import get_model, get_loss_function
from metrics import SegmentationMetrics

def get_default_args():
    # Define default arguments
    class Args:
        def __init__(self):
            self.data_dir = "D:\\Datasets\\BUSI Dataset"  # Update this path to your dataset location
            self.batch_size = 8
            self.epochs = 20
            self.img_size = [128, 128]
            self.output_dir = 'results'
            self.loss = 'bce'  # Will be overridden in the training loop
            self.learning_rate = 1e-4
            self.gpu = 0
            self.seed = 42
    
    return Args()

def train_and_evaluate(loss_name, train_ds, val_ds, test_ds, args):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create output directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}_{loss_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize metrics
    metrics_calculator = SegmentationMetrics()
    test_metrics = []
    inference_times = []
    
    # Get loss function
    print(f"\nUsing {loss_name} loss...")
    
    # Get model with the specified loss function
    model = get_model(
        input_shape=(*args.img_size, 1),
        loss=loss_name,  # Pass the loss name directly and let get_model handle it
        learning_rate=args.learning_rate
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(run_dir, 'best_model.h5'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, 'training_log.csv'),
            append=True
        )
    ]
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time = (time.time() - start_time) / 60  # in minutes
    
    # Load best weights
    model.load_weights(os.path.join(run_dir, 'best_model.h5'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    
    # Calculate additional metrics
    print("\nCalculating metrics on test set...")
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for x, y in tqdm(test_ds, desc="Processing test set"):
        y_true.extend(y.numpy())
        y_pred_batch = model.predict(x, verbose=0)
        y_pred.extend(y_pred_batch)
        
        # Calculate metrics for this batch
        batch_metrics = metrics_calculator.get_metrics(y.numpy(), y_pred_batch)
        test_metrics.append(batch_metrics)
        
        # Calculate inference time for this batch
        batch_time = (time.time() - start_time) / len(x)  # Per image
        inference_times.append(batch_time)
        start_time = time.time()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Aggregate metrics
    results = {
        'loss': loss_name,
        'dice': np.mean([m['dice'] for m in test_metrics]),
        'iou': np.mean([m['iou'] for m in test_metrics]),
        'hd95': np.mean([m['hd95'] for m in test_metrics]),
        'precision': np.mean([m['precision'] for m in test_metrics]),
        'recall': np.mean([m['recall'] for m in test_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in test_metrics]),
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'inference_time': np.mean(inference_times) if inference_times else 0,
        'epochs_trained': len(history.history['loss'])
    }
    
    return results

def main():
    # Get default arguments
    args = get_default_args()
    
    # Set random seeds for reproducibility
    tf.keras.utils.set_random_seed(args.seed)
    
    print("Using the following settings:")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.img_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using GPU: {args.gpu}")
    print(f"Random seed: {args.seed}")
    print("\nTraining with the following loss functions:", ['bce', 'dice', 'tversky', 'focal', 'frs'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(
        args.data_dir,
        img_size=tuple(args.img_size)
    )
    
    # Create datasets
    train_ds = create_dataset(x_train, y_train, batch_size=args.batch_size, augment=True)
    val_ds = create_dataset(x_val, y_val, batch_size=args.batch_size, augment=False)
    test_ds = create_dataset(x_test, y_test, batch_size=args.batch_size, augment=False)
    
    # Loss functions to compare
    loss_functions = ['bce', 'dice', 'tversky', 'focal', 'frs']
    
    # Define the metrics we want to track
    metrics_columns = ['loss_function', 'dice', 'iou', 'hd95', 'test_loss', 'test_accuracy', 'training_time']
    
    # Initialize results list with default values
    all_results = {col: [] for col in metrics_columns}
    
    # Train and evaluate models
    for loss_name in loss_functions:
        try:
            print(f"\n{'='*50}")
            print(f"Training with {loss_name.upper()} loss function")
            print(f"{'='*50}")
            
            # Train and evaluate model
            results = train_and_evaluate(loss_name, train_ds, val_ds, test_ds, args)
            
            # Append results
            all_results['loss_function'].append(loss_name)
            all_results['dice'].append(results.get('dice', np.nan))
            all_results['iou'].append(results.get('iou', np.nan))
            all_results['hd95'].append(results.get('hd95', np.nan))
            all_results['test_loss'].append(results.get('test_loss', np.nan))
            all_results['test_accuracy'].append(results.get('test_accuracy', np.nan))
            all_results['training_time'].append(results.get('training_time', np.nan))
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
            
        except Exception as e:
            print(f"Error training with {loss_name}: {str(e)}")
            # Add NaN values for failed runs
            all_results['loss_function'].append(loss_name)
            for col in metrics_columns[1:]:
                all_results[col].append(np.nan)
            
            # Save intermediate results even if some runs fail
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
            continue
    
    # Create final DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print final comparison
    print("\n" + "="*50)
    print("Final Comparison of All Loss Functions")
    print("="*50)
    
    if not results_df.empty:
        # Calculate mean values for each metric
        metrics_summary = results_df[metrics_columns[1:]].describe().loc[['mean', 'std', 'min', 'max']]
        
        print("\nResults Summary:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-" * 80)
        
        for metric in metrics_columns[1:]:  # Skip 'loss_function' column
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            print(f"{metric:<20} {mean_val:<15.4f} {std_val:<15.4f} {min_val:<15.4f} {max_val:<15.4f}")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f'final_results_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        
        # Save summary to a separate file
        summary_file = os.path.join(args.output_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write("Final Comparison of All Loss Functions\n")
            f.write("="*50 + "\n\n")
            f.write(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}\n")
            f.write("-" * 80 + "\n")
            for metric in metrics_columns[1:]:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                f.write(f"{metric:<20} {mean_val:<15.4f} {std_val:<15.4f} {min_val:<15.4f} {max_val:<15.4f}\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    else:
        print("\nNo results to display. All training runs failed.")

if __name__ == "__main__":
    main()
