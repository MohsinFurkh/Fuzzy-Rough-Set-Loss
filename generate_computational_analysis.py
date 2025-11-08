import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import psutil
import GPUtil
import pandas as pd
from pynvml import *
import time
from typing import Dict, List, Tuple, Optional, Any

# Add the current directory to the path to import FRS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the exact model architecture and dependencies from FRS
from FRS import build_unet, dice_coefficient, iou_coef, tverskyloss, dice_loss

# Configuration
IMG_SIZE = 128
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
NUM_RUNS = 5  # Number of runs to average for stable measurements
 # Define dataset paths and their display names
image_dir_path = {
        'BUSI': "D:\\Datasets\\BUSI Dataset\\images",
        'HAM10000': "D:\\Datasets\\Skin Dataset\\skin_images.npy",
        'Kvasir': "D:\\Datasets\\Kvasir-SEG\\images",
        'Brain MRI': "D:\\Datasets\\MRI Dataset\\MRI_images.npy",
        'Chest CT': "D:\\Datasets\\Chest CT\\images.npy"
    }

# Set memory growth to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def build_model():
    """Build the model with the exact architecture used in training."""
    # Create model with the same architecture as in FRS.py
    model = build_unet()
    
    # Compile with dummy optimizer and loss (we're just doing inference)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model

def load_model(model_path):
    """Load a model with the correct architecture and weights."""
    try:
        # First try to load as a complete model
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_coefficient': dice_coefficient,
                    'iou_coef': iou_coef,
                    'tverskyloss': tverskyloss,
                    'dice_loss': dice_loss
                },
                compile=False
            )
            print(f"Successfully loaded complete model from {model_path}")
            return model
        except Exception as e:
            print(f"Could not load as complete model: {e}")
            
        # If that fails, build the model and load weights
        print("Building model and loading weights...")
        model = build_model()
        
        # Try different weight loading approaches
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print(f"Successfully loaded weights into model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading weights with by_name=True: {e}")
            
        try:
            model.load_weights(model_path)
            print(f"Successfully loaded weights (strict) from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading weights strictly: {e}")
            
        print("Could not load model or weights.")
        return None
        
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None

# Function to load and preprocess images
def load_images(image_dir):
    images = []
    
    image_filenames = os.listdir(image_dir)
    for image_filename in image_filenames:
        # Load and preprocess the image
        image_path = os.path.join(image_dir, image_filename)
        image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
        image = img_to_array(image) / 255.0  # Normalize image
        images.append(image)
        
    
    return np.array(images)
def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    gpu_info = {}
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_info[f'gpu_{i}_used_mb'] = info.used / 1024**2
            gpu_info[f'gpu_{i}_total_mb'] = info.total / 1024**2
        nvmlShutdown()
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
    return gpu_info

def get_system_memory_usage() -> Dict[str, float]:
    """Get current system memory usage in MB."""
    mem = psutil.virtual_memory()
    return {
        'system_used_mb': (mem.total - mem.available) / 1024**2,
        'system_total_mb': mem.total / 1024**2,
        'system_available_mb': mem.available / 1024**2
    }

def measure_memory_usage(model: tf.keras.Model, data: np.ndarray, batch_size: int) -> Dict[str, float]:
    """Measure memory usage during model inference."""
    # Clear any existing Keras session and garbage collect
    K.clear_session()
    import gc
    gc.collect()
    
    # Get baseline memory
    baseline_gpu = get_gpu_memory_usage()
    baseline_sys = get_system_memory_usage()
    
    # Run inference
    _ = model.predict(data[:batch_size], batch_size=batch_size, verbose=0)
    
    # Get memory after inference
    after_gpu = get_gpu_memory_usage()
    after_sys = get_system_memory_usage()
    
    # Calculate memory deltas
    memory_metrics = {}
    for k in baseline_gpu:
        if k in after_gpu:
            memory_metrics[f'{k}_delta'] = after_gpu[k] - baseline_gpu.get(k, 0)
    
    for k in baseline_sys:
        if k in after_sys:
            memory_metrics[f'{k}_delta'] = after_sys[k] - baseline_sys.get(k, 0)
    
    return memory_metrics

def measure_inference_time(model: tf.keras.Model, data: np.ndarray, batch_size: int, 
                         num_runs: int = NUM_RUNS) -> Tuple[float, Dict[str, float]]:
    """Measure average inference time and memory usage for a given batch size.
    
    Returns:
        tuple: (mean_time, memory_metrics)
    """
    import time
    import numpy as np
    
    # Warm-up run
    _ = model.predict(data[:batch_size], batch_size=batch_size, verbose=0)
    
    # Ensure we have enough data for all runs
    num_batches = len(data) // batch_size
    if num_batches < 1:
        raise ValueError(f"Not enough data for batch size {batch_size}")
    
    # Use the first num_runs batches, or all batches if there are fewer
    num_runs = min(num_runs, num_batches)
    
    times = []
    memory_metrics_list = []
    
    for i in range(num_runs):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        
        # Measure memory before inference
        if i == 0:  # Only measure memory on first run to reduce overhead
            memory_metrics = measure_memory_usage(model, batch_data, batch_size)
            memory_metrics_list.append(memory_metrics)
        
        # Measure inference time
        start_time = time.perf_counter()
        _ = model.predict(batch_data, batch_size=batch_size, verbose=0)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate mean and standard deviation
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Average memory metrics if we have multiple measurements
    avg_memory_metrics = {}
    if memory_metrics_list:
        for k in memory_metrics_list[0]:
            avg_memory_metrics[k] = np.mean([m.get(k, 0) for m in memory_metrics_list])
    
    print(f"  - Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
    for k, v in avg_memory_metrics.items():
        if 'delta' in k and 'mb' in k.lower():
            print(f"  - {k}: {v:.2f} MB")
    
    return mean_time, avg_memory_metrics

def plot_computational_analysis(results, output_dir='.'):
    """
    Plot computational analysis results with memory metrics.
    
    Args:
        results: Dictionary containing inference stats for each model
        output_dir: Directory to save the output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Subplot 1: Inference Time vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    # Subplot 2: GPU Memory Usage vs Batch Size
    ax2 = fig.add_subplot(gs[0, 1])
    # Subplot 3: Throughput vs Batch Size
    ax3 = fig.add_subplot(gs[1, 0])
    # Subplot 4: Memory Efficiency (Throughput/Memory)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Data for the summary table
    summary_data = []
    
    # Process results for plotting
    processed_results = {}
    for model_name, data in results.items():
        stats = data.get('stats', [])
        if not stats:
            continue
            
        # Extract data for plotting
        batch_sizes = [s['batch_size'] for s in stats]
        inference_times = [s['mean_time'] for s in stats]
        throughputs = [s['batch_size']/s['mean_time'] if s['mean_time'] > 0 else 0 for s in stats]
        gpu_memory = [s['memory_metrics'].get('gpu_0_used_mb_delta', 0) for s in stats]
        
        # Store processed data
        processed_results[model_name] = {
            'batch_sizes': batch_sizes,
            'inference_times': inference_times,
            'throughputs': throughputs,
            'gpu_memory': gpu_memory
        }
        
        # Calculate memory efficiency (images per second per MB)
        memory_efficiency = [t/m if m > 0 else 0 for t, m in zip(throughputs, gpu_memory)]
        
        # Add to summary data
        if len(inference_times) > 0 and len(throughputs) > 0:
            max_throughput_idx = np.argmax(throughputs)
            summary_data.append({
                'Model': model_name,
                'Min Latency (s)': f"{min(inference_times):.4f}",
                'Max Throughput (img/s)': f"{max(throughputs):.2f}",
                'Optimal Batch Size': batch_sizes[max_throughput_idx],
                'Max GPU Memory (MB)': f"{max(gpu_memory):.1f}",
                'Peak Memory Efficiency': f"{max(memory_efficiency):.2f} img/s/MB"
            })
    
    # Process results for each model
    processed_results = {}
    for model_name, data in results.items():
        if 'stats' not in data or not data['stats']:
            continue
            
        # Extract data for plotting
        batch_sizes = [s['batch_size'] for s in data['stats']]
        inference_times = [s['mean_time'] for s in data['stats']]
        throughputs = [s['throughput'] for s in data['stats']]
        gpu_memory = [s['memory_metrics'].get('gpu_0_used_mb_delta', 0) for s in data['stats']]
        
        # Store processed data
        processed_results[model_name] = {
            'batch_sizes': batch_sizes,
            'inference_times': inference_times,
            'throughputs': throughputs,
            'gpu_memory': gpu_memory
        }
    
    # Plot the data
    for (model_name, data), color in zip(processed_results.items(), colors):
        try:
            batch_sizes = np.array(data['batch_sizes'])
            inference_times = np.array(data['inference_times'])
            throughputs = np.array(data['throughputs'])
            gpu_memory = np.array(data['gpu_memory'])
            
            # Filter out any invalid data points
            valid_mask = ~np.isnan(inference_times) & ~np.isnan(throughputs) & ~np.isnan(gpu_memory)
            if not np.any(valid_mask):
                print(f"Skipping {model_name} - no valid data points")
                continue
                
            batch_sizes = batch_sizes[valid_mask]
            inference_times = inference_times[valid_mask]
            throughputs = throughputs[valid_mask]
            gpu_memory = gpu_memory[valid_mask]
            
            # Calculate memory efficiency (images per second per MB)
            memory_efficiency = np.divide(throughputs, gpu_memory, out=np.zeros_like(throughputs, dtype=float), where=gpu_memory>0)
            
            # Plot 1: Inference Time vs Batch Size
            ax1.plot(batch_sizes, inference_times, 'o-', color=color, label=model_name)
            
            # Plot 2: GPU Memory Usage vs Batch Size
            ax2.plot(batch_sizes, gpu_memory, 'o-', color=color, label=model_name)
            
            # Plot 3: Throughput vs Batch Size
            ax3.plot(batch_sizes, throughputs, 'o-', color=color, label=model_name)
            
            # Plot 4: Memory Efficiency (Throughput/Memory)
            ax4.plot(batch_sizes, memory_efficiency, 'o-', color=color, label=model_name)
            
        except Exception as e:
            print(f"Error plotting {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Configure subplot 1: Inference Time vs Batch Size
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Inference Time (s)', fontsize=12)
    ax1.set_title('(a) Inference Time vs Batch Size', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Configure subplot 2: GPU Memory Usage
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax2.set_title('(b) GPU Memory Usage vs Batch Size', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Configure subplot 3: Throughput vs Batch Size
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Throughput (images/s)', fontsize=12)
    ax3.set_title('(c) Throughput vs Batch Size', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Configure subplot 4: Memory Efficiency
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Memory Efficiency (img/s/MB)', fontsize=12)
    ax4.set_title('(d) Memory Efficiency vs Batch Size', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the figure
    output_path = os.path.join(output_dir, 'computational_analysis_with_memory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save a summary table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'computational_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        
        # Create a formatted table for visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        table = ax.table(
            cellText=df_summary.values,
            colLabels=df_summary.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Save the table
        table_path = os.path.join(output_dir, 'computational_summary_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComputational analysis with memory metrics saved to: {os.path.abspath(output_path)}")
        print(f"Summary data saved to: {os.path.abspath(csv_path)}")
        print(f"Summary table saved to: {os.path.abspath(table_path)}")
    else:
        print("\nNo valid data to generate summary.")

def main():
    # Define model paths and their display names
    models = {
        'BUSI': "D:\\FRS Loss\\models\\best_model_BUSI.weights.h5",
        'HAM10000': 'D:\\FRS Loss\\models\\best_model_HAM10000.weights.h5',
        'Kvasir': 'D:\\FRS Loss\\models\\best_model_Kvasir.weights.h5',
        'Brain MRI': 'D:\\FRS Loss\\models\\best_model_brain_MRI.weights.h5',
        'Chest CT': 'D:\\FRS Loss\\models\\best_model_chest_CT.weights.h5'
    }
    
    # Clear any existing Keras session
    K.clear_session()
    # For TensorFlow 2.8.0, we'll let Keras handle GPU memory management
    
    results = {}
    
    # Test each model with its own test data
    results = {}
    for model_name, model_path in models.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing {model_name}...")
            print(f"{'='*50}")
            
            # Load the model
            model = load_model(model_path)
            if model is None:
                print(f"Skipping {model_name} due to loading error")
                continue
            
            # Load images
            if image_dir_path[model_name].endswith('.npy'):
                images = np.load(image_dir_path[model_name])
            else:
                images = load_images(image_dir_path[model_name])
            
            # Split data into train and test sets (80% train, 20% test)
            _, test_images = train_test_split(
                images, 
                test_size=0.2, 
                random_state=42
            )
            
            # Debug: Print initial shape and type
            print(f"Initial test images shape: {test_images.shape}, dtype: {test_images.dtype}")
            
            # Handle 5D input (samples, height, width, channels, 1)
            if len(test_images.shape) == 5:
                print("Reshaping 5D input to 4D...")
                # Remove the last dimension if it's 1
                if test_images.shape[-1] == 1:
                    test_images = np.squeeze(test_images, axis=-1)
            
            # Normalize to [0, 1] if needed
            if test_images.max() > 1.0:
                test_images = test_images.astype('float32') / 255.0
            
            # Ensure we have the right number of dimensions
            if len(test_images.shape) == 3:  # If (samples, height, width)
                # Add channel dimension
                test_images = np.expand_dims(test_images, axis=-1)
            
            # Handle color images (should be 4D at this point: samples, height, width, channels)
            if len(test_images.shape) == 4:
                if test_images.shape[-1] == 3:  # RGB
                    print("Converting RGB to grayscale...")
                    # Convert to grayscale using the standard formula
                    test_images = np.dot(test_images[...,:3], [0.2989, 0.5870, 0.1140])
                    # Add back the channel dimension
                    test_images = np.expand_dims(test_images, axis=-1)
                elif test_images.shape[-1] > 3:  # RGBA or other
                    print(f"Converting {test_images.shape[-1]}-channel image to grayscale...")
                    # Take only first 3 channels and convert to grayscale
                    test_images = np.dot(test_images[...,:3], [0.2989, 0.5870, 0.1140])
                    test_images = np.expand_dims(test_images, axis=-1)
            
            # Final shape check and squeeze if needed
            if len(test_images.shape) == 5 and test_images.shape[-1] == 1:
                test_images = np.squeeze(test_images, axis=-1)
            
            print(f"Final test images shape: {test_images.shape}, dtype: {test_images.dtype}")
            print(f"Value range: {test_images.min()} - {test_images.max()}")
            
            # Ensure we have the correct shape (samples, height, width, 1)
            if len(test_images.shape) != 4 or test_images.shape[-1] != 1:
                raise ValueError(f"Expected shape (samples, height, width, 1), but got {test_images.shape}")
            print(f"Total test samples: {len(test_images)}")
            
            # Initialize stats list for this model
            model_stats = []
            
            for batch_size in BATCH_SIZES:
                try:
                    print(f"\nTesting batch size: {batch_size}")
                    # Calculate how many complete batches we can make
                    num_batches = len(test_images) // batch_size
                    if num_batches < 1:
                        print(f"Skipping batch size {batch_size} - not enough test samples")
                        continue
                        
                    # Use the first complete batch for testing
                    batch_data = test_images[:batch_size]
                    
                    # Measure both time and memory
                    avg_time, memory_metrics = measure_inference_time(model, batch_data, batch_size)
                    throughput = batch_size / avg_time if avg_time > 0 else 0
                    
                    # Store stats for this batch size
                    model_stats.append({
                        'batch_size': batch_size,
                        'mean_time': avg_time,
                        'throughput': throughput,
                        'memory_metrics': memory_metrics
                    })
                    
                    # Print summary
                    print(f"Batch size {batch_size}:")
                    print(f"  - Time: {avg_time:.4f}s")
                    print(f"  - Throughput: {throughput:.2f} img/s")
                    for k, v in memory_metrics.items():
                        if 'delta' in k and 'mb' in k.lower():
                            print(f"  - {k}: {v:.2f} MB")
                            
                except Exception as e:
                    print(f"Error processing batch size {batch_size}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Store all stats for this model
            results[model_name] = {
                'stats': model_stats
            }
            
            # Clean up to free memory
            del model
            K.clear_session()
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print(f"\nCompleted {model_name}\n")
    
    # Plot results if we have valid data
    if any('stats' in data and data['stats'] for data in results.values()):
        plot_computational_analysis(results)
    else:
        print("\nNo valid data to plot. Check if any models were processed successfully.")

if __name__ == "__main__":
    # Print TensorFlow version and GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Clear any existing Keras session
    tf.keras.backend.clear_session()
    
    # Run the main function
    main()
