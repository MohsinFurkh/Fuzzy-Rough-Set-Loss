"""
Visualization utilities for image segmentation results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd

# Set style for plots
plt.style.use('seaborn')
sns.set_style("whitegrid")
sns.set_palette("husl")

# Type aliases
ColorType = Union[str, Tuple[float, float, float, float]]

def plot_image_mask_prediction(images: ArrayLike, 
                            masks: ArrayLike, 
                            predictions: ArrayLike,
                            num_samples: int = 5,
                            figsize: Tuple[int, int] = (10, 15),
                            cmap: str = 'gray',
                            alpha: float = 0.5,
                            save_path: Optional[str] = None) -> None:
    """
    Plot images with their ground truth masks and predictions.
    
    Args:
        images: Input images (batch_size, height, width, channels)
        masks: Ground truth masks (batch_size, height, width, 1)
        predictions: Predicted masks (batch_size, height, width, 1)
        num_samples: Number of samples to display
        figsize: Figure size
        cmap: Colormap for images
        alpha: Transparency for mask overlay
        save_path: Path to save the figure (optional)
    """
    # Ensure we don't exceed the number of available samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    
    # If only one sample, ensure axes is 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Define titles for each column
    titles = ['Input Image', 'Ground Truth', 'Prediction']
    
    for i in range(num_samples):
        # Get current sample
        img = images[i].squeeze()
        mask = masks[i].squeeze()
        pred = predictions[i].squeeze()
        
        # Plot input image
        axes[i, 0].imshow(img, cmap=cmap)
        axes[i, 0].set_title(titles[0])
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(img, cmap=cmap)
        axes[i, 1].imshow(mask, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[i, 1].set_title(titles[1])
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(img, cmap=cmap)
        axes[i, 2].imshow(pred, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[i, 2].set_title(titles[2])
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

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
    
    # Get the number of epochs from the first available metric
    available_metrics = [m for m in metrics if m in history_dict]
    if not available_metrics:
        print("No valid metrics found in history.")
        return
        
    epochs = range(1, len(history_dict[available_metrics[0]]) + 1)
    
    plt.figure(figsize=figsize)
    
    # Plot training metrics
    for metric in metrics:
        if metric in history_dict:
            plt.plot(epochs, history_dict[metric], label=f'Training {metric}')
    
    # Plot validation metrics
    for metric in val_metrics:
        if metric in history_dict:
            plt.plot(epochs, history_dict[metric], label=f'Validation {metric[4:]}', linestyle='--')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_confusion_matrix(y_true: ArrayLike, 
                         y_pred: ArrayLike,
                         class_names: List[str] = ['Background', 'Foreground'],
                         normalize: bool = False,
                         cmap: str = 'Blues',
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap for the plot
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten() > 0.5, y_pred.flatten() > 0.5)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_roc_curve(y_true: ArrayLike, 
                  y_scores: ArrayLike,
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores/probabilities
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_scores.flatten())
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Format plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_precision_recall_curve(y_true: ArrayLike, 
                              y_scores: ArrayLike,
                              figsize: Tuple[int, int] = (8, 6),
                              save_path: Optional[str] = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores/probabilities
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_scores.flatten())
    average_precision = average_precision_score(y_true.flatten(), y_scores.flatten())
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot precision-recall curve
    plt.step(recall, precision, where='post', label=f'AP = {average_precision:.2f}')
    
    # Format plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_uncertainty_map(image: ArrayLike, 
                        prediction: ArrayLike, 
                        uncertainty: ArrayLike,
                        alpha: float = 0.7,
                        cmap: str = 'viridis',
                        figsize: Tuple[int, int] = (10, 15),
                        save_path: Optional[str] = None) -> None:
    """
    Plot uncertainty map alongside the image and prediction.
    
    Args:
        image: Input image
        prediction: Model prediction
        uncertainty: Uncertainty map
        alpha: Transparency for overlay
        cmap: Colormap for uncertainty
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot input image
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot prediction
    axes[1].imshow(prediction.squeeze(), cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Plot uncertainty
    im = axes[2].imshow(uncertainty.squeeze(), cmap=cmap)
    axes[2].set_title('Uncertainty Map')
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_learning_curves(train_scores: ArrayLike, 
                        val_scores: ArrayLike, 
                        metric_name: str = 'Score',
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves from cross-validation.
    
    Args:
        train_scores: Training scores from cross-validation
        val_scores: Validation scores from cross-validation
        metric_name: Name of the metric being plotted
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot training scores
    plt.plot(train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(
        range(len(train_mean)),
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='blue'
    )
    
    # Plot validation scores
    plt.plot(val_mean, label='Cross-validation score', color='red', marker='s')
    plt.fill_between(
        range(len(val_mean)),
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color='red'
    )
    
    # Format plot
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_feature_maps(model: Any, 
                     image: ArrayLike, 
                     layer_name: str,
                     num_features: int = 16,
                     figsize: Tuple[int, int] = (10, 15),
                     save_path: Optional[str] = None) -> None:
    """
    Plot feature maps from a specific layer of the model.
    
    Args:
        model: Keras model
        image: Input image (should be preprocessed)
        layer_name: Name of the layer to visualize
        num_features: Number of feature maps to display
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    # Create a model that outputs the specified layer's output
    layer_output = model.get_layer(layer_name).output
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer_output)
    
    # Get feature maps
    feature_maps = feature_extractor.predict(np.expand_dims(image, axis=0))
    
    # Get number of feature maps
    num_features = min(num_features, feature_maps.shape[-1])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot feature maps
    for i in range(num_features):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f'Feature maps from {layer_name}')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_3d_segmentation(image: ArrayLike, 
                        mask: ArrayLike,
                        alpha: float = 0.3,
                        cmap: str = 'jet',
                        figsize: Tuple[int, int] = (10, 15),
                        save_path: Optional[str] = None) -> None:
    """
    Create a 3D visualization of a segmentation mask overlaid on the image.
    
    Args:
        image: 2D or 3D image
        mask: 2D or 3D segmentation mask
        alpha: Transparency for the mask overlay
        cmap: Colormap for the mask
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure
    
    # Ensure mask is binary
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of the mask
    verts, faces, _, _ = measure.marching_cubes(mask_binary, 0.5)
    
    # Plot the surface
    mesh = ax.plot_trisurf(
        verts[:, 0], 
        verts[:, 1], 
        faces, 
        verts[:, 2],
        cmap=cmap,
        alpha=alpha,
        linewidth=0.1,
        antialiased=True
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Segmentation')
    
    # Add colorbar
    fig.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_reliability_diagram(confidences: ArrayLike,
                           accuracies: ArrayLike,
                           num_bins: int = 10,
                           figsize: Tuple[int, int] = (8, 8),
                           save_path: Optional[str] = None) -> None:
    """
    Plot a reliability diagram showing model calibration.
    
    Args:
        confidences: Model confidence scores (0-1)
        accuracies: Corresponding accuracies (0-1)
        num_bins: Number of confidence bins
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_acc[i] = np.mean(accuracies[mask])
            bin_conf[i] = np.mean(confidences[mask])
            bin_counts[i] = np.sum(mask)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Use the count of samples in each bin to determine the marker size
    sizes = bin_counts / np.max(bin_counts) * 100  # Scale for visibility
    plt.scatter(bin_conf, bin_acc, s=sizes, alpha=0.7, 
               label='Bins (size = #samples)')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_uncertainty_error_correlation(uncertainties: ArrayLike,
                                     errors: ArrayLike,
                                     figsize: Tuple[int, int] = (10, 5),
                                     save_path: Optional[str] = None) -> None:
    """
    Plot the correlation between model uncertainty and prediction error.
    
    Args:
        uncertainties: Array of uncertainty values
        errors: Array of error values
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(uncertainties, errors, alpha=0.3, s=5)
    plt.xlabel('Uncertainty')
    plt.ylabel('Error')
    plt.title('Uncertainty vs Error')
    
    # Add correlation coefficient
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    plt.text(0.05, 0.95, f'Corr: {corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # Hexbin plot for density visualization
    plt.subplot(1, 2, 2)
    plt.hexbin(uncertainties, errors, gridsize=30, cmap='viridis', bins='log')
    plt.colorbar(label='log10(N)')
    plt.xlabel('Uncertainty')
    plt.ylabel('Error')
    plt.title('Density of Uncertainty vs Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_boundary_uncertainty(image: ArrayLike,
                            mask: ArrayLike,
                            uncertainty: ArrayLike,
                            alpha: float = 0.7,
                            cmap: str = 'viridis',
                            figsize: Tuple[int, int] = (15, 5),
                            save_path: Optional[str] = None) -> None:
    """
    Plot image with boundary and uncertainty overlay.
    
    Args:
        image: Input image
        mask: Ground truth or predicted mask
        uncertainty: Uncertainty map
        alpha: Transparency for overlays
        cmap: Colormap for uncertainty
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Boundary overlay
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(mask.squeeze(), mode='inner')
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(boundaries, cmap='Reds', alpha=alpha)
    axes[1].set_title('Boundary')
    axes[1].axis('off')
    
    # Uncertainty overlay
    axes[2].imshow(image, cmap='gray')
    im = axes[2].imshow(uncertainty, cmap=cmap, alpha=alpha)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title('Uncertainty')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_inference_time_comparison(models: List[str],
                                inference_times: List[float],
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Optional[str] = None) -> None:
    """
    Plot comparison of inference times across different models.
    
    Args:
        models: List of model names
        inference_times: List of average inference times in seconds
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(models, inference_times, color='skyblue')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom')
    
    plt.xlabel('Models')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Model Inference Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_convergence_comparison(histories: Dict[str, Dict[str, List[float]]],
                              metric: str = 'loss',
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
    """
    Plot training convergence for multiple models.
    
    Args:
        histories: Dictionary of training histories
        metric: Metric to plot (e.g., 'loss', 'accuracy')
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    for model_name, history in histories.items():
        if f'val_{metric}' in history:
            plt.plot(history[metric], '--', label=f'{model_name} (train)')
            plt.plot(history[f'val_{metric}'], '-', label=f'{model_name} (val)')
        else:
            plt.plot(history[metric], '-', label=model_name)
    
    plt.title(f'Training Convergence ({metric})')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          metric_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple metrics across different models.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metric dicts as values
        metric_names: List of metric names to include (if None, include all)
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    if not metric_names:
        # Use all metrics present in the first model's metrics
        metric_names = list(next(iter(metrics_dict.values())).keys())
    
    models = list(metrics_dict.keys())
    num_metrics = len(metric_names)
    
    # Create subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    # If only one metric, axes won't be iterable
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        
        # Create bar plot for this metric
        bars = axes[i].bar(models, values, color=f'C{i}')
        axes[i].set_title(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_attention_map(image: ArrayLike,
                      attention_map: ArrayLike,
                      alpha: float = 0.5,
                      cmap: str = 'jet',
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None) -> None:
    """
    Plot an attention map overlaid on the original image.
    
    Args:
        image: Input image
        attention_map: Attention map
        alpha: Transparency for the attention map overlay
        cmap: Colormap for the attention map
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot attention map
    im = ax2.imshow(attention_map.squeeze(), cmap=cmap)
    ax2.set_title('Attention Map')
    ax2.axis('off')
    
    # Plot overlay
    ax3.imshow(image.squeeze(), cmap='gray')
    ax3.imshow(attention_map.squeeze(), cmap=cmap, alpha=alpha)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
