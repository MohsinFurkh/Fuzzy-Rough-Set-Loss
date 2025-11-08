"""
Custom loss functions for image segmentation tasks.
"""
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from typing import Callable, Dict, Union

# Type alias for loss functions
LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Calculate the Dice coefficient.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Dice loss for segmentation tasks.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def binary_crossentropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Standard binary crossentropy loss."""
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
                alpha: float = 0.3, beta: float = 0.7, 
                smooth: float = 1.0) -> tf.Tensor:
    """
    Tversky loss - generalization of Dice loss with adjustable FP/FN weights.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: False positive weight
        beta: False negative weight
        smooth: Smoothing factor
        
    Returns:
        Tversky loss value
    """
    # Flatten tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    # Calculate true positives, false positives, and false negatives
    true_pos = K.sum(y_true_flat * y_pred_flat)
    false_neg = K.sum(y_true_flat * (1 - y_pred_flat))
    false_pos = K.sum((1 - y_true_flat) * y_pred_flat)
    
    # Calculate Tversky index
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    
    # Return Tversky loss
    return 1 - tversky_index

def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
              gamma: float = 2.0, alpha: float = 0.25) -> tf.Tensor:
    """
    Focal loss for dealing with class imbalance.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        gamma: Focusing parameter (higher values focus more on hard examples)
        alpha: Weighting factor for positive class
        
    Returns:
        Focal loss value
    """
    # Flatten the tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    # Calculate binary cross entropy
    bce = K.binary_crossentropy(y_true_flat, y_pred_flat)
    
    # Calculate the focal loss
    p_t = (y_true_flat * y_pred_flat) + ((1 - y_true_flat) * (1 - y_pred_flat))
    alpha_factor = y_true_flat * alpha + (1 - y_true_flat) * (1 - alpha)
    modulating_factor = K.pow((1.0 - p_t), gamma)
    
    focal_loss = K.mean(alpha_factor * modulating_factor * bce, axis=-1)
    return focal_loss

def hausdorff_distance_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate an approximation of the Hausdorff distance loss using only TensorFlow operations.
    
    Note: This is a differentiable approximation of the Hausdorff distance.
    """
    # Ensure we're working with float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate boundary maps using sobel filters
    def get_boundary(x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = tf.expand_dims(x, -1)
            
        # Sobel filters for gradient in x and y directions
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Apply sobel filters
        gx = tf.nn.conv2d(x, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        gy = tf.nn.conv2d(x, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        # Calculate gradient magnitude
        boundary = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-6)
        return boundary
    
    # Get boundaries for true and predicted masks
    boundary_true = get_boundary(y_true)
    boundary_pred = get_boundary(y_pred)
    
    # Calculate distance transform approximation using max pooling
    def approx_distance_transform(x, num_iterations=3):
        # Initialize distance transform
        dt = tf.cast(x > 0.5, tf.float32)
        
        # Apply max pooling to propagate distances
        for _ in range(num_iterations):
            dt = tf.nn.max_pool2d(dt, ksize=3, strides=1, padding='SAME')
        
        # Invert and normalize
        dt = 1.0 - dt
        return dt
    
    # Calculate distance transforms
    dt_true = approx_distance_transform(y_true)
    dt_pred = approx_distance_transform(y_pred)
    
    # Calculate boundary errors
    boundary_error = tf.square(dt_true * boundary_pred + dt_pred * boundary_true)
    
    # Return mean boundary error
    return tf.reduce_mean(boundary_error)

def weighted_membership_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Weighted membership loss that emphasizes uncertain regions (close to 0.5).
    
    Args:
        y_true: Ground truth masks (values in [0, 1])
        y_pred: Predicted masks (logits)
        
    Returns:
        Weighted cross-entropy loss
    """
    # Calculate weights: higher weights for uncertain regions (close to 0.5)
    weights = tf.abs(y_true - 0.5) * 2

    # Calculate binary cross-entropy with logits for numerical stability
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # Apply weights without broadcasting issues
    weighted_loss = tf.multiply(bce, weights)
    return tf.reduce_mean(weighted_loss)


def rough_set_approx_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Rough set approximation loss that measures the discrepancy between
    lower and upper approximations of the predicted segmentation.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks (logits)
        
    Returns:
        Combined lower and upper approximation loss
    """
    # Apply sigmoid if logits are provided
    y_pred_prob = tf.sigmoid(y_pred) if y_pred.dtype != tf.float32 else y_pred
    
    lower_approx = tf.minimum(y_pred_prob, y_true)
    upper_approx = tf.maximum(y_pred_prob, y_true)
    
    lower_loss = tf.reduce_mean(tf.square(lower_approx - y_true))
    upper_loss = tf.reduce_mean(tf.square(upper_approx - y_true))
    
    return lower_loss + upper_loss


def frs_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
            alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
            **kwargs) -> tf.Tensor:
    """
    Enhanced Fuzzy Rough Set (FRS) based loss function that combines:
    1. Fuzzy boundary loss for handling boundary uncertainty
    2. Weighted membership loss for class imbalance
    3. Rough set approximation loss for set-based approximation
    
    Args:
        y_true: Ground truth masks (values in [0, 1])
        y_pred: Predicted masks (logits or probabilities)
        alpha: Weight for fuzzy boundary loss (default: 0.4)
        beta: Weight for weighted membership loss (default: 0.3)
        gamma: Weight for rough set approximation loss (default: 0.3)
        **kwargs: Additional arguments to pass to the component loss functions
        
    Returns:
        Combined FRS loss value
        
    Note:
        The weights (alpha, beta, gamma) are automatically normalized to sum to 1.0
    """
    # Normalize weights to ensure they sum to 1.0
    total = alpha + beta + gamma + 1e-7  # Add small epsilon to avoid division by zero
    alpha_norm, beta_norm, gamma_norm = alpha/total, beta/total, gamma/total
    
    # Calculate component losses
    fuzzy_bound = fuzzy_boundary_loss(y_true, y_pred, **kwargs.get('fuzzy_boundary', {}))
    weighted_mem = weighted_membership_loss(y_true, y_pred)
    rough_set = rough_set_approx_loss(y_true, y_pred)
    
    # Combine losses with normalized weights
    total_loss = (alpha_norm * fuzzy_bound + 
                 beta_norm * weighted_mem + 
                 gamma_norm * rough_set)
    
    return total_loss



def fuzzy_boundary_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
                       alpha: float = 0.7, beta: float = 0.3,
                       sigma: float = 1.0, epsilon: float = 1e-7) -> tf.Tensor:
    """
    Fuzzy boundary loss that models uncertainty in lesion boundaries using fuzzy set theory.
    
    This loss function combines:
    1. Fuzzy membership-based cross-entropy for lesion and background regions
    2. Boundary-aware weighting that focuses on uncertain boundary regions
    
    Args:
        y_true: Ground truth masks (values in [0, 1])
        y_pred: Predicted logits or probabilities
        alpha: Weight for lesion region loss (default: 0.7)
        beta: Weight for boundary region loss (default: 0.3)
        sigma: Controls the width of the boundary region (default: 1.0)
        epsilon: Small constant for numerical stability
        
    Returns:
        Fuzzy boundary loss value
    """
    # Ensure y_pred are probabilities
    y_pred_prob = tf.sigmoid(y_pred) if y_pred.dtype != tf.float32 else y_pred
    
    # Calculate boundary region using Gaussian smoothing
    # This creates a smooth transition between foreground and background
    boundary_region = tf.exp(-(y_true - 0.5)**2 / (2 * sigma**2 + epsilon))
    
    # Calculate membership degrees
    # For lesion region (foreground)
    mu_lesion = y_true * (1 - boundary_region) + boundary_region * 0.5
    # For background region
    mu_background = (1 - y_true) * (1 - boundary_region) + boundary_region * 0.5
    
    # Calculate fuzzy cross-entropy terms
    # Lesion term
    lesion_term = -mu_lesion * tf.math.log(y_pred_prob + epsilon)
    # Background term
    background_term = -mu_background * tf.math.log(1 - y_pred_prob + epsilon)
    
    # Combine terms
    fuzzy_ce = lesion_term + background_term
    
    # Calculate boundary-aware weighting
    # Higher weight for boundary regions
    boundary_weight = 1.0 + boundary_region
    
    # Apply boundary weighting
    weighted_loss = fuzzy_ce * boundary_weight
    
    # Combine with boundary loss
    boundary_loss = tf.reduce_mean(tf.abs(y_pred_prob - y_true) * boundary_region)
    
    # Final loss combines fuzzy cross-entropy and boundary loss
    total_loss = (alpha * tf.reduce_mean(weighted_loss) + 
                 beta * boundary_loss)
    
    return total_loss


# Dictionary of available loss functions
LOSS_FUNCTIONS = {
    'binary_crossentropy': binary_crossentropy_loss,
    'dice_loss': dice_loss,
    'tversky_loss': tversky_loss,
    'focal_loss': focal_loss,
    'hausdorff_loss': hausdorff_distance_loss,
    'weighted_membership': weighted_membership_loss,
    'rough_set_approx': rough_set_approx_loss,
    'fuzzy_boundary': fuzzy_boundary_loss,
    'frs_loss': frs_loss
}

def get_loss_function(loss_name: str) -> LossFunction:
    """
    Get a loss function by name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        The corresponding loss function
        
    Raises:
        ValueError: If the loss function is not found
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[loss_name]
