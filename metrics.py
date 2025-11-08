import tensorflow as tf
import numpy as np
from medpy.metric.binary import hd95

class SegmentationMetrics:
    """
    Class to compute various segmentation metrics.
    """
    
    def __init__(self, threshold=0.5, smooth=1e-5):
        self.threshold = threshold
        self.smooth = smooth
    
    def _threshold_predictions(self, y_pred):
        return tf.cast(y_pred > self.threshold, tf.float32)
    
    def dice_coefficient(self, y_true, y_pred):
        """
        Compute Dice coefficient (F1 score).
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Flatten predictions and labels
        y_pred_f = tf.reshape(y_pred, [-1])
        y_true_f = tf.reshape(y_true, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
    
    def iou_score(self, y_true, y_pred):
        """
        Compute Intersection over Union (Jaccard index).
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Flatten predictions and labels
        y_pred_f = tf.reshape(y_pred, [-1])
        y_true_f = tf.reshape(y_true, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        return (intersection + self.smooth) / (union + self.smooth)
    
    def hd95_distance(self, y_true, y_pred):
        """
        Compute 95th percentile of the Hausdorff Distance.
        Note: This is a simplified version and might be slow for large batches.
        For more accurate results, consider using the full implementation from medpy.
        """
        y_pred = self._threshold_predictions(y_pred)
        y_true = y_true.numpy() if tf.is_tensor(y_true) else y_true
        y_pred = y_pred.numpy() if tf.is_tensor(y_pred) else y_pred
        
        batch_size = y_true.shape[0]
        hd95_scores = []
        
        for i in range(batch_size):
            try:
                # Ensure binary masks
                y_true_bin = (y_true[i, ..., 0] > 0.5).astype(np.uint8)
                y_pred_bin = (y_pred[i, ..., 0] > 0.5).astype(np.uint8)
                
                # Calculate HD95 if both masks are not empty
                if np.any(y_true_bin) and np.any(y_pred_bin):
                    hd95_val = hd95(y_pred_bin, y_true_bin)
                    hd95_scores.append(hd95_val)
            except:
                continue
                
        return np.mean(hd95_scores) if hd95_scores else 0.0
    
    def get_metrics(self, y_true, y_pred):
        """
        Compute all metrics.
        Returns:
            dict: Dictionary containing all metrics
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        metrics = {
            'dice': self.dice_coefficient(y_true, y_pred).numpy(),
            'iou': self.iou_score(y_true, y_pred).numpy(),
        }
        
        # HD95 is more expensive to compute, so we'll do it separately
        hd95_val = self.hd95_distance(y_true, y_pred)
        metrics['hd95'] = hd95_val
        
        return metrics
