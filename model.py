import tensorflow as tf
from tensorflow.keras import layers, Model
from loss_function import frs_loss

class UNet:
    def __init__(self, input_shape=(128, 128, 1), num_classes=1):
        """
        Initialize U-Net model.
        
        Args:
            input_shape: Shape of input images (height, width, channels).
                       For grayscale images, channels=1
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def conv_block(self, x, num_filters):
        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    def encoder_block(self, x, num_filters):
        x = self.conv_block(x, num_filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, x, skip_features, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x
    
    def build_unet(self):
        inputs = layers.Input(self.input_shape)
        
        # Encoder
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)
        
        # Bridge
        b1 = self.conv_block(p4, 1024)
        
        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, 1, padding="same", activation='sigmoid')(d4)
        
        return Model(inputs, outputs, name="U-Net")

def get_loss_function(loss_name):
    """
    Get the loss function by name.
    """
    if loss_name == 'bce':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_name == 'dice':
        def dice_loss(y_true, y_pred):
            smooth = 1e-5
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.sigmoid(y_pred)
            
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1.0 - dice
        return dice_loss
    elif loss_name == 'tversky':
        def tversky_loss(y_true, y_pred, alpha=0.7):
            smooth = 1e-5
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.sigmoid(y_pred)
            
            # Flatten
            y_true_pos = tf.reshape(y_true, [-1])
            y_pred_pos = tf.reshape(y_pred, [-1])
            
            # Calculate true positives, false positives and false negatives
            true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
            false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
            false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
            
            tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
            return 1.0 - tversky
        return tversky_loss
    elif loss_name == 'focal':
        def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
            """
            Focal loss for binary classification.
            FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)
            """
            # Ensure inputs are float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Flatten the predictions and labels
            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred, [-1])
            
            # Clip predictions to avoid log(0) and log(1)
            epsilon = tf.keras.backend.epsilon()
            y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1. - epsilon)
            
            # Calculate cross-entropy
            cross_entropy = -y_true_flat * tf.math.log(y_pred_flat) - (1 - y_true_flat) * tf.math.log(1 - y_pred_flat)
            
            # Calculate p_t
            p_t = y_true_flat * y_pred_flat + (1 - y_true_flat) * (1 - y_pred_flat)
            
            # Calculate alpha factor
            alpha_factor = y_true_flat * alpha + (1 - y_true_flat) * (1 - alpha)
            
            # Calculate modulating factor
            modulating_factor = tf.pow(1.0 - p_t, gamma)
            
            # Calculate focal loss
            focal_loss = alpha_factor * modulating_factor * cross_entropy
            
            # Return mean over the batch
            return tf.reduce_mean(focal_loss)
        return focal_loss
    elif loss_name == 'frs':
        return frs_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def get_model(input_shape=(128, 128, 1), num_classes=1, loss='bce', learning_rate=1e-4):
    """
    Create and compile a U-Net model.
    
    Args:
        input_shape: Input shape (height, width, channels). Default is (128, 128, 1) for grayscale images.
        num_classes: Number of output classes
        loss: Loss function to use (can be string name or callable)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Ensure input shape has 3 dimensions
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)  # Add channel dimension for grayscale
    
    # Build the model
    model = UNet(input_shape, num_classes).build_unet()
    
    # Get loss function if a string is provided
    if isinstance(loss, str):
        loss_fn = get_loss_function(loss)
    else:
        loss_fn = loss
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model
