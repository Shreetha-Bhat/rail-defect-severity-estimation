"""
Rail Defect Segmentation using U-Net++ with EfficientNet-B0 Encoder
Complete implementation with severity calculation based on MM Agarwal criteria
"""

# ============================================================================
# SECTION 1: INSTALLATIONS AND IMPORTS
# ============================================================================

# Run these installations in Colab
"""
!pip install segmentation-models
!pip install albumentations
!pip install opencv-python
!pip install scikit-image
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Segmentation Models
import segmentation_models as sm
sm.set_framework('tf.keras')

# Augmentation
import albumentations as A

# Metrics
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import directed_hausdorff

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class Config:
    # Paths - Update these to your actual paths
    BASE_PATH = '/content/drive/MyDrive/RailDefects'  # Update this
    TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'Train/Images')
    TRAIN_MASK_PATH = os.path.join(BASE_PATH, 'Train/Masks')
    TEST_IMG_PATH = os.path.join(BASE_PATH, 'Test/Images')
    TEST_MASK_PATH = os.path.join(BASE_PATH, 'Test/Masks')
    
    # Model parameters
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Encoder
    BACKBONE = 'efficientnetb0'
    ENCODER_WEIGHTS = 'imagenet'
    
    # Classes - Update based on your actual class names
    CLASSES = [
        'Longitudinal',
        'Transverse', 
        'Wear',
        'Batter',
        'Scabbing',
        'WheelBurns',
        'Corrugation'
    ]
    NUM_CLASSES = len(CLASSES)
    
    # Training
    VALIDATION_SPLIT = 0.2
    
config = Config()

# ============================================================================
# SECTION 3: SEVERITY CALCULATION LOGIC (MM Agarwal)
# ============================================================================

class SeverityCalculator:
    """
    Severity calculation based on MM Agarwal book criteria
    """
    
    @staticmethod
    def calculate_wear_severity(vertical_wear_mm, lateral_wear_mm, rail_type='52kg'):
        """
        Calculate severity for rail wear defects
        vertical_wear_mm: vertical wear in millimeters
        lateral_wear_mm: lateral wear in millimeters
        """
        # Vertical wear limits (from MM Agarwal Table)
        vertical_limits = {
            '52kg': {'moderate': 8, 'severe': 13},
            '60kg': {'moderate': 10, 'severe': 13},
            '90R': {'moderate': 10, 'severe': 13}
        }
        
        # Lateral wear limits
        lateral_limits = {
            '52kg': {'moderate': 10, 'severe': 13},
            '60kg': {'moderate': 10, 'severe': 13},
            '90R': {'moderate': 10, 'severe': 13}
        }
        
        v_limit = vertical_limits.get(rail_type, vertical_limits['52kg'])
        l_limit = lateral_limits.get(rail_type, lateral_limits['52kg'])
        
        if vertical_wear_mm > v_limit['severe'] or lateral_wear_mm > l_limit['severe']:
            severity = 'Critical'
            action = 'Immediate rail renewal required'
        elif vertical_wear_mm > v_limit['moderate'] or lateral_wear_mm > l_limit['moderate']:
            severity = 'Severe'
            action = 'Schedule rail renewal within 6 months'
        elif vertical_wear_mm > v_limit['moderate']*0.5 or lateral_wear_mm > l_limit['moderate']*0.5:
            severity = 'Moderate'
            action = 'Monitor closely, plan renewal'
        else:
            severity = 'Minor'
            action = 'Continue routine maintenance'
            
        return severity, action
    
    @staticmethod
    def calculate_batter_severity(batter_mm):
        """
        Calculate severity for rail end batter
        batter_mm: difference in height at rail end vs 30cm away
        """
        if batter_mm > 3:
            severity = 'Critical'
            action = 'Crop rail ends immediately or replace'
        elif batter_mm > 2:
            severity = 'Severe'
            action = 'Crop rail ends or schedule replacement'
        elif batter_mm > 1:
            severity = 'Moderate'
            action = 'Monitor and plan cropping'
        else:
            severity = 'Minor'
            action = 'Continue monitoring'
            
        return severity, action
    
    @staticmethod
    def calculate_defect_severity_from_mask(mask, defect_type, pixel_to_mm=0.5):
        """
        Calculate severity from predicted mask
        mask: binary mask of the defect
        defect_type: type of defect
        pixel_to_mm: conversion factor from pixels to millimeters
        """
        # Calculate defect area
        defect_area_pixels = np.sum(mask > 0)
        defect_area_mm2 = defect_area_pixels * (pixel_to_mm ** 2)
        
        # Calculate defect dimensions
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 'None', 'No defect detected', {}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to mm
        width_mm = w * pixel_to_mm
        height_mm = h * pixel_to_mm
        
        metrics = {
            'area_mm2': defect_area_mm2,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'defect_percentage': (defect_area_pixels / (mask.shape[0] * mask.shape[1])) * 100
        }
        
        # Severity logic based on defect type
        if defect_type.lower() in ['wear', 'lateral_wear', 'vertical_wear']:
            # Estimate wear depth based on defect intensity/area
            estimated_wear = (defect_area_mm2 / 1000) * 2  # Simplified estimation
            severity, action = SeverityCalculator.calculate_wear_severity(estimated_wear, estimated_wear)
            
        elif defect_type.lower() in ['batter', 'rail_end_batter']:
            # Estimate batter based on defect characteristics
            estimated_batter = min(height_mm / 10, 5)  # Simplified estimation
            severity, action = SeverityCalculator.calculate_batter_severity(estimated_batter)
            
        elif defect_type.lower() in ['scabbing', 'shelling']:
            if defect_area_mm2 > 500:
                severity = 'Critical'
                action = 'Replace rail section immediately'
            elif defect_area_mm2 > 200:
                severity = 'Severe'
                action = 'Schedule urgent rail replacement'
            elif defect_area_mm2 > 50:
                severity = 'Moderate'
                action = 'Monitor and plan replacement'
            else:
                severity = 'Minor'
                action = 'Continue monitoring'
                
        elif defect_type.lower() in ['wheelburns', 'wheel_burns']:
            if defect_area_mm2 > 300:
                severity = 'Severe'
                action = 'Grind rail surface or replace section'
            elif defect_area_mm2 > 100:
                severity = 'Moderate'
                action = 'Plan grinding operation'
            else:
                severity = 'Minor'
                action = 'Monitor condition'
                
        elif defect_type.lower() in ['corrugation']:
            if defect_area_mm2 > 1000:
                severity = 'Severe'
                action = 'Rail grinding required immediately'
            elif defect_area_mm2 > 500:
                severity = 'Moderate'
                action = 'Schedule rail grinding'
            else:
                severity = 'Minor'
                action = 'Monitor development'
                
        else:  # Longitudinal, Transverse cracks
            if defect_area_mm2 > 200 or width_mm > 50:
                severity = 'Critical'
                action = 'Immediate rail replacement - safety hazard'
            elif defect_area_mm2 > 100 or width_mm > 30:
                severity = 'Severe'
                action = 'Urgent rail replacement required'
            elif defect_area_mm2 > 50:
                severity = 'Moderate'
                action = 'Schedule rail replacement'
            else:
                severity = 'Minor'
                action = 'Increase inspection frequency'
        
        return severity, action, metrics

# ============================================================================
# SECTION 4: DATA LOADING AND AUGMENTATION
# ============================================================================

def get_augmentation_pipeline(is_training=True):
    """Create augmentation pipeline"""
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
            A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        ])
    else:
        return A.Compose([
            A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        ])

class RailDefectDataset(tf.keras.utils.Sequence):
    """Custom dataset for rail defect segmentation"""
    
    def __init__(self, image_paths, mask_paths, class_names, batch_size=8, 
                 augmentation=None, preprocessing=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_names = class_names
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            # Load image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply augmentation
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            
            if self.preprocessing:
                image = self.preprocessing(image)
            
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)

def load_dataset_paths(base_img_path, base_mask_path, class_names):
    """Load all image and mask paths organized by class"""
    all_image_paths = []
    all_mask_paths = []
    
    for class_name in class_names:
        img_class_path = os.path.join(base_img_path, class_name)
        mask_class_path = os.path.join(base_mask_path, class_name)
        
        if os.path.exists(img_class_path) and os.path.exists(mask_class_path):
            images = sorted(glob(os.path.join(img_class_path, '*.jpg')) + 
                          glob(os.path.join(img_class_path, '*.jpeg')))
            masks = sorted(glob(os.path.join(mask_class_path, '*.png')))
            
            all_image_paths.extend(images)
            all_mask_paths.extend(masks)
            
            print(f"Class {class_name}: {len(images)} images, {len(masks)} masks")
    
    return all_image_paths, all_mask_paths

# ============================================================================
# SECTION 5: MODEL BUILDING - U-Net++ with EfficientNet-B0
# ============================================================================

def build_unetplusplus_model(input_shape=(256, 256, 3), num_classes=1):
    """
    Build U-Net++ model with EfficientNet-B0 encoder
    """
    # Use segmentation_models library
    model = sm.Unet(
        backbone_name=config.BACKBONE,
        encoder_weights=config.ENCODER_WEIGHTS,
        input_shape=input_shape,
        classes=num_classes,
        activation='sigmoid',
        encoder_freeze=False
    )
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for model training"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                            tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for training"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined BCE and Dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ============================================================================
# SECTION 6: PERFORMANCE METRICS CALCULATION
# ============================================================================

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_dice_score(y_true, y_pred, smooth=1e-6):
        """Dice Score / F1 Score"""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    @staticmethod
    def calculate_iou(y_true, y_pred, smooth=1e-6):
        """Standard IoU (Intersection over Union)"""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def calculate_giou(y_true, y_pred):
        """Generalized IoU"""
        # Get bounding boxes
        def get_bbox(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return 0, 0, 0, 0
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return rmin, cmin, rmax, cmax
        
        # Standard IoU
        iou = PerformanceMetrics.calculate_iou(y_true, y_pred)
        
        # Get bounding boxes
        y_true_bbox = get_bbox(y_true)
        y_pred_bbox = get_bbox(y_pred)
        
        # Calculate enclosing box
        c_rmin = min(y_true_bbox[0], y_pred_bbox[0])
        c_cmin = min(y_true_bbox[1], y_pred_bbox[1])
        c_rmax = max(y_true_bbox[2], y_pred_bbox[2])
        c_cmax = max(y_true_bbox[3], y_pred_bbox[3])
        
        c_area = (c_rmax - c_rmin + 1) * (c_cmax - c_cmin + 1)
        
        # Union area
        true_area = np.sum(y_true)
        pred_area = np.sum(y_pred)
        union_area = true_area + pred_area - np.sum(y_true * y_pred)
        
        # GIoU
        giou = iou - (c_area - union_area) / (c_area + 1e-6)
        return giou
    
    @staticmethod
    def calculate_hausdorff_distance(y_true, y_pred):
        """Hausdorff Distance"""
        # Get coordinates of True pixels
        true_coords = np.argwhere(y_true > 0.5)
        pred_coords = np.argwhere(y_pred > 0.5)
        
        if len(true_coords) == 0 or len(pred_coords) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        hd1 = directed_hausdorff(true_coords, pred_coords)[0]
        hd2 = directed_hausdorff(pred_coords, true_coords)[0]
        
        return max(hd1, hd2)
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, threshold=0.5):
        """Calculate all metrics"""
        # Binarize predictions
        y_pred_bin = (y_pred > threshold).astype(np.float32)
        y_true_bin = (y_true > threshold).astype(np.float32)
        
        # Flatten for pixel-wise metrics
        y_true_f = y_true_bin.flatten()
        y_pred_f = y_pred_bin.flatten()
        
        # Confusion matrix elements
        tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
        tn = np.sum((y_true_f == 0) & (y_pred_f == 0))
        fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
        fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Segmentation metrics
        dice = PerformanceMetrics.calculate_dice_score(y_true_bin, y_pred_bin)
        iou = PerformanceMetrics.calculate_iou(y_true_bin, y_pred_bin)
        giou = PerformanceMetrics.calculate_giou(y_true_bin, y_pred_bin)
        hausdorff = PerformanceMetrics.calculate_hausdorff_distance(y_true_bin, y_pred_bin)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'dice_score': dice,
            'iou': iou,
            'giou': giou,
            'hausdorff_distance': hausdorff
        }

def evaluate_model_classwise(model, image_paths, mask_paths, class_names, batch_size=8):
    """Evaluate model performance class-wise"""
    results = []
    
    for class_name in class_names:
        print(f"\nEvaluating class: {class_name}")
        
        # Filter paths for this class
        class_img_paths = [p for p in image_paths if class_name in p]
        class_mask_paths = [p for p in mask_paths if class_name in p]
        
        if len(class_img_paths) == 0:
            continue
        
        class_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
            'dice_score': [], 'iou': [], 'giou': [], 'hausdorff_distance': []
        }
        
        # Process each image
        for img_path, mask_path in zip(class_img_paths, class_mask_paths):
            # Load and preprocess
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH))
            image = image.astype(np.float32) / 255.0
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (config.IMG_HEIGHT, config.IMG_WIDTH))
            mask = mask.astype(np.float32) / 255.0
            
            # Predict
            pred_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            
            # Calculate metrics
            metrics = PerformanceMetrics.calculate_metrics(mask, pred_mask.squeeze())
            
            for key in class_metrics:
                class_metrics[key].append(metrics[key])
        
        # Average metrics for this class
        avg_metrics = {k: np.mean(v) for k, v in class_metrics.items()}
        avg_metrics['class'] = class_name
        avg_metrics['num_samples'] = len(class_img_paths)
        results.append(avg_metrics)
    
    return pd.DataFrame(results)

# ============================================================================
# SECTION 7: TRAINING
# ============================================================================

def train_model():
    """Main training function"""
    print("Loading dataset...")
    
    # Load training data
    train_img_paths, train_mask_paths = load_dataset_paths(
        config.TRAIN_IMG_PATH, 
        config.TRAIN_MASK_PATH, 
        config.CLASSES
    )
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        train_img_paths, train_mask_paths, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create data generators
    train_dataset = RailDefectDataset(
        X_train, y_train, config.CLASSES,
        batch_size=config.BATCH_SIZE,
        augmentation=get_augmentation_pipeline(is_training=True)
    )
    
    val_dataset = RailDefectDataset(
        X_val, y_val, config.CLASSES,
        batch_size=config.BATCH_SIZE,
        augmentation=get_augmentation_pipeline(is_training=False)
    )
    
    # Build model
    print("\nBuilding U-Net++ model with EfficientNet-B0 encoder...")
    model = build_unetplusplus_model(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        num_classes=1
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, 'accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_unetplusplus_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ============================================================================
# SECTION 8: VISUALIZATION - CUSTOM STYLE
# ============================================================================

def plot_training_history(history):
    """Plot training history with custom visualization style"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Training Performance', fontsize=20, fontweight='bold', y=0.995)
    
    # Set style
    sns.set_style("whitegrid")
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', 
             linewidth=2.5, color=colors[0], marker='o', markersize=4, alpha=0.8)
    ax1.plot(history.history['val_loss'], label='Validation Loss', 
             linewidth=2.5, color=colors[1], marker='s', markersize=4, alpha=0.8)
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Dice Coefficient
    ax2 = axes[0, 1]
    ax2.plot(history.history['dice_coefficient'], label='Training Dice', 
             linewidth=2.5, color=colors[2], marker='o', markersize=4, alpha=0.8)
    ax2.plot(history.history['val_dice_coefficient'], label='Validation Dice', 
             linewidth=2.5, color=colors[3], marker='s', markersize=4, alpha=0.8)
    ax2.set_title('Dice Coefficient Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Accuracy
    ax3 = axes[1, 0]
    ax3.plot(history.history['accuracy'], label='Training Accuracy', 
             linewidth=2.5, color=colors[0], marker='o', markersize=4, alpha=0.8)
    ax3.plot(history.history['val_accuracy'], label='Validation Accuracy', 
             linewidth=2.5, color=colors[1], marker='s', markersize=4, alpha=0.8)
    ax3.set_title('Model