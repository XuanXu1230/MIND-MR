import os
import gc
import pandas as pd
import numpy as np
import time
import logging
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import catboost as cb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training_progress.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import CTGAN and sdmetrics
try:
    from ctgan import CTGAN
    import ctgan
    logger.info(f"CTGAN imported successfully from ctgan (version {ctgan.__version__})")
except ImportError as e:
    logger.error(f"Failed to import CTGAN: {str(e)}. Falling back to SMOTE-only augmentation.")
    CTGAN = None

try:
    from sdmetrics.reports.single_table import QualityReport
    import sdmetrics
    logger.info(f"QualityReport imported successfully from sdmetrics (version {sdmetrics.__version__})")
except ImportError as e:
    logger.error(f"Failed to import QualityReport: {str(e)}. Synthetic data quality validation skipped.")
    QualityReport = None

# Initialize CUDA
if torch.cuda.is_available():
    torch.cuda.init()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Configuration
CONFIG = {
    'batch_size': 32,
    'min_batch_size': 8,
    'n_jobs': min(4, psutil.cpu_count(logical=False)),
    'epochs': 10,
    'k_folds': 5,
    'n_repeats': 2,
    'early_stopping_patience': 3,
    'gpu_memory_threshold': 1000,
    'gradient_clip_val': 0.5,
    'label_smoothing': 0.05,
    'min_dataset_size': 30,  # Reduced from 100 to allow datasets with 70 samples
    'resample_target_size': 200,
    'min_fold_size': 10,  # Reduced from 20 to allow smaller folds
    'min_transformer_dataset': 100,  # Reduced to allow small datasets
    'min_cnn_dataset': 100,  # Reduced to allow small datasets
    'min_neural_dataset': 100,  # Reduced to allow small datasets
    'small_dataset_epochs': 5,
    'ctgan_epochs': 100,
    'smote_ratio': 0.1,
    'max_features': 20,
    'rfe_step': 0.1,
    'warmup_epochs': 2,
    'tabnet_sparsity': 1e-5,
}

# GPU memory monitoring
def check_gpu_memory(model_name, file_name):
    if device.type == "cuda":
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory = free_memory / 1024**2
            total_memory = total_memory / 1024**2
            if free_memory < total_memory * 0.2:
                torch.cuda.empty_cache()
                free_memory, _ = torch.cuda.mem_get_info()
                free_memory = free_memory / 1024**2
            logger.info(f"GPU memory for {model_name} on {file_name}: Free={free_memory:.2f} MB")
            return free_memory, total_memory
        except Exception as e:
            logger.warning(f"GPU memory check failed: {str(e)}")
            return None, None
    return None, None

# Adjust batch size
def adjust_batch_size(batch_size, free_memory, total_memory, dataset_size):
    batch_size = max(CONFIG['min_batch_size'], min(batch_size, dataset_size // 4))
    if free_memory is not None and free_memory < total_memory * 0.3:
        batch_size = max(CONFIG['min_batch_size'], batch_size // 2)
    if dataset_size % batch_size == 1:
        batch_size = max(CONFIG['min_batch_size'], batch_size - 1)
    if batch_size < 2:
        batch_size = 2  # Ensure minimum batch size of 2
    return batch_size

# Memory cleanup
def cleanup_memory(model_name, file_name):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Cleaned memory for {model_name} on {file_name}")

# Dynamic SMOTE ratio
def get_dynamic_smote_ratio(y):
    class_counts = np.bincount(y.astype(int))
    imbalance_ratio = max(class_counts) / min(class_counts)
    if imbalance_ratio > 20:
        return 0.1
    elif imbalance_ratio > 10:
        return 0.2
    elif imbalance_ratio > 5:
        return 0.15
    else:
        return 0.1

# Stratified resampling
def stratified_resample(X, y, file_name):
    try:
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning(f"Only one class in {file_name}. Skipping resampling.")
            return X, y
        if not np.all(np.isin(unique_labels, [0, 1])):
            logger.error(f"Non-binary labels detected in {file_name}: {unique_labels}. Using original data.")
            return X, y
        class_counts = np.bincount(y.astype(int))
        logger.info(f"Dataset details for {file_name}: Total samples={len(y)}, Class counts={class_counts}")
        min_class_count = min(class_counts)
        imbalance_ratio = max(class_counts) / min_class_count if min_class_count > 0 else float('inf')
        if len(y) < 20 or min_class_count <= 3 or imbalance_ratio > 50:
            logger.warning(f"Dataset too small, too few minority samples, or too imbalanced in {file_name} "
                          f"(size={len(y)}, min_class_count={min_class_count}, imbalance_ratio={imbalance_ratio}). "
                          f"Using original data.")
            return X, y
        class_proportions = class_counts / len(y)
        logger.info(f"Original class proportions in {file_name}: {class_proportions}")
        target_samples = min(CONFIG['resample_target_size'], len(y) * 2)
        logger.info(f"Target samples for {file_name}: {target_samples}")
        smote_ratio = get_dynamic_smote_ratio(y)
        k_neighbors = min(5, max(1, min_class_count - 1))
        if min_class_count <= k_neighbors + 1:
            k_neighbors = max(1, min_class_count - 2)
            logger.info(f"Adjusted k_neighbors to {k_neighbors} due to small minority class in {file_name}")
        logger.info(f"SMOTE parameters for {file_name}: ratio={smote_ratio}, k_neighbors={k_neighbors}")
        smote = SMOTE(
            random_state=42,
            k_neighbors=k_neighbors,
            sampling_strategy=smote_ratio
        )
        X_smote, y_smote = smote.fit_resample(X, y)
        smote_samples = len(y_smote)
        logger.info(f"After SMOTE in {file_name}: Total samples={smote_samples}")
        remaining_samples = target_samples - smote_samples
        if remaining_samples > 0 and CTGAN is not None:
            logger.info(f"Attempting CTGAN for {file_name}")
            data = pd.DataFrame(X_smote, columns=[f'feature_{i}' for i in range(X_smote.shape[1])])
            data['label'] = y_smote
            discrete_columns = ['label']
            ctgan_epochs = CONFIG['ctgan_epochs'] if len(data) > 100 else 50
            logger.info(f"CTGAN epochs for {file_name}: {ctgan_epochs}")
            try:
                ctgan = CTGAN(epochs=ctgan_epochs, verbose=False)
                ctgan.fit(data, discrete_columns)
                X_synthetic = []
                y_synthetic = []
                unique_labels = np.unique(y_smote)
                samples_per_class = {label: int(remaining_samples * class_proportions[label]) for label in unique_labels}
                for label in unique_labels:
                    if samples_per_class[label] > 0:
                        synthetic_data = ctgan.sample(
                            n=samples_per_class[label],
                            condition_column='label',
                            condition_value=label
                        )
                        X_synthetic.append(synthetic_data.drop('label', axis=1).values)
                        y_synthetic.append(synthetic_data['label'].values)
                if X_synthetic:
                    X_synthetic = np.vstack(X_synthetic)
                    y_synthetic = np.hstack(y_synthetic)
                    if QualityReport is not None:
                        try:
                            quality_report = QualityReport()
                            synthetic_data_full = pd.DataFrame(
                                X_synthetic,
                                columns=[f'feature_{i}' for i in range(X_synthetic.shape[1])]
                            )
                            synthetic_data_full['label'] = y_synthetic
                            metadata = {
                                'columns': {
                                    col: {'sdtype': 'numerical'} for col in data.columns if col != 'label'
                                } | {
                                    'label': {'sdtype': 'categorical'}
                                }
                            }
                            quality_report.generate(data, synthetic_data_full, metadata)
                            logger.info(f"Synthetic data quality for {file_name}: {quality_report.get_score()}")
                        except Exception as e:
                            logger.warning(f"Synthetic data quality validation failed for {file_name}: {str(e)}")
                    X_resampled = np.vstack([X_smote, X_synthetic])
                    y_resampled = np.hstack([y_smote, y_synthetic])
                else:
                    raise ValueError("No synthetic data generated by CTGAN")
            except Exception as e:
                logger.warning(f"CTGAN failed in {file_name}: {str(e)}. Using SMOTE-only augmentation.")
                X_resampled, y_resampled = X_smote, y_smote
        else:
            if CTGAN is None:
                logger.info(f"CTGAN not available for {file_name}")
            X_resampled, y_resampled = X_smote, y_smote
            if smote_samples < target_samples:
                k_neighbors = min(5, max(1, min(np.bincount(y_smote.astype(int))) - 1))
                if min(np.bincount(y_smote.astype(int))) <= k_neighbors + 1:
                    k_neighbors = max(1, min(np.bincount(y_smote.astype(int))) - 2)
                    logger.info(f"Adjusted k_neighbors to {k_neighbors} for additional SMOTE in {file_name}")
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'
                )
                try:
                    X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)
                    logger.info(f"After additional SMOTE in {file_name}: Total samples={len(y_resampled)}")
                except Exception as e:
                    logger.warning(f"Additional SMOTE failed in {file_name}: {str(e)}. Using SMOTE-only data.")
        final_class_counts = np.bincount(y_resampled.astype(int))
        final_proportions = final_class_counts / len(y_resampled)
        logger.info(f"Final class proportions in {file_name}: {final_proportions}")
        return X_resampled, y_resampled
    except Exception as e:
        logger.warning(f"Initial SMOTE failed in {file_name}: {str(e)}. Attempting conservative SMOTE.")
        try:
            min_class_count = min(np.bincount(y.astype(int)))
            k_neighbors = min(2, max(1, min_class_count - 2))
            smote_ratio = 0.1
            logger.info(f"Conservative SMOTE parameters for {file_name}: ratio={smote_ratio}, k_neighbors={k_neighbors}")
            smote = SMOTE(
                random_state=42,
                k_neighbors=k_neighbors,
                sampling_strategy=smote_ratio
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"After conservative SMOTE in {file_name}: Total samples={len(y_resampled)}")
            final_class_counts = np.bincount(y_resampled.astype(int))
            final_proportions = final_class_counts / len(y_resampled)
            logger.info(f"Final class proportions in {file_name}: {final_proportions}")
            return X_resampled, y_resampled
        except Exception as e2:
            logger.warning(f"Conservative SMOTE failed in {file_name}: {str(e2)}. Attempting random oversampling.")
            try:
                ros_target_samples = min(CONFIG['resample_target_size'], len(y) * 2)
                ros_sampling_strategy = 'auto' if min_class_count > 3 else {0: class_counts[0], 1: class_counts[0]}
                logger.info(f"Random oversampling parameters for {file_name}: target_samples={ros_target_samples}, "
                           f"sampling_strategy={ros_sampling_strategy}")
                ros = RandomOverSampler(random_state=42, sampling_strategy=ros_sampling_strategy)
                X_resampled, y_resampled = ros.fit_resample(X, y)
                logger.info(f"After random oversampling in {file_name}: Total samples={len(y_resampled)}")
                final_class_counts = np.bincount(y_resampled.astype(int))
                final_proportions = final_class_counts / len(y_resampled)
                logger.info(f"Final class proportions in {file_name}: {final_proportions}")
                return X_resampled, y_resampled
            except Exception as e3:
                logger.error(f"Random oversampling failed in {file_name}: {str(e3)}. Using original data.")
                return X, y

# Dataset class
class GeneDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, torch.Tensor):
            self.X = X.clone().detach().to(dtype=torch.float32, device=device)
        else:
            self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        if isinstance(y, torch.Tensor):
            self.y = y.clone().detach().to(dtype=torch.float32, device=device)
        else:
            self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Label smoothing loss
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        target = target * (1 - 2 * self.smoothing) + self.smoothing
        return nn.BCEWithLogitsLoss()(pred, target)

# SklearnWrapper
class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device, epochs=10, batch_size=32, lr=0.001, patience=3, physics_weight=0.0):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.physics_weight = physics_weight
        self.scaler = GradScaler('cuda') if device.type == 'cuda' else None
        self.classes_ = None
        self.best_model_state = None

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device) if isinstance(X, np.ndarray) else X
        y = torch.tensor(y, dtype=torch.float32, device=self.device) if isinstance(y, np.ndarray) else y
        self.classes_ = np.unique(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)
        
        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size
        if train_size < CONFIG['min_dataset_size'] or val_size < CONFIG['min_batch_size']:
            logger.warning(f"Dataset too small for {self.model.__class__.__name__} (train_size={train_size}, val_size={val_size}). Skipping.")
            return self
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            GeneDataset(X, y), [train_size, val_size]
        )
        
        if len(train_dataset) < CONFIG['min_neural_dataset']:
            logger.warning(f"Training dataset too small for {self.model.__class__.__name__} (size={len(train_dataset)}). Skipping.")
            return self
        if isinstance(self.model, AdvancedTransformerModel) and len(train_dataset) < CONFIG['min_transformer_dataset']:
            logger.warning(f"Dataset too small for transformer (size={len(train_dataset)}). Skipping.")
            return self
        if isinstance(self.model, AdvancedCNNModel) and len(train_dataset) < CONFIG['min_cnn_dataset']:
            logger.warning(f"Dataset too small for CNN (size={len(train_dataset)}). Skipping.")
            return self
        
        batch_size = min(self.batch_size, len(train_dataset) // 4)
        if batch_size < 2:
            batch_size = 2
            logger.info(f"Adjusted batch_size to {batch_size} for {self.model.__class__.__name__} on small dataset")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Batch Size={batch_size}")
        
        criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=CONFIG['label_smoothing'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - CONFIG['warmup_epochs'])
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=CONFIG['warmup_epochs'] * len(train_loader)
        )
        
        self.model.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in tqdm(range(self.epochs), desc=f"Training {self.model.__class__.__name__}", leave=False):
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            batch_count = 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
            for batch_X, batch_y in train_bar:
                if batch_X.size(0) < 2:
                    logger.warning(f"Skipping batch with size {batch_X.size(0)} for {self.model.__class__.__name__}.")
                    continue
                
                optimizer.zero_grad(set_to_none=True)
                try:
                    if self.device.type == 'cuda':
                        with autocast('cuda'):
                            outputs = self.model(batch_X).view(-1)
                            classification_loss = criterion(outputs, batch_y)
                            physics_loss = self.model.physics_loss(batch_X) if isinstance(self.model, PINNModel) else 0.0
                            loss = classification_loss + self.physics_weight * physics_loss
                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['gradient_clip_val'])
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(batch_X).view(-1)
                        classification_loss = criterion(outputs, batch_y)
                        physics_loss = self.model.physics_loss(batch_X) if isinstance(self.model, PINNModel) else 0.0
                        loss = classification_loss + self.physics_weight * physics_loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['gradient_clip_val'])
                        optimizer.step()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (preds == batch_y).sum().item()
                    train_total += batch_y.size(0)
                    total_train_loss += loss.item()
                    batch_count += 1
                    
                    train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    if epoch < CONFIG['warmup_epochs']:
                        warmup_scheduler.step()
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"GPU out of memory. Clearing cache and skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    logger.error(f"Error in training {self.model.__class__.__name__}: {str(e)}")
                    return self
                
                cleanup_memory("batch", self.model.__class__.__name__)
            
            if batch_count == 0:
                logger.warning(f"No valid batches processed in epoch {epoch+1}. Skipping epoch.")
                continue
            
            self.model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            val_batch_count = 0
            
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch_X, batch_y in val_bar:
                    if batch_X.size(0) < 2:
                        logger.warning(f"Skipping validation batch with size {batch_X.size(0)} for {self.model.__class__.__name__}.")
                        continue
                    try:
                        outputs = self.model(batch_X).view(-1)
                        loss = criterion(outputs, batch_y)
                        total_val_loss += loss.item()
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        val_batch_count += 1
                        val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
                    except RuntimeError as e:
                        logger.warning(f"Error in validation for {self.model.__class__.__name__}: {str(e)}")
                        continue
                
                cleanup_memory("val_batch", self.model.__class__.__name__)
            
            avg_train_loss = total_train_loss / batch_count
            avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict()
                logger.info(f"Saved best model state for {self.model.__class__.__name__} at epoch {epoch+1}")
            
            if avg_val_loss >= best_val_loss:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                epochs_no_improve = 0
            
            if epoch >= CONFIG['warmup_epochs']:
                scheduler.step()
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        cleanup_memory("SklearnWrapper", "fit")
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device) if isinstance(X, np.ndarray) else X
        dataset = GeneDataset(X, np.zeros(len(X)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                try:
                    outputs = torch.sigmoid(self.model(batch_X)).view(-1)
                    predictions.append(outputs.cpu().numpy())
                except RuntimeError as e:
                    logger.warning(f"Error in prediction for {self.model.__class__.__name__}: {str(e)}")
                    predictions.append(np.zeros(batch_X.size(0)))
        return (np.concatenate(predictions) > 0.5).astype(float)

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device) if isinstance(X, np.ndarray) else X
        dataset = GeneDataset(X, np.zeros(len(X)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                try:
                    outputs = torch.sigmoid(self.model(batch_X)).view(-1)
                    probabilities.append(outputs.cpu().numpy())
                except RuntimeError as e:
                    logger.warning(f"Error in predict_proba for {self.model.__class__.__name__}: {str(e)}")
                    probabilities.append(np.zeros(batch_X.size(0)))
        probs = np.concatenate(probabilities)
        return np.vstack((1 - probs, probs)).T

# Model Definitions
class AdvancedCNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self._calculate_fc_input(input_dim)
        self.fc1 = nn.Linear(self.fc_input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
        self.shortcut = nn.Linear(input_dim, self.fc_input_dim)
        self.to(device)

    def _calculate_fc_input(self, input_dim):
        x = torch.zeros((1, 1, input_dim))
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        self.fc_input_dim = x.view(1, -1).size(1)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
        else:
            self.bn1.train(self.training)
            self.bn2.train(self.training)
            self.bn3.train(self.training)
        
        x_input = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x_input)))
        if x.size(2) > 1:
            x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        if x.size(2) > 1:
            x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        if x.size(2) > 1:
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        shortcut = self.shortcut(x_input.squeeze(1))
        x = x + shortcut
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AdvancedTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, nhead=4, num_layers=2, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        nhead = max(1, min(nhead, hidden_dim // 8))
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nhead={nhead}")
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size < 2 and self.training:
            return torch.zeros((batch_size, 1), device=x.device)
        x_input = x.unsqueeze(1)
        x = self.embedding(x_input)
        shortcut = self.shortcut(x_input)
        x = x + shortcut
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

class PINNModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.k = nn.Parameter(torch.tensor(0.1))
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and batch_size == 1:
                layer.eval()
            else:
                layer.train(self.training)
        return self.layers(x)

    def physics_loss(self, x):
        if x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        else:
            x = x.clone().requires_grad_(True)
        output = self.forward(x)
        dx_dt = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        physics_residual = dx_dt + self.k * x
        return torch.mean(physics_residual ** 2)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 16, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(16)
        self.fc = nn.Linear(16 * 2, output_dim)
        self.to(device)
        if next(self.lstm.parameters()).is_cuda:
            try:
                self.lstm.flatten_parameters()
            except RuntimeError as e:
                logger.warning(f"Failed to flatten LSTM parameters: {str(e)}")

    def forward(self, x):
        if self.training and next(self.lstm.parameters()).is_cuda:
            try:
                self.lstm.flatten_parameters()
            except RuntimeError as e:
                logger.warning(f"Failed to flatten LSTM parameters: {str(e)}")
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        x = self.fc(context)
        return x

class AdvancedMLPModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.Linear(16, output_dim)
        )
        self.shortcut = nn.Linear(input_dim, output_dim)
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1:
            for layer in self.layers:
                if isinstance(layer, nn.BatchNorm1d):
                    layer.eval()
        else:
            for layer in self.layers:
                if isinstance(layer, nn.BatchNorm1d):
                    layer.train(self.training)
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return out + shortcut

class AdvancedRNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 16, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(16)
        self.fc = nn.Linear(16 * 2, output_dim)
        self.to(device)
        if next(self.rnn.parameters()).is_cuda:
            try:
                self.rnn.flatten_parameters()
            except RuntimeError as e:
                logger.warning(f"Failed to flatten RNN parameters: {str(e)}")

    def forward(self, x):
        if self.training and next(self.rnn.parameters()).is_cuda:
            try:
                self.rnn.flatten_parameters()
            except RuntimeError as e:
                logger.warning(f"Failed to flatten RNN parameters: {str(e)}")
        x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        context, _ = self.attention(rnn_out)
        x = self.fc(context)
        return x

class TabNetModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_d=16, n_a=16, n_steps=5, gamma=1.3, n_independent=2, n_shared=2):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = CONFIG['tabnet_sparsity']
        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.initial_splitter = nn.Linear(input_dim, n_d + n_a)
        self.shared_glus = nn.ModuleList([nn.Sequential(
            nn.Linear(n_d + n_a, 2 * (n_d + n_a)),
            nn.BatchNorm1d(2 * (n_d + n_a)),
            nn.GLU(dim=1),
            nn.LayerNorm(n_d + n_a)
        ) for _ in range(n_shared)])
        self.independent_glus = nn.ModuleList([
            nn.ModuleList([nn.Sequential(
                nn.Linear(n_d + n_a, 2 * (n_d + n_a)),
                nn.BatchNorm1d(2 * (n_d + n_a)),
                nn.GLU(dim=1),
                nn.LayerNorm(n_d + n_a)
            ) for _ in range(n_independent)]) for _ in range(n_steps)
        ])
        self.attention_transformers = nn.ModuleList([nn.Linear(n_a, input_dim) for _ in range(n_steps)])
        self.final_mapping = nn.Linear(n_d * n_steps, output_dim)
        self.final_bn = nn.BatchNorm1d(n_d)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1:
            self.initial_bn.eval()
            self.final_bn.eval()
            for glu in self.shared_glus:
                for layer in glu:
                    if isinstance(layer, nn.BatchNorm1d):
                        layer.eval()
            for step_glus in self.independent_glus:
                for glu in step_glus:
                    for layer in glu:
                        if isinstance(layer, nn.BatchNorm1d):
                            layer.eval()
        else:
            self.initial_bn.train(self.training)
            self.final_bn.train(self.training)
            for glu in self.shared_glus:
                for layer in glu:
                    if isinstance(layer, nn.BatchNorm1d):
                        layer.train(self.training)
            for step_glus in self.independent_glus:
                for glu in step_glus:
                    for layer in glu:
                        if isinstance(layer, nn.BatchNorm1d):
                            layer.train(self.training)
        x = self.initial_bn(x)
        prior = torch.ones_like(x)
        outputs = []
        M_sum = 0
        x = self.initial_splitter(x)
        feature = x[:, :self.n_d]
        attention = x[:, self.n_d:self.n_d + self.n_a]
        for step in range(self.n_steps):
            M = torch.softmax(self.attention_transformers[step](attention) * prior, dim=1)
            M_sum += M
            prior = self.gamma - torch.sum(M, dim=1, keepdim=True)
            feature_combined = torch.cat([feature, attention], dim=1)
            for shared_glu in self.shared_glus:
                feature_combined = shared_glu(feature_combined)
            for indep_glu in self.independent_glus[step]:
                feature_combined = indep_glu(feature_combined)
            feature = feature_combined[:, :self.n_d]
            attention = feature_combined[:, self.n_d:self.n_d + self.n_a]
            outputs.append(self.final_bn(feature))
        output = torch.cat(outputs, dim=1)
        output = self.final_mapping(output)
        output = self.dropout(output)
        sparsity_loss = self.lambda_sparse * torch.mean(torch.abs(M_sum))
        return output + sparsity_loss

# Model Training Functions
def train_neural_network(model_fn, X_train, y_train, X_test, y_test, model_name, feature_names, output_path, file_name, batch_size):
    logger.info(f"Training {model_name} on {file_name}...")
    if len(X_train) < CONFIG['min_dataset_size'] or len(X_train) < 4 * CONFIG['min_batch_size']:
        logger.warning(f"Dataset too small for {model_name} on {file_name} (size={len(X_train)}). Skipping.")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
    try:
        free_memory, total_memory = check_gpu_memory(model_name, file_name)
        local_device = torch.device('cpu') if (free_memory is not None and free_memory < CONFIG['gpu_memory_threshold'] and model_name in ['Advanced LSTM', 'Advanced RNN']) else device
        batch_size = adjust_batch_size(batch_size, free_memory, total_memory, len(X_train))
        if batch_size < 2 and model_name in ['Advanced CNN', 'TabNet', 'PINN', 'Advanced MLP']:
            logger.warning(f"Batch size {batch_size} too small for {model_name} on {file_name}. Skipping.")
            return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
        model = model_fn(X_train.shape[1]).to(local_device)
        epochs = CONFIG['small_dataset_epochs'] if len(X_train) < CONFIG['min_neural_dataset'] else CONFIG['epochs']
        wrapped_model = SklearnWrapper(model, local_device, epochs=epochs, batch_size=batch_size, patience=CONFIG['early_stopping_patience'])
        wrapped_model.fit(X_train, y_train)
        result = evaluate_model(wrapped_model, X_train, X_test, y_test, model_name, feature_names, file_name)
        cleanup_memory(model_name, file_name)
        return result
    except Exception as e:
        logger.error(f"Error training {model_name} on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_pinn(X_train, y_train, X_test, y_test, model_name, feature_names, output_path, file_name, batch_size):
    if len(X_train) < CONFIG['min_dataset_size'] or len(X_train) < 4 * CONFIG['min_batch_size']:
        logger.warning(f"Dataset too small for {model_name} on {file_name} (size={len(X_train)}). Skipping.")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
    try:
        free_memory, total_memory = check_gpu_memory(model_name, file_name)
        local_device = torch.device('cpu') if free_memory is not None and free_memory < CONFIG['gpu_memory_threshold'] else device
        batch_size = adjust_batch_size(batch_size, free_memory, total_memory, len(X_train))
        if batch_size < 2:
            logger.warning(f"Batch size {batch_size} too small for {model_name} on {file_name}. Skipping.")
            return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
        model = PINNModel(X_train.shape[1]).to(local_device)
        epochs = CONFIG['small_dataset_epochs'] if len(X_train) < CONFIG['min_neural_dataset'] else CONFIG['epochs']
        wrapped_model = SklearnWrapper(model, local_device, epochs=epochs, batch_size=batch_size, patience=CONFIG['early_stopping_patience'], physics_weight=0.1)
        wrapped_model.fit(X_train, y_train)
        result = evaluate_model(wrapped_model, X_train, X_test, y_test, model_name, feature_names, file_name)
        cleanup_memory(model_name, file_name)
        return result
    except Exception as e:
        logger.error(f"Error training {model_name} on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_tabnet(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size):
    if len(X_train) < CONFIG['min_dataset_size'] or len(X_train) < 4 * CONFIG['min_batch_size']:
        logger.warning(f"Dataset too small for TabNet on {file_name} (size={len(X_train)}). Skipping.")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
    try:
        free_memory, total_memory = check_gpu_memory('TabNet', file_name)
        local_device = torch.device('cpu') if free_memory is not None and free_memory < CONFIG['gpu_memory_threshold'] else device
        batch_size = adjust_batch_size(batch_size, free_memory, total_memory, len(X_train))
        if batch_size < 2:
            logger.warning(f"Batch size {batch_size} too small for TabNet on {file_name}. Skipping.")
            return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
        model = TabNetModel(X_train.shape[1]).to(local_device)
        epochs = CONFIG['small_dataset_epochs'] if len(X_train) < CONFIG['min_neural_dataset'] else CONFIG['epochs']
        wrapped_model = SklearnWrapper(model, local_device, epochs=epochs, batch_size=batch_size, patience=CONFIG['early_stopping_patience'])
        wrapped_model.fit(X_train, y_train)
        result = evaluate_model(wrapped_model, X_train, X_test, y_test, 'TabNet', feature_names, file_name)
        cleanup_memory('TabNet', file_name)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU out of memory for TabNet on {file_name}. Retrying with smaller batch size.")
            torch.cuda.empty_cache()
            return train_tabnet(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=batch_size//2)
        logger.error(f"Error training TabNet on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
    except Exception as e:
        logger.error(f"Error training TabNet on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_catboost(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        param_dist = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
            'bagging_temperature': [0, 0.5, 1]
        }
        class_counts = np.bincount(y_train.astype(int))
        class_weights = {0: 1.0, 1: len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0}
        base_model = cb.CatBoostClassifier(verbose=0, random_state=42, task_type='CPU', early_stopping_rounds=20, class_weights=class_weights)
        search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=CONFIG['n_jobs'], random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)
        result = evaluate_model(model, X_train, X_test, y_test, 'CatBoost', feature_names, file_name)
        cleanup_memory("CatBoost", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training CatBoost on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_random_forest(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        }
        class_counts = np.bincount(y_train.astype(int))
        class_weights = {0: 1.0, 1: len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0}
        base_model = RandomForestClassifier(random_state=42, n_jobs=1, class_weight=class_weights)
        search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=CONFIG['n_jobs'], random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        result = evaluate_model(model, X_train, X_test, y_test, 'RandomForest', feature_names, file_name)
        cleanup_memory("RandomForest", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training RandomForest on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=CONFIG['n_jobs'], random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        result = evaluate_model(model, X_train, X_test, y_test, 'GradientBoosting', feature_names, file_name)
        cleanup_memory("GradientBoosting", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training GradientBoosting on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_svm(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        param_dist = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
        class_counts = np.bincount(y_train.astype(int))
        class_weights = {0: 1.0, 1: len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0}
        base_model = SVC(probability=True, random_state=42, class_weight=class_weights)
        search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=CONFIG['n_jobs'], random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        result = evaluate_model(model, X_train, X_test, y_test, 'SVM', feature_names, file_name)
        cleanup_memory("SVM", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training SVM on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_voting_classifier(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        cb_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0, random_state=42, task_type='CPU')
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, max_features='sqrt')
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        voting_clf = VotingClassifier(estimators=[('cb', cb_model), ('rf', rf_model), ('gb', gb_model)], voting='soft', n_jobs=CONFIG['n_jobs'])
        voting_clf.fit(X_train, y_train)
        result = evaluate_model(voting_clf, X_train, X_test, y_test, 'VotingClassifier', feature_names, file_name)
        cleanup_memory("VotingClassifier", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training VotingClassifier on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

def train_stacking_classifier(X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size=None):
    try:
        cb_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0, random_state=42, task_type='CPU')
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, max_features='sqrt')
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        meta_learner = LogisticRegression(random_state=42)
        stacking_clf = StackingClassifier(
            estimators=[('cb', cb_model), ('rf', rf_model), ('gb', gb_model)],
            final_estimator=meta_learner,
            cv=3,
            n_jobs=CONFIG['n_jobs']
        )
        stacking_clf.fit(X_train, y_train)
        result = evaluate_model(stacking_clf, X_train, X_test, y_test, 'StackingClassifier', feature_names, file_name)
        cleanup_memory("StackingClassifier", file_name)
        return result
    except Exception as e:
        logger.error(f"Error training StackingClassifier on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None

# Model evaluation
def evaluate_model(model, X_train, X_test, y_test, model_name, feature_names, file_name):
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    try:
        preds = model.predict(X_test_np)
        probs = model.predict_proba(X_test_np)[:, 1] if hasattr(model, 'predict_proba') else preds
        metrics = {
            'Accuracy': accuracy_score(y_test_np, preds),
            'Precision': precision_score(y_test_np, preds, zero_division=1),
            'Recall': recall_score(y_test_np, preds, zero_division=1),
            'F1 Score': f1_score(y_test_np, preds, zero_division=1),
            'ROC AUC': roc_auc_score(y_test_np, probs)
        }
    except Exception as e:
        logger.warning(f"Error evaluating {model_name} on {file_name}: {str(e)}")
        return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']}, None
    cm = confusion_matrix(y_test_np, preds)
    logger.info(f"{model_name} on {file_name}: {metrics}")
    logger.info(f"Confusion Matrix:\n{cm}")
    feature_importances = None
    if feature_names and len(feature_names) == X_test_np.shape[1]:
        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            else:
                perm_importance = permutation_importance(model, X_test_np, y_test_np, scoring='roc_auc', n_repeats=3, random_state=42, n_jobs=CONFIG['n_jobs'])
                importance_values = perm_importance.importances_mean
            if importance_values.min() < 0:
                importance_values = importance_values + abs(importance_values.min())
            feature_importances = pd.DataFrame({
                'Gene': feature_names,
                'Importance': importance_values
            }).sort_values(by='Importance', ascending=False)
        except Exception as e:
            logger.warning(f"Feature importance failed for {model_name} on {file_name}: {str(e)}")
    return metrics, feature_importances

# Preprocess data
def preprocess_data(X_train, y_train, X_test, fold, file_name, cache, cache_key, feature_names_orig):
    try:
        fold_key = f"{cache_key}_fold_{fold}"
        if fold_key not in cache:
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            selector = VarianceThreshold(threshold=0.01)
            X_train = selector.fit_transform(X_train)
            selected_indices = selector.get_support(indices=True)
            X_test = X_test[:, selected_indices]
            feature_names = [feature_names_orig[i] for i in selected_indices]
            if X_train.shape[1] > CONFIG['max_features']:
                selector = SelectKBest(mutual_info_classif, k=min(50, X_train.shape[1]))
                X_train = selector.fit_transform(X_train, y_train)
                selected_indices = selector.get_support(indices=True)
                X_test = X_test[:, selected_indices]
                feature_names = [feature_names[i] for i in selected_indices]
                estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
                rfe = RFE(estimator, n_features_to_select=CONFIG['max_features'], step=CONFIG['rfe_step'])
                X_train = rfe.fit_transform(X_train, y_train)
                selected_indices = rfe.get_support(indices=True)
                X_test = X_test[:, selected_indices]
                feature_names = [feature_names[i] for i in selected_indices]
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            cache[fold_key] = (X_train, y_train, X_test, feature_names)
        else:
            X_train, y_train, X_test, feature_names = cache[fold_key]
        return X_train, y_train, X_test, feature_names
    except Exception as e:
        logger.error(f"Preprocessing failed for fold {fold} in {file_name}: {str(e)}")
        return X_train, y_train, X_test, feature_names_orig

# Process file
def process_file(file_path, file_name, folder_name, models, output_path):
    results = []
    importances = []
    error_count = 0
    error_summary = []
    file_id = f"{folder_name}/{file_name}"
    try:
        df = pd.read_csv(file_path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:-1].astype(float).values
        feature_names_orig = df.iloc[:, 1:-1].columns.tolist()
        logger.info(f"Dataset details for {file_id}: Total samples={len(y)}, Class counts={np.bincount(y.astype(int))}, Unique labels={np.unique(y)}")
        X, y = stratified_resample(X, y, file_id)
        if len(X) < CONFIG['min_dataset_size']:
            logger.warning(f"Skipping {file_id}: Too few samples ({len(X)})")
            return [], [], 1
        selected_models = models.copy()
        if len(X) < CONFIG['min_neural_dataset']:
            neural_network_models = ['Advanced Transformer', 'Advanced CNN', 'Advanced LSTM', 'Advanced MLP', 'Advanced RNN', 'TabNet', 'PINN']
            for nn_model in neural_network_models:
                selected_models.pop(nn_model, None)
            logger.info(f"Skipped neural network models for {file_id} due to small dataset size ({len(X)})")
        
        # Dynamic k_folds adjustment
        k_folds = CONFIG['k_folds']
        samples_per_fold = len(X) // k_folds
        if samples_per_fold < CONFIG['min_fold_size']:
            k_folds = max(2, len(X) // CONFIG['min_fold_size'])
            logger.info(f"Adjusted k_folds to {k_folds} for {file_id} to ensure at least {CONFIG['min_fold_size']} samples per fold (dataset size={len(X)})")
        
        cache = {}
        kf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=CONFIG['n_repeats'], random_state=42)
        for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(X, y), desc=f"Folds in {file_id}", total=k_folds * CONFIG['n_repeats'])):
            try:
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
                if len(X_train) < CONFIG['min_fold_size']:
                    logger.warning(f"Skipping fold {fold} in {file_id}: Too few training samples ({len(X_train)})")
                    error_summary.append(f"Fold {fold}: Too few training samples ({len(X_train)})")
                    continue
                X_train, y_train, X_test, feature_names = preprocess_data(
                    X_train, y_train, X_test, fold, file_id, cache, file_id, feature_names_orig
                )
                tasks = []
                for model_name, model_fn in selected_models.items():
                    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
                    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
                    X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
                    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
                    tasks.append(delayed(lambda fn, name: (
                        name, fn(
                            X_train_np, y_train_np, X_test_np, y_test_np, feature_names, output_path, file_id, CONFIG['batch_size']
                        )
                    ))(model_fn, model_name))
                fold_results = Parallel(n_jobs=CONFIG['n_jobs'])(tasks)
                for model_name, (metrics, feature_importances) in fold_results:
                    result = {'ID': file_id, 'File': file_name, 'Folder': folder_name, 'Model': model_name, 'Fold': fold}
                    result.update(metrics)
                    results.append(result)
                    if feature_importances is not None:
                        importance = feature_importances.copy()
                        importance['Model'] = model_name
                        importance['ID'] = file_id
                        importance['File'] = file_name
                        importance['Folder'] = folder_name
                        importance['Fold'] = fold
                        importances.append(importance)
                logger.info(f"Completed fold {fold+1}/{k_folds*CONFIG['n_repeats']} for {file_id}")
                cleanup_memory("fold", file_id)
            except Exception as e:
                logger.error(f"Error processing fold {fold} in {file_id}: {str(e)}")
                error_summary.append(f"Fold {fold}: {str(e)}")
                continue
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.to_csv(os.path.join(output_path, f"{folder_name}_{file_name}_results.csv"), index=False)
        if importances:
            pd.concat(importances, ignore_index=True).to_csv(
                os.path.join(output_path, f"{folder_name}_{file_name}_importances.csv"), index=False
            )
        if error_summary:
            logger.info(f"Error summary for {file_id}: {error_summary}")
    except Exception as e:
        logger.error(f"Error processing {file_id}: {str(e)}")
        error_count += 1
        error_summary.append(f"File-level: {str(e)}")
    return results, importances, error_count

# Process files
def process_files(root_path, output_path):
    start_time = time.time()
    file_count = 0
    total_error_count = 0
    error_summary = []
    os.makedirs(output_path, exist_ok=True)
    models = {
        'Advanced Transformer': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_neural_network(
            lambda input_dim: AdvancedTransformerModel(input_dim, 1), X_train, y_train, X_test, y_test, 'Advanced Transformer', feature_names, output_path, file_name, batch_size
        ),
        'Advanced CNN': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_neural_network(
            lambda input_dim: AdvancedCNNModel(input_dim, 1), X_train, y_train, X_test, y_test, 'Advanced CNN', feature_names, output_path, file_name, batch_size
        ),
        'Advanced LSTM': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_neural_network(
            lambda input_dim: AdvancedLSTMModel(input_dim, 1), X_train, y_train, X_test, y_test, 'Advanced LSTM', feature_names, output_path, file_name, batch_size
        ),
        'Advanced MLP': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_neural_network(
            lambda input_dim: AdvancedMLPModel(input_dim, 1), X_train, y_train, X_test, y_test, 'Advanced MLP', feature_names, output_path, file_name, batch_size
        ),
        'Advanced RNN': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_neural_network(
            lambda input_dim: AdvancedRNNModel(input_dim, 1), X_train, y_train, X_test, y_test, 'Advanced RNN', feature_names, output_path, file_name, batch_size
        ),
        'TabNet': train_tabnet,
        'PINN': lambda X_train, y_train, X_test, y_test, feature_names, output_path, file_name, batch_size: train_pinn(
            X_train, y_train, X_test, y_test, 'PINN', feature_names, output_path, file_name, batch_size
        ),
        'RandomForest': train_random_forest,
        'GradientBoosting': train_gradient_boosting,
        'CatBoost': train_catboost,
        'SVM': train_svm,
        'VotingClassifier': train_voting_classifier,
        'StackingClassifier': train_stacking_classifier
    }
    all_results = []
    all_importances = []
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name == 'result':
            continue
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for file_name in tqdm(files, desc=f"Files in {folder_name}"):
            file_path = os.path.join(folder_path, file_name)
            results, importances, errors = process_file(file_path, file_name, folder_name, models, output_path)
            all_results.extend(results)
            all_importances.extend(importances)
            file_count += 1
            total_error_count += errors
            logger.info(f"Completed processing {file_name} in {folder_name}")
        folder_results_df = pd.DataFrame([r for r in all_results if r['Folder'] == folder_name])
        if not folder_results_df.empty:
            folder_results_df = folder_results_df.sort_values(by='ROC AUC', ascending=False)
            folder_results_df.to_csv(os.path.join(output_path, f"{folder_name}_performance_results.csv"), index=False)
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Model', y='ROC AUC', data=folder_results_df)
            plt.title(f'Model Performance: ROC AUC ({folder_name})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{folder_name}_model_performance.png"), dpi=300)
            plt.close()
        if all_importances:
            folder_importances = pd.concat([imp for imp in all_importances if imp['Folder'].iloc[0] == folder_name], ignore_index=True)
            folder_importances.to_csv(
                os.path.join(output_path, f"{folder_name}_feature_importance_results.csv"), index=False
            )
    consolidated_results_df = pd.DataFrame(all_results)
    if not consolidated_results_df.empty:
        consolidated_results_df = consolidated_results_df.sort_values(by='ROC AUC', ascending=False)
        consolidated_results_df.to_csv(os.path.join(output_path, "consolidated_performance_results.csv"), index=False)
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y='ROC AUC', hue='Folder', data=consolidated_results_df)
        plt.title('Model Performance: ROC AUC (All Folders)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "consolidated_model_performance.png"), dpi=300)
        plt.close()
    if all_importances:
        pd.concat(all_importances, ignore_index=True).to_csv(
            os.path.join(output_path, "consolidated_feature_importance_results.csv"), index=False
        )
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time / 60:.2f} minutes")
    logger.info(f"Files processed: {file_count}")
    logger.info(f"Errors encountered: {total_error_count}")
    if error_summary:
        logger.info(f"Error summary across all files: {error_summary}")
    cleanup_memory("process_files", "final")


if __name__ == "__main__":
    root_path = "/DATA/mergeomics/module/SCZ/"
    output_path = "/DATA/mergeomics/module/SCZ/result/"
    if not os.path.exists(root_path):
        logger.error(f"Root folder {root_path} does not exist")
        raise FileNotFoundError(f"Root folder {root_path} does not exist")
    os.makedirs(output_path, exist_ok=True)
    process_files(root_path, output_path)