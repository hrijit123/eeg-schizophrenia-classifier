import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score
import seaborn as sns

from src.config import Config

def evaluate_and_report(model, loader):
    """
    Evaluates the model on a test dataloader and prints confusion matrix,
    sensitivity, specificity, precision, and F1-score.
    """
    device = Config.DEVICE
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions = (probs >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute confusion matrix with explicit labels to ensure it is always 2x2
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    
    # Check if confusion matrix is 2x2
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        # Fallback if binary class representation is not complete in test data
        TN = FP = FN = TP = 0
        if len(np.unique(all_targets)) == 1:
            val = int(all_targets[0])
            if val == 0:
                TN = len(all_targets)
            else:
                TP = len(all_targets)
    
    # Metrics
    accuracy = (all_preds == all_targets).mean()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print("\n" + "="*40)
    print("           EVALUATION REPORT")
    print("="*40)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print("-"*40)
    print("Confusion Matrix:")
    print(f"  [TN={TN:4d}  FP={FP:4d}]")
    print(f"  [FN={FN:4d}  TP={TP:4d}]")
    print("="*40)
    
    # Print scikit-learn classification report with explicit labels
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, labels=[0, 1], target_names=["Healthy", "Schizophrenia"], zero_division=0))
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    return metrics


def plot_confusion_matrix(cm, save_path=None):
    """Plots a nice heatmap of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Healthy', 'Schizophrenia'],
        yticklabels=['Healthy', 'Schizophrenia']
    )
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved confusion matrix heatmap to {save_path}")
    else:
        plt.show()
    plt.close()
