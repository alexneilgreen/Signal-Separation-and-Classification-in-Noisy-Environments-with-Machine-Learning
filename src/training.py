import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from tqdm import tqdm

class ErrorMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.running_loss = 0.0
        self.num_samples = 0
    
    def update(self, predictions, targets, loss, batch_size):
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.running_loss += loss * batch_size
        self.num_samples += batch_size
    
    def compute(self):
        metrics = {
            'loss': self.running_loss / self.num_samples,
            'accuracy': accuracy_score(self.targets, self.predictions),
            'precision': precision_score(self.targets, self.predictions, zero_division=0),
            'recall': recall_score(self.targets, self.predictions, zero_division=0),
            'f1': f1_score(self.targets, self.predictions, zero_division=0)
        }
        return metrics

def load_dataset(base_dir):
    # Define data directories
    capuchin_dir = Path(base_dir, "Parsed_Capuchinbird_Clips")
    not_capuchin_dir = Path(base_dir, "Parsed_Not_Capuchinbird_Clips")
    
    # Get file paths and labels
    capuchin_files = list(capuchin_dir.glob("*.wav"))
    not_capuchin_files = list(not_capuchin_dir.glob("*.wav"))
    
    all_files = capuchin_files + not_capuchin_files
    labels = [1] * len(capuchin_files) + [0] * len(not_capuchin_files)
    
    # First split: separate test (15%) set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # Second split: separate train (70%) and validation (15%) sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=(15/85), random_state=42, stratify=train_val_labels
    )
    
    # Print dataset sizes
    print(f"\tTraining set size: {len(train_files)}")
    print(f"\tValidation set size: {len(val_files)}")
    print(f"\tTest set size: {len(test_files)}\n")
    
    return train_files, val_files, test_files, train_labels, val_labels, test_labels

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    metrics = ErrorMetrics()
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            predicted = (outputs > 0.5).float()
            metrics.update(predicted.squeeze(), labels, loss.item(), labels.size(0))
    
    return metrics.compute()

def train_model(model, train_loader, val_loader, test_loader, device, num_epochs=50, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )

    # Enhanced scheduler with more parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,  # reduce LR by factor of 10
        patience=2,
        verbose=False,  # print message when LR changes
        min_lr=1e-6,  # don't reduce LR below this
        cooldown=1    # wait this many epochs after a LR change before resuming normal operation
    )
    
    best_val_loss = float('inf')
    history = {
        'ein': [], 'eval': [], 'eout': [],
        'train_metrics': [], 'val_metrics': [], 'test_metrics': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = ErrorMetrics()
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            predicted = (outputs > 0.5).float()
            train_metrics.update(predicted.squeeze(), batch_labels, loss.item(), batch_labels.size(0))
        
        # Potentially need to add model.eval()
        model.eval()

        # Calculate metrics for all sets
        train_results = train_metrics.compute()
        val_results = evaluate_model(model, val_loader, criterion, device)
        test_results = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_results['loss'])
        
        # Save metrics
        history['ein'].append(train_results['loss'])
        history['eval'].append(val_results['loss'])
        history['eout'].append(test_results['loss'])
        history['train_metrics'].append(train_results)
        history['val_metrics'].append(val_results)
        history['test_metrics'].append(test_results)
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), 'Results/best_model.pth')

        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch + 1}/{num_epochs} '+ ('-' * 100))
        print(f'Current Learning Rate:\t{current_lr:.6f}')
        print(f'Ein (Training Error):\t{train_results["loss"]:.6f}')
        print(f'Eval (Validation Error):{val_results["loss"]:.6f}')
        print(f'Eout (Test Error):\t{test_results["loss"]:.6f}')
        print(f'Training Accuracy:\t{train_results["accuracy"]:.6f}')
        print(f'Validation Accuracy:\t{val_results["accuracy"]:.6f}')
        print(f'Test Accuracy:\t\t{test_results["accuracy"]:.6f}\n')
    
    return history

def plot_training_history(history):
    # Plot errors
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['ein'], label='Ein (Training)')
    plt.plot(history['eval'], label='Eval (Validation)')
    plt.plot(history['eout'], label='Eout (Test)')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training, Validation, and Test Errors')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    train_acc = [metrics['accuracy'] for metrics in history['train_metrics']]
    val_acc = [metrics['accuracy'] for metrics in history['val_metrics']]
    test_acc = [metrics['accuracy'] for metrics in history['test_metrics']]
    
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Results/1.TrainingHistory.png')
    plt.close()

def print_final_metrics(history):
    final_epoch = -1
    
    print("\nFinal Model Metrics:")
    print("-" * 100)
    print("Error Metrics:")
    print(f"\tEin (Training Error):\t{history['ein'][final_epoch]:.6f}")
    print(f"\tEval (Validation Error):{history['eval'][final_epoch]:.6f}")
    print(f"\tEout (Test Error):\t{history['eout'][final_epoch]:.6f}")
    
    print("\nDetailed Metrics:")
    print("\nTraining:")
    for metric, value in history['train_metrics'][final_epoch].items():
        print(f"\t{metric}: {value:.6f}")
    
    print("\nValidation:")
    for metric, value in history['val_metrics'][final_epoch].items():
        print(f"\t{metric}: {value:.6f}")
    
    print("\nTest:")
    for metric, value in history['test_metrics'][final_epoch].items():
        print(f"\t{metric}: {value:.6f}")
    
    # Visualize final metrics
    plt.figure(figsize=(18, 6))
    
    # Prepare data for plotting
    sets = ['Training', 'Validation', 'Test']
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    colors = ['blue', 'orange', 'green']
    
    # Create a horizontal bar plot for each metric
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 5, i)
        values = [
            history['train_metrics'][final_epoch][metric],
            history['val_metrics'][final_epoch][metric],
            history['test_metrics'][final_epoch][metric]
        ]
        bars = plt.barh(range(len(sets)), values, color=colors, tick_label=sets)
        
        # Make titles larger and bold
        plt.title(metric.capitalize(), fontsize=16, fontweight='bold')
        plt.xlim(0, 1)
        plt.yticks([])  # Remove y-axis labels
        
        # Customize text placement and formatting
        for j, bar in enumerate(bars):
            width = bar.get_width()
            # For loss, place text on the right; for others, center
            if metric == 'loss':
                plt.text(width, bar.get_y() + bar.get_height() / 2, 
                         f'{width:.6f}', 
                         ha='left', va='center', fontsize=12, fontweight='bold')
            else:
                plt.text(width / 2, bar.get_y() + bar.get_height() / 2, 
                         f'{width:.6f}', 
                         ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Key (Legend)
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Training Data'),
        Patch(facecolor='orange', edgecolor='black', label='Validation Data'),
        Patch(facecolor='green', edgecolor='black', label='Testing Data')
    ]

    # Position the legend below the entire figure
    plt.legend(
        handles=legend_elements,
        loc='upper center',  # Place legend at the top of the extra space
        bbox_to_anchor=(0.5, -0.15),  # Centered below the figure
        fontsize=12
    )

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.savefig('Results/2.FinalMetrics.png')
    plt.close()