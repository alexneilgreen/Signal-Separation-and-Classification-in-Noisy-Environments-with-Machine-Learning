import argparse
import torch
from torch.utils.data import DataLoader

from src.analysis import ContinuousAudioAnalyzer, check_and_compare_results
from src.data_processing import AudioDataset, augment_dataset
from src.model import AudioCNN
from src.training import train_model, load_dataset, plot_training_history, print_final_metrics
from src.utils import generate_demo

def train_only(args, device, data_dir):
    """Function to handle training mode"""
    
    # Load dataset with three-way split
    train_files, val_files, test_files, train_labels, val_labels, test_labels = load_dataset(data_dir)
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    test_dataset = AudioDataset(test_files, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = AudioCNN().to(device)
    
    # Train model
    history = train_model(model, train_loader, val_loader, test_loader, device, 
                         num_epochs=args.epochs, learning_rate=args.learning_rate)
    
    # Plot training history
    plot_training_history(history)
    
    # Print final metrics
    print_final_metrics(history)

def analyze_only():
    """Function to handle analysis mode"""
    print("Analysis mode selected")
    analyzer = ContinuousAudioAnalyzer('Results/best_model.pth')
    analyzer.analyze_directory('data/Forest Recordings')

    metrics = check_and_compare_results('Results/ResultsSummary.csv', 'Results/GroundTruth.csv')

    if metrics:
        mae, rmse, r2 = metrics
        print(f"\nResults:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Bird Call Detection and Analysis')
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3, 4],
                      help='1: Full pipeline, 2: Training only, 3: Analysis only, 4: Generate Demo Figures')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for training')
    parser.add_argument('--augment', type=str, default='T', choices=['T', 'F'],
                      help='Use augmented dataset: T (True) or F (False)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Handle augmentation if needed
    if args.augment == 'T' and args.mode != 4:
        data_dir = 'Data/Augmented'
        print("\nUsing Augmented Dataset")
        augment_dataset('Data', 'Data/Augmented')
    else:
        data_dir = 'Data'
        print("\nUsing Original Dataset")
    
    # Execute based on mode
    if args.mode == 1:
        print("\nFull Pipeline mode selected\n")
        train_only(args, device, data_dir)
        analyze_only()
    elif args.mode == 2:
        print("\nTraining mode selected\n")
        train_only(args, device, data_dir)
    elif args.mode == 3:
        print("\nAnalysis mode selected\n")
        analyze_only()
    elif args.mode == 4:
        print("\nGenerate Example mode selected\n")
        generate_demo()

if __name__ == "__main__":
    main()