import argparse
import os
import torch

from src.config import Config
from src.dataset import get_data_loaders
from src.model import get_model
from src.train import train_model, plot_training_history
from src.evaluate import evaluate_and_report, plot_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="EEG Schizophrenia Classifier CLI")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["preprocess", "train", "evaluate", "all"],
        help="Pipeline phase: preprocess (cache dataset), train (run training), evaluate (run testing), or all"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="original", 
        choices=["original", "eeg1d", "eegnet"],
        help="PyTorch model architecture to use"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Number of epochs to train (overrides config)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None, 
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--force-preprocess", 
        action="store_true", 
        help="Overwrites cached preprocessed data and re-runs preprocessing from raw files"
    )
    
    args = parser.parse_args()
    
    # Run setup
    Config.setup_directories()
    
    # 1. Preprocess mode
    if args.mode == "preprocess":
        print("Starting preprocessing phase...")
        # get_data_loaders will trigger loading and caching
        _, _ = get_data_loaders(force_preprocess=True)
        print("Preprocessing successfully finished!")
        return

    # For other modes, we need loaders
    try:
        train_loader, test_loader = get_data_loaders(force_preprocess=args.force_preprocess)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing Dataset: {e}")
        print("Please read instructions in 'data/README.md' or run a mock test first.")
        return

    # Define model specific checkpoint path
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_{args.model}.pth")
    
    # Define model
    model = get_model(
        model_name=args.model, 
        num_channels=Config.ELECTRODES_COUNT, 
        timesteps=Config.SAMPLE_RATE_PER_TRIAL // Config.N_AVERAGED
    )
    print(f"\nInitialized '{args.model}' model architecture.")
    
    # 2. Train mode
    if args.mode == "train" or args.mode == "all":
        print(f"\n--- Starting Training (Model: {args.model}) ---")
        model, history = train_model(
            model=model, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            epochs=args.epochs, 
            lr=args.lr,
            checkpoint_path=checkpoint_path
        )
        
        # Save plots
        plot_path = os.path.join(Config.BASE_DIR, "reports", f"training_history_{args.model}.png")
        plot_training_history(history, save_path=plot_path)
        print("Training phase finished.")
        
    # 3. Evaluate mode
    if args.mode == "evaluate" or args.mode == "all":
        print(f"\n--- Starting Evaluation ---")
        # Load best model weights if they exist and we didn't just train
        if args.mode == "evaluate":
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
                print(f"Loaded saved weights from {checkpoint_path}")
            else:
                print(f"Warning: Checkpoint weights not found at {checkpoint_path}. Evaluating with initial random weights.")
                
        metrics = evaluate_and_report(model, test_loader)
        
        # Save confusion matrix plot
        cm_plot_path = os.path.join(Config.BASE_DIR, "reports", f"confusion_matrix_{args.model}.png")
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_plot_path)
        print("Evaluation phase finished.")

if __name__ == "__main__":
    main()
