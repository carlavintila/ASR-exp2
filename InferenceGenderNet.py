import torch
from torch.utils.data import DataLoader
from GenderNet import AudioDataset, GenderNet, evaluate
from GenderNet import load_model  # this loads model+optimizer+epoch
import argparse
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

def load_models(checkpoint_dir, folds, device):
    """
    Load trained models from specified checkpoint directory.

    Parameters:
        checkpoint_dir (str or Path): Directory containing model checkpoints for each fold.
        folds (int): Number of model folds to load.
        device (str): Device to load models on ("cuda" or "cpu").

    Returns:
        List[torch.nn.Module]: List of loaded and evaluated GenderNet models.
    """
    models = []
    for fold in range(1, folds + 1):
        fold_path = Path(checkpoint_dir) / f"fold_{fold}"
        checkpoints = list(fold_path.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No model found in {fold_path}")
        checkpoint_path = checkpoints[0]
        model = GenderNet().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        load_model(model, optimizer, checkpoint_path, device)
        model.eval()
        models.append(model)
        print(f"Loaded model from fold {fold}: {checkpoint_path.name}")
    return models

@torch.no_grad()
def evaluate_ensemble(models, dataloader, device):
    """
    Evaluate ensemble of models on a dataset.

    Parameters:
        models (List[torch.nn.Module]): List of trained models.
        dataloader (DataLoader): DataLoader with evaluation dataset.
        device (str): Device to run inference on.

    Returns:
        Tuple[float, float]: Average loss and accuracy over dataset.
    """
    total_loss = 0
    correct = 0
    total = len(dataloader.dataset)
    loss_fn = torch.nn.CrossEntropyLoss()

    for inputs, _, targets in tqdm(dataloader, desc="Evaluating Ensemble"):
        inputs, targets = inputs.to(device), targets.to(device)

        logits_sum = None
        for model in models:
            outputs = model(inputs)
            if logits_sum is None:
                logits_sum = outputs
            else:
                logits_sum += outputs

        avg_logits = logits_sum / len(models)
        loss = loss_fn(avg_logits, targets)
        total_loss += loss.item()
        correct += (avg_logits.argmax(dim=1) == targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total

def main(args_dict=None):
    """
    Main execution function for ensemble evaluation.

    Parameters:
        args_dict (dict or None): Optional dictionary of arguments. If None, uses argparse.

    Executes:
        - Loads dataset and trained models.
        - Runs ensemble evaluation.
        - Prints loss and accuracy.
    """
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_dir', required=True)
        parser.add_argument('--folds', type=int, default=5)
        parser.add_argument('--audio_list', required=True)
        parser.add_argument('--speakers_json', required=True)
        parser.add_argument('--gender_json', required=True)
        parser.add_argument('--segment_length', type=int, default=8192)
        parser.add_argument('--sampling_rate', type=int, default=16000)
        parser.add_argument('--batch_size', type=int, default=128)
        args = parser.parse_args()
    else:
        args = SimpleNamespace(**args_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test data
    test_dataset = AudioDataset(
        audio_list_file=args.audio_list,
        segment_length=args.segment_length,
        sampling_rate=args.sampling_rate,
        speakers_json=args.speakers_json,
        gender_json=args.gender_json,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Load models
    models = load_models(args.checkpoint_dir, args.folds, device)

    # Evaluate ensemble
    test_loss, test_acc = evaluate_ensemble(models, test_loader, device)
    print(f"\nEnsemble Test Loss: {test_loss:.4f}")
    print(f"Ensemble Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    config = {
        "audio_list": ".\\text_files\\audio_outputs_gengan_polish.txt",
        "speakers_json": ".\\json\\mls_test_speakers_polish.json",
        "gender_json": ".\\json\\mls_test_gender_polish.json",
        "segment_length": 8192,
        "sampling_rate": 16000,
        "batch_size": 128,
        "checkpoint_dir": ".\\checkpoints\\polish",
        "folds": 5,
    }
    main(config)
