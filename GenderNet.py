import torch
import random
import json
from pathlib import Path
import torchaudio
import torch.nn.functional as F
from sklearn.model_selection import KFold
from types import SimpleNamespace
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import numpy as np



class AudioDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading audio files and converting them into mel spectrograms
    for gender classification.
    """
    def __init__(self, audio_list_file, segment_length, sampling_rate, speakers_json, gender_json, augment=True, n_mels=80):
        """
        Initialize the dataset.

        Parameters:
            audio_list_file (str or list): Path to text file or list containing audio file paths.
            segment_length (int): Length of audio segments in samples.
            sampling_rate (int): Target sampling rate for audio.
            speakers_json (str): JSON file with speaker IDs.
            gender_json (str): JSON file with gender labels corresponding to speaker IDs.
            augment (bool): Whether to apply volume augmentation.
            n_mels (int): Number of Mel filter banks.
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.augment = augment
        self.n_mels = n_mels
        self.audio_files = self._load_file_list(audio_list_file)
        self.speakers = sorted(json.load(open(speakers_json)))
        gender_list = json.load(open(gender_json))
        self.gender = {speaker: gender for speaker, gender in zip(self.speakers, gender_list)}
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels
        )
        random.shuffle(self.audio_files)

    def _load_file_list(self, filepath):
        """Load audio file paths from a file or list."""
        if isinstance(filepath, list):
            return filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.audio_files)

    def _load_audio(self, path):
        """
        Load an audio file and apply optional augmentation and resampling.

        Parameters:
            path (str): Path to audio file.

        Returns:
            torch.Tensor: Processed waveform.
        """
        waveform, sr = torchaudio.load(path)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sampling_rate)
        waveform = waveform.mean(dim=0)
        waveform = 0.95 * (waveform / waveform.abs().max())
        if self.augment:
            waveform *= random.uniform(0.3, 1.0)
        return waveform

    def __getitem__(self, index):
        """
        Get a data sample.

        Parameters:
            index (int): Index of the sample.

        Returns:
            tuple: (Mel spectrogram, speaker ID, gender label)
        """
        filepath = Path(self.audio_files[index])
        speaker = filepath.stem.split("_")[0]
        gender_label = torch.tensor(self.gender[speaker]).long()

        audio = self._load_audio(str(filepath))
        if audio.size(0) >= self.segment_length:
            start = random.randint(0, audio.size(0) - self.segment_length) 
            audio = audio[start:start + self.segment_length]

        mel = self.mel_transform(audio.unsqueeze(0))
        return mel, speaker, gender_label

class GenderNet(nn.Module):
    """
    CNN-based model for gender classification from mel spectrograms.
    """
    def __init__(self, in_channels=1, n_mels=80, filters=32, layers=4):
        """
        Initialize GenderNet.

        Parameters:
            in_channels (int): Number of input channels.
            n_mels (int): Number of mel spectrogram bins.
            filters (int): Number of convolution filters.
            layers (int): Number of convolutional blocks.
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ) for _ in range(layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(filters, 2)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.

    Parameters:
        model (nn.Module): GenderNet model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        loss_fn (Loss): Loss function.
        device (str): Device to train on.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.train()
    total_loss, correct = 0, 0
    total = len(dataloader.dataset)

    for inputs, _, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def load_audio_list(filepath):
    """
    Load a list of audio file paths from a text file.

    Parameters:
        filepath (str): Path to the text file.

    Returns:
        list: List of file paths.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model.

    Parameters:
        model (nn.Module): GenderNet model.
        dataloader (DataLoader): DataLoader for validation/test data.
        loss_fn (Loss): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    total_loss, correct = 0, 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for inputs, _, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total

def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Parameters:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, epoch, path):
    """
    Save model checkpoint.

    Parameters:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer state.
        epoch (int): Current epoch.
        path (str or Path): Directory to save the checkpoint.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(checkpoint, Path(path) / 'best_model.pt')  

def load_model(model, optimizer, checkpoint_path, device):
    """
    Load model and optimizer state from a checkpoint.

    Parameters:
        model (nn.Module): Model to load into.
        optimizer (Optimizer): Optimizer to load into.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint.

    Returns:
        int: Epoch number loaded from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def main(args_dict=None):
    """
    Main function to train GenderNet using K-Fold cross-validation.

    Parameters:
        args_dict (dict or None): Optional arguments dictionary. If None, parsed from CLI.

    Procedure:
        - Loads audio file paths
        - Splits data into K folds
        - Trains and validates model per fold
        - Saves best model for each fold
    """
    if args_dict is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--audio_list', required=True)
        parser.add_argument('--speakers_json', required=True)
        parser.add_argument('--gender_json', required=True)
        parser.add_argument('--segment_length', type=int, default=8192)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
        parser.add_argument('--folds', type=int, default=5)
        parser.add_argument('--patience', type=int, default=5)
        parser.add_argument('--sampling_rate', type=int, default=16000)
        args = parser.parse_args()
    else:
        args = SimpleNamespace(**args_dict)

    set_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    audio_list = load_audio_list(args.audio_list)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=123)

    for fold, (train_idx, val_idx) in enumerate(kf.split(audio_list)):
        print(f"\n======== Fold {fold + 1}/{args.folds} ========")
        train_subset = [audio_list[i] for i in train_idx]
        val_subset = [audio_list[i] for i in val_idx]

        train_dataset = AudioDataset(train_subset, args.segment_length, args.sampling_rate, args.speakers_json, args.gender_json, augment=True)
        val_dataset = AudioDataset(val_subset, args.segment_length, args.sampling_rate, args.speakers_json, args.gender_json, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = GenderNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_val_acc, epochs_no_improve = 0, 0

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, optimizer, epoch, Path(args.checkpoint_dir) / f"fold_{fold + 1}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    config = {
        "audio_list": ".\\text_files\\audio_opus_train_dutch.txt",
        "speakers_json": ".\\json\\mls_train_speakers_dutch.json",
        "gender_json": ".\\json\\mls_train_gender_dutch.json",
        "segment_length": 8192,
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "checkpoint_dir": ".\\checkpoints\\dutch",
        "folds": 5,
        "patience": 5,
        "sampling_rate": 16000
    }
    main(config)