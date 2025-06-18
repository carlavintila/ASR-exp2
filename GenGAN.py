from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import *
import argparse
from pathlib import Path
import os
import time
from networks import UNetFilter
from torch.autograd import Variable
import numpy as np
from modules import MelGAN_Generator, Audio2Mel
from pydub import AudioSegment
import math
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
from scipy.io import wavfile

LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor

def parse_args():
    """
    Parse command-line arguments for GenGAN inference.

    Returns:
        argparse.Namespace: Parsed arguments with attributes for configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--filter_receptive_field", type=int, default=3)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--seeds", type=int, nargs='+', default=[123])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--noise_dim", type=int, default=65)
    parser.add_argument("--max_duration", type=int, default=16.7)
    parser.add_argument("--path_to_dir", type=str, default='C:\\Users\\C\\Documents\\Coding Projects\\ASR\\mls_polish_opus\\dev\\audio')
    parser.add_argument("--path_to_models", type=str, default='.\\models')
    args = parser.parse_args()
    return args

def load_opus_to_torch(full_path, sampling_rate, max_duration):
    """
    Convert an .opus audio file to a padded Torch tensor using ffmpeg and scipy.

    Parameters:
        full_path (str): Path to the .opus file.
        sampling_rate (int): Target sampling rate.
        max_duration (float): Maximum duration (in seconds) to pad to.

    Returns:
        Tuple[torch.Tensor, int]: Padded waveform tensor and original duration in samples.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        # Use ffmpeg to decode .opus to temporary .wav
        result = subprocess.run([
            "ffmpeg", "-y", "-i", full_path,
            "-ar", str(sampling_rate),  # resample
            "-ac", "1",                 # mono
            "-f", "wav", tmp_wav.name
        ], capture_output=True)

        if result.returncode != 0:
            print("FFmpeg stderr:", result.stderr.decode())
            raise RuntimeError(f"FFmpeg failed to convert {full_path}")

        # Load the decoded WAV using scipy
        sr, samples = wavfile.read(tmp_wav.name)

    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.float32:
        pass  # already in good shape
    else:
        raise ValueError(f"Unsupported audio dtype: {samples.dtype}")

    duration = len(samples)
    samples = torch.from_numpy(samples).float()
    if samples.size(0) <= max_duration * sampling_rate:
        samples = F.pad(samples, (0, int(max_duration * sampling_rate) - samples.size(0)), "constant").data

    return samples, duration

def save_opus_sample(path, sample_rate, audio_tensor):
    """
    Save a waveform tensor to a .opus file.

    Parameters:
        path (str): Output file path.
        sample_rate (int): Audio sampling rate.
        audio_tensor (torch.Tensor): Audio data tensor.
    """
    import soundfile as sf
    sf.write(path, audio_tensor.numpy(), sample_rate, format='OPUS')

def main():
    """
    Main function to run the GenGAN voice transformation pipeline.

    - Parses arguments
    - Loads audio and model components
    - Processes each .opus audio file in the input directory
    - Transforms it using a trained GenGAN model
    - Saves the output as a new .opus file
    """
    args = parse_args()
    root = Path(os.getcwd())
    device = 'cuda:' + str(args.device)

    set_seed(args.seeds[0])
    audio_dir = Path(args.path_to_dir)
    run_dir = root / 'audio_outputs_polish'
    run_dir.mkdir(parents=True, exist_ok=True)

    fft = Audio2Mel(sampling_rate=args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    Mel2Audio.load_state_dict(torch.load(Path(args.path_to_models) / 'multi_speaker.pt'))

    netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128],
                      kernel_size=args.filter_receptive_field,
                      image_width=32, image_height=80, noise_dim=args.noise_dim,
                      nb_classes=2, embedding_dim=16, use_cond=False).to(device)
    netG.load_state_dict(torch.load(Path(args.path_to_models) / 'netG_epoch_25.pt'))
    netG.eval()

    for root_dir, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".opus"):
                full_path = os.path.join(root_dir, file)
                print(f"Processing {file}")
                x, dur = load_opus_to_torch(full_path, args.sampling_rate, args.max_duration)

            x = torch.unsqueeze(x, 1)
            spectrograms = fft(x.reshape(1, x.size(0))).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

            z = torch.randn(spectrograms.shape[0], 1, 5, 65).to(device)
            gen_secret = Variable(LongTensor(np.random.choice([1.0], spectrograms.shape[0]))).to(device)
            y_n = gen_secret * np.random.normal(0.5, math.sqrt(0.05))
            generated_neutral = netG(spectrograms, z, y_n).detach()

            generated_neutral = torch.squeeze(generated_neutral, 1).to(device) * 3 * stds.to(device) + means.to(device)
            inverted_neutral = Mel2Audio(generated_neutral).squeeze().detach().cpu()

            output_stem = Path(file).stem
            output_path = run_dir / f"{output_stem}_transformed.opus"
            save_opus_sample(str(output_path), args.sampling_rate, inverted_neutral[:dur])

    print(f"All files processed. Output saved in: {run_dir}")

if __name__ == "__main__":
    main()
