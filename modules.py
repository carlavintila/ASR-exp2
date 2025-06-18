import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    """
    Initialize weights of convolutional and batch normalization layers.

    Parameters:
        m (nn.Module): A PyTorch layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    """
    Wrapper for weight-normalized 1D convolution layer.
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """
    Wrapper for weight-normalized 1D transposed convolution layer.
    """
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Audio2Mel(nn.Module):
    """
    Converts waveform audio to log-mel spectrogram.
    """
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        """
        Initialize Audio2Mel module.

        Parameters:
            n_fft (int): FFT size.
            hop_length (int): Hop size.
            win_length (int): Window size.
            sampling_rate (int): Sampling rate.
            n_mel_channels (int): Number of mel bins.
            mel_fmin (float): Minimum mel frequency.
            mel_fmax (float or None): Maximum mel frequency.
        """
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = mel_basis = librosa_mel_fn(
                                                sr=sampling_rate,
                                                n_fft=n_fft,
                                                n_mels=n_mel_channels,
                                                fmin=mel_fmin,
                                                fmax=mel_fmax
                                            )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        """
        Compute log-mel spectrogram from audio.

        Parameters:
            audio (torch.Tensor): Input waveform.

        Returns:
            torch.Tensor: Log-mel spectrogram.
        """
        p = (self.n_fft - self.hop_length) // 2
        # audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        audio = F.pad(audio, (p, p), "constant").squeeze(1)

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,  
        )
        magnitude = torch.abs(fft) 
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class ResnetBlock(nn.Module):
    """
    Residual block with dilation for MelGAN.
    """
    def __init__(self, dim, dilation=1):
        """
        Initialize residual block.

        Parameters:
            dim (int): Number of input/output channels.
            dilation (int): Dilation rate for convolutions.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        Compute log-mel spectrogram from audio.

        Parameters:
            audio (torch.Tensor): Input waveform.

        Returns:
            torch.Tensor: Log-mel spectrogram.
        """
        return self.shortcut(x) + self.block(x)


class MelGAN_Generator(nn.Module):
    """
    MelGAN generator network to convert mel spectrograms to waveforms.
    """
    def __init__(self, input_size, ngf, n_residual_layers):
        """
        Initialize MelGAN generator.

        Parameters:
            input_size (int): Number of mel bins.
            ngf (int): Base number of filters.
            n_residual_layers (int): Number of residual blocks per upsampling stage.
        """
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        """
        Generate waveform from mel spectrogram.

        Parameters:
            x (torch.Tensor): Mel spectrogram input.

        Returns:
            torch.Tensor: Generated waveform.
        """
        return self.model(x)
