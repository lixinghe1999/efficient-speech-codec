import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from dac.model.base import CodecMixin
#import base
#from dac.model.base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize
from dac.model.subband_quantize import Subband_Quantizer
import time
from dac.model.frequency_nn import Encoder, Decoder

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

# default setting in spatial codec
default_config = {
    "encoder_channels": [2, 16, 32, 64, 128, 256, 256],
    "n_fft": 640, "hop_length": 320, "f_kernel_size": [5,3,3,3,3,4], "f_stride_size": [2,2,2,2,2,1], 
    "f_padding_size": [0, 0, 0, 0, 0, 0],
    "decoder_channels": [16, 32, 64, 128, 256, 256, 256],
    "latent_dim": 256 * 6,
    "n_code_groups": 6, "residual_layer": 2,
    "remove_zero_frequency": False
}

# use subband quantization
v1_config = {
    "encoder_channels": [2, 4, 8, 16, 64, 128, 128],
    "n_fft": 1024, "hop_length": 512, "f_kernel_size": [4, 4, 4, 4, 4, 4], "f_stride_size": [2, 2, 2, 2, 2, 2], 
    "f_padding_size": [1, 1, 1, 1, 1, 1], 
    "decoder_channels": [16, 32, 64, 128, 128, 128, 128],
    "latent_dim": 128 * 8,
    "quantizer": "subband",
    "n_code_groups": 8, 
    "residual_layer": 1,
    "remove_zero_frequency": True,  # Whether to remove the zero frequency component in the encoder
}

# use RVQ
v2_config = {
    "encoder_channels": [2, 4, 8, 16, 64, 128, 128],
    "n_fft": 1024, "hop_length": 512, "f_kernel_size": [4, 4, 4, 8, 8, 8], "f_stride_size": [2, 2, 2, 4, 4, 4], 
    "f_padding_size": [1, 1, 1, 2, 2, 2], 
    "decoder_channels": [16, 32, 64, 128, 128, 128, 128],
    "latent_dim": 128 * 1,
    "quantizer": "rvq",
    "n_codebooks": 14,
    "codebook_size": 1024,
    "codebook_dim": 8,
    "quantizer_dropout": 0.5,
    "remove_zero_frequency": True,  # Whether to remove the zero frequency component in the encoder
}

class DAC_FREQ(BaseModel, CodecMixin):
    def __init__(
        self,
        config,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.config = config
        self.sample_rate = sample_rate

        self.latent_dim = self.config["latent_dim"]
        self.hop_length = self.config["hop_length"]

        self.encoder = Encoder(
            channels=self.config["encoder_channels"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            f_kernel_size=self.config["f_kernel_size"],
            f_stride_size=self.config["f_stride_size"],
            f_padding_size=self.config["f_padding_size"],
            remove_zero_frequency=self.config["remove_zero_frequency"]
        )
            
        self.decoder = Decoder(
            channels=self.config["decoder_channels"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            f_kernel_size=self.config["f_kernel_size"],
            f_stride_size=self.config["f_stride_size"],
            f_padding_size=self.config["f_padding_size"],
            remove_zero_frequency=self.config["remove_zero_frequency"]

        )
        if self.config["quantizer"] == "rvq":
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.latent_dim,
                n_codebooks=self.config['n_codebooks'],
                codebook_size=self.config['codebook_size'],
                codebook_dim=self.config['codebook_dim'],
                quantizer_dropout=self.config['quantizer_dropout'],
            )
        elif self.config["quantizer"] == "subband":
            self.quantizer = Subband_Quantizer(n_dim=self.latent_dim, n_code_groups=self.config['n_code_groups'], 
                                           residual_layer=self.config['residual_layer'])
        else:
            raise ValueError("Invalid quantizer type. Use 'rvq' or 'subband'.")
        
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC_FREQ(config=v2_config).to("cuda")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # # Create gradient variable
    # grad = torch.zeros_like(out)
    # grad[:, :, grad.shape[-1] // 2] = 1

    # # Make a backward pass
    # out.backward(grad)

    # # Check non-zero values
    # gradmap = x.grad.squeeze(0)
    # gradmap = (gradmap != 0).sum(0)  # sum across features
    # rf = (gradmap != 0).sum()

    # print(f"Receptive field: {rf.item()}")

    # x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    # model.decompress(model.compress(x, verbose=True), verbose=True)
