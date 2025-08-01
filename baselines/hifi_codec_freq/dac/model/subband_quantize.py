import torch
import torch.nn as nn
from dac.nn.quantize import ResidualVectorQuantize

class Subband_Quantizer(torch.nn.Module):
    def __init__(self, n_dim=1024, n_code_groups=6, residual_layer=2):
        super(Subband_Quantizer, self).__init__()
        assert n_dim % n_code_groups == 0

        self.subband_RVQ = nn.ModuleList([
            ResidualVectorQuantize(n_dim // n_code_groups, n_codebooks=residual_layer, codebook_size=1024) for _ in range(n_code_groups)
        ])
        
        self.residul_layer = residual_layer
        self.n_code_groups = n_code_groups
        self.n_dim = n_dim
   
    def forward(self, z, n_quantizers: int = None):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
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
        B, C, T = z.shape
        z = z.reshape(B, self.n_code_groups, C // self.n_code_groups, T)
        zq, codes, latents, commitment_loss, codebook_loss = [], [], [], 0, 0
        for i in range(self.n_code_groups):
            _z = z[:, i, :, :] # B, C // n_code_groups, T
            _z_q, _codes, _latents, _commitment_loss, _codebook_loss = self.subband_RVQ[i](_z, n_quantizers=n_quantizers)
            zq.append(_z_q)
            codes.append(_codes)        
            latents.append(_latents)
            commitment_loss = commitment_loss + _commitment_loss
            codebook_loss = codebook_loss + _codebook_loss
        zq = torch.cat(zq, dim=1)
        codes = torch.cat(codes, dim=1)
        latents = torch.cat(latents, dim=1)
        commitment_loss = commitment_loss / self.n_code_groups
        codebook_loss = codebook_loss / self.n_code_groups
        return zq, codes, latents, commitment_loss, codebook_loss
    
