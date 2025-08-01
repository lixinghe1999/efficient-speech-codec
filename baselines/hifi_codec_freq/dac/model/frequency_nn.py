import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd import Variable
from dac.nn.quantize import ResidualVectorQuantize

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(dilation[0], 1),
                               padding=(get_padding(kernel_size, dilation[0]), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[1], 1),
                               padding=(get_padding(kernel_size, dilation[1]), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[2], 1),
                               padding=(get_padding(kernel_size, dilation[2]), 2))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
        ])
        self.convs2.apply(init_weights)
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

def cal_dimension(n_fft, f_kernel_size, f_stride_size, f_padding_size):
    """
    calculate the output dimension of the convolutional layer
    n_fft: n_fft // 2 == the frequency dimension
    f_kernel_size: the kernel size of the convolutional layer
    f_stride_size: the stride size of the convolutional layer
    f_padding_size: the padding size of the convolutional layer
    """
    freq_dim = n_fft // 2 + 1
    for i in range(len(f_kernel_size)):
        freq_dim = (freq_dim + 2 * f_padding_size[i] - f_kernel_size[i]) // f_stride_size[i] + 1
    return freq_dim


class Encoder(torch.nn.Module):
    def __init__(self, channels=[2, 16, 32, 64, 128, 256, 256], n_fft=640, hop_length=320,
                    f_kernel_size=[5,3,3,3,3,4], f_stride_size=[2,2,2,2,2,1], f_padding_size=[0,0,0,0,0,0], remove_zero_frequency=False):
        super(Encoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.out_freq_dim = cal_dimension(n_fft, f_kernel_size, f_stride_size, f_padding_size)
        self.out_channel_dim = channels[-1]

        resblock_kernel_sizes = [3,7]
        resblock_dilation_sizes = [[1,3,5], [1,3,5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_layers = len(channels) - 1
        self.normalize = nn.ModuleList()

        conv_list = []
        norm_list = []
        res_list = []
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            conv_list.append(
                nn.Conv2d(channels[c_idx], channels[c_idx+1], (3, f_kernel_size[c_idx]), stride=(1, f_stride_size[c_idx]), 
                                        padding=(1, f_padding_size[c_idx])),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock(channels[c_idx+1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[c_idx+1], eps=1e-6, affine=True))
       
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
        
        self.remove_zero_frequency = remove_zero_frequency
        self.window = torch.hann_window(self.n_fft)

    def forward(self, audio):
        '''
        x: bs, 2, T, F
        out: bs, 256, n_frames, 2
        '''
        audio = audio.squeeze(1)
        device = audio.device
        x = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window.to(device), center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        # x: bs, F, T
        x = torch.view_as_real(x).permute(0, 3, 2, 1)
        bs, _, n_frames, n_freqs = x.shape
        if self.remove_zero_frequency:
            x = x[..., 1:]
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
                else:
                    xs += self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU_SLOPE) # bs, 256, n_frames, 2
        # bs, 256, n_frames, 9
        x = x.permute(0,3,1,2).reshape(bs, self.out_freq_dim * self.out_channel_dim, n_frames)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        
class Decoder(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 128, 256, 256, 256], n_fft=640, hop_length=320,
                    f_kernel_size=[5,3,3,3,3,4], f_stride_size=[2,2,2,2,2,1], f_padding_size=[0,0,0,0,0,0], remove_zero_frequency=False):
        super(Decoder, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.out_freq_dim = cal_dimension(n_fft, f_kernel_size, f_stride_size, f_padding_size)
        self.out_channel_dim = channels[-1]

        resblock_kernel_sizes = [3, 7]
        resblock_dilation_sizes = [[1,3,5], [1,3,5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_layers = len(channels) - 1
        self.normalize = nn.ModuleList()

        conv_list = []
        norm_list = []
        res_list = []
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            conv_list.append(
                nn.ConvTranspose2d(channels[self.num_layers-c_idx], channels[self.num_layers-c_idx-1], (3, f_kernel_size[self.num_layers-c_idx-1]), stride=(1, f_stride_size[self.num_layers-c_idx-1]), padding=(1, f_padding_size[self.num_layers-c_idx-1])),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock(channels[self.num_layers-c_idx-1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[self.num_layers-c_idx-1], eps=1e-6, affine=True))
       
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
        self.conv_post = weight_norm(nn.Conv2d(channels[0], 2, (5,5), (1,1), padding=(2,2)))
        self.conv_post.apply(init_weights)
        
        self.remove_zero_frequency = remove_zero_frequency
        self.window = torch.hann_window(self.n_fft)
        
    def forward(self, x):
        '''
        x: bs, 9*256, T
        out: bs, 
        '''
        bs, _, n_frames = x.shape

        x = x.reshape(bs, self.out_freq_dim, self.out_channel_dim, n_frames)
        x = x.permute(0,2,3,1).contiguous()
        
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
                else:
                    xs += self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x) # bs, 2, T, F
        # bs, 2, T, F
        
        x = x.permute(0,3,2,1).contiguous()
        if self.remove_zero_frequency:
            x = torch.cat([x, torch.zeros(bs, 1, n_frames, 2).to(x.device)], dim=1)
        x = torch.istft(torch.view_as_complex(x), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window.to(x.device), center=True, normalized=False, onesided=True, length=n_frames*self.hop_length)
        x = x.unsqueeze(1)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        
if __name__=='__main__':
    encoder = Encoder()
    decoder = Decoder()
    # quantizer = Subband_Quantizer(256*6, 6)
    quantizer = ResidualVectorQuantize(input_dim=256*6)
    input = torch.randn(4, 1, 441280)
    emb = encoder(input)
    print(emb.shape)
    z_q, codes, latents, commitment_loss, codebook_loss = quantizer(emb)
    out = decoder(z_q)
    print(out.shape)
    
    x = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    y = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(x/1000000, y/1000000)