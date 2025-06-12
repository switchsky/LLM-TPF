import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Conv_Blocks import Inception_Block_V2
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def FFT_for_Period(x, weights, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    normalized_weights = weights / weights.sum()

    frequency_list = (abs(xf) * normalized_weights).mean(0).sum(-1)
    frequency_list[0] = 0

    _, top_list = random_select_with_temperature(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    period = x.shape[1] // top_list

    weighted_mean = (abs(xf) * normalized_weights).sum(-1)[:, top_list]
    return period, weighted_mean


def random_select_with_temperature(frequency_list, k, temperature=0.9):
    frequency_tensor = torch.tensor(frequency_list, dtype=torch.float32)
    probabilities = torch.softmax(frequency_tensor / temperature, dim=0)
    selected_indices = torch.multinomial(probabilities, k, replacement=False)
    selected_values = frequency_tensor[selected_indices]
    return selected_values,selected_indices,



class TimesNet(nn.Module):
    def __init__(self, configs, device,word_embedding):
        super().__init__()
        self.weights = None
        self.configs = configs
        self.device = device
        self.seq_len = configs.seq_len
        self.k = configs.timesnet_k
        self.word_embedding = word_embedding.T
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)

    def forward(self, x):
        B, T, N = x.size()
        # word_embedding
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1) #word_embedding shape: 100,500,768
        self.n = N
        self.out_channel = 32
        self.conv = nn.Sequential(
            Inception_Block_V2(self.n, self.out_channel,
                               num_kernels=self.configs.num_kernels),
            nn.GELU(),
            Inception_Block_V2(self.out_channel, self.n,
                               num_kernels=self.configs.num_kernels)
        ).to(self.device)
        self.weights = nn.Parameter(torch.ones(x.shape[-1])).to(self.device)
        period_list, period_weight = FFT_for_Period(x,self.weights ,self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 确保 period 不为 0，默认设置为 1
            if period == 0:
                period = 1

            # padding
            if (self.seq_len) % period != 0:
                length = (
                                 ((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len)
                out = x
            # reshape
            try:
                out = out.reshape(B, length // period, period,
                                    N).permute(0, 3, 1, 2).contiguous()
            except RuntimeError:
                print("length:{},period:{}".format(length, period))
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = out[:, :(self.seq_len), :]
            Linear = nn.Linear(out.shape[-1], 768).to(self.device)
            Linear2 = nn.Linear(768, out.shape[-1]).to(self.device)
            out = Linear(out.transpose(0, 1))
            out, _ = self.cross_attention(out, self.word_embedding.transpose(0, 1), self.word_embedding.transpose(0, 1))
            out = Linear2(out.transpose(0, 1))
            res.append(out)
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = x
        return res
