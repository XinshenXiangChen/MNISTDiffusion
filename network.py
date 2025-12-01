import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

    def forward(self, input):

        return input

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

    def forward(self, input):
        return input

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()

    def forward(self, input):
        return input

def noise_scheduler():
    pass

class DDPM(nn.Module):
    def __init__(self, t_n=1000, batch_size=32):
        super(DDPM, self).__init__()
        self.t_n = t_n
        self.batch_size = batch_size

    def forward(self, x):

        # this returns a list of size batch_size with numbers between
        time_step = torch.randn(1, self.t_n + 1, self.batch_size)
        real_noise = torch.randn_like(x)

        predicted_noise = torch.randn_like(x)
        return mse_loss(real_noise, predicted_noise)


