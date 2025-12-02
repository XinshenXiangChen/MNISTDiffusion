import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn.functional import mse_loss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(

            # from in_channels to out_channels goes from a tensor of shape (1, 28, 28) where 28 are the (channels, height width)
            # converts it into a tensor of shape (128, 28, 28)
            # keeping the shape is thanks to the padding, stride = 1 and kernelsize = 3x3

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.conv2 = Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


    def forward(self, input):
        """

        we keep an original to add to the convoluted image
        y = F(x, weights) + x
        where x is the original
        and F(x, weights) is the convoluted image
        we are training it to predict the right convolution weights

        PROBLEM: x and x2 are not the same size so out = x + x2 would give an error
        Solutions:
        i. 1x1 convolution projection

        original (x) = conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=1)
        y - F(x, weights) + original (x)


        ii. pseudo residual trick

        Instead of doing y = F(x, weights) + x
        we can do y = x1 + x2
        where x1 = F(x, weights)
        and x2 = G(x1, weights)
        To make sure the outputs are the same
        Currently implemented is the pseudo residual trick
        """

        x1 = self.conv1(input)
        x2 = self.conv2(x1)

        out = x1 + x2

        return out / 1.414

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        # from in_channels to out_channels goes from a tensor of shape (1, 28, 28) where 28 are the (channels, height width)
        # converts it into a tensor of shape (128, 28, 28)
        # keeping the shape is thanks to the padding, stride = 1 and kernelsize = 3x3

        self.conv1 = Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



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

def noise_scheduler(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }



class DDPM(nn.Module):
    def __init__(self, t_n=1000, batch_size=32, betas=(1e-4, 0.02), n_T=1000):
        super(DDPM, self).__init__()
        self.t_n = t_n
        self.batch_size = batch_size


        for k, v in noise_scheduler(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)


    def forward(self, x):

        # this returns a list of size batch_size with numbers between
        time_step = torch.randn(1, self.t_n + 1, self.batch_size)
        real_noise = torch.randn_like(x)

        predicted_noise = torch.randn_like(x)
        return mse_loss(real_noise, predicted_noise)


