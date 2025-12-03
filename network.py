import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, MaxPool2d
from torch.nn.functional import mse_loss, max_pool2d


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(

            # from in_channels to out_channels goes from a tensor of shape (1, 28, 28) where 28 are the (channels, height width)
            # converts it into a tensor of shape (128, 28, 28)
            # keeping the shape is thanks to the padding, stride = 1 and kernelsize = 3x3

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # from in_channels to out_channels goes from a tensor of shape (1, 28, 28) where 28 are the (channels, height width)
        # converts it into a tensor of shape (128, 28, 28)
        # keeping the shape is thanks to the padding, stride = 1 and kernelsize = 3x3

        self.conv1 = Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
    def __init__(self, in_channels, out_channels):
        super(Encoder,self).__init__()

        self.encode = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            MaxPool2d(2),
        )

    def forward(self, input):
        input = self.encode(input)
        return input


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder,self).__init__()

        self.encode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),

        )
    def forward(self, input, skip):
        # we concatenate the previous upsampled + convolution tensor at dimension 1
        # dimension 1 becuase its the channels dimension
        # this concatenation is part of the unet

        input = torch.cat((input, skip), 1)
        input = self.encode(input)
        return input

class Embedded(nn.Module):
    def __init__(self, input_dim, embed_output_dim):
        super(Embedded, self).__init__()
        self.input_dim = input_dim
        self.embed_output_dim = embed_output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, embed_output_dim),
            nn.GELU(),
            nn.Linear(embed_output_dim, embed_output_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.model(x)
        x = x.view(-1, self.embed_output_dim, 1, 1)
        return x


class UNET(nn.Module):
    # in_channels defaults to 1 because grayscale mnist
    def __init__(self, in_channels=1, n_features=128, n_classes=10):
        super(UNET,self).__init__()

        self.n_features = n_features
        self.in_channels = in_channels
        self.n_classes = n_classes


        self.encode0 = ResidualBlock(in_channels, n_features)
        # referenced code doesnt change the in / out channels for the first Encoding
        # which my guess i for efficiency reasons and because mnist is very small
        self.encode1 = Encoder(n_features, n_features)
        self.encode2 = Encoder(n_features, 2*n_features)

        # 2 encode does 2 maxpoll2d(2) which means the size would become 28/2*2 = 7
        # to reduce it to a tensor of 2*nfeaturesx1x1 to embed label and t
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = Embedded(1, 2 * n_features)
        self.timeembed2 = Embedded(1, 1 * n_features)
        self.contextembed1 = Embedded(n_classes, 2 * n_features)
        self.contextembed2 = Embedded(n_classes, 1 * n_features)

        self.decode0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_features, 2*n_features, 7, 7),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU(),

        )
        # decode0 returns out_channels = 2*nfeatures concatenated with encode2 with out_channels also 2*nfeatures
        # input_channels = 2*nfeatures + 2*nfeatures = 4*nfeatures
        self.decode1 = Decoder(4 * n_features, n_features)

        #previous up outputs n_features, with concatenation from encoded1
        # input_channels = nfeatures + nfeatures = 2*nfeatures
        self.decode2 = Decoder(2 * n_features, n_features)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features, 3, 1, 1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, self.in_channels, 3, 1, 1),
        )
    def forward(self, x, c, t, context_mask):
        x = self.encode0(x)
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        hiddenvec = self.to_vec(encode2)

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # this is for classifier free model, we sometimes drop the label


        #  context_mask[:, None] converts a tensor of shape tensor([0, 1, 0, 1])
        # to [[0],
        #    [1],
        #    [0],
        #    [1]]

        context_mask = context_mask[:, None]

        # context_mask.repeat(1, self.n_classes), copies the values of eahch row
        # self.n_classes times
        # to [[0, ... self.n_classes times],
        #    [1, ... self.n_classes times],
        #    [0, ... self.n_classes times],
        #    [1, ... self.n_classes times]]
        # resulting shape [batch_size,self.n_classes]
        context_mask = context_mask.repeat(1, self.n_classes)

        # given c is in one_hot its shape is also
        # [batch_size, ... self.n_classes times]
        c = c * context_mask

        # contextembed1/2 and timeembed1/2 return shape []
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t)
        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t)

        decode0 = self.decode0(hiddenvec)
        decode1 = self.decode1(decode0*cemb1 + temb1, encode2)
        decode2 = self.decode2(decode1*cemb2 + temb2, encode1)
        # Concatenate decode2 with the original transformed input x to match expected input channels
        out = self.out(torch.cat((decode2, x), 1))
        return out


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
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in noise_scheduler(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        # this returns a list of size batch_size with numbers between

        # IMPORTANT (x.shape[0], ) is the only way
        # (x.shape[0] is the dimension of the batch size (batch_size, 1, 28, 28)
        # DOING ONLY (x.shape[0]) python will NOT parse it as a tuple
        time_step = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        real_noise = torch.randn_like(x)
        x_t = (
                self.sqrtab[time_step, None, None, None] * x
                + self.sqrtmab[time_step, None, None, None] * real_noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + (1 - self.drop_prob)).to(self.device)

        # Normalize time_step to [0, 1] range and convert to float for embedding
        time_step_normalized = (time_step / self.n_T).float()
        predicted_noise = self.nn_model(x_t, c, time_step_normalized, context_mask)
        return self.loss_mse(real_noise, predicted_noise)

    def prediction(self, n_sample, size, c, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = c.to(device)
        # c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        # c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        x_i_store = []  # keep track of generated steps in case want to plot something
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
