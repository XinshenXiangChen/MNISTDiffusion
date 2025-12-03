
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from scipy import io

from network import DDPM, UNET


def train_mnist():
    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400  # 500
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 256  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './data/diffusion_output/'
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    ddpm = DDPM(nn_model=UNET(in_channels=1, n_features=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    LOSS = []

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            LOSS.append(loss_ema)
            optim.step()

        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")
            io.savemat('loss.mat', {'loss': LOSS})


if __name__ == "__main__":
    train_mnist()


