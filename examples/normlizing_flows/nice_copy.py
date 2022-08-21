import torchvision
import os
import torch
import numpy as np
import torch.nn as nn
from examples.utils import fetch_dataloaders, save_img, check_dir
import torch.nn.functional as F
import IPython


def dequantize(x, dataset):
    '''Dequantize data.

    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).

    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    noise = torch.distributions.Uniform(0., 1.).sample(x.size())
    return (x * 255. + noise) / 256.


def prepare_data(x, dataset, zca=None, mean=None, reverse=False):
    """Prepares data for NICE.

    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        dataset: name of dataset.
        zca: ZCA whitening transformation matrix.
        mean: center of original dataset.
        reverse: True if in inference mode, False if in training mode.
    Returns:
        transformed data.
    """
    if reverse:
        assert len(list(x.size())) == 2
        [B, W] = list(x.size())

        if dataset in ['mnist', 'fashion-mnist']:
            assert W == 1 * 28 * 28
            x += mean
            x = x.reshape((B, 1, 28, 28))
        elif dataset in ['svhn', 'cifar10']:
            assert W == 3 * 32 * 32
            x = torch.matmul(x, zca.inverse()) + mean
            x = x.reshape((B, 3, 32, 32))
    else:
        assert len(list(x.size())) == 4
        [B, C, H, W] = list(x.size())

        if dataset in ['mnist', 'fashion-mnist']:
            assert [C, H, W] == [1, 28, 28]
        elif dataset in ['svhn', 'cifar10']:
            assert [C, H, W] == [3, 32, 32]

        x = dequantize(x, dataset)
        x = x.reshape((B, C * H * W))

        if dataset in ['mnist', 'fashion-mnist']:
            x -= mean
        elif dataset in ['svhn', 'cifar10']:
            x = torch.matmul((x - mean), zca)
    return x


class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim),
            nn.ReLU()
        )
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()
            ) for _ in range(hidden - 1)
        ])
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, reverse=False):
        [B, W] = list(x.size())
        x = x.reshape((B, W // 2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)

        if reverse:
            on = on - shift

        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)

        return x.reshape((B, W))


class Scaling(nn.Module):
    def __init__(self, dim):
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim), requires_grad=True)
        )

    def forward(self, x, reverse=False):
        log_det_j = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_j


class NICE(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config):
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=(mask_config + i) % 2
                     ) for i in range(coupling)]
        )
        self.scaling = Scaling(in_out_dim)

    def g(self, z):
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def sample(self, size):
        z = self.prior.sample((size, self.in_out_dim))
        return self.g(z)

    def log_porb(self, x):
        z, log_det_j = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_j

    def forward(self, x):
        return self.log_porb(x)


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size)
        return torch.log(z) - torch.log(1. - z)


def main():
    batch_size = 200
    epoch_size = 40
    sample_size = 64
    coupling = 4
    mask_config = 1.

    # Optim Parameters
    lr = 1e-3

    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 5
    prior = StandardLogistic()

    model = NICE(prior=prior,
                 coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 hidden=hidden,
                 mask_config=mask_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-4)

    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                          train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size, shuffle=True, num_workers=2)

    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=False, dequantify=False)
    num_iter = -1
    mean = torch.load('./mnist_mean.pt')
    # for i in model.parameters():
    #     print(i.shape)
    for epoch in range(epoch_size):
        stats = []
        for _, data in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            inputs = data[0]
            inputs = prepare_data(
                inputs, 'mnist', zca=None, mean=mean)
            # loss = -model.nodes['x'].log_prob(inputs).mean()
            loss = -model(inputs).mean()
            loss.backward()
            optimizer.step()
            num_iter += 1
            stats.append(loss.detach().numpy())
            if num_iter % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    path = os.path.join(os.getcwd(), 'results', 'NICE')
                    check_dir(path)
                    samples = model.sample(sample_size).cpu()
                    samples = prepare_data(
                        samples, 'mnist', zca=None, mean=mean, reverse=True)

                    # torchvision.utils.save_image(torchvision.utils.make_grid(samples.clone()),
                    #                              path + '/samples/' + "sample" + 'iter%d.png' % num_iter)
                    # IPython.embed()
                    # save_img(reconst.numpy(), os.path.join(path, "recons{}.png".format(num_iter)))
                    # samples = model.sample(64)
                    # samples = prepare_data(
                    #     samples, 'mnist', zca=None, mean=mean, reverse=True)
                    samples = samples.reshape([sample_size, -1])
                    save_img(samples.numpy(), os.path.join(path, "sample{}.png".format(num_iter)))
        print("Epoch:[{}/{}], Log Likelihood: {:.4f} iter{}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats)), num_iter
        ))

    model.eval()
    with torch.no_grad():
        samples = model.sample(64)
        print(samples.shape)
        path = os.path.join(os.getcwd(), 'results', 'NICE')
        check_dir(path)
        save_img(samples.detach().cpu().numpy(), os.path.join(path, 'sample-NICE2.png'))


if __name__ == '__main__':
    main()
