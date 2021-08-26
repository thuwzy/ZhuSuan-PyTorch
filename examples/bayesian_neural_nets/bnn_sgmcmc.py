import torch

import os
import math
import numpy as np
import sys

sys.path.append('..')

from zhusuan.framework.bn import BayesianNet
from zhusuan.mcmc.SGLD import SGLD

from examples.utils import load_uci_boston_housing, standardize


class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        with torch.no_grad():
            self.y_logstd = torch.nn.init.constant_(torch.empty([1], dtype=torch.float32), val=-1.95)

        self.w_logstds = []

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_logstd_ = torch.nn.init.constant_(torch.empty([n_out, n_in + 1], dtype=torch.float32), val = 0.0)
            _name = 'w_logstd_' + str(i)
            self.__dict__[_name] = w_logstd_
            self.w_logstds.append(w_logstd_)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']
        h = x.repeat([self.n_particles, *len(x.shape) * [1]])

        batch_size = x.shape[0]

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w = self.sn('Normal',
                        name='w' + str(i),
                        mean=torch.zeros([n_out, n_in + 1]),
                        logstd=self.w_logstds[i],
                        group_ndims=2,
                        n_samples=self.n_particles,
                        reduce_mean_dims=[0])
            w = torch.unsqueeze(w, 1)
            w = w.repeat([1, batch_size, 1, 1])
            h = torch.cat([h, torch.ones([*h.shape[:-1], 1]).to(self.device)], -1)
            h = torch.unsqueeze(h, -1)
            p = torch.sqrt(torch.as_tensor(h.shape[2], dtype=torch.float32))
            h = torch.matmul(w, h) / p
            h = torch.squeeze(h, -1)

            if i < len(self.layer_sizes) - 2:
                h = torch.nn.ReLU()(h)
        y_mean = torch.squeeze(h, 2)

        y = self.observed['y']
        y_pred = torch.mean(y_mean, 0)
        self.cache['rmse'] = torch.sqrt(torch.mean((y - y_pred) ** 2))

        self.sn('Normal',
                name='y',
                mean=y_mean,
                logstd=self.y_logstd,
                reparameterize=True,
                reduce_mean_dims=[0, 1],
                multiplier=456)
        return self


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.join('data', 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape

    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = standardize(y_train, y_test)

    print('data size:', len(x_train))

    lb_samples = 20
    epoch_size = 5000
    batch_size = 114

    n_hiddens = [50]

    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size: ', layer_sizes)

    net = Net(layer_sizes, lb_samples).to(device)
    print('parameters length: ', len([_ for _ in net.parameters()]))

    lr = 1e-3
    model = SGLD(lr).to(device)

    len_ = len(x_train)
    num_batches = math.floor(len_ / batch_size)

    test_freq = 20

    x_train = torch.as_tensor(x_train).to(device)
    y_train = torch.as_tensor(y_train).to(device)
    x_test = torch.as_tensor(x_test).to(device)
    y_test = torch.as_tensor(y_test).to(device)

    for epoch in range(epoch_size):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm, :]
        y_train = y_train[perm]

        for step in range(num_batches):
            x = torch.as_tensor(x_train[step * batch_size:(step + 1) * batch_size])
            y = torch.as_tensor(y_train[step * batch_size:(step + 1) * batch_size])

            re_sample = True if epoch == 0 and step == 0 else False
            w_samples = model.sample(net, {'x': x, 'y': y}, re_sample)

            for i, (k, w) in enumerate(w_samples.items()):
                assert (w.shape[0] == lb_samples)
                esti_logstd = 0.5 * torch.log(torch.mean(w * w, [0]))
                net.w_logstds[i] = esti_logstd.detach()

            if (step + 1) % num_batches == 0:
                net.forward({**w_samples, 'x': x, 'y': y})
                rmse = net.cache['rmse'].clone().cpu().detach().numpy()
                print("Epoch[{}/{}], Step [{}/{}], RMSE: {:.4f}".format(epoch + 1, epoch_size, step + 1, num_batches,
                                                                        float(rmse) * std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = torch.as_tensor(x_test)
            y_t = torch.as_tensor(y_test)
            net.forward({**w_samples, 'x': x_t, 'y': y_t})
            rmse = net.cache['rmse'].clone().cpu().detach().numpy()
            print('>> TEST')
            print('>> Test RMSE: {:.4f}'.format(float(rmse) * std_y_train))


if __name__ == '__main__':
    main()

