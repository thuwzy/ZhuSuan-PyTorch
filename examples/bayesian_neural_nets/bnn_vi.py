import torch

import os
import math
import numpy as np
import sys

sys.path.append('..')

from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from examples.utils import load_uci_boston_housing, standardize


class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.y_logstd = torch.nn.parameter.Parameter(torch.nn.init.constant_(torch.empty([1], dtype = torch.float32), 0.0), requires_grad=True)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']
        h = x.repeat([self.n_particles, *len(x.shape) * [1]])

        batch_size = x.shape[0]

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w = self.sn('Normal',
                        name='w' + str(i),
                        mean=torch.zeros([n_out, n_in + 1]),
                        std=torch.ones([n_out, n_in + 1]),
                        group_ndims=2,
                        n_samples=self.n_particles,
                        reduce_mean_dims=[0])
            w = torch.unsqueeze(w, 1)
            w = w.repeat([1, batch_size, 1, 1])
            h = torch.cat((h, torch.ones([*h.shape[:-1], 1]).to(self.device)), -1)
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
                multiplier=456)  # training data size
        return self


class Variational(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles

        self.w_means = []
        self.w_logstds = []

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_mean = torch.nn.init.constant_(torch.empty([n_out, n_in + 1], dtype = torch.float32), 0)
            _name = 'w_mean_' + str(i)
            self.__dict__[_name] = w_mean
            w_logstd = torch.nn.init.constant_(torch.empty([n_out, n_in + 1], dtype = torch.float32), 0)
            _name = 'w_logstd_' + str(i)
            self.__dict__[_name] = w_logstd
            w_mean = torch.nn.parameter.Parameter(w_mean, requires_grad=True)
            w_logstd = torch.nn.parameter.Parameter(w_logstd, requires_grad=True)
            self.w_means.append(w_mean)
            self.w_logstds.append(w_logstd)

        self.w_means = torch.nn.ParameterList(self.w_means)    
        self.w_logstds = torch.nn.ParameterList(self.w_logstds)   

    def forward(self, observed):
        self.observe(observed)
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.sn('Normal',
                    name='w' + str(i),
                    mean=self.w_means[i],
                    logstd=self.w_logstds[i],
                    group_ndims=2,
                    n_samples=self.n_particles,
                    reparameterize=True,
                    reduce_mean_dims=[0])
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

    lb_samples = 512
    epoch_size = 5000
    batch_size = 114

    n_hiddens = [50]
    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size:', layer_sizes)

    net = Net(layer_sizes, lb_samples)
    variational = Variational(layer_sizes, lb_samples)
    # for param in variational.parameters():
    #     print("a",type(param), param.size())
    model = ELBO(net, variational)

    model.to(device)
    x_train = torch.as_tensor(x_train).to(device)
    y_train = torch.as_tensor(y_train).to(device)
    x_test = torch.as_tensor(x_test).to(device)
    y_test = torch.as_tensor(y_test).to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # print('parameters length:', len(model.parameters()))

    len_ = len(x_train)
    num_batches = math.floor(len_ / batch_size)

    test_freq = 20

    for epoch in range(epoch_size):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm, :]
        y_train = y_train[perm]

        for step in range(num_batches):
            x = x_train[step * batch_size:(step + 1) * batch_size]
            y = y_train[step * batch_size:(step + 1) * batch_size]
            lbs = model({'x': x, 'y': y})
            optimizer.zero_grad()
            lbs.backward()
            optimizer.step()

            if (step + 1) % num_batches == 0:
                rmse = net.cache['rmse'].clone().cpu().detach().numpy()
                print("Epoch[{}/{}], Step [{}/{}], Lower bound: {:.4f}, RMSE: {:.4f}".format(epoch + 1, epoch_size,
                                                                                             step + 1,
                                                                                             num_batches,
                                                                                             float(lbs.clone().cpu().detach().numpy()),
                                                                                             float(rmse) * std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = x_test
            y_t = y_test
            lbs = model({'x': x_t, 'y': y_t})
            rmse = net.cache['rmse'].clone().cpu().detach().numpy()
            print('>> TEST')
            print('>> Test Lower bound: {:.4f}, RMSE: {:.4f}'.format(float(lbs.clone().cpu().detach().numpy()), float(rmse) * std_y_train))


if __name__ == '__main__':
    main()
