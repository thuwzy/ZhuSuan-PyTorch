import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.append("..")
from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.importance_weighted_objective import ImportanceWeightedObjective
from examples.utils import load_mnist_realval, save_img

hidden_dim = 500


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, n_samples):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.gen_sq = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Sigmoid()
        )

    def forward(self, observed: dict):
        """
        this step will add the observed (name:tensor) pair to the
        self._obseved dict
        """
        # print("forward of decoder")
        self.observe(observed)

        try:
            batch_len = self.observed['z'].shape[1]
        except:
            batch_len = 64
        mean = torch.zeros([batch_len, self.z_dim])
        std = torch.ones([batch_len, self.z_dim])

        # the prior of z is sampled from standard normal distribution
        z = self.sn("Normal",
                    name="z",
                    mean=mean,
                    std=std,
                    reparameterize=False,
                    is_reparameterized=False,
                    n_samples=self.n_samples,
                    reduce_mean_dims=None,
                    reduce_sum_dims=[2]
                    )
        x_probs = self.gen_sq(z)
        self.cache["x_mean"] = x_probs

        x_sample = self.sn(
            'Bernoulli',
            name='x',
            probs=x_probs,
            reduce_mean_dims=None,
            reduce_sum_dims=[2]
        )
        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, n_samples):
        super(Variational, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        # define common forward layer
        self.output_logits = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # each of them output the mean and logstd of posterior of q(z|x)
        self.output_mean = nn.Linear(hidden_dim, z_dim)
        self.output_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']
        z_logits = self.output_logits(x)
        z_mean = self.output_mean(z_logits)
        z_logstd = self.output_logstd(z_logits)
        z_std = torch.exp(z_logstd)
        # get samples from StochasticTensor, z is not observed, so self.sn return new samples
        # z.shape [n_samples, batch_shape, z_dim]
        # the z sample will be stored to self._nodes["z"]._dist.sample_cache
        z = self.sn("Normal",
                    name="z",
                    mean=z_mean,
                    std=z_std,
                    reparameterize=False,
                    is_reparameterized=False,
                    n_samples=self.n_samples,
                    reduce_mean_dims=None,
                    reduce_sum_dims=[2]
                    )
        return self


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    epoch_size = 10
    batch_size = 64

    z_dim = 40
    x_dim = 28 * 28 * 1

    lr = 0.001

    lb_samples = 40

    generator = Generator(x_dim, z_dim, lb_samples)
    variational = Variational(x_dim, z_dim, lb_samples)
    model = ImportanceWeightedObjective(generator, variational, axis=0, estimator="vimco").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()

    x_train = torch.as_tensor(x_train).to(device)
    x_test = torch.as_tensor(x_test).to(device)

    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = x_train[step * batch_size:min((step + 1) * batch_size, len_)]
            x = torch.reshape(x, [-1, x_dim])
            if x.shape[0] != batch_size:
                break
            loss = model({'x': x})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (step + 1) % 100 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, epoch_size, step + 1, num_batches,
                                                                        loss.clone().cpu().detach().numpy()))
                # float(loss.clone().detach().numpy())))
    end = time.time()
    print("using", end - start, "s")
    batch_x = x_test[0:64]
    nodes_q = variational({'x': batch_x}).nodes
    z = nodes_q['z'].tensor
    cache = generator({'z': z}).cache
    sample = cache['x_mean'][0].cpu().detach().numpy()

    cache = generator({}).cache
    sample_gen = cache['x_mean'][0].cpu().detach().numpy()

    result_fold = './result'
    if not os.path.exists(result_fold):
        os.mkdir(result_fold)

    batch_x = batch_x.cpu().detach().numpy()

    save_img(batch_x, os.path.join(result_fold, 'iw_origin_x_.png'))
    save_img(sample, os.path.join(result_fold, 'iw_reconstruct_x_.png'))
    save_img(sample_gen, os.path.join(result_fold, 'iw_sample_x_.png'))


if __name__ == '__main__':
    main()
