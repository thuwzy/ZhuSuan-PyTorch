import os
import math
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from zhusuan.framework.bn import BayesianNet
from zhusuan.distributions import Normal, Bernoulli
from zhusuan.variational.elbo import ELBO
from zhusuan.invertible import MaskCoupling, get_coupling_mask, Scaling, RevSequential, RevNet
from examples.utils import load_mnist_realval, save_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
option = ["NICE", "Planar", "HouseHolder"]
using_method = option[2]

class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, x_dim),
            nn.Sigmoid()
        )

    def forward(self, observed):
        self.observe(observed)
        batch_len = self.observed['z'].shape[0]

        prior = Normal(mean=torch.zeros([batch_len, self.z_dim]),
                       std=torch.ones([batch_len, self.z_dim]), dtype=torch.float32)
        z = self.sn(prior, "z", reduce_mean_dims=[0], reduce_sum_dims=[1])
        x_probs = self.gen(z)
        self.cache['x_mean'] = x_probs

        dis = Bernoulli(probs=x_probs)
        sample_x = self.sn(dis, "x",
                           reduce_mean_dims=[0],
                           reduce_sum_dims=[1])
        assert (sample_x.shape[0] == batch_len)

        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super(Variational, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.common_layer = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )

        self.output_mean = nn.Linear(500, z_dim)
        self.output_sd = nn.Linear(500, z_dim)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']

        z_logits = self.common_layer(x)
        self.cache['z_logits'] = z_logits

        z_mean = self.output_mean(z_logits)
        z_sd = torch.exp(self.output_sd(z_logits))

        normal = Normal(mean=z_mean, std=z_sd, is_reparameterized=True)
        z = self.sn(normal, "z",
                    reduce_mean_dims=[0],
                    reduce_sum_dims=[1])
        assert (z.shape[1] == self.z_dim)
        return self


class NICEFlow(nn.Module):
    def __init__(self, z_dim, mid_dim, num_coupling, num_hidden):
        super(NICEFlow, self).__init__()
        masks = get_coupling_mask(z_dim, 1, num_coupling)
        flows = [MaskCoupling(
            in_out_dim=z_dim,
            mid_dim=mid_dim,
            hidden=num_hidden,
            mask=masks[i].to(device)
        ) for i in range(num_coupling)]
        flows.append(Scaling(z_dim))
        self.flow = RevSequential(flows)

    def forward(self, z, **kwargs):
        out, log_det_J = self.flow.forward(z[0], **kwargs)
        res = {"z": out}
        return res, log_det_J


class HF(RevNet):
    def __init__(self, z_dim, is_first=False, v_dim=None):
        super(HF, self).__init__()
        if is_first:
            self.v_layer = nn.Linear(v_dim, z_dim)
        else:
            self.v_layer = nn.Linear(z_dim, z_dim)

    def _forward(self, inputs, **kwargs):
        z = inputs[0]
        v = inputs[1]
        v_new = self.v_layer(v)
        vvT = torch.bmm(v_new.unsqueeze(2), v_new.unsqueeze(1))
        vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(2)
        norm_sq = torch.sum(v_new * v_new, dim=1, keepdim=True)
        norm_sq = norm_sq.expand(norm_sq.shape[0], v_new.shape[1])
        z_new = z - 2 * vvTz / norm_sq
        return (z_new, v_new), torch.zeros([1, 1])


class HouseHolderFlow(RevNet):
    def __init__(self, z_dim, v_dim, n_flows):
        super(HouseHolderFlow, self).__init__()
        _first_kwarg = {'is_first': True, 'v_dim': v_dim}

        self.flow = RevSequential([HF(z_dim, **_first_kwarg) if _ == 0 else HF(z_dim) for _ in range(n_flows)])

    def _forward(self, inputs, **kwargs):
        out, log_det = self.flow.forward(inputs, **kwargs)
        res = {"z": out[0]}
        return res, log_det
    

class PF(nn.Module):
    def __init__(self, z_dim):
        super(PF, self).__init__()
        self.u = nn.Parameter(torch.rand([1, z_dim]))
        self.w = nn.Parameter(torch.rand([1, z_dim]))
        self.b = nn.Parameter(torch.rand([1]))

    def forward(self, z, **kwargs):
        def m(z):
            return F.softplus(z) - 1.

        def h(z):
            return torch.tanh(z)

        def h_prime(z):
            return 1. - torch.square(h(z))

        inner = torch.sum(self.w * self.u)
        u = self.u + (m(inner) - inner) * self.w / torch.square(self.w.norm())
        activation = torch.sum(self.w * z, dim=1, keepdim=True) + self.b
        z_new = z + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1. + torch.sum(u * psi, dim=1, keepdim=True)))
        return z_new, log_det


class PlanarFlow(nn.Module):
    def __init__(self, z_dim, n_flows):
        super(PlanarFlow, self).__init__()
        self.flows = nn.Sequential(*[PF(z_dim) for _ in range(n_flows)])

    def forward(self, z, **kwargs):
        out, log_det = self.flows(z[0])
        res = {'z': out}
        return res, log_det



def main():
    # Define model parameters
    epoch_size = 10
    batch_size = 64

    z_dim = 40
    x_dim = 28 * 28 * 1
    lr = 0.001

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)

    if using_method == "NICE":
        mid_dim_flow = 64
        num_coupling = 10
        num_hidden_per_coupling = 4
        nice_flow = NICEFlow(z_dim, mid_dim_flow, num_coupling, num_hidden_per_coupling)
        model = ELBO(generator, variational, transform=nice_flow, transform_var=['z'])
    elif using_method == "Planar":
        planar_flow = PlanarFlow(z_dim, 1)
        model = ELBO(generator, variational, transform=planar_flow, transform_var=['z'])
    elif using_method == "HouseHolder":
        householder_flow = HouseHolderFlow(z_dim, 500, 5)
        model = ELBO(generator, variational, transform=householder_flow, transform_var=['z'], auxillary_var=['z_logits'])
    else:
        raise NotImplementedError("please select correct method")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()

    # do train
    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = torch.as_tensor(x_train[step * batch_size:min((step + 1) * batch_size, len_)])
            x = torch.reshape(x, [-1, x_dim])
            if x.shape[0] != batch_size:
                continue
            loss = model({'x': x})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(loss.detach().cpu().numpy())))

    # eval
    batch_x = x_test[0:batch_size]
    nodes_q = variational({'x': torch.as_tensor(batch_x)}).nodes
    z = nodes_q['z'].tensor
    cache = generator({'z': z}).cache
    sample = cache['x_mean'].detach().cpu().numpy()

    z = nodes_q['z'].tensor
    cache = generator({'z': z}).cache
    sample_gen = cache['x_mean'].detach().cpu().numpy()

    result_fold = './results/flow-VAE'
    if not os.path.exists(result_fold):
        os.mkdir(result_fold)

    save_img(batch_x, os.path.join(result_fold, 'origin_x_VAE+NICE.png'))
    save_img(sample, os.path.join(result_fold, 'reconstruct_x_VAE+NICE.png'))
    save_img(sample_gen, os.path.join(result_fold, 'sample_x_VAE+NICE.png'))


if __name__ == '__main__':
    main()
