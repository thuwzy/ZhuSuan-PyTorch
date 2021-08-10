import torch


import math
import os
import sys
import numpy as np

sys.path.append('..')
from zhusuan.transforms import *
from zhusuan.transforms.invertible import *
from zhusuan.framework import BayesianNet
from zhusuan.flow import Flow

from examples.utils import load_mnist_realval, save_img, save_image

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class NICE(BayesianNet):
    def __init__(self, num_coupling, in_out_dim, mid_dim, num_hidden, device):
        super().__init__()
        self.in_out_dim = in_out_dim
        masks = get_coupling_mask(in_out_dim, 1, num_coupling, device = device)
        coupling_layer = [AdditiveCoupling(in_out_dim, mid_dim, num_hidden, masks[i])
                     for i in range(num_coupling)]
        scaling_layer = Scaling(in_out_dim, device)
        self.flow = Sequential(coupling_layer + [scaling_layer])
        
        loc = torch.zeros([in_out_dim]).to(device)
        scale = torch.ones([in_out_dim]).to(device)

        self.sn('Logistic',
                name='z',
                loc=loc,
                scale=scale,
                device = device)
        self.sn('Flow',
                name='x',
                latent=self.nodes['z'].dist,
                transform=self.flow,
                n_samples=-1,
                device = device) # Not sample when initializing
    
    def sample(self, n_samples=1):
        return self.nodes['x'].dist.sample(n_samples)
    
    def forward(self, x):
        return self.nodes['x'].log_prob(x)

def main():


    print(device)

    batch_size = 200
    epoch_size = 1000
    sample_size = 64
    coupling = 4
    
    lr = 1e-3

    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 5

    model = NICE(num_coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 num_hidden=hidden,
                 device = device)

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-4)
    # for param in model.parameters():
    #     print("a",type(param), param.size())
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval(dequantify=True)
    x_train = torch.as_tensor(x_train).to(device)


    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)

    
    for epoch in range(epoch_size):
        model.train()
        stats = []
        for step in range(num_batches):
            x = torch.as_tensor(x_train[step * batch_size:min((step + 1) * batch_size, len_)])
            x = torch.reshape(x, [-1, full_dim])
            #print(model.nodes['x'].tensor.device)
            loss = -model.nodes['x'].log_prob(x)
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.append(loss.cpu().clone().detach().numpy())
        print("Epoch:[{}/{}], Log Likelihood: {:.4f}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats))
        ))
        if epoch%10 == 0:
            model.eval()
            sample_x = model.sample(n_samples=sample_size)
            result_fold = './result'
            if not os.path.exists(result_fold):
                os.mkdir(result_fold)
            save_img((sample_x).cpu().detach().numpy(), os.path.join(result_fold, str(epoch)+'sample-NICE.png'))
            
    
    model.eval()
    sample_x = model.sample(n_samples=sample_size)
    # print(sample_x.shape)
    # sample_x = torch.reshape(sample_x, [-1, 784])
    # print(sample_x.shape)
    # sample_x = torch.reshape(sample_x, [-1, 1, 28, 28])
    result_fold = './result'
    if not os.path.exists(result_fold):
        os.mkdir(result_fold)
    save_img((sample_x).cpu().detach().numpy(), os.path.join(result_fold, 'sample-NICE.png'))

    x = torch.as_tensor(x_train[0:64])
    save_img((x).cpu().detach().numpy(), os.path.join(result_fold, 'ox.png'))
    x = torch.reshape(x, [-1, full_dim])
    z = model.flow.forward(x)[0][0]
    print(z)
    save_img((z).cpu().detach().numpy(), os.path.join(result_fold, 'oz0.png'))
    z = model.flow.forward(torch.nn.init.constant_(z,0), inverse=True)[0]
    print(z)
    print(z.shape)
    save_img((z).cpu().detach().numpy(), os.path.join(result_fold, 'oz1.png'))

if __name__ == '__main__':
    main()
