import torch
import time
import os, sys
import numpy as np
sys.path.append("..")
from zhusuan.framework.bn import BayesianNet
from zhusuan.distributions import Logistic, FlowDistribution
from zhusuan.invertible import get_coupling_mask, MaskCoupling, Scaling, RevSequential
from examples.utils import fetch_dataloaders, save_img, check_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NICE(BayesianNet):
    def __init__(self, num_coupling, in_out_dim, mid_dim, hidden):
        super(NICE, self).__init__()
        self.in_out_dim = in_out_dim
        masks = get_coupling_mask(in_out_dim, 1, num_coupling)
        flow_layers = []
        for i in range(num_coupling):
            flow_layers.append(MaskCoupling(
                in_out_dim=in_out_dim,
                mid_dim=mid_dim,
                hidden=hidden,
                mask=masks[i].to(device)
            ))
        flow_layers.append(Scaling(in_out_dim))
        self.flow = RevSequential(flow_layers)
        # for gpu use, masks and dis both must move to gpu device
        dis = Logistic(loc=torch.zeros([in_out_dim]), scale=torch.ones([in_out_dim]), device=device)
        flow_dis = FlowDistribution(latents=dis, transformation=self.flow, device=device)
        # do not sample at init using n_samples=-1 for FlowDistribution, only FlowDistribution has this property
        self.sn(flow_dis, name="x", n_samples=-1)

    def sample(self, size):
        return self.nodes["x"].dist.sample(size)

    def forward(self, x):
        return self.nodes['x'].log_prob(x)


def main():
    batch_size = 200
    epoch_size = 14
    sample_size = 64
    coupling = 4
    # Optim Parameters
    lr = 1e-3
    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 5
    num_iter = -1

    start = time.time()
    model = NICE(num_coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=False, dequantify=True)
    for epoch in range(epoch_size):
        stats = []
        for _, data in enumerate(train_dataloader):
            num_iter += 1
            model.train()
            optimizer.zero_grad()
            inputs = data[0].to(device)
            loss = -model(inputs).mean()
            loss.backward()
            optimizer.step()
            stats.append(loss.detach().cpu().numpy())
            if num_iter % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    path = os.path.join(os.getcwd(), 'results', 'NICE')
                    check_dir(path)
                    samples = model.sample(sample_size).cpu()
                    save_img(samples.numpy(), os.path.join(path, "nice_sample_iter{}.png".format(num_iter)))
        print("Epoch:[{}/{}], loss: {:.4f} iter:{}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats)), num_iter
        ))

    end = time.time()
    print("using", end - start, "s")
    model.eval()
    with torch.no_grad():
        samples = model.sample(sample_size)
        path = os.path.join(os.getcwd(), 'results', 'NICE')
        check_dir(path)
        save_img(samples.detach().cpu().numpy(), os.path.join(path, 'sample-NICE.png'))

if __name__ == '__main__':
    main()
