import torch

import math

from zhusuan.mcmc.SGMCMC import SGMCMC


class SGLD(SGMCMC):
    """
    Subclass of SGMCMC which implements Stochastic Gradient Langevin Dynamics
    (Welling & Teh, 2011) (SGLD) update. The updating equation implemented
    below follows Equation (3) in the paper.
    
    * **var_list** - The updated values of latent variables.
    
    :param learning_rate: A 0-D `float32` Var.
    """
    def __init__(self, learning_rate):
        super().__init__()
        self.lr = torch.as_tensor(learning_rate)
        self.lr_min = torch.as_tensor(1e-4)

    def _update(self, bn, observed):
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)

        log_joint_ = bn.log_joint()
        grad = torch.autograd.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            epsilon = torch.normal(0., math.sqrt(self.lr), size=self._var_list[i].shape)
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * grad[i] + epsilon
            self._var_list[i] = self._var_list[i].detach()
