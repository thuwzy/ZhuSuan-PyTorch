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

        self._device = torch.device('cpu') #! NOTICE: device default as CPU

    @property
    def device(self):
        """
        The device this module lies at.
        
        :return: torch.device
        """    
        try: 
            return next(self.parameters()).device
        except:
            return self._device

    def to(self, device):
        self._device = device
        return super().to(device)


    def _update(self, bn, observed):
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)

        log_joint_ = bn.log_joint()

        grad = torch.autograd.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            epsilon = torch.normal(0., math.sqrt(self.lr), size=self._var_list[i].shape).to(self.device)
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * grad[i] + epsilon
            self._var_list[i] = self._var_list[i].detach()
            self._var_list[i].requires_grad = True

class PSGLD(SGLD):
    """
        PSGLD with RMSprop preconditioner, "Preconditioned stochastic gradient Langevin dynamics for deep neural networks"
    """

    def __init__(self, learning_rate, decay=0.9, epsilon=1e-3):
        super().__init__(learning_rate)
        self.aux = None
        self.decay = decay
        self.epsilon = epsilon

    def _update(self, bn, observed):
        if not self.aux:
            self.aux = [torch.zeros_like(q) for q in self._var_list]
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)

        log_joint_ = bn.log_joint()
        grad = torch.autograd.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            self.aux[i] = self.decay * self.aux[i] + (1 - self.decay) * torch.pow(grad[i], 2)
            g = 1 / (self.epsilon + torch.sqrt(self.aux[i]))
            e = torch.normal(0.0, torch.sqrt(self.lr * g)).to(self.device)
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * g * grad[i] + e
            self._var_list[i] = self._var_list[i].detach()
            self._var_list[i].requires_grad = True