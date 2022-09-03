import warnings

import IPython
import torch
import torch.nn as nn
from torch import Tensor
from copy import copy
from zhusuan.framework import BayesianNet
from zhusuan.distributions import Normal

__all__ = [
    "HMC"
]


def velocity(momentum, mass):
    """
    get the velocity using momentum divided by mass
    """
    return map(lambda z: z[0] / z[1], zip(momentum, mass))


def random_momentum(shapes, mass):
    """
    output a list of random momentum with given listed shapes and mass
    """
    return [torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)) * torch.sqrt(m)
            for shape, m in zip(shapes, mass)]


def hamiltonian(q, p, log_posterior, mass, data_axes):
    """
    Hamilton operator, return the value of H(q,p) and - potential energy
    """
    potential = -log_posterior(q)
    kinetic = 0.5 * sum([torch.sum(torch.square(momentum) / m, dim=axis)
                         for momentum, m, axis in zip(p, mass, data_axes)])
    return potential + kinetic, -potential


def leapfrog_integrator(q, p, step_size1, step_size2, grad, mass):
    q = [x + step_size1 * y for x, y in zip(q, velocity(p, mass))]
    grads = grad(q)
    p = [x + step_size2 * y for x, y in zip(p, grads)]
    return q, p


def get_acceptance_rate(q, p, new_q, new_p, log_posterior, mass, data_axes):
    old_H, old_log_prob = hamiltonian(q, p, log_posterior, mass, data_axes)
    new_H, new_log_prob = hamiltonian(new_q, new_p, log_posterior, mass, data_axes)
    if not torch.isfinite(old_log_prob).any():
        raise ValueError('HMC: old_log_prob has numeric errors! Try better initialization.')

    acceptance_rate = torch.exp(torch.minimum(old_H - new_H, torch.zeros([])))
    is_finite = torch.logical_and(torch.isfinite(acceptance_rate), torch.isfinite(new_log_prob))
    acceptance_rate = torch.where(is_finite, acceptance_rate, torch.zeros_like(acceptance_rate))
    return old_H, new_H, old_log_prob, new_log_prob, acceptance_rate


class StepSizeTuner:
    def __init__(self, init_step_size, adapt_step_size, gamma, t0, kappa, delta, dtype=torch.float32):
        self.adapt_step_size = torch.as_tensor(adapt_step_size, dtype=torch.bool)
        self.init_step_size = init_step_size
        self.gamma = torch.as_tensor(gamma, dtype=dtype)
        self.t0 = torch.as_tensor(t0, dtype=dtype)
        self.kappa = torch.as_tensor(kappa, dtype=dtype)
        self.delta = torch.as_tensor(delta, dtype=dtype)
        self.mu = torch.tensor(10 * init_step_size, dtype=dtype)  # constant in tf
        self.step = torch.as_tensor(0., dtype=dtype)
        self.step.requires_grad = False
        self.log_epsilon_bar = torch.as_tensor(0., dtype=dtype)
        self.log_epsilon_bar.requires_grad = False
        self.h_bar = torch.as_tensor(0., dtype=dtype)
        self.h_bar.requires_grad = False

    def tune(self, acceptance_rate, fresh_start):
        # def adapt_step_size():

        if self.adapt_step_size:
            self.step = (1 - fresh_start) * self.step + 1
            rate1 = 1. / (self.step + self.t0)
            self.h_bar = (1 - fresh_start) * (1 - rate1) * self.h_bar + rate1 * (self.delta - acceptance_rate)
            log_epsilon = self.mu - torch.sqrt(self.step) / self.gamma * self.h_bar
            rate = torch.pow(self.step, -self.kappa)
            self.log_epsilon_bar = rate * log_epsilon + (1 - fresh_start) * (1 - rate) * self.log_epsilon_bar
            return torch.exp(log_epsilon)
        else:
            return torch.exp(self.log_epsilon_bar)


class ExponentialWeightedMovingVariance:
    def __init__(self, decay, shape, num_chain_dims):
        self.t = torch.zeros([], requires_grad=False, dtype=torch.int32)
        # mean, var: (1,...,1 data_dims)
        self.mean = [torch.zeros(s, requires_grad=False) for s in shape]
        self.var = [torch.zeros(s, requires_grad=False) for s in shape]
        self.decay = decay
        self.one = torch.ones([], requires_grad=False, dtype=torch.float32)  # constant in tf
        self.num_chain_dims = num_chain_dims
        self.chain_axes = torch.arange(self.num_chain_dims)

    def update(self, x):
        # x: (chain_dims data_dims)
        self.t += 1
        weight = (1 - self.decay) / (1 - torch.pow(self.decay, self.t))
        # incr: (chain_dims data_dims)
        incr = [weight * (q - mean) for q, mean in zip(x, self.mean)]
        # mean: (1,...,1 data_dims)
        self.mean = [mean.add(torch.mean(i, dim=self.chain_axes, keepdim=True))
                     for mean, i in zip(self.mean, incr)]  # TODO: is inplace add ok?
        # var: (1,...,1 data_dims)
        self.var = [(1 - weight) * var +
                    torch.mean(i * (q - mean), dim=self.chain_axes, keepdim=True)
                    for var, i, q, mean in zip(self.var, incr, x, self.mean)]

        return self.var

    def get_precision(self, var_in):
        return [(self.one / var) for var in var_in]

    def get_updated_precision(self, x):
        # Should be called only once
        return self.get_precision(self.update(x))

    def precision(self):
        return self.get_precision(self.var)


class HMCInfo(object):
    """
    Contains information about a sampling iteration by :class:`HMC`. Users
    can get fine control of the sampling process by monitoring these
    statistics.

    .. note::

        Attributes provided in this structure must be fetched together with the
        corresponding sampling operation and should not be fetched anywhere
        else. Otherwise you would get undefined behaviors.

    :param samples: A dictionary of ``(string, Tensor)`` pairs. Samples
        generated by this HMC iteration.
    :param acceptance_rate: A Tensor. The acceptance rate in this iteration.
    :param updated_step_size: A Tensor. The updated step size (by adaptation)
        after this iteration.
    :param init_momentum: A dictionary of ``(string, Tensor)`` pairs. The
        initial momentum for each latent variable in this sampling iteration.
    :param orig_hamiltonian: A Tensor. The original hamiltonian at the
        beginning of the iteration.
    :param hamiltonian: A Tensor. The current hamiltonian at the end of the
        iteration.
    :param orig_log_prob: A Tensor. The log joint probability at the
        beginning position of the iteration.
    :param log_prob: A Tensor. The current log joint probability at the end
        position of the iteration.
    """

    def __init__(self, samples, acceptance_rate, updated_step_size,
                 init_momentum, orig_hamiltonian, hamiltonian, orig_log_prob,
                 log_prob):
        self.samples = samples
        self.acceptance_rate = acceptance_rate
        self.updated_step_size = updated_step_size
        self.init_momentum = init_momentum
        self.orig_hamiltonian = orig_hamiltonian
        self.hamiltonian = hamiltonian
        self.orig_log_prob = orig_log_prob
        self.log_prob = log_prob


class HMC:
    """
    Hamiltonian Monte Carlo (Neal, 2011) with adaptation for stepsize (Hoffman &
    Gelman, 2014) and mass. The usage is similar with a Tensorflow optimizer.

    The :class:`HMC` class supports running multiple MCMC chains in parallel. To
    use the sampler, the user first creates a (list of) tensorflow `Variable`
    storing the initial sample, whose shape is ``chain axes + data axes``. There
    can be arbitrary number of chain axes followed by arbitrary number of data
    axes. Then the user provides a `log_joint` function which returns a tensor
    of shape ``chain axes``, which is the log joint density for each chain.
    Finally, the user runs the operation returned by :meth:`sample`, which
    updates the sample stored in the `Variable`.

    .. note::

        Currently we do not support invoking the :meth:`sample` method
        multiple times per :class:`HMC` class. Please declare one :class:`HMC`
        class per each invoke of the :meth:`sample` method.

    .. note::

        When the adaptations are on, the sampler is not reversible.
        To guarantee current equilibrium, the user should only turn on
        the adaptations during the burn-in iterations, and turn them off
        when collecting samples. To achieve this, the best practice is to
        set `adapt_step_size` and `adapt_mass` to be placeholders and feed
        different values (True/False) when needed.

    :param step_size: A 0-D `float32` Tensor. Initial step size.
    :param n_leapfrogs: A 0-D `int32` Tensor. Number of leapfrog steps.
    :param adapt_step_size: A `bool` Tensor, if set, indicating whether to
        adapt the step size.
    :param target_acceptance_rate: A 0-D `float32` Tensor. The desired
        acceptance rate for adapting the step size.
    :param gamma: A 0-D `float32` Tensor. Parameter for adapting the step
        size, see (Hoffman & Gelman, 2014).
    :param t0: A 0-D `float32` Tensor. Parameter for adapting the step size,
        see (Hoffman & Gelman, 2014).
    :param kappa: A 0-D `float32` Tensor. Parameter for adapting the step
        size, see (Hoffman & Gelman, 2014).
    :param adapt_mass: A `bool` Tensor, if set, indicating whether to adapt
        the mass, adapt_step_size must be set.
    :param mass_collect_iters: A 0-D `int32` Tensor. The beginning iteration
        to change the mass.
    :param mass_decay: A 0-D `float32` Tensor. The decay of computing
        exponential moving variance.
    """

    def __init__(self, step_size=1., n_leapfrogs=10,
                 adapt_step_size=None, target_acceptance_rate=0.8,
                 gamma=0.05, t0=100, kappa=0.75,
                 adapt_mass=None, mass_collect_iters=10, mass_decay=0.99, dtype=torch.float32):
        # TODO: Maintain the variables somewhere else to let the sample be
        # called multiple times
        self.step_size = torch.tensor(step_size, requires_grad=False, dtype=dtype)
        self.n_leapfrogs = torch.as_tensor(n_leapfrogs, dtype=torch.int32)
        self.target_acceptance_rate = torch.as_tensor(target_acceptance_rate, dtype=dtype)
        self.t = torch.tensor(0, requires_grad=False, dtype=torch.int32)
        self.adapt_step_size = adapt_step_size
        self.dtype = dtype
        if adapt_step_size is not None:
            self.step_size_tuner = StepSizeTuner(
                step_size, adapt_step_size, gamma, t0, kappa,
                target_acceptance_rate)
        if adapt_mass is not None:
            if adapt_step_size is None:
                raise ValueError('If adapt mass is set, we should also adapt step size')
            self.adapt_mass = torch.as_tensor(adapt_mass, dtype=torch.bool)
        else:
            mass_collect_iters = 0
            self.adapt_mass = None
        self.mass_collect_iters = torch.as_tensor(mass_collect_iters, dtype=torch.int32)
        self.mass_decay = torch.as_tensor(mass_decay, dtype=dtype)

    def _adapt_mass(self, t: Tensor, num_chain_dims) -> list[Tensor]:
        ewmv = ExponentialWeightedMovingVariance(
            self.mass_decay, self.data_shapes, num_chain_dims)
        if self.adapt_mass:
            new_mass = ewmv.get_updated_precision(self.q)
        else:
            new_mass = ewmv.precision()
        if not isinstance(new_mass, list):
            new_mass = [new_mass]

        # print('New mass is = {}'.format(new_mass))
        # TODO incorrect shape? why filling with ones?
        # print('New mass={}'.format(new_mass))
        # print('q={}, NMS={}'.format(self.q[0].get_shape(),
        #                             new_mass[0].get_shape()))
        if torch.less(torch.as_tensor(t, dtype=torch.int32), self.mass_collect_iters):
            current_mass = [torch.ones(shape) for shape in self.data_shapes]
        else:
            current_mass = new_mass
        if not isinstance(current_mass, list):
            current_mass = [current_mass]
        return current_mass

    def _init_step_size(self, q, p, mass, get_gradient, get_log_posterior):
        factor = 1.5

        def loop_body(step_size, last_acceptance_rate, cond):
            # Calculate acceptance_rate
            new_q, new_p = leapfrog_integrator(q, p, 0.0, step_size / 2, get_gradient, mass)
            new_q, new_p = leapfrog_integrator(new_q, new_p, step_size, step_size / 2, get_gradient, mass)
            acceptance_rate = get_acceptance_rate(q, p, new_q, new_p, get_log_posterior, mass, self.data_axes)[-1]
            acceptance_rate = torch.mean(acceptance_rate)

            # Change step size and stopping criteria
            if torch.less(acceptance_rate, self.target_acceptance_rate):
                new_step_size = step_size * (1. / factor)
            else:
                new_step_size = step_size * factor

            cond = torch.logical_not(torch.logical_xor(
                torch.less(last_acceptance_rate, self.target_acceptance_rate),
                torch.less(acceptance_rate, self.target_acceptance_rate)))
            return [new_step_size, acceptance_rate, cond]

        new_step_size_ = self.step_size
        last_acceptance_rate_ = 1.0
        input_cond = torch.tensor(True)

        while input_cond:
            new_step_size_, last_acceptance_rate_, input_cond = loop_body(new_step_size_, last_acceptance_rate_,
                                                                          input_cond)

        return new_step_size_

    def _leapfrog(self, q, p, step_size, get_gradient, mass):
        # with torch.no_grad():
        i = torch.tensor(0, dtype=torch.int32)
        while i < self.n_leapfrogs + 1:
            if i > 0:
                step_size1 = step_size
            else:
                step_size1 = torch.tensor(0.0, dtype=self.dtype)

            if torch.logical_and(torch.less(i, self.n_leapfrogs), torch.less(torch.zeros([]), i)):
                step_size2 = step_size
            else:
                step_size2 = step_size / 2

            q, p = leapfrog_integrator(q, p, step_size1, step_size2,
                                       lambda x: get_gradient(x), mass)
            i += 1
        return q, p

    def _adapt_step_size(self, acceptance_rate, if_initialize_step_size):
        self.step_size = self.step_size_tuner.tune(
            torch.mean(acceptance_rate),
            torch.as_tensor(if_initialize_step_size, dtype=self.dtype))
        return self.step_size.detach()

    def sample(self, meta_bn: BayesianNet, observed, latent: dict):
        """
        Return the sampling `Operation` that runs a HMC iteration and
        the statistics collected during it, given the log joint function (or a
        :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance), observed
        values and latent variables.

        :param meta_bn: A function or a
            :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance. If it
            is a function, it accepts a dictionary argument of ``(string,
            Tensor)`` pairs, which are mappings from all `StochasticTensor`
            names in the model to their observed values. The function should
            return a Tensor, representing the log joint likelihood of the
            model. More conveniently, the user can also provide a
            :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance
            instead of directly providing a log_joint function. Then a
            log_joint function will be created so that `log_joint(obs) =
            meta_bn.observe(**obs).log_joint()`.
        :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping
            from names of observed `StochasticTensor` s to their values.
        :param latent: A dictionary of ``(string, Variable)`` pairs.
            Mapping from names of latent `StochasticTensor` s to corresponding
            tensorflow `Variables` for storing their initial values and
            samples.

        :return: A Tensorflow `Operation` that runs a HMC iteration.
        :return: A :class:`HMCInfo` instance that collects sampling statistics
            during an iteration.
        """
        if isinstance(meta_bn, BayesianNet):
            self._log_joint = lambda obs: meta_bn.forward(obs).log_joint()
        else:
            warnings.warn("regard as function")
            self._log_joint = meta_bn

        self.t += 1
        new_t = self.t
        # 取出k,v
        latent_k = list(latent.keys())
        latent_v = list(latent.values())
        # for i, v in enumerate(latent_v):
        #     if not isinstance(v, tf.Variable):
        #         raise TypeError("latent['{}'] is not a tensorflow Variable."
        #                         .format(latent_k[i]))
        self.q = copy(latent_v)

        def get_log_posterior(var_list):
            joint_obs = {**dict(zip(latent_k, var_list)), **observed}
            return self._log_joint(joint_obs)

        def get_gradient(var_list):
            log_p = get_log_posterior(var_list)
            return torch.autograd.grad(log_p, var_list)

        self.dynamic_shapes = [q.shape for q in self.q]
        self.static_chain_shape = get_log_posterior(self.q).shape

        # if not self.static_chain_shape:
        #     raise ValueError(
        #         "HMC requires that the static shape of the value returned "
        #         "by log joint function should be at least partially defined. "
        #         "(shape: {})".format(self.static_chain_shape))

        self.n_chain_dims = len(self.static_chain_shape)
        self.data_shapes = [
            tuple(torch.cat(
                (torch.tensor([1] * self.n_chain_dims, dtype=torch.int32),
                 torch.as_tensor(q.shape[self.n_chain_dims:], dtype=torch.int32))
            ).tolist()) for q in self.q]
        self.data_axes = [list(range(self.n_chain_dims, len(data_shape)))
                          for data_shape in self.data_shapes]

        # Adapt mass
        if self.adapt_mass is not None:
            mass = [t.detach() for t in
                    self._adapt_mass(new_t, self.n_chain_dims)]
        else:
            mass = [torch.ones(shape) for shape in self.data_shapes]

        p = random_momentum(self.dynamic_shapes, mass)
        current_p = copy(p)
        current_q = copy(self.q)

        # Initialize step size
        if self.adapt_step_size is None:
            new_step_size = self.step_size
        else:
            if torch.equal(new_t, torch.ones([], dtype=torch.int32)) or torch.equal(new_t, self.mass_collect_iters):
                if_initialize_step_size = torch.tensor(True)
            else:
                if_initialize_step_size = torch.tensor(False)

            def iss():
                return self._init_step_size(current_q, current_p, mass,
                                            get_gradient, get_log_posterior)

            if if_initialize_step_size:
                new_step_size = iss()
            else:
                new_step_size = self.step_size
            new_step_size = new_step_size.detach()

        # Leapfrog
        current_q, current_p = self._leapfrog(
            current_q, current_p, new_step_size, get_gradient, mass)

        # MH-Test
        (old_hamiltonian, new_hamiltonian, old_log_prob,
         new_log_prob, acceptance_rate) = get_acceptance_rate(
            self.q, p, current_q, current_p,
            get_log_posterior, mass, self.data_axes)

        # shape of acceptance_rate
        sp = acceptance_rate.shape
        u01 = torch.distributions.uniform.Uniform(torch.zeros(sp), torch.ones(sp)).sample()
        if_accept = torch.less(u01, acceptance_rate)

        new_q = []
        for nq, oq, da in zip(current_q, self.q, self.data_axes):
            expanded_if_accept = if_accept
            for i in range(len(da)):
                expanded_if_accept = torch.unsqueeze(expanded_if_accept, -1)
            expanded_if_accept = torch.logical_and(
                expanded_if_accept, torch.ones_like(nq, dtype=torch.bool))
            new_q.append(torch.where(expanded_if_accept, nq, oq))

        update_q = [new for old, new in zip(latent_v, new_q)]
        new_log_prob = torch.where(if_accept, new_log_prob, old_log_prob)

        # Adapt step size
        if self.adapt_step_size is not None:
            update_step_size = self._adapt_step_size(acceptance_rate,
                                                     if_initialize_step_size)
        else:
            update_step_size = self.step_size

        # Pack HMC statistics
        hmc_info = HMCInfo(
            samples=dict(zip(latent_k, new_q)),
            acceptance_rate=acceptance_rate,
            updated_step_size=update_step_size,
            init_momentum=dict(zip(latent_k, p)),
            orig_hamiltonian=old_hamiltonian,
            hamiltonian=new_hamiltonian,
            orig_log_prob=old_log_prob,
            log_prob=new_log_prob,
        )
        return dict(zip(latent_k, update_q)), hmc_info
