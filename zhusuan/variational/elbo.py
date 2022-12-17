import torch
import torch.nn as nn


class ELBO(nn.Module):
    """
    The class that represents the evidence lower bound (ELBO) objective for
    variational inference. It can be constructed like a Jittor's `Module` by passing 2
    :class:`~zhusuan.framework.bn.BayesianNet` instances. For example, the generator network and the variational
    inference network in VAE. The model can calculate the ELBO's value with observations passed.

    .. seealso::
        For more details and examples, please refer to
        :doc:`/tutorials/vae` and :doc:`/tutorials/bnn`

    :param generator: A :class'~zhusuan.framework.BayesianNet` instance or a log joint probability function.
        For the latter, it must accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        node names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param variational: A :class:`~zhusuan.framework.bn.BayesianNet` instance
        that defines the variational family.
    :param estimator: gradient estimate method, including ``sgvb`` and ``reinforce``
    :param transform: A :class:`~zhusuan.invertible.RevNet` instance that transform Specified variables,
    returns the transformed variable and the log_det_J i.e log-determinant of transition Jacobian matrix
    :param transform_var: a list of names of variable to be transformed, all tensor
    that correspond to these names will be placed into tuple by order and feed to the transform
    network
    :param auxillary_var: auxillary variable name list that need to be passed to transform network
    """

    def __init__(self, generator, variational, estimator='sgvb', transform=None, transform_var=[], auxillary_var=[]):
        super(ELBO, self).__init__()

        self.generator = generator
        self.variational = variational

        supported_estimator = ['sgvb', 'reinforce']

        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator
        if estimator == 'reinforce':
            mm = torch.zeros(size=[1], dtype=torch.float32)
            ls = torch.zeros(size=[1], dtype=torch.int32)
            self.register_buffer('moving_mean', mm)
            self.register_buffer('local_step', ls)
            self.moving_mean.requires_grad = False

        if transform:
            self.transform = transform
            self.transform_var = transform_var
            self.auxillary_var = auxillary_var
        else:
            self.transform = None

    def log_joint(self, nodes):
        """
        The default log joint probability function.
        It works by summing over all the conditional log probabilities of
        stochastic nodes evaluated at their current values (samples or
        observations).

        :return: A Var.
        """
        log_joint_ = None
        for n_name in nodes.keys():
            '''
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()  # TODO: figure it out
            '''
            if log_joint_ is None:
                log_joint_ = nodes[n_name].log_prob()
            else:
                log_joint_ += nodes[n_name].log_prob()
        return log_joint_

    def forward(self, observed, reduce_mean=True, **kwargs):
        """
        observe nodes, transform latent variables, return evidence lower bound
        :return: evidence lower bound
        """
        self.variational(observed)
        nodes_q = self.variational.nodes

        log_det = None
        if self.transform is not None:
            _transformed_inputs = {}
            _v_inputs = {}

            # Build input tuple for flow
            flow_inputs = []
            for k in self.transform_var:
                # Only latent variable can be transformed
                assert k not in observed.keys()
                assert k in nodes_q.keys()
                flow_inputs.append(nodes_q[k].tensor)
            for k in self.auxillary_var:
                flow_inputs.append(self.variational.cache[k])
            flow_inputs = tuple(flow_inputs)

            # Transform
            output, log_det = self.transform(flow_inputs)
            # All transformed var should be returned
            assert len(output) == len(self.transform_var)
            for k in self.transform_var:
                _transformed_inputs[k] = output[k]

            for k, v in nodes_q.items():
                if k not in _transformed_inputs.keys():
                    _v_inputs[k] = v.tensor
            _observed = {**_transformed_inputs, **_v_inputs, **observed}
            self.generator(_observed)
            nodes_p = self.generator.nodes
            logpxz = self.log_joint(nodes_p)
            logqz = self.log_joint(nodes_q)

        else:
            _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
            _observed = {**_v_inputs, **observed}
            self.generator(_observed)
            nodes_p = self.generator.nodes
            logpxz = self.log_joint(nodes_p)
            logqz = self.log_joint(nodes_q)

        if self.estimator == "sgvb":
            return self.sgvb(logpxz, logqz, reduce_mean, log_det)
        elif self.estimator == "reinforce":
            return self.reinforce(logpxz, logqz, reduce_mean, **kwargs)

    def sgvb(self, logpxz, logqz, reduce_mean=True, log_det=None):
        """
           Implements the stochastic gradient variational bayes (SGVB) gradient
           estimator for the objective, also known as "reparameterization trick"
           or "path derivative estimator". It was first used for importance
           weighted objectives in (Burda, 2015), where it's named "IWAE".

           It only works for latent `StochasticTensor` s that can be
           reparameterized (Kingma, 2013). For example,
           :class:`~zhusuan.distribution.Normal`
           and :class:`~zhusuan.framework.stochastic.Concrete`.

           .. note::

               To use the :meth:`sgvb` estimator, the ``is_reparameterized``
               property of each latent `StochasticTensor` must be True (which is
               the default setting when they are constructed).

           :return: A Tensor. The surrogate cost for optimizers to
               minimize.
        """
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = torch.mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz
        if log_det is not None:
            elbo += torch.mean(torch.sum(log_det)).squeeze()
        return -elbo

    def reinforce(self, logpxz, logqz, reduce_mean=True, baseline=None, variance_reduction=True, decay=0.8):
        """
        Implements the score function gradient estimator for the ELBO, with
        optional variance reduction using moving mean estimate or "baseline".
        Also known as "REINFORCE" (Williams, 1992), "NVIL" (Mnih, 2014),
        and "likelihood-ratio estimator" (Glynn, 1990).

        It works for all types of latent `StochasticTensor` s.

        .. note::

            To use the :meth:`reinforce` estimator, the ``is_reparameterized``
            property of each reparameterizable latent `StochasticTensor` must
            be set False.

        :param logpxz: log joint of generator
        :param logqz: log joint of variational
        :param reduce_mean: whether reduce to a scalar by mean operation

        :param baseline: A Tensor that can broadcast to match the shape
            returned by `log_joint`. A trainable estimation for the scale of
            the elbo value, which is typically dependent on observed values,
            e.g., a neural network with observed values as inputs. This will be
            additional.

        :param variance_reduction: Bool. Whether to use variance reduction.
            By default will subtract the learning signal with a moving mean
            estimation of it. Users can pass an additional customized baseline
            using the baseline argument, in that way the returned will be a
            tuple of costs, the former for the gradient estimator, the latter
            for adapting the baseline.

        :param decay: Float. The moving average decay for variance
            normalization.

        :return: A Tensor. The surrogate cost for optimizers to
            minimize.
        """

        decay_tensor = torch.ones(size=[1], dtype=torch.float32) * decay
        l_signal = logpxz - logqz
        l_signal = l_signal.detach()
        l_signal.require_grads = False
        baseline_cost = None
        if variance_reduction:
            if baseline is not None:
                baseline_cost = 0.5 * torch.square(
                    l_signal.detach() - baseline
                )
                if len(logqz.shape) > 0 and reduce_mean:
                    baseline_cost = torch.mean(baseline_cost)
                l_signal = l_signal - baseline
            # TODO: extend to non-scalar
            if len(logqz.shape) > 0 and reduce_mean:
                bc = torch.mean(l_signal)
            else:
                bc = l_signal
            # Moving average
            self.moving_mean -= (self.moving_mean - bc) * (1.0 - decay)
            self.local_step += 1
            bias_factor = 1 - torch.pow(decay_tensor, self.local_step)
            self.moving_mean /= bias_factor
            l_signal -= self.moving_mean.detach()
        l_signal = l_signal.detach()
        l_signal.require_grads = False
        cost = - (logpxz + l_signal * logqz)
        if baseline_cost is not None:
            if len(logqz.shape) > 0 and reduce_mean:
                loss = torch.mean(cost + baseline_cost)
            else:
                loss = cost + baseline_cost
            return loss, torch.mean(logpxz - logqz)
        else:
            if len(logqz.shape) > 0 and reduce_mean:
                cost = torch.mean(cost)
            return cost


class EvidenceLowerBoundObjective(ELBO):
    """
    A alias of ELBO.

    .. seealso::
        For more details and examples, please refer to
        :doc:`/api/zhusuan.variational.elbo`

    """

    def __init__(self, generator, variational, estimator='sgvb', transform=None, transform_var=[], auxillary_var=[]):
        super().__init__(generator, variational, estimator,
                         transform, transform_var, auxillary_var)
