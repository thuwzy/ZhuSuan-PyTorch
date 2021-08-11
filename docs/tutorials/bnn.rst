Bayesian Neural Networks
========================

.. note::

    This tutorial assumes that readers have been familiar with ZhuSuan's
    :doc:`basic concepts <concepts>`.

Recent years have seen neural networks' powerful abilities in fitting complex
transformations, with successful applications on speech recognition, image
classification, and machine translation, etc.
However, typical training of neural networks requires lots of labeled data
to control the risk of overfitting.
And the problem becomes harder when it comes to real world regression tasks.
These tasks often have smaller amount of training data to use, and the
high-frequency characteristics of these data often makes neural networks
easier to get trapped in overfitting.

A principled approach for solving this problem is **Bayesian Neural Networks**
(BNN).
In BNN, prior distributions are put upon the neural network's weights
to consider the modeling uncertainty.
By doing Bayesian inference on the weights, one can learn a predictor
which both fits to the training data and reasons about the uncertainty of
its own prediction on test data.
In this tutorial, we show how to implement BNNs in ZhuSuan.
The full script for this tutorial is at
`examples/bayesian_neural_nets/bnn_vi.py <https://github.com/McGrady00H/Zhusuan-Jittor/blob/main/examples/bayesian_neural_nets/bnn_vi.py>`_.

We use a regression dataset called
`Boston housing <https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>`_.
This has :math:`N = 506` data points, with :math:`D = 13` dimensions.
The generative process of a BNN for modeling multivariate regression is
as follows:

.. math::

    W_i &\sim \mathrm{N}(W_i|0, I),\quad i=1\cdots L. \\
    y_{mean} &= f_{NN}(x, \{W_i\}_{i=1}^L) \\
    y &\sim \mathrm{N}(y|y_{mean}, \sigma^2)

This generative process starts with an input feature (:math:`x`), which
is forwarded through a deep neural network (:math:`f_{NN}`) with :math:`L`
layers, whose parameters in each layer (:math:`W_i`) satisfy a factorized
multivariate standard Normal distribution.
With this forward transformation, the model is able to learn complex
relationships between the input (:math:`x`) and the output (:math:`y`).
Finally, some noise is added to the output to get a tractable likelihood
for the model, which is typically a Gaussian noise in regression problems.
A graphical model representation for bayesian neural network is as follows.

.. image:: ../_static/images/bnn.png
    :align: center
    :width: 25%

Build the model
---------------

We start by the model building function (we shall see the meanings of
these arguments later)::

    class Net(BayesianNet):
        def __init__(self, layer_sizes, n_particles):
            super().__init__()

Following the generative process, we need standard Normal
distributions to generate the weights (:math:`\{W_i\}_{i=1}^L`) in each layer.
For a layer with ``n_in`` input units and ``n_out`` output units, the weights
are of shape ``[n_out, n_in + 1]`` (one additional column for bias).
To support multiple samples (useful in inference and prediction), a common
practice is to set the `n_samples` argument to a placeholder, which we
choose to be ``n_particles`` here::

    h = x.repeat([self.n_particles, *len(x.shape) * [1]])
    for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
        w = self.sn('Normal',
                    name='w' + str(i),
                    mean=torch.zeros([n_out, n_in + 1]),
                    std=torch.ones([n_out, n_in + 1]),
                    group_ndims=2,
                    n_samples=self.n_particles,
                    reduce_mean_dims=[0])

Note that we expand ``x`` with a new dimension and tile it to enable
computation with multiple particles of weight samples.
To treat the weights in each layer as a whole and evaluate the probability of
them together, ``group_ndims`` is set to 2.
If you are unfamiliar with this property, see :ref:`dist` for details.

Then we write the feed-forward process of neural networks, through which the
connection between output ``y`` and input ``x`` is established::

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
        h = torch.cat((h, torch.ones([*h.shape[:-1], 1])), -1)
        h = torch.unsqueeze(h, -1)
        p = torch.sqrt(torch.as_tensor(h.shape[2], dtype=torch.float32))
        h = torch.matmul(w, h) / p
        h = torch.squeeze(h, -1)
        if i < len(self.layer_sizes) - 2:
            h = torch.nn.ReLU()(h)

Next, we add an observation distribution (noise) to get a tractable
likelihood when evaluating the probability::

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

Putting together and adding model reuse, the code for constructing a BNN is::

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
                h = torch.cat((h, torch.ones([*h.shape[:-1], 1])), -1)
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

Inference
---------

Having built the model, the next step is to infer the posterior distribution,
or uncertainty of weights given the training data.

.. math::

    p(W|x_{1:N}, y_{1:N}) \propto p(W)\prod_{n=1}^N p(y_n|x_n, W)

Because the normalizing constant is intractable, we cannot directly
compute the posterior distribution of network parameters
(:math:`\{W_i\}_{i=1}^L`).
In order to solve this problem, we use
`Variational Inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_,
i.e., using a variational distribution
:math:`q_{\phi}(\{W_i\}_{i=1}^L)=\prod_{i=1}^L{q_{\phi_i}(W_i)}` to
approximate the true posterior.
The simplest variational posterior (:math:`q_{\phi_i}(W_i)`) we can specify
is factorized (also called mean-field) Normal distribution parameterized
by its mean and log standard deviation.

.. math::

    q_{\phi_i}(W_i) = \mathrm{N}(W_i|\mu_i, {\sigma_i}^2)

The code for above definition is::

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

In Variational Inference, to make :math:`q_{\phi}(W)` approximate
:math:`p(W|x_{1:N}, y_{1:N})` well.
We need to maximize a lower bound of the marginal log probability
(:math:`\log p(y|x)`):

.. math::

    \log p(y_{1:N}|x_{1:N}) &\geq \log p(y_{1:N}|x_{1:N})
    - \mathrm{KL}(q_{\phi}(W)\|p(W|x_{1:N},y_{1:N})) \\
    &= \mathbb{E}_{q_{\phi}(W)} \left[\log (p(y_{1:N}|x_{1:N}, W)p(W))
    - \log q_{\phi}(W)\right] \\
    &\triangleq \mathcal{L}(\phi)

The lower bound is equal to the marginal log
likelihood if and only if :math:`q_{\phi}(W) = p(W|x_{1:N}, y_{1:N})`,
for :math:`i` in :math:`1\cdots L`, when the
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
between them (:math:`\mathrm{KL}(q_{\phi}(W)\|p(W|x_{1:N}, y_{1:N})`)
is zero.

This lower bound is usually called Evidence Lower Bound (ELBO). Note that the
only probabilities we need to evaluate in it is the joint likelihood and
the probability of the variational posterior.
The log conditional likelihood is

.. math::
    \log p(y_{1:N}|x_{1:N}, W) = \sum_{n=1}^N\log p(y_n|x_n, W)

Computing log conditional likelihood for the whole dataset is very
time-consuming.
In practice, we sub-sample a minibatch of data to approximate the conditional
likelihood

.. math::
    \log p(y_{1:N}|x_{1:N}, W) \approx \frac{N}{M}\sum_{m=1}^M\log p(y_m| x_m, W)

Here :math:`\{(x_m, y_m)\}_{m=1:M}` is a subset including :math:`M`
random samples from the training set :math:`\{(x_n, y_n)\}_{n=1:N}`.
:math:`M` is called the batch size.
By setting the batch size relatively small, we can compute the lower bound
above efficiently.

.. Note::

    Different from models like VAEs, BNN's latent variables
    :math:`\{W_i\}_{i=1}^L` are global for all the data, therefore we don't
    explicitly condition :math:`W` on each data in the variational posterior.

We optimize this lower bound by
`stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_.
As we have done in the :doc:`VAE tutorial <vae>`,
the **Stochastic Gradient Variational Bayes** (SGVB) estimator is used.
The code for this part is::

    net = Net(layer_sizes, n_particles)
    variational = Variational(layer_sizes, n_particles)

    model = zs.variational.ELBO(net, variational)

Evaluation
----------

What we've done above is to define the model and infer the parameters.
The main purpose of doing this is to predict about new data.
The probability distribution of new data (:math:`y`) given its input
feature (:math:`x`) and our training data (:math:`D`) is

.. math::

    p(y|x, D) = \int_W p(y|x, W)p(W|D)

Because we have learned the approximation of :math:`p(W|D)` by the variational
posterior :math:`q(W)`, we can substitute it into the equation

.. math::

    p(y|x, D) \simeq \int_W p(y|x, W)q(W)

Although the above integral is still intractable, Monte Carlo estimation
can be used to get an unbiased estimate of it by sampling from the variational
posterior

.. math::

    p(y|x, D) \simeq \frac{1}{M}\sum_{i=1}^M p(y|x, W^i)\quad W^i \sim q(W)

We can choose the mean of this predictive distribution to be our prediction
on new data

.. math::

    y^{pred} = \mathbb{E}_{p(y|x, D)} \; y \simeq \frac{1}{M}\sum_{i=1}^M \mathbb{E}_{p(y|x, W^i)} \; y \quad W^i \sim q(W)

The above equation can be implemented by passing the samples from the
variational posterior as observations into the model, and averaging over the
samples of ``y_mean`` from the resulting
:class:`~zhusuan.framework.bn.BayesianNet`.
The trick here is that the procedure of observing :math:`W` as samples from
:math:`q(W)` has been implemented when constructing the evidence lower bound. ::

    # prediction: rmse & log likelihood
    # In Net
    y_mean = torch.squeeze(h, 2)

    y = self.observed['y']
    y_pred = torch.mean(y_mean, 0)
    self.cache['rmse'] = torch.sqrt(torch.mean((y - y_pred) ** 2))
    # During training
    lower_bound = model({'x': x, 'y': y})

The predictive mean is given by ``y_mean``.
To see how this performs, we would like to compute some quantitative
measurements including
`Root Mean Squared Error (RMSE) <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_
and `log likelihood <https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood>`_.

RMSE is defined as the square root of the predictive mean square error,
smaller RMSE means better predictive accuracy:

.. math::
    RMSE = \sqrt{\frac{1}{N}\sum_{n=1}^N(y_n^{pred}-y_n^{target})^2}

Log likelihood (LL) is defined as the natural logarithm of the likelihood
function, larger LL means that the learned model fits the test data better:

.. math::

    LL &= \log p(y|x, D) \\
       &\simeq \log \int_W p(y|x, W)q(W) \\

This can also be computed by Monte Carlo estimation

.. math::

    LL \simeq \log \frac{1}{M}\sum_{i=1}^M p(y|x, W^i)\quad W^i\sim q(W)

To be noted, as we usually standardized the data to make
them have unit variance at beginning (check the full script
`examples/bayesian_neural_nets/bnn_vi.py <https://github.com/McGrady00H/Zhusuan-Jittor/blob/main/examples/bayesian_neural_nets/bnn_vi.py>`_),
we need to count its effect in our evaluation formulas.
RMSE is proportional to the amplitude, therefore the final RMSE should be
multiplied with the standard deviation.
For log likelihood, it needs to be subtracted by a log term.
All together, the code for evaluation is::

    # prediction: rmse & log likelihood
    rese = net.cache['rmse']
    log_ll = model({'x': x, 'y': y})

Run gradient descent
--------------------

Again, everything is good before a run. Now add the following codes to
run the training loop and see how your BNN performs::

    for epoch in range(epoch_size):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm, :]
        y_train = y_train[perm]

        for step in range(num_batches):
            x = torch.as_tensor(x_train[step * batch_size:(step + 1) * batch_size])
            y = torch.as_tensor(y_train[step * batch_size:(step + 1) * batch_size])
            lbs = model({'x': x, 'y': y})
            optimizer.zero_grad()
            lbs.backward()
            optimizer.step()

            if (step + 1) % num_batches == 0:
                rmse = net.cache['rmse'].clone().detach().numpy()
                print("Epoch[{}/{}], Step [{}/{}], Lower bound: {:.4f}, RMSE: {:.4f}".format(epoch + 1, epoch_size,
                                                                                            step + 1,
                                                                                            num_batches,
                                                                                            float(lbs.clone().detach().numpy()),
                                                                                            float(rmse) * std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = torch.as_tensor(x_test)
            y_t = torch.as_tensor(y_test)
            lbs = model({'x': x_t, 'y': y_t})
            rmse = net.cache['rmse'].clone().detach().numpy()
            print('>> TEST')
            print('>> Test Lower bound: {:.4f}, RMSE: {:.4f}'.format(float(lbs.clone().detach().numpy()), float(rmse) * std_y_train))
