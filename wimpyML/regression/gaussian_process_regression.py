import jax
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
import itertools
import matplotlib.pyplot as plt
import optax
jax.config.update("jax_enable_x64", True)

PI = jnp.pi

test_points = 30
training_points = 20
params = {"sigma": 1.0, "linscale": 1.0, "period": 1.0}


def rbf_kernel(params, x, y):
    sigma = params["sigma"]
    linscale = params["linscale"]

    return (sigma ** 2) * jnp.exp(-0.5 * (x - y) ** 2 / (2 * linscale ** 2))


def periodic_kernel(params, x, y):
    sigma = params["sigma"]
    linscale = params["linscale"]
    period = params["period"]

    return (sigma ** 2) * jnp.exp((-2 / linscale**2) * jnp.sin(PI * jnp.abs(x - y) / period) ** 2)


def kernel(params, x, y):
    # return rbf_kernel(params, x, y)
    return rbf_kernel(params, x, y)


def construct_covariance_matrix(params, P, Q, kernel=kernel):

    sigma = params["sigma"]
    linscale = params["linscale"]
    def tupleize(x, y): return [(kernel(params, x, _)).astype(float) for _ in y] if type(
        x) == jnp.float64 or type(x) == jnp.int64 else [(kernel(params, _, y)).astype(float) for _ in x]
    return jnp.array([jnp.array(list(tupleize(i, P))) for i in Q]).reshape(len(Q), len(P))


X_train = jnp.array([i.astype(jnp.float16)
                    for i in (jnp.linspace(0, 2*PI, training_points))])
Y_train = jnp.array([i.astype(jnp.float16) for i in np.sin(X_train)])

X_test = jnp.array(jnp.linspace(-1*PI, 3*PI, test_points))
X_test = jnp.array([i.astype(jnp.float16) for i in X_test])
Y_test = jnp.array(jnp.sin(X_test)).astype(jnp.float16)
Y_test = jnp.array([i.astype(jnp.float16) for i in Y_test])

X_test = X_test.reshape(-1, 1)
X_train = X_train.reshape(-1, 1)

y_2 = Y_train  # The values from the training set

sigma_11 = construct_covariance_matrix(params, X_train, X_train)
sigma_12 = construct_covariance_matrix(params, X_train, X_test)
sigma_22 = construct_covariance_matrix(params, X_test, X_test)
sigma_21 = construct_covariance_matrix(params, X_test, X_train)

sigma_11 = sigma_11.reshape(training_points, training_points)
sigma_12 = sigma_12.reshape(test_points, training_points)
sigma_22 = sigma_22.reshape(test_points, test_points)
sigma_21 = sigma_21.reshape(training_points, test_points)

mu_posterior = jnp.linalg.solve(sigma_11.T, sigma_21).T @ Y_train
sigma_posterior = sigma_22 - \
    jnp.linalg.solve(sigma_11.T, sigma_21).T @ sigma_21

[
    plt.plot(
        X_test,
        np.random.multivariate_normal(mu_posterior,
                                      sigma_posterior,
                                      100)[i],
        linewidth=1,
    )
    for i in range(20)
]
plt.plot(X_test, mu_posterior, linewidth=1,
         color="black", label="Posterior Mean")
plt.plot(X_test, Y_test, linestyle="dashed",
         linewidth=3, color="red", label="Truth")
plt.scatter(X_train, Y_train, color="yellow",
            zorder=4, label="Training points")

plt.legend()
plt.savefig("gp_untrained.png")

try:
    from scipy.stats import multivariate_normal

    # add jitter to the diagonal
    sigma_posterior += 1e-6 * np.eye(sigma_posterior.shape[0])
    var = multivariate_normal(
        mean=mu_posterior, cov=sigma_posterior, allow_singular=True)

    # add jitter
    var.pdf(X_test)

except ValueError:
    print("ValueError: The covariance matrix of the distribution must be symmetric and positive definite.")
    pass


def gp_posterior(params, X_train, X_test, Y_train):

    sigma_11 = construct_covariance_matrix(params, X_train, X_train)
    sigma_12 = construct_covariance_matrix(params, X_train, X_test)
    sigma_22 = construct_covariance_matrix(params, X_test, X_test)
    sigma_21 = construct_covariance_matrix(params, X_test, X_train)

    mu_posterior = jnp.linalg.solve(sigma_11.T, sigma_21).T @ Y_train
    sigma_posterior = sigma_22 - \
        jnp.linalg.solve(sigma_11.T, sigma_21).T @ sigma_21

    return mu_posterior, sigma_posterior, sigma_22


def dictionarize(params):
    return {"sigma": (params[0]), "linscale": (params[1]), "period": (params[2])}


def gp_loss(params, noise=0.1):

    params = dictionarize(params)

    mu_posterior, sigma_posterior, sigma_22 = gp_posterior(
        params, X_train, X_test, Y_train)

    noise_variance = (noise**2) * jnp.eye(sigma_22.shape[0])
    nll = - 0.5 * jnp.linalg.solve(sigma_22.T + noise_variance, Y_test - mu_posterior).T @ (
        Y_test - mu_posterior) - 0.5 * jnp.log(jnp.linalg.det(sigma_22 + noise_variance))

    return -1*nll


params = [1.0, 1.0, 1.0]

optimizer = optax.adam(1e-1)
opt_state = optimizer.init(params)

for _ in range(20):
    loss_value, grads = jax.value_and_grad(gp_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(
        f"Loss: {loss_value}, Params: {float(params[0])}, {float(params[1])}, {float(params[2])}")


[
    plt.plot(
        X_test,
        np.random.multivariate_normal(mu_posterior,
                                      sigma_posterior,
                                      100)[i],
        linewidth=1,
    )
    for i in range(20)
]
plt.plot(X_test, mu_posterior, linewidth=1,
         color="black", label="Posterior Mean")
plt.plot(X_test, Y_test, linestyle="dashed",
         linewidth=3, color="red", label="Truth")
plt.scatter(X_train, Y_train, color="yellow",
            zorder=4, label="Training points")

plt.legend()
plt.savefig("gp_trained.png")
