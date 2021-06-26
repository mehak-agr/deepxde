from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""

import numpy as np

import deepxde as dde
from deepxde.backend import tf

c = 10
A = np.pi ** 2


def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


def main():
    def pde(x, y):
        dy_tt = dde.grad.hessian(y, x, i=2, j=2)
        dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
        dy_x1x1 = dde.grad.hessian(y, x, i=0, j=0)
        return dy_tt - c ** 2 * (2 * A) * (dy_x1x1 + dy_x2x2)

    def func(x):
        x1, x2, t = x[:, 0:1], x[:, 1:2], x[:, 2:]
        p1 = np.sin(np.pi * x1) * np.sin(np.pi * x2)
        p2 = A * (np.cos(c * np.pi * np.sqrt(2) * t) + np.sin(c * np.pi * np.sqrt(2) * t))
        return p1 * p2
    
    def dfunc_t(x, y, _):
        dy_t = dde.grad.jacobian(y, x, i=0, j=2)
        x1, x2, t = x[:, 0:1], x[:, 1:2], x[:, 2:]
        p1 = tf.sin(np.pi * x1) * tf.sin(np.pi * x2)
        p2 = A * (tf.cos(c * np.pi * np.sqrt(2) * t) + tf.sin(c * np.pi * np.sqrt(2) * t))
        dp2_t = c * np.pi * np.sqrt(2) * p2
        return dy_t - p1 * dp2_t

    geom = dde.geometry.Rectangle([-1, -1], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime,
                         func,
                         lambda _, on_boundary: on_boundary)
    ic_1 = dde.IC(geomtime,
                  func,
                  lambda x, _: np.isclose(x[2], 0))
    ic_2 = dde.OperatorBC(geomtime,
                          dfunc_t,
                          lambda x, _: np.isclose(x[2], 0))

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic_1, ic_2],
        num_domain=360,
        num_boundary=360,
        num_initial=360,
        solution=func,
        num_test=10000,
    )

    layer_size = [3] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.STMsFFN(
        layer_size, activation, initializer, sigmas_x=[1, 1], sigmas_t=[1, 10]
    )
    net.apply_feature_transform(lambda x: x * 2 * np.sqrt(2))

    model = dde.Model(data, net)
    initial_losses = get_initial_loss(model)
    loss_weights = 5 / initial_losses
    model.compile(
        "adam",
        lr=0.001,
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
        decay=("inverse time", 2000, 0.9),
    )
    pde_residual_resampler = dde.callbacks.PDEResidualResampler(period=1)
    losshistory, train_state = model.train(
        epochs=10000, callbacks=[pde_residual_resampler], display_every=500
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()

X = geomtime.random_points(10000)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
