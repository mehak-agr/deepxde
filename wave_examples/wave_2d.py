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

A = 2
C = 10


def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


def main():
    def pde(x, y):
        # dy_tt = dde.grad.hessian(y, x, i=2, j=2)
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        print(dy_xx)
        return dy_tt - C ** 2 * dy_xx_1 - C ** 2 * dy_xx_0

    def func_bc(x):
        return 0
    
    def func_ic_1(x):
        return np.sqrt((x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2)
    
    def func_ic_2(x, y, _):
        dy_t = dde.grad.jacobian(y, x, i=0, j=2)
        return dy_t

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, func_bc, lambda _, on_boundary: on_boundary)
    ic_1 = dde.IC(geomtime, func_ic_1, lambda _, on_initial: on_initial)
    ic_2 = dde.OperatorBC(geomtime,
                          func_ic_2,
                          lambda x, _: np.isclose(x[2], 0))

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic_1, ic_2],
        num_domain=360,
        num_boundary=360,
        num_initial=360,
        num_test=10000,
    )

    layer_size = [2] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.STMsFFN(
        layer_size, activation, initializer, sigmas_x=[1, 1], sigmas_t=[1, 10]
    )
    net.apply_feature_transform(lambda x: (x[:, 0:1] - 0.5) * 2 * np.sqrt(3) + (x[:, 1:2] - 0.5) * 2 * np.sqrt(3))

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


