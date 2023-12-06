backend = "pytorch"

import torch

# assert torch.cuda.is_available()

import deepxdeAbeta.deepxde as dde

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import os
import shutil
import time


dde.config.set_default_float("float64")
dde.backend.set_default_backend(backend)

"""
TODO: Add more constraints to limit possible rate constant values (non-negative, relational equations to add to residuals)
"""

####### No constant ratios
# a1, a2, a3, a4, a5, b3 = np.float64([0.001, 1, 1, 1, 0.0001, 1])
a1, a2, a3, a4, a5, b3 = np.float64([0, 0, 0, 0, 0, 0])
# a1, a2, a3, a4, a5 = [dde.Variable(a1), dde.Variable(a2), dde.Variable(a3), dde.Variable(a4), dde.Variable(a5)]
# b1, b2, b3, b4, b5, b6 = [dde.Variable(b1), dde.Variable(b2), dde.Variable(b3), dde.Variable(b4), dde.Variable(b5),
#                           dde.Variable(b6)]
a1, a2, a3, a4, a5, b3 = dde.Variable(a1), dde.Variable(a2), dde.Variable(a3), dde.Variable(a4), dde.Variable(a5), dde.Variable(b3)
at, bt = [0.001, 1.0, 1.0, 1.0, 0.0001], [0, 0, 1.0, 0, 0, 0]
# constants = [a2, b2, b3, b6]
n, m = np.float64([12.0, 24.0])
all_init_vals = np.float64([
    # [1, 0, 0]
    # [1, 1.5, 0, 0, 0, 0],
    # [1, 0, 1.5, 0, 0, 0],
    # [1, 0, 0, 1.5, 0, 0],
    # [1, 0, 0, 0, 1.5, 0],
    # [1, 0, 0, 0, 0, 1.5],
    [1.39E-06,1.46E-05,2.14E-04]
])

max_time = 10
num_rows = 199
time_steps = {
    # "log": np.exp(np.linspace(0, np.log(max_time+1), num_rows))-1,
    # "linear": np.linspace(0, max_time, num_rows),
    # "pcwise": np.concatenate([np.linspace(0, 5, num_rows//2), np.linspace(5, max_time, num_rows//2)]),
    # "loglog": np.exp(np.exp(np.linspace(0, np.log(np.log(max_time+1)), num_rows))-1)-1,
    "sqrt": np.square(np.linspace(0, np.sqrt(max_time), num_rows))
}
#
# a2, b2, b3, b6 = [dde.Variable(0.0) for i in range(4)]
# b6 = dde.Variable(10.0)
# at, bt = [0.001, 1.0, 1.0, 1.0, 0.0001], [0.005, 10.0, 1.0, 1.0, 0.005, 10.0]
constants = [a1, a2, a3, a4, a5, b3]
# a2, a3, a4, b2, b3, b4 = 1, 1, 1, 10, 1, 1


def to_npz(filepath, out_path, x_cols, y_cols):
    df = pd.read_csv(filepath)
    t = []
    y = []
    for _, v in df.iterrows():
        t.append(v[x_cols].to_numpy())
        y.append(v[y_cols].to_numpy())
    np.savez(out_path, t=t, y=y)


def load_training_data(data_path):
    tr_data = np.load(data_path)
    return tr_data['t'], tr_data['y']


def abetaODE(y0, a, b, time):
    # abetaODE(numIter,A,B,C,D,E
    # numIter - total number of initial conditions to simulate
    # A,B,C,D,E - relative concentration IC seedings of B1', Bn, Bn', Bm, Bm'

    # define the model parameters


    # initialize initial condition, and bridge rate loops

    # Define the system of equations
    def f(y, t):
        # y is initial condition vector
        # t is time grid
        B1t = y[0]
        Bnt = y[1]
        Bmt = y[2]

        B1_t = n * a[0] * Bnt - n * a[1] * pow(B1t, n) - a[2] * B1t
        Bn_t = a[1] * pow(B1t, n) - a[0] * Bnt + (m / n) * a[4] * Bmt - a[3] * Bnt - (
                m / n) * b[2] * pow(Bnt, (m / n))
        Bm_t = b[2] * pow(Bnt, (m / n)) - a[4] * Bmt

        return [B1_t, Bn_t, Bm_t]

    soln = odeint(f, y0, time)  # solve the system of equations

    # extract solution for each agent
    df_dict = {
        't': time,
        'B1': soln[:, 0],
        'Bn': soln[:, 1],
        'Bm': soln[:, 2],
    }

    df = pd.DataFrame.from_dict(df_dict, orient='index').transpose()

    df.to_csv('sim_abeta.csv', index=False)
    to_npz('sim_abeta.csv', 'sim_abeta.npz', x_cols=['t'], y_cols=['B1', 'Bn', 'Bm'])

    return df
def six_species_equations(x, y):
    # deepxde solvers require function to be signature: f(x, y)
    B1, Bn, Bm = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    # tf.print(x, output_stream=sys.stderr, summarize=10)

    B1t = dde.grad.jacobian(y, x, i=0)
    Bnt = dde.grad.jacobian(y, x, i=1)
    Bmt = dde.grad.jacobian(y, x, i=2)

    # calculates residuals by subtracting right side from left side of each equation.
    # Left side is calculated with respect to time, right side is calculated with the other species.
    # Finding the correct constants to make residuals = 0; essentially residual = loss to be minimized.
    r_b1 = B1t - (B1tp := (n * a1 * Bn - n * a2 * pow(B1, n) - a3 * B1))
    r_bn = Bnt - (Bntp := (a2 * pow(B1, n) - a1 * Bn + (m / n) * a5 * Bm - a4 * Bn - (m / n) * b3 * pow(Bn, (m / n))))
    r_bm = Bmt - (Bmtp := b3 * pow(Bn, (m / n)) - a5 * Bm)

    # res = torch.cat([r_b1, r_b1p, r_bn, r_bnp, r_bm, r_bmp])
    # diff = torch.cat([B1t, B1pt, Bnt, Bnpt, Bmt, Bmpt])
    # pred = torch.cat([B1tp, B1ptp, Bntp, Bnptp, Bmtp, Bmptp])
    # res2 = torch.where(pred < 0, res*100, res)
    # # print(res[torch.ge(res, diff)])
    #
    # return torch.chunk(res, 6)

    return [r_b1, r_bn, r_bm]


# abetaODE(*init_vals, [a1t, a2t, a3t, a4t, a5t], [b1t, b2t, b3t, b4t, b5t, b6t])


geom = dde.geometry.TimeDomain(0, max_time)


# Helper function that is used to check whether a point is an initial point or not. This is only used by DeepXDE
def boundary(_, on_initial):
    return on_initial


num_hidden_layers = 3
hidden_layer_size = 6
output_layer = 3

layers = [1] + [hidden_layer_size] * num_hidden_layers + [output_layer]

activation = ['tanh']*num_hidden_layers + ['tanh']

iterations = 70_000
optimizer = "adam"
learning_rate = 1e-03



if __name__ == "__main__":
    try:
        os.mkdir('timesteps')
        os.mkdir('timesteps/model')
    except FileExistsError:
        shutil.rmtree('timesteps')
        os.mkdir('timesteps')
        os.mkdir('timesteps/model')

    for name, step in time_steps.items():

        os.mkdir(f'timesteps/{name}')
        fnamevar = f"timesteps/{name}/variables.dat"
        variable = dde.callbacks.VariableValue(constants, period=(iterations // 100 if iterations > 99 else 1),
                                               filename=fnamevar)

        start = time.time()

        for ind, ic in enumerate(all_init_vals):

            # abetaODE(ic, at, bt, step)
            to_npz('amylofit.csv', 'amylofit.npz', x_cols=['t'], y_cols=['B1', 'Bn', 'Bm'])
            ob_t, ob_y = load_training_data('amylofit.npz')

            ic1 = dde.icbc.IC(geom, lambda X: ic[0], boundary, component=0)
            ic2 = dde.icbc.IC(geom, lambda X: ic[1], boundary, component=1)
            ic3 = dde.icbc.IC(geom, lambda X: ic[2], boundary, component=2)

            ob_y1 = dde.icbc.PointSetBC(ob_t, ob_y[:, 0:1], component=0)
            ob_y2 = dde.icbc.PointSetBC(ob_t, ob_y[:, 1:2], component=1)
            ob_y3 = dde.icbc.PointSetBC(ob_t, ob_y[:, 2:3], component=2)

            data = dde.data.PDE(
                geom,
                six_species_equations,
                [ic1, ic2, ic3, ob_y1, ob_y2, ob_y3],
                num_domain=1149,
                num_boundary=1,
                anchors=ob_t,
            )

            if ind == 0:
                network = dde.nn.FNN(layers, activation, 'Glorot uniform')

                model = dde.Model(data, network)
            else:
                model.set_weights(weights)

            model.compile(optimizer, lr=learning_rate, external_trainable_variables=constants,
                          clamp=True
                          )

            # model.constrain_params(mn=0)

            loss_history, train_state = model.train(
                epochs=iterations, callbacks=[variable], display_every=(iterations // 100 if iterations > 99 else 1),
                disregard_previous_best=True
            )

            weights = model.get_weights()

        dde.saveplot(loss_history, train_state, issave=True, isplot=False, output_dir=f'timesteps/{name}')
        dde.utils.external.plot_loss_history(loss_history, fname=f'timesteps/{name}/loss_history')
        dde.utils.external.plot_best_state(train_state, fname=f'timesteps/{name}/train_state')

        pred = model.predict(ob_t, operator=six_species_equations)
        model.save('timesteps/model/nonneg_constr')

        with open(f'timesteps/{name}/info.dat', 'x') as f:

            lines = [
                f'training time: {time.time() - start}\n',
                f'residual: {np.mean(np.absolute(pred))}\n'
                f'best model at: {train_state.best_step}\n'
                f'train loss: {train_state.best_loss_train}\n'
            ]

            f.writelines(lines)
