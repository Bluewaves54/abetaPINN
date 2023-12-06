backend = "pytorch"

import deepxde.deepxde as dde
dde.config.set_default_float("float64")
dde.backend.set_default_backend(backend)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import os
import shutil
import time


a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 = np.float64([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
a1, a2, a3, a4, a5 = [dde.Variable(a1), dde.Variable(a2), dde.Variable(a3), dde.Variable(a4), dde.Variable(a5)]
b1, b2, b3, b4, b5, b6 = [dde.Variable(b1), dde.Variable(b2), dde.Variable(b3), dde.Variable(b4), dde.Variable(b5), dde.Variable(b6)]
at, bt = [0.001, 1.0, 1.0, 1.0, 0.0001], [0.005, 10.0, 1.0, 1.0, 0.005, 10.0]
constants = [a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6]
n, m = np.float64([12.0, 24.0])
all_init_vals = np.float64([
    [1, 0, 0, 0, 0, 0],
    [1, 1.5, 0, 0, 0, 0],
    [1, 0, 1.5, 0, 0, 0],
    [1, 0, 0, 1.5, 0, 0],
    [1, 0, 0, 0, 1.5, 0],
    [1, 0, 0, 0, 0, 1.5],
    [1, 1.5, 1.5, 1.5, 1.5, 1.5]
])
max_time = 200.0
num_rows = 10000
time_steps = {
    "log": np.exp(np.linspace(0, np.log(max_time+1), num_rows))-1,
    "linear": np.linspace(0, max_time, num_rows),
    "pcwise": np.concatenate([np.linspace(0, 5, num_rows//2), np.linspace(5, max_time, num_rows//2)]),
    "loglog": np.exp(np.exp(np.linspace(0, np.log(np.log(max_time+1)), num_rows))-1)-1,
    "sqrt": np.square(np.linspace(0, np.sqrt(max_time), num_rows))
}





def to_npz(filepath, outpath, x_cols, y_cols):
    df = pd.read_csv(filepath)
    t = []
    y = []
    for _, v in df.iterrows():
        t.append(v[x_cols].to_numpy())
        y.append(v[y_cols].to_numpy())

    np.savez(outpath, t=t, y=y)

def load_training_data(datapath):
    data = np.load(datapath)
    return data['t'], data['y']

def abetaODE(y0, a, b, time):
    # abetaODE(numIter,A,B,C,D,E
    # numIter - total number of initial conditions to simulate
    # A,B,C,D,E - relative concentration IC seedings of B1', Bn, Bn', Bm, Bm'

    # define the model parameters

    a1, a2, a3, a4, a5 = a
    b1, b2, b3, b4, b5, b6 = b

    # initialize initial condition, and bridge rate loops

    # Define the system of equations
    def f(y, t):
        # y is initial condition vector
        # t is time grid
        B1t = y[0]
        B1pt = y[1]
        Bnt = y[2]
        Bnpt = y[3]
        Bmt = y[4]
        Bmpt = y[5]

        B1_t = n * a1 * Bnt - n * a2 * pow(B1t, n) + B1pt - a3 * B1t
        B1p_t = n * b1 * Bnpt - n * b2 * pow(B1pt, n) + a3 * B1pt - B1pt
        Bn_t = a2 * pow(B1t, n) - a1 * Bnt + (m / n) * a5 * Bmt + b4 * Bnpt - a4 * Bnt - (
                    m / n) * b3 * pow(Bnt, (m / n))
        Bnp_t = b2 * pow(B1pt, n) - b1 * Bnpt + a4 * Bnt + (m / n) * b5 * Bmpt - (m / n) * b6 * pow(
            Bnpt, (m / n)) - b4 * Bnpt
        Bm_t = b3 * pow(Bnt, (m / n)) - a5 * Bmt
        Bmp_t = b6 * pow(Bnpt, (m / n)) - b5 * Bmpt

        return [B1_t, B1p_t, Bn_t, Bnp_t, Bm_t, Bmp_t]

    soln = odeint(f, y0, time)  # solve the system of equations

    # extract solution for each agent
    df_dict = {
        't': time,
        'B1': soln[:, 0],
        'B1p': soln[:, 1],
        'Bn': soln[:, 2],
        'Bnp': soln[:, 3],
        'Bm': soln[:, 4],
        'Bmp': soln[:, 5]
    }

    df = pd.DataFrame.from_dict(df_dict, orient='index').transpose()

    df.to_csv('sim_abeta.csv', index=False)
    to_npz('sim_abeta.csv', 'sim_abeta.npz', x_cols=['t'], y_cols=['B1', 'B1p', 'Bn', 'Bnp', 'Bm', 'Bmp'])

    return df

def six_species_equations(x, y):

    # deepxde solvers require function to be signatured: f(x, y)
    B1, B1p, Bn, Bnp, Bm, Bmp = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4], y[:,4:5], y[:,5:6]
    # tf.print(x, output_stream=sys.stderr, summarize=10)

    B1t = dde.grad.jacobian(y, x, i=0)
    Bnt = dde.grad.jacobian(y, x, i=1)
    Bmt = dde.grad.jacobian(y, x, i=2)
    B1pt = dde.grad.jacobian(y, x, i=3)
    Bnpt = dde.grad.jacobian(y, x, i=4)
    Bmpt = dde.grad.jacobian(y, x, i=5)


    # calculates residuals by subtracting right side from left side of each equation.
    # Left side is calculated with respect to time, right side is calculated with the other species.
    # Finding the correct constants to make residuals = 0; essentially residual = loss to be minimized.
    r_b1 = B1t - (n * a1 * Bn - n * a2 * pow(B1, n) + B1p - a3 * B1)
    r_b1p = B1pt - (n * b1 * Bnp - n * b2 * pow(B1p, n) + a3 * B1 - B1p)
    r_bn = Bnt - (a2 * pow(b1, n) - a1 * Bn + (m / n) * a5 * Bm + b4 * Bnp - a4 * Bn - (m / n) * b3 * pow(Bn, (m / n)))
    r_bnp = Bnpt - (b2 * pow(B1p, n) - b1 * Bnp + a4 * Bn + (m / n) * b5 * Bmp - (m / n) * b6 * pow(Bnp, (m / n)) - b4 * Bnp)
    r_bm = Bmt - (b3 * pow(Bn, (m / n)) - a5 * Bm)
    r_bmp = Bmpt - (b6 * pow(Bnp, (m / n)) - b5 * Bmp)

    return [r_b1, r_b1p, r_bn, r_bnp, r_bm, r_bmp]


# abetaODE(*init_vals, [a1t, a2t, a3t, a4t, a5t], [b1t, b2t, b3t, b4t, b5t, b6t])






geom = dde.geometry.TimeDomain(0, max_time)

# Helper function that is used to check whether a point is an initial point or not. This is only used by DeepXDE
def boundary(_, on_initial):
    return on_initial

num_hidden_layers = 3
hidden_layer_size = 40
output_layer = 6

layers = [1] + [hidden_layer_size]*num_hidden_layers + [output_layer]

activation = 'tanh'

iterations = 10000
optimizer = "adam"
learning_rate = 1e-02


try:
    os.mkdir('timesteps')
except:
    shutil.rmtree('timesteps')
    os.mkdir('timesteps')

for name, step in time_steps.items():

    os.mkdir(f'timesteps/{name}')
    fnamevar = f"timesteps/{name}/variables.dat"
    variable = dde.callbacks.VariableValue(constants, period=(iterations//100 if iterations > 99 else 1), filename=fnamevar)

    start = time.time()

    for ind, ic in enumerate(all_init_vals):


        abetaODE(ic, at, bt, step)
        data_path ='sim_abeta.npz'
        ob_t, ob_y = load_training_data(data_path)


        ic1 = dde.icbc.IC(geom, lambda X: ic[0], boundary, component=0)
        ic2 = dde.icbc.IC(geom, lambda X: ic[1], boundary, component=1)
        ic3 = dde.icbc.IC(geom, lambda X: ic[2], boundary, component=2)
        ic4 = dde.icbc.IC(geom, lambda X: ic[3], boundary, component=3)
        ic5 = dde.icbc.IC(geom, lambda X: ic[4], boundary, component=4)
        ic6 = dde.icbc.IC(geom, lambda X: ic[5], boundary, component=5)

        ob_y1 = dde.icbc.PointSetBC(ob_t, ob_y[:,0:1], component=0)
        ob_y2 = dde.icbc.PointSetBC(ob_t, ob_y[:,1:2], component=1)
        ob_y3 = dde.icbc.PointSetBC(ob_t, ob_y[:,2:3], component=2)
        ob_y4 = dde.icbc.PointSetBC(ob_t, ob_y[:,3:4], component=3)
        ob_y5 = dde.icbc.PointSetBC(ob_t, ob_y[:,4:5], component=4)
        ob_y6 = dde.icbc.PointSetBC(ob_t, ob_y[:,5:6], component=5)

        data = dde.data.PDE(
            geom,
            six_species_equations,
            [ic1, ic2, ic3, ic4, ic5, ic6, ob_y1, ob_y2, ob_y3, ob_y4, ob_y5, ob_y6],
            num_domain=1149,
            num_boundary=1,
            anchors=ob_t,
            num_test=int(num_rows*(8/10)),
            train_distribution='uniform'
        )


        if ind == 0:
            network = dde.nn.FNN(layers, activation, 'Glorot uniform')

            model = dde.Model(data, network)
        else:
            model.set_weights(weights)

        model.compile(optimizer, lr=learning_rate, external_trainable_variables=constants)

        loss_history, train_state = model.train(
            epochs=iterations, callbacks=[variable], display_every=(iterations//100 if iterations > 99 else 1), disregard_previous_best=True
        )

        weights = model.get_weights()
    



    dde.saveplot(loss_history, train_state, issave=True, isplot=False, output_dir=f'timesteps/{name}')
    dde.utils.external.plot_loss_history(loss_history, fname=f'timesteps/{name}/loss_history')
    dde.utils.external.plot_best_state(train_state, fname=f'timesteps/{name}/train_state')


    pred = model.predict(ob_t, operator=six_species_equations)


    with open(f'timesteps/{name}/info.dat', 'x') as f:

        lines = [
            f'training time: {time.time()-start}\n',
            f'residual: {np.mean(np.absolute(pred))}\n'
            f'best model at: {train_state.best_step}\n'
            f'train loss: {train_state.best_loss_train}\n'
        ]

        f.writelines(lines)
