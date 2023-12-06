backend = "tensorflow"

import deepxde.deepxde as dde
dde.config.set_default_float("float64")
dde.backend.set_default_backend(backend)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import tensorflow as tf
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error

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

def abetaODE(y0, a, b):
    # abetaODE(numIter,A,B,C,D,E
    # numIter - total number of initial conditions to simulate
    # A,B,C,D,E - relative concentration IC seedings of B1', Bn, Bn', Bm, Bm'

    # define the model parameters

    a1, a2, a3, a4, a5 = a
    b1, b2, b3, b4, b5, b6 = b

    # initialize initial condition, and bridge rate loops

    # y0 initial condition vector

    t = np.linspace(0, max_time, num_rows)  # time grid

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

    soln = odeint(f, y0, t)  # solve the system of equations

    # extract solution for each agent
    df_dict = {
        't': t,
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




a1t, a2t, a3t, a4t, a5t, b1t, b2t, b3t, b4t, b5t, b6t = [0.001, 1.0, 1.0, 1.0, 0.0001, 0.005, 10.0, 1.0, 1.0, 0.005, 10.0]
at, bt = [a1t, a2t, a3t, a4t, a5t], [b1t, b2t, b3t, b4t, b5t, b6t]
a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 = np.float64([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
a1, a2, a3, a4, a5 = [dde.Variable(a1), dde.Variable(a2), dde.Variable(a3), dde.Variable(a4), dde.Variable(a5)]
b1, b2, b3, b4, b5, b6 = [dde.Variable(b1), dde.Variable(b2), dde.Variable(b3), dde.Variable(b4), dde.Variable(b5), dde.Variable(b6)]
constants = [a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6]
n, m = np.float64([12.0, 24.0])
max_time = 100.0
num_rows = 3*1150
activation = 'tanh'
fnamevar = "variables.dat"
iterations = 10000
optimizer = "nadam"
learning_rate = 1e-02
num_hidden_layers = 1
hidden_layer_size = 40
output_layer = 6
layers = [1] + [hidden_layer_size]*num_hidden_layers + [output_layer]
all_init_vals = np.float64([
    [1, 0, 0, 0, 0, 0],
    [1, 1.5, 0, 0, 0, 0],
    [1, 0, 1.5, 0, 0, 0],
    [1, 0, 0, 1.5, 0, 0],
    [1, 0, 0, 0, 1.5, 0],
    [1, 0, 0, 0, 0, 1.5]
])
test_ic = [1, 1.5, 1.5, 1.5, 1.5, 1.5]
abetaODE(test_ic, at, bt)
t, test_y = load_training_data('sim_abeta.npz')
geom = dde.geometry.TimeDomain(0, max_time)
network = dde.nn.FNN([1] + [hidden_layer_size]*num_hidden_layers + [output_layer], activation, 'Glorot uniform')
variable = dde.callbacks.VariableValue(constants, period=(iterations//100 if iterations > 99 else 1), filename=fnamevar)

class Loss(dde.callbacks.Callback):
    def __init__(self, train_y, test_y):
        super().__init__()
        self.testerr = []
        self.trainerr = []
        self.epoch = 0
        self.test_y = test_y
        self.train_y = train_y
    

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch == (iterations//100 if iterations > 99 else 1):

            model = dde.Model(setup_deepxde_data(test_ic, (t, self.test_y)), network)
            model.set_weights(self.model.get_weights())
            pred = model.predict(t, operator=six_species_equations)
            pred = (pr:=np.array(pred)).reshape(pr.shape[:2])
            self.testerr.append(mean_squared_error(pred, self.test_y.T))


            pred = self.model.predict(t, operator=six_species_equations)
            pred = (pr:=np.array(pred)).reshape(pr.shape[:2])
            self.trainerr.append(mean_squared_error(pred, self.train_y.T))


            self.epoch = 0

loss_callback = Loss(test_y, test_y)



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

def setup_deepxde_data(iv, ob):
    ob_t, ob_y = ob
    def boundary(_, on_initial):
        return on_initial

    ic2 = dde.icbc.IC(geom, lambda X: iv[0], boundary, component=0)
    ic1 = dde.icbc.IC(geom, lambda X: iv[1], boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda X: iv[2], boundary, component=2)
    ic4 = dde.icbc.IC(geom, lambda X: iv[3], boundary, component=3)
    ic5 = dde.icbc.IC(geom, lambda X: iv[4], boundary, component=4)
    ic6 = dde.icbc.IC(geom, lambda X: iv[5], boundary, component=5)

    ob_y1 = dde.icbc.PointSetBC(ob_t, ob_y[:,0:1], component=0)
    ob_y2 = dde.icbc.PointSetBC(ob_t, ob_y[:,1:2], component=1)
    ob_y3 = dde.icbc.PointSetBC(ob_t, ob_y[:,2:3], component=2)
    ob_y4 = dde.icbc.PointSetBC(ob_t, ob_y[:,3:4], component=3)
    ob_y5 = dde.icbc.PointSetBC(ob_t, ob_y[:,4:5], component=4)
    ob_y6 = dde.icbc.PointSetBC(ob_t, ob_y[:,5:6], component=5)

    data = dde.data.TimePDE(
        geom,
        six_species_equations,
        [ic1, ic2, ic3, ic4, ic5, ic6, ob_y1, ob_y2, ob_y3, ob_y4, ob_y5, ob_y6],
        num_domain=num_rows-1
        ,
        num_boundary=1
        ,anchors=ob_t
    )

    return data

def train(weights, iv, y):
    loss_callback.train_y, loss_callback.test_y = y

    data = setup_deepxde_data(iv, (t, y[0]))

    model = dde.Model(data, network)
    if weights is not None:
        model.set_weights(weights)
    model.compile(optimizer, lr=learning_rate, external_trainable_variables=constants)

    hist, state = model.train(
        epochs=iterations, callbacks=[variable, loss_callback], display_every=(iterations//100 if iterations > 99 else 1), disregard_previous_best=True
    )

    return model, (hist, state)

for i, ic in enumerate(all_init_vals):
    abetaODE(ic, at, bt)
    _, tr_y = load_training_data('sim_abeta.npz')

    if i == 0:
        model, results = train(None, ic, (test_y, tr_y))
    else:
        model, results = train(model.get_weights(), ic, (test_y, tr_y))
    # dde.saveplot(results[0], results[1], issave=True, isplot=True)

print(loss_callback.trainerr, loss_callback.testerr)

plt.plot(range(len(loss_callback.testerr)), loss_callback.testerr, label='test')
plt.show()
plt.plot(range(len(loss_callback.trainerr)), loss_callback.trainerr, label='train')
plt.show()
