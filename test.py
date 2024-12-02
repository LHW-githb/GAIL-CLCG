from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from random import randint, uniform, choice
import numpy as np
import os
import math
import scipy.io
from math import sin, cos, atan2, sqrt
import scipy.io
from scipy import interpolate
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import scipy.io
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
state_dim = 6
action_dim = 1
hidden_dim = 256
device = 'cpu'
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal
from nets import My_Model
from nets import ProgNet
from nets import SimpleColumnGenerator
us1976 = scipy.io.loadmat('./cld_parmeters/us1976.mat')
cxdata0 = scipy.io.loadmat('./cld_parmeters/cxdata0.mat')
cydata0 = scipy.io.loadmat('./cld_parmeters/cydata0.mat')
us1976 = us1976['us1976']
cxdata0 = cxdata0['cxdata0']
cydata0 = cydata0['cydata0']
pi = 3.141592653589793
RAD = 180 / pi

task = [ "1","2", "3"][1]

if task == "1":

    config = {'save_path': './experts/Expert1.pth' }
    prog_net = ProgNet(colGen=SimpleColumnGenerator(6, 50, 1, 1))
    for _ in range(1):
        prog_net.addColumn()
    checkpoint = torch.load('./ckpts/best1.ckpt')



if task == "2":

    config = {'save_path': './experts/Expert2.pth' }
    prog_net = ProgNet(colGen=SimpleColumnGenerator(6, 50, 1, 1))
    for _ in range(2):
        prog_net.addColumn()
    checkpoint = torch.load('./ckpts/best2.ckpt')


if task == "3":

    config = {'save_path': './experts/Expert3.pth'}
    prog_net = ProgNet(colGen=SimpleColumnGenerator(6, 50, 1, 1))
    for _ in range(3):
        prog_net.addColumn()
    checkpoint = torch.load('./ckpts/best3.ckpt')


model = prog_net
model.load_state_dict(checkpoint)
model.eval()

model_dnn =  My_Model(input_dim=6)
model_dnn.load_state_dict(torch.load(config['save_path'],map_location='cpu'))
model_dnn.eval()

class MISSILE():
    def __init__(self):
        self.done = False
        self.xf = 20000.
        self.yf = 2000.
        self.fpa0 = 0

        self.pos_x = 0.
        self.pos_y = 5000.
        self.time = 0.
        self.v = 300.
        self.lamda = atan2((self.yf - self.pos_y), (self.xf - self.pos_x))
        self.fpa = (math.pi / 180) * self.fpa0
        self.mass = 400.
        self.state = np.array(
            [self.time, self.pos_x, self.pos_y, self.v, self.fpa, self.mass])

        self.refs = 0.06
        self.thrust = 2000.
        self.Isp = 1600.
        self.r = sqrt((self.pos_x - self.xf) ** 2 + (self.pos_y - self.yf) ** 2)
        self.potential = 1 / self.r
        self.vlamda = self.v * sin(self.fpa - self.lamda)
        self.vr = self.v * cos(self.fpa - self.lamda)
        self.z = self.r * self.vlamda / sqrt(self.vlamda ** 2 + self.vr ** 2)
        self.z0 = self.z

    def modify(self, state=None, ad=None):
        if state is None:
            state = [0., 0, 5000, 300, (math.pi / 180) * uniform(0, 10), 400]
        self.fpa0 = state[4]
        self.state = np.array(state)
        self.done = False


    def step(self, action):
        Rr = sqrt((self.state[1] - self.xf) ** 2 + (self.state[2] - self.yf) ** 2)
        if Rr>300:
            Inth = 0.05
        else:
            Inth = 0.001
        nextstate = self.rk4(self.dyfunc, self.state, action, Inth)
        nextr = sqrt((nextstate[1] - self.xf) ** 2 + (nextstate[2] - self.yf) ** 2)
        if (self.state[2] < 2001 and self.state[1] > 0.85 * self.xf) or self.state[0] > 150:
            self.done = True
        return nextstate, self.done

    def dyfunc(self, state, action): # state = t, x, y, fpa, v, mass
        x = state[1]
        y = state[2]
        v = state[3]
        fpa = state[4]
        mass = state[5]
        S = self.refs
        rho = self.rhoInterp(y)
        sound = self.soundInterp(y)
        Ma = v / sound
        CD0 = self.CD0Interp(Ma)
        CD = self.CDInterp(Ma)
        CL = self.CLInterp(Ma)
        r = sqrt((x - self.xf) ** 2 + (y - self.yf) ** 2)
        lamda = atan2((self.yf - state[2]), (self.xf - state[1]))
        lamdadot = -v * sin(fpa - lamda) / r
        a0 = 3 * v * lamdadot
        ab = action
        ac = a0 + ab
        q = 0.5 * rho * v ** 2
        alpha = mass * (ac + 9.81 * cos(fpa)) / (CL * q * S + self.thrust)
        if alpha > 10 / RAD:
            alpha =  10 / RAD
        elif alpha < -10 / RAD:
            alpha =  -10 / RAD
        D = (CD0 + CD * alpha ** 2) * q * S
        L = CL * alpha * q * S
        dstate = np.zeros((6))
        dstate[0] = 1
        dstate[1] = v * cos(fpa)
        dstate[2] = v * sin(fpa)
        dstate[3] = (self.thrust * cos(alpha) - D) / mass - 9.81 * sin(fpa)
        dstate[4] = (L + self.thrust * sin(alpha)) / (mass * v) - 9.81 * cos(fpa) / v
        dstate[5] = -self.thrust / self.Isp
        return dstate

    def rk4(self, dyfunc, state, action, h):
        k1 = dyfunc(state, action)
        k2 = dyfunc(state + 0.5 * h * k1, action)
        k3 = dyfunc(state + 0.5 * h * k2, action)
        k4 = dyfunc(state + h * k3, action)
        nextstate = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state = nextstate
        return nextstate

    def rhoInterp(self, data):
        interp = interpolate.interp1d(us1976[:, 0], us1976[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)

    def soundInterp(self, data):
        interp = interpolate.interp1d(us1976[:, 0], us1976[:, 2], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CD0Interp(self, data):
        interp = interpolate.interp1d(cxdata0[:, 0], cxdata0[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CDInterp(self, data):
        interp = interpolate.interp1d(cxdata0[:, 0], cxdata0[:, 2], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CLInterp(self, data):
        interp = interpolate.interp1d(cydata0[:, 0], cydata0[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)

class MISSILE_dnn():
    def __init__(self):
        self.done = False
        self.xf = 20000.
        self.yf = 2000.
        self.fpa0 = 0

        self.pos_x = 0.
        self.pos_y = 5000.
        self.time = 0.
        self.v = 300.
        self.lamda = atan2((self.yf - self.pos_y), (self.xf - self.pos_x))
        self.fpa = (math.pi / 180) * self.fpa0
        self.mass = 400.
        self.state = np.array(
            [self.time, self.pos_x, self.pos_y, self.v, self.fpa, self.mass])

        self.refs = 0.06
        self.thrust = 2000.
        self.Isp = 1600.
        self.r = sqrt((self.pos_x - self.xf) ** 2 + (self.pos_y - self.yf) ** 2)
        self.potential = 1 / self.r
        self.vlamda = self.v * sin(self.fpa - self.lamda)
        self.vr = self.v * cos(self.fpa - self.lamda)
        self.z = self.r * self.vlamda / sqrt(self.vlamda ** 2 + self.vr ** 2)
        self.z0 = self.z

    def modify(self, state=None, ad=None):
        if state is None:
            state = [0., 0, 5000, 300, missile.fpa0, 400]
        self.xf = missile.xf
        self.fpa0 = state[4]
        self.state = np.array(state)
        self.done = False


    def step(self, action):
        Rr = sqrt((self.state[1] - self.xf) ** 2 + (self.state[2] - self.yf) ** 2)
        if Rr>200:
            Inth = 0.05
        else:
            Inth = 0.01
        nextstate = self.rk4(self.dyfunc, self.state, action, Inth)
        nextr = sqrt((nextstate[1] - self.xf) ** 2 + (nextstate[2] - self.yf) ** 2)
        if (self.state[2] < 2001 and self.state[1] > 0.85 * self.xf) or self.state[0] > 120:
            self.done = True
        return nextstate, self.done

    def dyfunc(self, state, action):
        x = state[1]
        y = state[2]
        v = state[3]
        fpa = state[4]
        mass = state[5]
        S = self.refs
        rho = self.rhoInterp(y)
        sound = self.soundInterp(y)
        Ma = v / sound
        CD0 = self.CD0Interp(Ma)
        CD = self.CDInterp(Ma)
        CL = self.CLInterp(Ma)
        r = sqrt((x - self.xf) ** 2 + (y - self.yf) ** 2)
        lamda = atan2((self.yf - state[2]), (self.xf - state[1]))
        lamdadot = -v * sin(fpa - lamda) / r
        a0 = 3 * v * lamdadot
        ab = action
        ac = a0 + ab
        q = 0.5 * rho * v ** 2
        alpha = mass * (ac + 9.81 * cos(fpa)) / (CL * q * S + self.thrust)
        if alpha > 10 / RAD:
            alpha =  10 / RAD
        elif alpha < -10 / RAD:
            alpha =  -10 / RAD
        D = (CD0 + CD * alpha ** 2) * q * S
        L = CL * alpha * q * S
        dstate = np.zeros((6))
        dstate[0] = 1
        dstate[1] = v * cos(fpa)
        dstate[2] = v * sin(fpa)
        dstate[3] = (self.thrust * cos(alpha) - D) / mass - 9.81 * sin(fpa)
        dstate[4] = (L + self.thrust * sin(alpha)) / (mass * v) - 9.81 * cos(fpa) / v
        dstate[5] = -self.thrust / self.Isp
        return dstate

    def rk4(self, dyfunc, state, action, h):
        k1 = dyfunc(state, action)
        k2 = dyfunc(state + 0.5 * h * k1, action)
        k3 = dyfunc(state + 0.5 * h * k2, action)
        k4 = dyfunc(state + h * k3, action)
        nextstate = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state = nextstate
        return nextstate

    def rhoInterp(self, data):
        interp = interpolate.interp1d(us1976[:, 0], us1976[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)

    def soundInterp(self, data):
        interp = interpolate.interp1d(us1976[:, 0], us1976[:, 2], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CD0Interp(self, data):
        interp = interpolate.interp1d(cxdata0[:, 0], cxdata0[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CDInterp(self, data):
        interp = interpolate.interp1d(cxdata0[:, 0], cxdata0[:, 2], 'cubic', fill_value="extrapolate")
        return interp(data)

    def CLInterp(self, data):
        interp = interpolate.interp1d(cydata0[:, 0], cydata0[:, 1], 'cubic', fill_value="extrapolate")
        return interp(data)



def normalization(s):
    if task == "1":
        smax = np.array([2.121320343559642e+04, 4.585058037319488e+02, 0.449400520557010, -0.141897054604164, 0.174532925199433,21000])
        smin = np.array([3.296435241333737e+02, 2.524581821838549e+02, -1.371547875898444, -1.384000166625609, 0, 20000])
    if task == "2":
        smax = np.array([21213.20343559642, 426.3092218811059, 0.225686305842656, -0.141897054604164, 0.174532925199433, 21000])
        smin = np.array([29.313792923509720, 292.6769967324946, -1.561709693908308, -1.566251953977234, 0, 20000])
    if task == "3":
        smax = np.array(
            [2.121320343559642e+04, 4.606040333749587e+02,0.475257710133108, -0.141897054604164, 0.174532925199433, 21000])
        smin = np.array([36.801294717784380, 2.473182654171175e+02, -1.221522708593687, -1.221633054380580, 0, 20000])
    snorm = 2 * (s - smin) / (smax - smin) - 1
    return snorm


def unnorm(snorm):
    if task == "1":
        smax = np.array([15.674090553681594])
        smin = np.array([5.269040280106307])
    if task == "2":
        smax = np.array([31.887927302595884])
        smin = np.array([3.561949136154314])
    if task == "3":
        smax = np.array([16.457259723994778])
        smin = np.array([0.380935527758852])
    s = (snorm.sample() + 1) * (smax - smin) / 2 + smin
    return s

def unnorm_dnn(snorm):

    if task == "1":
        smax = np.array([15.674090553681594])
        smin = np.array([5.269040280106307])
    if task == "2":
        smax = np.array([31.887927302595884])
        smin = np.array([3.561949136154314])
    if task == "3":
        smax = np.array([16.457259723994778])
        smin = np.array([0.380935527758852])
    s = (snorm.detach().numpy() + 1) * (smax - smin) / 2 + smin
    return s


def simulate_missile(missile, model, max_steps=10000000):


    state_history = []
    action_history = []
    am_history = []
    R_history =[]
    lamda_history = []
    missile.modify()

    for step in range(max_steps):
        print(f"Before update, step {step}, missile.state = {missile.state}")
        R = sqrt((missile.state[1] - missile.xf) ** 2 + (missile.state[2] - missile.yf) ** 2)
        V = missile.state[3]
        Fpa = missile.state[4]
        Lamd = atan2((missile.yf - missile.state[2]), (missile.xf - missile.state[1]))
        input_DNN = torch.Tensor([R, V, Fpa, Lamd, missile.fpa0,missile.xf]).to(device)
        input_DNN_np = input_DNN.detach().cpu().numpy()
        input_DNN_norm = normalization(input_DNN_np)
        input_DNN_norm = torch.from_numpy(input_DNN_norm).float()
        if task == "1":
            predicted_action = model.forward(0, input_DNN_norm).mean
        if task == "2":
            predicted_action = model.forward(1, input_DNN_norm).mean
        if task == "3":
            predicted_action = model.forward(2, input_DNN_norm).mean

        predicted_action = unnorm_dnn( predicted_action)
        lamdadot = -V* sin(Fpa - Lamd) / R
        a0 = 3 * V * lamdadot
        am = predicted_action + a0
        next_state, done1 = missile.step(predicted_action)
        R_history.append(R)
        lamda_history.append(Lamd/3.1415926*180)
        action_history.append(predicted_action)
        state_history.append(next_state)
        am_history.append(am)
        if done1:
            break
        print(f"After update, step {step}, missile.state = {missile.state}")
    return state_history, action_history,R_history,lamda_history,am_history




def simulate_missile_dnn( missile_dnn ,model_dnn, max_steps=10000000):
    state_history_dnn = []
    action_history_dnn = []
    am_history_dnn = []
    missile_dnn.modify()
    for step in range(max_steps):
        R_dnn = sqrt((missile_dnn.state[1] - missile_dnn.xf) ** 2 + (missile_dnn.state[2] - missile_dnn.yf) ** 2)
        V_dnn = missile_dnn.state[3]
        Fpa_dnn = missile_dnn.state[4]
        Lamd_dnn = atan2((missile_dnn.yf - missile_dnn.state[2]), (missile_dnn.xf - missile_dnn.state[1]))
        input_DNN_dnn = torch.Tensor([R_dnn, V_dnn, Fpa_dnn, Lamd_dnn, missile_dnn.fpa0,missile_dnn.xf]).to(device)
        input_DNN_np_dnn = input_DNN_dnn.detach().cpu().numpy()
        input_DNN_norm_dnn = normalization(input_DNN_np_dnn)
        input_DNN_norm_dnn = torch.from_numpy(input_DNN_norm_dnn).float()
        predicted_action_dnn = model_dnn(input_DNN_norm_dnn)
        predicted_action_dnn = unnorm_dnn(predicted_action_dnn)
        lamdadot = -V_dnn * sin(Fpa_dnn - Lamd_dnn) / R_dnn
        a0 = 3 * V_dnn * lamdadot
        am = predicted_action_dnn + a0
        next_state_dnn, done = missile_dnn.step(predicted_action_dnn)

        action_history_dnn.append(predicted_action_dnn)
        state_history_dnn.append(next_state_dnn)
        am_history_dnn.append(am)
        if done:
            break
        print(f"After update, step {step}, missile_dnn.state = {missile_dnn.state}")
    return  action_history_dnn, state_history_dnn,am_history_dnn



from scipy.io import savemat

missile = MISSILE()
missile_dnn = MISSILE_dnn()
errors = []
errors_dnn = []

num_simulations=1
for kkk in range(num_simulations):
    state_history, action_history,r_history,l_history,am_history = simulate_missile(missile,model)
    action_history_dnn,state_history_dnn ,am_history_dnn= simulate_missile_dnn(missile_dnn,model_dnn)

    time_values, x_values, y_values, v_values,angle_values = zip(*[(state[0], state[1], state[2],state[3], state[4]) for state in state_history])

    time_values_dnn, x_values_dnn, y_values_dnn,v_values_dnn, angle_values_dnn = zip(*[(state[0], state[1], state[2],state[3], state[4]) for state in state_history_dnn])
    action_history_squeezed = np.squeeze(action_history)
    action_history_squeezed_dnn = np.squeeze(action_history_dnn)
    am_history_dnn_squeezed = np.squeeze(am_history_dnn)
    am_history_squeezed = np.squeeze(am_history)

    if task == "1":
        print(f"err_x {x_values[-1] - 20000}, err_y{y_values[-1] - 2000},err_angle{angle_values[-1] / pi * 180 + 80},v{v_values[-1]}")
        print(f"err_x_dnn {x_values_dnn[-1] - 20000}, err_y_dnn{y_values_dnn[-1] - 2000},err_angle_dnn{angle_values_dnn[-1] / pi * 180 + 80},v_dnn{v_values_dnn[-1]}")
    if task == "2":
        print(f"err_x {x_values[-1] - 20000}, err_y{y_values[-1] - 2000},err_t{time_values[-1] - 70},v{v_values[-1]}")
        print( f"err_x_dnn {x_values_dnn[-1] - 20000}, err_y_dnn{y_values_dnn[-1] - 2000},err_t_dnn{time_values_dnn[-1] - 70},v_dnn{v_values_dnn[-1]}")
    if task == "3":
        print(f"err_x {x_values[-1] - 20000}, err_y{y_values[-1] - 2000},err_t{time_values[-1] -80},err_angle{angle_values[-1] / pi * 180 + 70},v{v_values[-1]}")
        print(f"err_x_dnn {x_values_dnn[-1] - 20000}, err_y_dnn{y_values_dnn[-1] - 2000},err_t_dnn{time_values_dnn[-1] - 80},err_angle_dnn{angle_values_dnn[-1] / pi * 180 + 70},v_dnn{v_values_dnn[-1]}")

    plt.plot(x_values, y_values, label='Agent')
    plt.plot(x_values_dnn, y_values_dnn, label='Expert')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Missile Path Comparison')
    plt.legend()
    plt.show()

    plt.plot(time_values, am_history_squeezed, label='Agent')
    plt.plot(time_values_dnn, am_history_dnn_squeezed, label='Expert')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('action')
    plt.legend()
    plt.show()

    plt.plot(time_values, v_values, label='Agent')
    plt.plot(time_values_dnn, v_values_dnn, label='Expert')
    plt.xlabel('time')
    plt.ylabel('volcity')
    plt.title('volcity')
    plt.legend()
    plt.show()

    plt.plot(time_values, [angle_value / 3.1415926 * 180 for angle_value in angle_values], label='Agent')
    plt.plot(time_values_dnn, [angle_value_dnn / 3.1415926 * 180 for angle_value_dnn in angle_values_dnn],
             label='Expert')
    plt.xlabel('time')
    plt.ylabel('angle')
    plt.title('Missile path')
    plt.legend()
    plt.show()