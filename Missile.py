import scipy.io
from scipy import interpolate
import scipy.io
from random import randint, uniform, choice
import numpy as np
import math
from math import sin, cos, atan2, sqrt
device = 'cpu'

us1976 = scipy.io.loadmat('./cld_parmeters/us1976.mat')
cxdata0 = scipy.io.loadmat('./cld_parmeters/cxdata0.mat')
cydata0 = scipy.io.loadmat('./cld_parmeters/cydata0.mat')
us1976 = us1976['us1976']
cxdata0 = cxdata0['cxdata0']
cydata0 = cydata0['cydata0']
RAD = 180 / 3.1415926

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
        self.xf =  uniform(20000, 20500)
        self.fpa0 = state[4]
        self.state = np.array(state)
        self.done = False


    def step(self, action):
        Rr = sqrt((self.state[1] - self.xf) ** 2 + (self.state[2] - self.yf) ** 2)
        if Rr>400:
            Inth = 0.05
        else:
            Inth = 0.002

        nextstate = self.rk4(self.dyfunc, self.state, action, Inth)
        nextr = sqrt((nextstate[1] - self.xf) ** 2 + (nextstate[2] - self.yf) ** 2)
        if (self.state[2]<2001 and self.state[1]>0.85*self.xf) or self.state[0]>120:
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