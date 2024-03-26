import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.m3_list = [0.1, 0.01, 1.0]
        self.r3_list = [0.05, 0.01, 0.3]
        self.models = []
        for i in range(len(self.m3_list)):
            self.models.append(ManiuplatorModel(Tp, self.m3_list[i], self.r3_list[i]))

        # self.models = [None, None, None]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        errors = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)


        #self.i = np.argmin(errors)
        # TODO: remove once implemented
        self.i = 0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot # TODO: add feedback
        v = q_r_ddot - self.models[self.i].Kp @ (x[:2] - q_r) - self.models[self.i].Kd @ (x[2:] - q_r_dot)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
