import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
#from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3*p[0], 0], [0, 3*p[1]], [3*p[0]**2, 0], [0, 3*p[1]**2], [p[0]**3, 0], [0, p[1]**3]])
        W = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
        self.A = np.array([[0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]])
        self.B = np.zeros((6, 2))
        self.eso = ESO(self.A, self.B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])
        inv_M = np.linalg.inv(self.model.M(x.flatten()))
        inv_M_C = inv_M @ self.model.C(x.flatten())
        self.A[2:4, 2:4] = - inv_M_C
        self.B[2:4, :] = inv_M
        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        z = self.eso.get_state()
        z = z[:, np.newaxis]
        v = q_d_ddot - self.Kp @ (x[:2] - q_d) - self.Kd @ (x[2:] - q_d_dot)
        # print(v.shape)
        # print(z[:4].shape)
        # print(z[4:].shape)
        # print(z[2:4].shape)
        # print(x.shape)
        #u = self.model.M(z[0:4]) @ (v[:, np.newaxis] - z[4:]) + self.model.C(z[0:4]) @ z[2:4]
        u = self.model.M(z[0:4].flatten()) @ (v[:, np.newaxis] - z[4:]) + self.model.C(z[0:4].flatten()) @ z[2:4]
        self.update_params(z[0:2], z[2:4])
        self.eso.update(x[:2], u)
        return u
        #return NotImplementedError
