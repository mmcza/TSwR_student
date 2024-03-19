import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        inverted_M = np.linalg.inv(self.model.M(x))
        v = -self.model.C(x) @ inverted_M @ x[2:] + np.eye(2) @ inverted_M @ np.ones((2, 1))

        # zeros = np.zeros((2, 2), dtype=np.float32)
        # A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        # b = np.concatenate([zeros, invM], 0)
        # return A @ x[:, np.newaxis] + b @ u

        return v
