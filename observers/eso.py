from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z_hat = self.state[:, np.newaxis]
        z_hat_dot = self.A @ z_hat + self.B @ u + np.array(self.L @ (q.reshape(-1, 1) - self.W @ z_hat))
        self.state = (z_hat + z_hat_dot * self.Tp).flatten()

    def get_state(self):
        return self.state
