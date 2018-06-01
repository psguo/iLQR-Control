from .control import Control
import numpy as np
import scipy.linalg as linalg

class LQR(Control):
    def __init__(self, **kwargs):
        super(Control, self).__init__(**kwargs)

        self.task_DOF = 2
        self.u = None

        self.eps = 0.00001


    def simulate_dynamics(self, x, u):
        self.arm.reset(q=x[:self.arm.DOF],
                       dq=x[self.arm.DOF:self.arm.DOF*2])
        self.arm.apply_torque(u, self.arm.dt)
        xnext = np.hstack([np.copy(self.arm.q), np.copy(self.arm.dq)])
        return xnext

    def finite_diff_method(self, x, u):
        x_1_perturbed = np.tile(x, (self.arm.DOF*2, 1)) + np.eye(self.arm.DOF*2) * self.eps
        x_2_perturbed = np.tile(x, (self.arm.DOF*2, 1)) - np.eye(self.arm.DOF*2) * self.eps
        u_same = np.tile(u, (self.arm.DOF*2, 1))
        f_1 = self.simulate_dynamics(x_1_perturbed, u_same)
        f_2 = self.simulate_dynamics(x_2_perturbed, u_same)

        A = (f_1 - f_2) / (2 * self.eps)

        u_1_perturbed = np.tile(x, (self.arm.DOF, 1)) + np.eye(self.arm.DOF) * self.eps
        u_2_perturbed = np.tile(x, (self.arm.DOF, 1)) - np.eye(self.arm.DOF) * self.eps
        x_same = np.tile(u, (self.arm.DOF, 1))
        f_1 = self.simulate_dynamics(x_same, u_1_perturbed)
        f_2 = self.simulate_dynamics(x_same, u_2_perturbed)

        B = (f_1 - f_2) / (2 * self.eps)

        return A, B

    def copy_arm(self, real_arm):
        arm = real_arm.__class__()
        arm.dt = real_arm.dt

        # reset arm position to x_0
        arm.reset(q = real_arm.q, dq = real_arm.dq)

        return arm, np.hstack([real_arm.q, real_arm.dq])

    def control(self, arm):
        x_desired = self.x - self.target

        self.arm, x = self.copy_arm(arm)
        self.Q = np.zeros((arm.DOF*2, arm.DOF*2))
        self.Q[:arm.DOF, :arm.DOF] = np.eye(arm.DOF) * 1000.0
        self.R = np.eye(arm.DOF) * 0.001

        A, B = self.finite_diff_method(x, self.u)
        P = linalg.solve_discrete_are(A, B, self.Q, self.R)
        K = -np.dot(np.linalg.pinv(self.R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))

        J = self.arm.gen_jacEE()
        x = np.hstack([np.dot(J.T, x_desired)], arm.dq)

        self.u = np.dot(K, x)

        return self.u

    def gen_target(self, arm):
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0

        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()
