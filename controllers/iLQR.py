from .LQR import LQR

import numpy as np
from numpy.linalg import multi_dot

class iLQR(LQR):
    def __init__(self, n=50, max_iter=100, **kwargs):
        super(iLQR, self).__init__(**kwargs)
        self.T = n
        self.max_iter = max_iter
        self.mu_factor = 10
        self.mu_max = 1000
        self.eps_converge = 0.001

    def control(self, arm):
        if self.t >= self.T-1:
            self.t = 0

        if self.t % 1 == 0:
            x0 = np.zeros(arm.DOF*2)
            self.arm, x0[:arm.DOF*2] = self.copy_arm(arm)
            U = np.copy(self.U[self.t:])
            self.X, self.U[self.t:], cost = self.ilqr(x0, U)
        self.u = self.U[self.t]

        self.t += 1

        return self.u

    def copy_arm(self, arm_to_copy):
        arm = arm_to_copy.__class__()
        arm.dt = arm_to_copy.dt
        arm.reset(q = arm_to_copy.q, dq = arm_to_copy.dq)
        return arm, np.hstack([arm_to_copy.q, arm_to_copy.dq])


    def ilqr(self, x0, U=None):
        U = self.U if U is None else U

        T = U.shape[0]
        dof = self.arm.DOF
        num_states = dof * 2
        dt = self.arm.dt

        mu = 1.0
        sim_trajectory = True

        for index in range(self.max_iter):
            if sim_trajectory:
                X, cost = self.simulate(x0, U)
                f_x = np.zeros((T, num_states, num_states))
                f_u = np.zeros((T, num_states, dof))
                l = np.zeros((T, 1))
                l_x = np.zeros((T, num_states))
                l_xx = np.zeros((T, num_states, num_states))
                l_u = np.zeros((T, dof))
                l_uu = np.zeros((T, dof, dof))
                l_ux = np.zeros((T, dof, num_states))
                for t in range(T - 1):
                    A, B = self.finite_differences(X[t], U[t])
                    f_x[t] = np.eye(num_states) + A * dt
                    f_u[t] = B * dt

                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = self.cost(X[t], U[t])
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt

                l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

                sim_trajectory = False

            V = l[-1].copy()
            V_x = l_x[-1].copy()
            V_xx = l_xx[-1].copy()
            k = np.zeros((T, dof))
            K = np.zeros((T, dof, num_states))

            # Backward pass
            for t in range(T - 2, -1, -1): # from second last time step to first time step
                Q_x = l_x[t] + np.dot(V_x, f_x[t])
                Q_u = l_u[t] + np.dot(V_x, f_u[t])

                Q_xx = l_xx[t] + multi_dot([f_x[t].T, V_xx, f_x[t]])
                Q_ux = l_ux[t] + multi_dot([f_u[t].T, V_xx, f_x[t]])
                Q_uu = l_uu[t] + multi_dot([f_u[t].T, V_xx, f_u[t]])

                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += mu
                Q_uu_inv = multi_dot([Q_uu_evecs, np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T])

                k[t] = -np.dot(Q_uu_inv, Q_u)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                V_x = Q_x - multi_dot([Q_u, Q_uu_inv, Q_ux])
                V_xx = Q_xx - multi_dot([Q_ux.T, Q_uu_inv, Q_ux])

            U_updated = np.zeros((T, dof))
            x_updated = x0.copy()

            # Forward pass
            for t in range(T - 1):
                U_updated[t] = U[t] + k[t] + np.dot(K[t], x_updated - X[t])
                _, x_updated = self.simulate_dynamics(x_updated, U_updated[t])

            # Evaluate the updated controller
            X_updated, cost_updated = self.simulate(x0, U_updated)

            if cost_updated < cost:
                mu /= self.mu_factor

                X = np.copy(X_updated)
                U = np.copy(U_updated)

                sim_trajectory = True

                if index > 0 and ((abs(cost-cost_updated)/cost_updated) < self.eps_converge):
                    break

                cost = np.copy(cost_updated)
            else:
                mu *= self.mu_factor
                if mu > self.mu_max:
                    break

        return X, U, cost

    def simulate_dynamics(self, x, u):
        self.arm.reset(q=x[:self.arm.DOF],
                       dq=x[self.arm.DOF:self.arm.DOF*2])
        self.arm.apply_torque(u, self.arm.dt)
        xnext = np.hstack([np.copy(self.arm.q), np.copy(self.arm.dq)])
        xdot = ((xnext - x) / self.arm.dt).squeeze()
        return xdot, xnext

    def simulate(self, x0, U):
        T = U.shape[0]
        num_states = x0.shape[0]
        dt = self.arm.dt

        X = np.zeros((T, num_states))
        X[0] = x0
        cost = 0

        for t in range(T - 1):
            _, X[t + 1] = self.simulate_dynamics(X[t], U[t])
            l, _, _, _, _, _ = self.cost(X[t], U[t])
            cost += + dt * l

        l_f, _, _ = self.cost_final(X[-1])
        cost = cost + l_f

        return X, cost

    def reset(self, arm):
        self.t = 0
        self.U = np.zeros((self.tN, arm.DOF))
        self.old_target = self.target.copy()


    def gen_target(self, arm):
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0

        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()

    def finite_diff_method(self, x, u):
        x_1_perturbed = np.tile(x, (self.arm.DOF*2, 1)) + np.eye(self.arm.DOF*2) * self.eps
        x_2_perturbed = np.tile(x, (self.arm.DOF*2, 1)) - np.eye(self.arm.DOF*2) * self.eps
        u_same = np.tile(u, (self.arm.DOF*2, 1))
        f_1, _ = self.simulate_dynamics(x_1_perturbed, u_same)
        f_2, _ = self.simulate_dynamics(x_2_perturbed, u_same)

        A = (f_1 - f_2) / (2 * self.eps)

        u_1_perturbed = np.tile(x, (self.arm.DOF, 1)) + np.eye(self.arm.DOF) * self.eps
        u_2_perturbed = np.tile(x, (self.arm.DOF, 1)) - np.eye(self.arm.DOF) * self.eps
        x_same = np.tile(u, (self.arm.DOF, 1))
        f_1, _ = self.simulate_dynamics(x_same, u_1_perturbed)
        f_2, _ = self.simulate_dynamics(x_same, u_2_perturbed)

        B = (f_1 - f_2) / (2 * self.eps)

        return A, B

    def cost(self, x, u):
        # compute cost
        dof = u.shape[0]
        num_states = x.shape[0]

        l = np.sum(u**2)

        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))
        l_u = 2 * u
        l_uu = 2 * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        return l, l_x, l_xx, l_u, l_uu, l_ux

    def cost_final(self, x):
        num_states = x.shape[0]
        l_x = np.zeros((num_states))
        l_xx = np.zeros((num_states, num_states))

        wp = 1e4 # terminal position cost weight
        wv = 1e4 # terminal velocity cost weight

        xy = self.arm.x
        xy_err = np.array([xy[0] - self.target[0], xy[1] - self.target[1]])
        l = (wp * np.sum(xy_err**2) +
                wv * np.sum(x[self.arm.DOF:self.arm.DOF*2]**2))

        l_x[0:self.arm.DOF] = wp * self.dif_end(x[0:self.arm.DOF])
        l_x[self.arm.DOF:self.arm.DOF*2] = (2 *
                wv * x[self.arm.DOF:self.arm.DOF*2])

        eps = 1e-4 # finite difference epsilon
        # calculate second derivative with finite differences
        for k in range(self.arm.DOF):
            veps = np.zeros(self.arm.DOF)
            veps[k] = eps
            d1 = wp * self.dif_end(x[0:self.arm.DOF] + veps)
            d2 = wp * self.dif_end(x[0:self.arm.DOF] - veps)
            l_xx[0:self.arm.DOF, k] = ((d1-d2) / 2.0 / eps).flatten()

        l_xx[self.arm.DOF:self.arm.DOF*2, self.arm.DOF:self.arm.DOF*2] = 2 * wv * np.eye(self.arm.DOF)

        # Final cost only requires these three values
        return l, l_x, l_xx

    def dif_end(self, x):

        xe = -self.target.copy()
        for ii in range(self.arm.DOF):
            xe[0] += self.arm.L[ii] * np.cos(np.sum(x[:ii+1]))
            xe[1] += self.arm.L[ii] * np.sin(np.sum(x[:ii+1]))

        edot = np.zeros((self.arm.DOF,1))
        for ii in range(self.arm.DOF):
            edot[ii,0] += (2 * self.arm.L[ii] *
                    (xe[0] * -np.sin(np.sum(x[:ii+1])) +
                     xe[1] * np.cos(np.sum(x[:ii+1]))))
        edot = np.cumsum(edot[::-1])[::-1][:]

        return edot
