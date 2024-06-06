import numpy as np
import sympy as sp


class Quadrotor3D():
    def __init__(self, prop_dict, params_dict):

        self.n_states = prop_dict['n_states']
        self.n_controls = prop_dict['n_controls']
        self.u1_constraint = prop_dict['u1_constraint']
        self.u2_constraint = prop_dict['u2_constraint']
        self.u3_constraint = prop_dict['u3_constraint']
        self.u4_constraint = prop_dict['u4_constraint']

        self.mass = params_dict['mass']
        self.inertia = np.diag(params_dict['inertia'])
        self.gravity = params_dict['gravity']
        self.delta_t = params_dict['delta_t']
        self.ueq = np.array([self.gravity, 0., 0., 0.])

        x, y, z, vx, vy, vz = sp.symbols('x y z vx vy vz')
        qx, qy, qz, qw, wx, wy, wz = sp.symbols('qx qy qz qw wx wy wz')
        u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')

        Q = sp.Matrix([[qw, -qz, qy],
                    [qz, qw, -qx],
                    [-qy, qx, qw],
                    [-qx, -qy, -qz]]) # becuase omega is in the body frame
        
        omega = sp.Matrix([wx, wy, wz])
        u = sp.Matrix([u1,u2,u3,u4])

        r13 = 2*(qx*qz+qw*qy)
        r23 = 2*(qy*qz-qw*qx)
        r33 = 1 - 2*(qx**2+qy**2)

        J = self.inertia
        J_inv = np.linalg.inv(J)

        states = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] # shape (13,)
        controls = [u1,u2,u3,u4]

        drift1 = sp.Matrix([vx, vy, vz]) # shape (3,1)
        drift2 = 0.5 * Q @ omega # shpae (4,1).
        drift3 = sp.Matrix([0,0,-self.gravity]) # shape (3,1)
        drift4 = - J_inv @ (omega.cross(J @ omega)) # shape (3,1)
        drift = drift1.col_join(drift2).col_join(drift3).col_join(drift4) # shape (13,1)

        self.drift_func = sp.lambdify(states, drift, 'numpy')

        drift_jac = drift.jacobian(states)
        self.drift_jac_func = sp.lambdify(states, drift_jac, 'numpy')

        actuation1 = sp.zeros(7,4) # shpae (7,4)
        actuation2 = sp.Matrix([[r13, 0, 0, 0],
                                [r23, 0, 0, 0],
                                [r33, 0, 0, 0]]) # shpae (3,4)
        actuation3 = sp.zeros(3,1).row_join(sp.Matrix(J_inv)) # shpae (3,4)
        actuation = actuation1.col_join(actuation2).col_join(actuation3) # shpae (13,4)

        self.actuation_func = sp.lambdify(states, actuation, 'numpy')

        full_dynamics = drift + actuation @ u
        discretized_dynamics = sp.Matrix(states) + self.delta_t*full_dynamics
        
        linearized_A = discretized_dynamics.jacobian(states)
        linearized_B = discretized_dynamics.jacobian(controls)
        self.linearized_A_func = sp.lambdify(states + controls, linearized_A, 'numpy')
        self.linearized_B_func = sp.lambdify(states + controls, linearized_B, 'numpy')

    def drift(self, x):
        """
        x: state vector = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
        """
        return self.drift_func(*x)

    def actuation(self, x):
        """
        x: state vector = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
        """
        return self.actuation_func(*x)
    
    def drift_jac(self, x):
        """
        x: state vector = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
        """
        return self.drift_jac_func(*x)
    
    def get_linearization(self, x, u):
        A = self.linearized_A_func(*x, *u)
        B = self.linearized_B_func(*x, *u)
        return A, B
    
    def get_next_state(self, x, u):
        """
        Inputs:
        x: state of the quadrotor as a numpy array [x, y, theta, vx, vy omega]
        u: control as a numpy array (u1, u2)

        Output:
        the new state of the quadrotor as a numpy array
        """
        u[0] = np.clip(u[0], self.u1_constraint[0], self.u1_constraint[1])
        u[1] = np.clip(u[1], self.u2_constraint[0], self.u2_constraint[1])
        u[2] = np.clip(u[2], self.u3_constraint[0], self.u3_constraint[1])
        u[3] = np.clip(u[3], self.u4_constraint[0], self.u4_constraint[1])
        dxdt = np.squeeze(self.drift(x)) + self.actuation(x) @ u
        x_next = x + self.delta_t*dxdt
        x_next[3:7] = x_next[3:7]/np.linalg.norm(x_next[3:7])
        x_next[3:7] = x_next[3:7] * np.sign(x_next[6])
        return x_next
    
    def simulate(self, x0, controller, horizon_length, disturbance=False):
        """
        This function simulates the quadrotor for horizon_length steps from initial state z0

        Inputs:
        x0: the initial conditions of the quadrotor as a numpy array [x, y, theta, vx, vy omega]
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length

        disturbance: if True will generate a random push every seconds during the simulation

        Output:
        t[time_horizon+1] contains the simulation time
        x[time_horizon+1, n_states, ] and u[time_horizon, n_controls] containing the time evolution of states and control
        """
    
        t = np.zeros([horizon_length+1,])
        x = np.empty([horizon_length+1, self.n_states])
        x[0,:] = x0
        u = np.zeros([horizon_length, self.n_controls])
        for i in range(horizon_length):
            u[i,:] = controller(x[i,:],i)
            x[i+1,:] = self.get_next_state(x[i,:], u[i,:])
            if disturbance and np.mod(i,100)==0:
                dist = np.zeros([self.n_states, ])
                dist[1::2] = np.random.uniform(-1.,1,(3,))
                x[i+1,:] += dist
            t[i+1] = t[i] + self.delta_t
        return t, x, u
    
    