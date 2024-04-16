import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse


class Quadrotor2D():
    def __init__(self, prop_dict, params_dict):
        """
        See https://cookierobotics.com/052/ for the derivation of the dynamics
        """
        self.n_states = prop_dict['n_states']
        self.n_controls = prop_dict['n_controls']
        self.u1_constraint = prop_dict['u1_constraint']
        self.u2_constraint = prop_dict['u2_constraint']

        self.mass = params_dict['mass']
        self.inertia = params_dict['inertia']
        self.length = params_dict['length']
        self.gravity = params_dict['gravity']
        self.delta_t = params_dict['delta_t']

        self.bounding_shape_config = {'shape': 'ellipse',
                                      'semi_major_axis': 1.5*self.length,
                                      'semi_minor_axis': 0.75*self.length}

        x, y, theta, vx, vy, omega = sp.symbols('x y theta v_x v_y omega')
        u1, u2 = sp.symbols('u_1 u_2')

        drift = sp.Matrix([vx,
                           vy,
                           omega,
                           0,
                           -self.gravity,
                           0])
        self.drift_func = sp.lambdify((x, y, theta, vx, vy, omega), drift, 'numpy')

        drift_jac = drift.jacobian([x, y, theta, vx, vy, omega])
        self.drift_jac_func = sp.lambdify((x, y, theta, vx, vy, omega), drift_jac, 'numpy')

        actuation1 = sp.Matrix([0,
                               0,
                               0,
                               -sp.sin(theta)/ self.mass,
                               sp.cos(theta)/ self.mass,
                               self.length/self.inertia,])
    
        actuation2 = sp.Matrix([0,
                               0,
                               0,
                               -sp.sin(theta)/ self.mass,
                               sp.cos(theta)/ self.mass,
                               -self.length/self.inertia,])
        self.actuation1_func = sp.lambdify((x, y, theta, vx, vy, omega), actuation1, 'numpy')
        self.actuation2_func = sp.lambdify((x, y, theta, vx, vy, omega), actuation2, 'numpy')

        full_dynamics = drift + u1*actuation1 + u2*actuation2
        discretized_dynamics = sp.Matrix([x, y, theta, vx, vy, omega]) + self.delta_t*full_dynamics
        linearized_A = discretized_dynamics.jacobian([x, y, theta, vx, vy, omega])
        linearized_B = discretized_dynamics.jacobian([u1, u2])
        self.linearized_A_func = sp.lambdify((x, y, theta, vx, vy, omega, u1, u2), linearized_A, 'numpy')
        self.linearized_B_func = sp.lambdify((x, y, theta, vx, vy, omega, u1, u2), linearized_B, 'numpy')

    def drift(self, x):
        """
        x: state vector = [x, y, theta, vx, vy omega]
        """
        return self.drift_func(*x)

    def actuation(self, x):
        """
        x: state vector = [x, y, theta, vx, vy omega]
        """
        g1 = self.actuation1_func(*x)
        g2 = self.actuation2_func(*x)
        return np.hstack((g1, g2))
    
    def drift_jac(self, x):
        """
        x: state vector = [x, y, theta, vx, vy omega]
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
        dxdt = np.squeeze(self.drift(x)) + self.actuation(x) @ u
        x_next = x + self.delta_t*dxdt
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
    
    def animate_robot(self, x, u, dt, save_video_path, plot_bounding_ellipse=True, plot_traj=True, obstacles=None,
                      robot_constraints=None, obstacle_constraints=None):
        """
        This function makes an animation showing the behavior of the quadrotor
        takes as input the result of a simulation (with dt=0.01s)
        """

        min_dt = 0.1
        if(dt < min_dt):
            steps = int(min_dt/dt)
            use_dt = int(np.round(min_dt * 1000)) #in ms
        else:
            steps = 1
            use_dt = int(np.round(dt * 1000)) #in ms

        #what we need to plot
        plotx = x[::steps,:]
        plotu = u[::steps,:]
        plotx = plotx[:len(plotu)]

        fig, ax = plt.subplots(figsize=[8.5, 8.5])
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.grid()

        list_of_lines = []

        #create the robot
        # the main frame
        line, = ax.plot([], [], 'k', lw=6, solid_capstyle='butt', zorder=2.1)
        list_of_lines.append(line)
        # the left propeller
        line, = ax.plot([], [], 'b', lw=4, solid_capstyle='butt', zorder=2.1)
        list_of_lines.append(line)
        # the right propeller
        line, = ax.plot([], [], 'b', lw=4, solid_capstyle='butt', zorder=2.1)
        list_of_lines.append(line)
        # the left thrust
        line, = ax.plot([], [], 'r', lw=1, solid_capstyle='butt', zorder=2.1)
        list_of_lines.append(line)
        # the right thrust
        line, = ax.plot([], [], 'r', lw=1, solid_capstyle='butt', zorder=2.1)
        list_of_lines.append(line)

        if plot_bounding_ellipse:
            # Create the bounding ellipse
            ellipse_width = 2 * self.bounding_shape_config['semi_major_axis']  # Modify the size as needed
            ellipse_height = 2 * self.bounding_shape_config['semi_minor_axis'] # Modify the size as needed
            ellipse = Ellipse((0, 0), ellipse_width, ellipse_height, angle=0, fill=False, linestyle='--', linewidth=1, zorder=2.1)
            ax.add_patch(ellipse)
        
        if plot_traj:
            # Initialize the trajectory line
            trajectory_line, = ax.plot([], [], 'g', lw=1.5, zorder=2)
            trajectory_data = []

        if obstacles is not None:
            for obstacle in obstacles:
                ax.add_patch(obstacle)
        
        if robot_constraints is not None:
            robot_constraints = robot_constraints[::steps]
            robot_constraint_points = ax.scatter([],[], color='g', alpha=0.01, zorder=2.2)
        
        if obstacle_constraints is not None:
            obstacle_constraints = obstacle_constraints[::steps]
            obstacles_constraint_points = ax.scatter([],[], color='r', alpha=0.01, zorder=2.2)
            
        def _animate(i):
            for l in list_of_lines: #reset all lines
                l.set_data([],[])

            x = plotx[i,0]
            y = plotx[i,1]
            theta = plotx[i,2]
            trans = np.array([[x,x],[y,y]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            main_frame = np.array([[-self.length, self.length],
                                   [-0.05,-0.05]])
            main_frame = rot @ main_frame + trans 

            left_propeller = np.array([[-1.3 * self.length, -0.5*self.length],
                                       [0.05,0.05]])
            left_propeller = rot @ left_propeller + trans

            right_propeller = np.array([[1.3 * self.length, 0.5*self.length],
                                        [0.05,0.05]])
            right_propeller = rot @ right_propeller + trans

            right_thrust = np.array([[self.length, self.length],
                                     [0.05, 0.05+plotu[i,0]*0.06]])
            right_thrust = rot @ right_thrust + trans

            left_thrust = np.array([[-self.length, -self.length],
                                    [0.05, 0.05+plotu[i,1]*0.06]])
            left_thrust = rot @ left_thrust + trans

            list_of_lines[0].set_data(main_frame[0,:], main_frame[1,:])
            list_of_lines[1].set_data(left_propeller[0,:], left_propeller[1,:])
            list_of_lines[2].set_data(right_propeller[0,:], right_propeller[1,:])
            list_of_lines[3].set_data(left_thrust[0,:], left_thrust[1,:])
            list_of_lines[4].set_data(right_thrust[0,:], right_thrust[1,:])

            all_artists = list_of_lines

            if plot_bounding_ellipse:
                # Update the ellipse
                x = plotx[i, 0]
                y = plotx[i, 1]
                theta = np.degrees(plotx[i, 2])  # Convert to degrees if necessary
                ellipse.set_center((x, y))
                ellipse.angle = theta
                all_artists = all_artists + [ellipse]
            
            if plot_traj:
                # Update the trajectory
                trajectory_data.append((plotx[i, 0], plotx[i, 1]))
                trajectory_line.set_data(*zip(*trajectory_data))
                all_artists = all_artists + [trajectory_line]
            
            if obstacles is not None:
                all_artists = all_artists + obstacles
            
            if robot_constraints is not None:
                robot_constraint_points.set_offsets(robot_constraints[i])
                all_artists = all_artists + [robot_constraint_points]
            
            if obstacle_constraints is not None:
                obstacles_constraint_points.set_offsets(obstacle_constraints[i])
                all_artists = all_artists + [obstacles_constraint_points]

            return all_artists

        def _init():
            return _animate(0)

        ani = animation.FuncAnimation(fig, _animate, np.arange(0, len(plotx)),
            interval=use_dt, blit=True, init_func=_init)
        plt.close(fig)
        plt.close(ani._fig)
        ani.save(save_video_path, writer='ffmpeg', fps=int(1000/use_dt))
