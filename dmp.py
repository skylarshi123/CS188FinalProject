import numpy as np
from scipy import interpolate

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0
        self.timesteps: int = int(self.run_time / dt)
        self.x: float = 1.0  # phase variable

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x = 1.0

    def __step_once(self, x: float, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Single step update of the canonical system.
        
        Args:
            x (float): Current phase value
            tau (float): Temporal scaling factor
            error_coupling (float): Error coupling term
            
        Returns:
            float: Updated phase value
        """
        dx = -self.ax * x * self.dt / tau
        return x + dx

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        self.x = self.__step_once(self.x, tau, error_coupling)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        xs = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
            xs[i] = self.step(tau, ec)
        return xs

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        
        self.y0: np.ndarray = np.ones(n_dmps) * y0 if np.isscalar(y0) else np.array(y0)
        self.goal: np.ndarray = np.ones(n_dmps) * goal if np.isscalar(goal) else np.array(goal)
        self.ay: np.ndarray = np.ones(n_dmps) * ay if np.isscalar(ay) else np.array(ay)
        
        if by is None:
            self.by: np.ndarray = self.ay /4
        else:
            self.by: np.ndarray = np.ones(n_dmps) * by if np.isscalar(by) else np.array(by)
        
        self.w: np.ndarray = np.zeros((n_dmps, n_bfs))
        
        self.cs: CanonicalSystem = CanonicalSystem(dt=dt)
        
        self.c = np.exp(-self.cs.ax * np.linspace(0, self.cs.run_time, n_bfs))
        self.h = np.ones(n_bfs) * n_bfs**1.5 / self.c / self.cs.ax
        
        self.y = None
        self.dy = None
        self.ddy = None
        
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()

    def basis_functions(self, x: float) -> np.ndarray:
        """
        Compute basis function activations for given phase x.
        
        Args:
            x (float): Phase variable value
            
        Returns:
            np.ndarray: Basis function activations
        """
        psi = np.exp(-self.h * (x - self.c)**2)
        return psi / np.sum(psi) 

    def forcing_term(self, x: float, dmp_idx: int = 0) -> float:
        """
        Compute the forcing term for a given phase and DMP dimension.
        
        Args:
            x (float): Phase variable value
            dmp_idx (int): DMP dimension index
            
        Returns:
            float: Forcing term value
        """
        psi = self.basis_functions(x)
        return x * (self.goal[dmp_idx] - self.y0[dmp_idx]) * np.dot(psi, self.w[dmp_idx, :])

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (T, D) or (T,) for 1D.

        Returns:
            np.ndarray: Interpolated demonstration (T', D).
        """
        if y_des.ndim == 1:
            y_des = y_des.reshape(-1, 1)
        
        T_demo, D = y_des.shape
        
        T_target = self.cs.timesteps
        t_demo = np.linspace(0, self.cs.run_time, T_demo)
        t_target = np.linspace(0, self.cs.run_time, T_target)
        
        y_interp = np.zeros((T_target, D))
        for d in range(D):
            f = interpolate.interp1d(t_demo, y_des[:, d], kind='cubic', fill_value='extrapolate')
            y_interp[:, d] = f(t_target)
        
        y_interp = y_interp[:, :self.n_dmps]
        
        dy_interp = np.gradient(y_interp, axis=0) / self.dt
        ddy_interp = np.gradient(dy_interp, axis=0) / self.dt
        
        x_track = self.cs.rollout()
        
        for d in range(self.n_dmps):
            f_target = (ddy_interp[:, d] - 
                       self.ay[d] * (self.by[d] * (self.goal[d] - y_interp[:, d]) - dy_interp[:, d]))
            
            Psi = np.zeros((T_target, self.n_bfs))
            for t in range(T_target):
                Psi[t, :] = self.basis_functions(x_track[t])
            
            scale = x_track[:, np.newaxis] * (self.goal[d] - self.y0[d])
            Psi_scaled = Psi * scale
            
            self.w[d, :] = np.linalg.lstsq(Psi_scaled, f_target, rcond=None)[0]
        
        return y_interp.T if self.n_dmps == 1 else y_interp

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        self.reset_state()
        
        goal = new_goal if new_goal is not None else self.goal
        if np.isscalar(goal):
            goal = np.ones(self.n_dmps) * goal
        
        y_track = np.zeros((self.cs.timesteps, self.n_dmps))
        dy_track = np.zeros((self.cs.timesteps, self.n_dmps))
        ddy_track = np.zeros((self.cs.timesteps, self.n_dmps))
        
        for t in range(self.cs.timesteps):
            x = self.cs.x
            
            f = np.zeros(self.n_dmps)
            for d in range(self.n_dmps):
                f[d] = self.forcing_term(x, d)
            
            self.ddy = (self.ay * (self.by * (goal - self.y) - self.dy * tau) + f) / (tau * tau)
            
            self.dy += self.ddy * self.dt
            self.y += self.dy * self.dt
            
            y_track[t, :] = self.y
            dy_track[t, :] = self.dy
            ddy_track[t, :] = self.ddy
            
            self.cs.step(tau, error)
        
        return y_track

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()