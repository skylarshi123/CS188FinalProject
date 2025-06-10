import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz

class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, demo_path='demos.npz', dt=0.01, n_bfs=20):
        print(square_pos)
        self.dt = dt
        self.n_bfs = n_bfs
        
        # Load and parse best demo
        demos = reconstruct_from_npz(demo_path)
        demo = demos['demo_0']
        besterror = 1000
        for i in range(200):
            positions = demos[f'demo_{i}']['obs_robot0_eef_pos']
            demo_nut_pos = min(positions, key=lambda x: x[0])
            error = np.linalg.norm(demo_nut_pos - square_pos) + 1 if demo_nut_pos[1] > square_pos[1] else np.linalg.norm(demo_nut_pos - square_pos)
            if error < besterror:
                besterror = error
                demo = demos[f'demo_{i}']

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        self.segments = segments
        self.segment_grasp_states = []
        self.dmps = []
        self.segment_trajectories = []

        self.rod_pos = []
        #pick up from the right
        if(offset[1] > 0):
            self.rod_pos = [0.237,  0.125,  0.97]
            offset += [0.002, -0.00, -0.02]
        #pick up from left
        else:
            self.rod_pos = [0.220,  0.08,  0.97]
            offset +=  [0.017, 0.02, -0.02]
            
        
        for i, (start, end) in enumerate(segments):
            segment_traj = ee_pos[start:end] 
            grasp_state = ee_grasp[start, 0]
            self.segment_grasp_states.append(grasp_state)
            dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt, y0=segment_traj[0], goal=segment_traj[-1])
            dmp.imitate(segment_traj)
            self.dmps.append(dmp)
            
            """ if i == 0:
                new_goal = new_obj_pos + offset
                traj = dmp.rollout(new_goal=new_goal)
            else:
                traj = dmp.rollout() """
            
            if i == 0:
                new_goal = new_obj_pos + offset
                traj = dmp.rollout(new_goal=new_goal)
            elif i == len(segments) - 2:
                traj = dmp.rollout(new_goal=self.rod_pos)
            else:
                traj = dmp.rollout()
            

            
            self.segment_trajectories.append(traj)
        
        initial_target = np.zeros(3) 
        self.pid = PID(kp=[10.0, 10.0, 10.0], ki=[0.1, 0.1, 0.1], kd=[1.0, 1.0, 1.0], target=initial_target)
        self.current_segment = 0
        self.current_step = 0
        self.total_steps_executed = 0
        self.segment_lengths = [len(traj) for traj in self.segment_trajectories]
        self.cumulative_lengths = np.cumsum([0] + self.segment_lengths)
        self.nudge_step = 0 


    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        segments = []
        grasp_flat = grasp_flags.flatten()
        
        transitions = []
        for i in range(1, len(grasp_flat)):
            if grasp_flat[i] != grasp_flat[i-1]:
                transitions.append(i)
        
        start_idx = 0
        for i, transition_idx in enumerate(transitions):
            segments.append((start_idx, transition_idx))
            start_idx = transition_idx
        
        if start_idx < len(grasp_flat):
            segments.append((start_idx, len(grasp_flat)))
        
        segments = [(start, end) for start, end in segments if end > start]
        
        return segments

    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        amt = 1
        if self.current_segment >= len(self.segments):
            #return np.zeros(7)
            directions = [
                np.array([amt, 0.0]),   # +X
                np.array([0.0, amt]),   # +Y
                np.array([-amt, 0.0]),  # -X
                np.array([0.0, -amt])   # -Y
            ]
            nudge_xy = directions[self.nudge_step % 4]
            self.nudge_step += 1

            action = np.array([
                nudge_xy[0] + 0.02, nudge_xy[1] + 0.02, -0.07,
                0.0, 0.0, 0.0,
                0.0
            ])
            return action
        
        current_traj = self.segment_trajectories[self.current_segment]
        
        if self.current_step >= len(current_traj):
            self.current_segment += 1
            self.current_step = 0
            self.pid.reset()
            
            if self.current_segment >= len(self.segments):
                return np.zeros(7)
            
            current_traj = self.segment_trajectories[self.current_segment]
        
        target_pos = current_traj[self.current_step]
        self.pid.target = target_pos
        delta_pos = self.pid.update(robot_eef_pos, dt=self.dt)
        grasp_state = self.segment_grasp_states[self.current_segment]
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, float(grasp_state)])
        self.current_step += 1
        self.total_steps_executed += 1
        
        return action