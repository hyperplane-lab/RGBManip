import numpy as np
import sapien.core as sapien

class OSCPlanner :

    def __init__ (self, pinocchio, eff_link_id, damping, qmask, dt = 0.1) :

        self.pinocchio : sapien.PinocchioModel = pinocchio
        self.link_id = eff_link_id
        self.damping = damping
        self.qmask = qmask
        self.dt = dt

    def control_ik(self, target_pose, dof_pos):

        result, success, error = self.pinocchio.compute_inverse_kinematics(
            self.link_id,
            target_pose,
            dof_pos,
            self.qmask,
            dt = self.dt,
            damp = self.damping,
            max_iterations=2048
        )

        return result, success, error


        self.pinocchio.compute_full_jacobian(dof_pos)
        jacobian = self.pinocchio.get_link_jacobian(self.link_id)[:, :7]

        # solve damped least squares
        j_eef_T = np.transpose(jacobian)
        lmbda = np.eye(6) * (self.damping ** 2)
        u = (j_eef_T @ np.linalg.inv(jacobian @ j_eef_T + lmbda) @ dpose.reshape(-1, 1)).reshape(-1)
        # print(dpose, u)

        # a = [[ 0.,         -0.24687966,  0.22048437,  0.20609066,  0.10563097, -0.0034682,\
        # -0.16370341,  0.,          0.        ],\
        # [-0.00000123,  0.22347132,  0.24358114,  0.09340601,  0.37274289,  0.0781759,\
        # -0.67183615,  0.,          0.,        ],\
        # [ 0.          ,0.00000001  ,0.00000084 ,-0.17650335 , 0.21965812 , 0.53936886\
        # ,0.00193711  ,0.          ,0.        ],\
        # [-0.00000367  ,0.67108505  ,0.73147261 ,-0.50465177 , 0.81164689 ,-0.22828374\
        # ,-0.28547254 , 0.         , 0.        ],\
        # [-0.00000001  ,0.74138037 ,-0.6621126  ,-0.36496345 , 0.10826459 , 0.96331716\
        # ,0.07231531  ,0.          ,0.        ],\
        # [ 1.          ,0.00000247 ,-0.16295621 ,-0.78238627 ,-0.57402796 ,-0.14109067\
        # ,0.95565471  ,0.          ,0.        ]]

        # b = [[0.         ,-0.2479053  , 0.21928855 , 0.20430644  ,0.09870769 ,-0.00941531\
        # ,-0.15290017 , 0.        ,  0.        ],\
        # [-0.00000123 , 0.22233299 , 0.24451201 , 0.09744147  ,0.37188696 , 0.06331096\
        # ,-0.67433819 , 0.        ,  0.        ],\
        # [ 0.         , 0.00000001 , 0.00000084 ,-0.17326357  ,0.22452759 , 0.54274221\
        # ,-0.01412352 , 0.        ,  0.        ],\
        # [-0.00000367 , 0.66766665 , 0.73426799 ,-0.4985721   ,0.81078597 ,-0.23614014\
        # ,-0.28776301 , 0.        ,  0.        ],\
        # [-0.00000001 , 0.74446037 ,-0.65852154 ,-0.35823141  ,0.12944958 , 0.96469461\
        # ,0.0452117   ,0.         , 0.        ],\
        # [1.         , 0.00000246 ,-0.16492394 ,-0.78936438 ,-0.5708493  ,-0.11662825\
        # ,0.95663387  ,0.         , 0.        ]]

        return u
    
    # def control_osc(self, dpose, hand_vel, dof_pos, dof_vel):

    #     self.pinocchio.compute_full_jacobian(dof_pos)
    #     mass_matrix = self.pinocchio.compute_generalized_mass_matrix(dof_pos)
    #     jacobian = self.pinocchio.get_link_jacobian(self.link_id)

    #     mm_inv = np.linalg.inv(mass_matrix) # torch.inverse(mm)
    #     m_eef_inv = jacobian @ mm_inv @ np.transpose(jacobian)
    #     m_eef = np.linalg.inv(m_eef_inv)
    #     u = np.transpose(jacobian) @ m_eef @ (
    #         self.kp * dpose.reshape(-1, 1) - self.kd * hand_vel.reshape(-1, 1))
    
    #     # Nullspace control torques `u_null` prevents large changes in joint configuration
    #     # They are added into the nullspace of OSC so that the end effector orientation remains constant
    #     # roboticsproceedings.org/rss07/p31.pdf
    #     j_eef_inv = m_eef @ jacobian @ mm_inv
    #     u_null = self.kd_null * -dof_vel.reshape(-1, 1) + self.kp_null * (
    #         (self.default_dof_pos.reshape(-1, 1) - dof_pos.reshape(-1, 1) + np.pi) % (2 * np.pi) - np.pi)
    #     u_null = mass_matrix @ u_null
    #     u += (np.eye(jacobian.shape[1]) - np.transpose(jacobian) @ j_eef_inv) @ u_null
    #     return u.reshape(-1)