import numpy as np
import sapien.core as sapien

class ImpedanceController :

    def __init__ (self,
                  pinocchio,
                  eff_link_id, 
                  cartesian_stiffness,
                  cartesian_damping,
                  nullspace_stiffness,
                  damping,
                  qmask
                ) :

        self.pinocchio : sapien.PinocchioModel = pinocchio
        self.link_id = eff_link_id
        self.damping = damping
        self.cartesian_stiffness = cartesian_stiffness
        self.cartesian_damping = cartesian_damping
        self.nullspace_stiffness = nullspace_stiffness
        self.qmask = qmask
        self.maskid = np.nonzero(qmask)

    def control_ik(self, target_pose : sapien.Pose, start_dof_pos, dof_pos, dof_vel):

        coriolis = self.pinocchio.compute_coriolis_matrix(dof_pos, dof_vel)
        jacobian = self.pinocchio.get_link_jacobian(self.link_id)[:, :7]
        q = dof_pos[self.maskid].reshape(-1, 1)
        q_nullspace = start_dof_pos[self.maskid].reshape(-1, 1)
        dq = dof_vel[self.maskid].reshape(-1, 1)
        error = np.zeros((6, 1))
        current_pose = self.pinocchio.get_link_pose(self.link_id)
        tau_task = np.zeros((7, 1))
        tau_null = np.zeros((7, 1))

        error[:3, 0] = current_pose.p - target_pose.p
        print(error)
        # error[3:, 0] = (current_pose.inv().q *  target_pose.q)[1:]

        # Computing Peudo Inverse
        lmbda = np.eye(6) * (self.damping ** 2)
        j_eef_T = np.transpose(jacobian)
        pinv =  np.linalg.inv(jacobian @ j_eef_T + lmbda) @ jacobian

        tau_task = j_eef_T @ (-self.cartesian_stiffness * error - self.cartesian_damping * (jacobian @ dq))
        tau_null = (np.eye(7) - j_eef_T @ pinv) @ (self.nullspace_stiffness * (q_nullspace - q) - (2.0 * np.sqrt(self.nullspace_stiffness)) * dq)

        tau = tau_task + tau_null
    
        # print(coriolis.shape)
        # print(jacobian.shape)

        return tau.transpose()[0]