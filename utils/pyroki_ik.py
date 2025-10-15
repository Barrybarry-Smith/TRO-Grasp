import argparse
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import yourdfpy
import numpy as np
import time
import tqdm
import pyroki as pk
from pyroki.costs._pose_cost_analytic_jac import _get_actuated_joints_applied_to_target

def _pos_cost_jac_multi(
    vals: jaxls.VarValues,
    jac_cache,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,  # Array of indices for all target links
    target_positions: jax.Array,  # (num_targets, 3) array of target positions
) -> jax.Array:
    """
    Returns:
        A JAX array representing the combined Jacobian of shape
        (num_targets * 3, robot.joints.num_actuated_joints).
    """
    del vals, joint_var, target_positions 

    Ts_world_joint, Ts_world_link, _ = jac_cache

    # Convert cached poses to jaxlie.SE3 objects for easier manipulation
    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)  # Shape: (n_total_joints, SE3_dim)
    T_world_target_links_se3 = jaxlie.SE3(Ts_world_link[target_link_indices])  # Shape: (num_targets, SE3_dim)

    # Get joint twists for all joints (linear and angular velocity components)
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_local = joint_twists[:, 3:]  # (n_total_joints, 3)
    vel_local = joint_twists[:, :3]    # (n_total_joints, 3)

    # Transform joint twists (angular and linear velocities) to the world frame
    omega_wrt_world = Ts_world_joint_se3.rotation() @ omega_local # (n_total_joints, 3)
    vel_wrt_world = Ts_world_joint_se3.rotation() @ vel_local     # (n_total_joints, 3)

    # Calculate the vector from each joint origin to each target link origin in world frame
    # This difference vector is crucial for the cross product term in the Jacobian.
    # By broadcasting, diff_vec becomes (num_targets, n_total_joints, 3)
    diff_vec = T_world_target_links_se3.translation()[:, None, :] - Ts_world_joint_se3.translation()[None, :, :]

    # Compute the cross product term (omega x r) for the Jacobian
    cross_product_term = jnp.cross(omega_wrt_world[None, :, :], diff_vec) # (num_targets, n_total_joints, 3)

    # Compute the full Jacobian for position for all target links with respect to all joints.
    jac_dp_all_targets = cross_product_term + vel_wrt_world[None, :, :]   # (num_targets, n_total_joints, 3)

    # Transpose each (n_total_joints, 3) block to (3, n_total_joints) to match jaxls format
    jac_dp_all_targets_transposed = jac_dp_all_targets.transpose(0, 2, 1) # (num_targets, 3, n_total_joints)

    # Determine which actuated joints apply to each target link.
    # First, get the parent joint indices for all links.
    base_link_mask = robot.links.parent_joint_indices == -1
    parent_joint_indices = jnp.where(
        base_link_mask, 0, robot.links.parent_joint_indices
    )
    target_joint_indices_for_all_targets = parent_joint_indices[target_link_indices]

    # Use jax.vmap to efficiently determine `joints_applied_to_target` for each target link.
    # This returns an array of shape (num_targets, robot.joints.num_actuated_joints)
    # Each element is the actuated joint index if it applies, or -1 otherwise.
    joints_applied_to_all_targets = jax.vmap(
        lambda idx: _get_actuated_joints_applied_to_target(robot, idx)
    )(target_joint_indices_for_all_targets)

    # Initialize the final combined Jacobian matrix.
    # Its shape is (num_residuals, num_actuated_joints), where num_residuals = num_targets * 3.
    full_jac = jnp.zeros((target_link_indices.shape[0] * 3, robot.joints.num_actuated_joints))

    def _process_one_target_jac(current_full_jac_carry, i):
        jac_i_all_joints = jac_dp_all_targets_transposed[i]
        applied_indices_i = joints_applied_to_all_targets[i]
        jac_i_actuated = (
            jnp.zeros((3, robot.joints.num_actuated_joints))
            .at[:, applied_indices_i]
            .add((applied_indices_i[None, :] != -1) * jac_i_all_joints)
        )
        start_row = i * 3
        updated_full_jac = jax.lax.dynamic_update_slice(current_full_jac_carry, jac_i_actuated, (start_row, 0))
        return updated_full_jac, None


    full_jac, _ = jax.lax.scan(
        _process_one_target_jac,
        full_jac,  
        jnp.arange(target_link_indices.shape[0]) 
    )

    return 10 * full_jac

class PyrokiRetarget:
    def __init__(
            self,
            urdf_path: str,
            target_link_name: list[str]
        ):
        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(urdf)
        self.target_link_index = jnp.array([
            self.robot.links.names.index(name) for name in target_link_name
        ])
        
    def solve_retarget(
        self,
        initial_q: jax.Array, 
        target_pos: jax.Array
    ):

        joint_var = self.robot.joint_var_cls(0)

        def solve_single(
            init_q: jax.Array,
            target_link_pos: jax.Array,
        )-> jax.Array:
            
            """analytical jacobion"""
            @jaxls.Cost.create_factory(jac_custom_with_cache_fn=_pos_cost_jac_multi)
            def pos_cost_analytical_jac_multi(
                vals: jaxls.VarValues,
                robot: pk.Robot,
                joint_var: jaxls.Var[jax.Array],
                target_link_indices: jax.Array,  # Array of indices
                target_positions: jnp.ndarray # (num_targets, 3) array of target positions
            ):
                joint_cfg = vals[joint_var]

                Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
                Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

                # Extract the translation components for all specified target links
                # T_world_target_link_translations: (num_targets, 3)
                T_world_target_link_translations = jaxlie.SE3(Ts_world_link[target_link_indices, :]).translation()

                # Calculate the position error for each target link
                # pos_error: (num_targets, 3)
                pos_error = T_world_target_link_translations - target_positions

                # Flatten the error vector to match jaxls's expectation for residuals
                # flattened_pos_error: (num_targets * 3,)
                flattened_pos_error = pos_error.flatten()

                # Return the weighted flattened error and cache necessary values for the Jacobian
                return (
                    10 * flattened_pos_error,
                    (Ts_world_joint, Ts_world_link, pos_error), # Cache for Jacobian
                )
                        
            factors = [
                pos_cost_analytical_jac_multi(
                    self.robot,
                    joint_var,
                    self.target_link_index, 
                    target_link_pos   
                ),
                pk.costs.limit_cost(self.robot, joint_var, weight=10.0),
            ]
            problem = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze()
            sol = problem.solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
                linear_solver="dense_cholesky",
                verbose=False,
                termination=jaxls.TerminationConfig(
                    max_iterations=64,
                    early_termination=False,
                ),
                trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0)
            )

            return sol[joint_var]
        
        return jax.vmap(solve_single)(initial_q, target_pos)

def main():

    args = argparse.ArgumentParser()
    args.add_argument("--config", required=True, type=str)
    args.add_argument("--out", required=True, type=str, default="tmp/out.npy")
    args = args.parse_args()

    cfg = np.load(args.config, allow_pickle=True).item()
    urdf_path = cfg["urdf"]
    target_links = cfg["target_links"]
    init_q = cfg["initial_q"]
    target_pos = cfg["target_pos"]

    solver = PyrokiRetarget(urdf_path, target_links)
    batch_retarget = jax.jit(solver.solve_retarget)
    time_1 = time.time()
    jax.block_until_ready(batch_retarget(
        initial_q=jnp.array(init_q),
        target_pos=jnp.array(target_pos)
    ))
    time_2 = time.time()
    print(time_2 - time_1)

    start_time = time.time()
    for i in tqdm.tqdm(range(20)):
        predict_q_pyroki=batch_retarget(
            initial_q=jnp.array(init_q),
            target_pos=jnp.array(target_pos)
        )
    jax.block_until_ready(predict_q_pyroki)
    end_time = time.time()
    print(f"Time: {(end_time - start_time)/20:.4f}")

if __name__ == '__main__':
    main()
