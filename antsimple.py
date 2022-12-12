import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Antsimple(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.max_episode_length = 200
        self.power_scale = 1.0

        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 2         # 2 for diffdrive, 8 for ant, how many for rocky? 4/5
        # self.cfg["env"]["numActions"] = 4
        # self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.gym.refresh_dof_state_tensor(self.sim)          
        self.gym.refresh_actor_root_state_tensor(self.sim)   

        self.initial_root_states = self.root_tensor.clone()
        self.initial_root_states[:, 7:13] = 0

        self.initial_dof_state = self.dof_state.clone()                             #scr

        print("self.initial_root_states")
        print(self.initial_root_states)

        print("self.initial_dof_state")
        print(self.initial_dof_state)

        self.dt = self.cfg["sim"]["dt"]

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        # asset_file = "mjcf/nv_ant.xml"
        # asset_file = "mjcf/diffdrivecp.xml"
        asset_file = "urdf/rocky.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.ant_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "diffdrivecp", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)

        self.num_dof = self.gym.get_asset_dof_count(ant_asset)


    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.initial_dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        if self.progress_buf[0] == 0:
            print("self.root_tensor    pre physics after reset")
            print(self.root_tensor)

        self.actions = actions.clone().to(self.device)
        forces = self.actions
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            print("self.root_tensor    post physics pre refresh")
            print(self.root_tensor)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        if len(env_ids) > 0:
            print("self.root_tensor    post physics post refresh")
            print(self.root_tensor) 

        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.root_tensor, 
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    root_states,
    reset,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # print("root_states")
    # print(root_states)

    height = root_states[::,2]
    # print("height")
    # print(height)

    # total_reward = torch.ones_like(reset_buf)
    reward = 1.0 - height
    # reward = height
    # reward = (height - 0.12)*10
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset), reset)

    return reward, reset

