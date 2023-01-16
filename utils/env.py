# import gym
# from gym.envs.mujoco import mujoco_env
# from gym import utils
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
    return end_sigma + (1 - min(t / duration, 1)) * (start_sigma - end_sigma)

# saving frames 

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale
from dm_env import StepType, specs
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)
    
    def _transform_reward(self, time_step):
        assert len(self._rewards) <= self._num_frames
        while len(self._rewards) < self._num_frames:
            self._rewards.append(time_step.reward)

        r = (np.array(list(self._rewards)) * self._reward_weight).sum() # weigheted sum over stacking rewards
        return time_step._replace(reward=r)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            # self._rewards.append(0)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        time_step = self._transform_observation(time_step)
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                            dtype,
                                            wrapped_action_spec.minimum,
                                            wrapped_action_spec.maximum,
                                            'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class TimeStepToGymWrapper(object):
    def __init__(self, env, domain, task, action_repeat, modality):
        try: # pixels
            obs_shp = env.observation_spec().shape
            assert modality == 'pixels'
        except: # state
            obs_shp = []
            for v in env.observation_spec().values():
                try:
                    shp = 1
                    _shp = v.shape
                    for s in _shp:
                        shp *= s
                except:
                    shp = 1
                obs_shp.append(shp)
            obs_shp = (np.sum(obs_shp),)
            assert modality != 'pixels'
        act_shp = env.action_spec().shape
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_shp, -np.inf if modality != 'pixels' else env.observation_spec().minimum),
            high=np.full(obs_shp, np.inf if modality != 'pixels' else env.observation_spec().maximum),
            shape=obs_shp,
            dtype=np.float32 if modality != 'pixels' else np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.full(act_shp, env.action_spec().minimum),
            high=np.full(act_shp, env.action_spec().maximum),
            shape=act_shp,
            dtype=env.action_spec().dtype)
        self.env = env
        self.domain = domain
        self.task = task
        self.ep_len = 1000//action_repeat
        self.modality = modality
        self.t = 0
    
    @property
    def unwrapped(self):
        return self.env

    @property
    def reward_range(self):
        return None

    @property
    def metadata(self):
        return None
    
    def _obs_to_array(self, obs):
        if self.modality != 'pixels':
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs

    def reset(self):
        self.t = 0
        return self._obs_to_array(self.env.reset().observation)
    
    def step(self, action):
        self.t += 1
        time_step = self.env.step(action)
        return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)

    def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
        camera_id = dict(quadruped=2).get(self.domain, camera_id)
        return self.env.physics.render(height, width, camera_id)


def make_env(env_name, seed, action_repeat):
    """
    Make environment for TD-MPC experiments.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = str(env_name).replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', humanoidCMU='humanoid_CMU').get(domain, domain)

    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                        task,
                        task_kwargs={'random': seed},
                        visualize_reward=False)
    else:
        raise ValueError
    
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(env, domain, task, action_repeat, 'state')

    return env
    



# # MBPO environments

# MBPO_ENVIRONMENT_SPECS = (
# 	{
#         'id': 'AntTruncatedObs-v2',
#         'entry_point': (f'utils.env:AntTruncatedObsEnv'),
#         'max_episode_steps': 1000,
#     },
# 	{
#         'id': 'HumanoidTruncatedObs-v2',
#         'entry_point': (f'utils.env:HumanoidTruncatedObsEnv'),
#         'max_episode_steps': 1000,
#     },
# )

# def _register_environments(register, specs):
#     for env in specs:
#         register(**env)

#     gym_ids = tuple(environment_spec['id'] for environment_spec in specs)
#     return gym_ids

# def register_mbpo_environments():
#     _register_environments(gym.register, MBPO_ENVIRONMENT_SPECS)

# def mass_center(model, sim):
#     mass = np.expand_dims(model.body_mass, 1)
#     xpos = sim.data.xipos
#     return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

# class HumanoidTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     """
#         COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator), 
#         and external forces (cfrc_ext) are removed from the observation.
#         Otherwise identical to Humanoid-v2 from
#         https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
#     """
#     def __init__(self):
#         mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
#         utils.EzPickle.__init__(self)

#     def _get_obs(self):
#         data = self.sim.data
#         return np.concatenate([data.qpos.flat[2:],
#                                data.qvel.flat,
#                                # data.cinert.flat,
#                                # data.cvel.flat,
#                                # data.qfrc_actuator.flat,
#                                # data.cfrc_ext.flat
#                                ])

#     def step(self, a):
#         pos_before = mass_center(self.model, self.sim)
#         self.do_simulation(a, self.frame_skip)
#         pos_after = mass_center(self.model, self.sim)
#         alive_bonus = 5.0
#         data = self.sim.data
#         lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
#         quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
#         quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
#         quad_impact_cost = min(quad_impact_cost, 10)
#         reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
#         qpos = self.sim.data.qpos
#         done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
#         return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

#     def reset_model(self):
#         c = 0.01
#         self.set_state(
#             self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
#             self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
#         )
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 1
#         self.viewer.cam.distance = self.model.stat.extent * 1.0
#         self.viewer.cam.lookat[2] = 2.0
#         self.viewer.cam.elevation = -20

# class AntTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     """
#         External forces (sim.data.cfrc_ext) are removed from the observation.
#         Otherwise identical to Ant-v2 from
#         https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
#     """
#     def __init__(self):
#         mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
#         utils.EzPickle.__init__(self)

#     def step(self, a):
#         xposbefore = self.get_body_com("torso")[0]
#         self.do_simulation(a, self.frame_skip)
#         xposafter = self.get_body_com("torso")[0]
#         forward_reward = (xposafter - xposbefore)/self.dt
#         ctrl_cost = .5 * np.square(a).sum()
#         contact_cost = 0.5 * 1e-3 * np.sum(
#             np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
#         survive_reward = 1.0
#         reward = forward_reward - ctrl_cost - contact_cost + survive_reward
#         state = self.state_vector()
#         notdone = np.isfinite(state).all() \
#             and state[2] >= 0.2 and state[2] <= 1.0
#         done = not notdone
#         ob = self._get_obs()
#         return ob, reward, done, dict(
#             reward_forward=forward_reward,
#             reward_ctrl=-ctrl_cost,
#             reward_contact=-contact_cost,
#             reward_survive=survive_reward)

#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[2:],
#             self.sim.data.qvel.flat,
#             # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.5

# class GymActionRepeatWrapper(gym.Wrapper):
# 	def __init__(self, env, num_repeats):
# 		assert '-v4' in env.unwrapped.spec.id
# 		super().__init__(env)
# 		self._env = env
# 		self._num_repeats = num_repeats

# 	def step(self, action):
# 		reward = 0.0
# 		notdone = True
# 		for i in range(self._num_repeats):
# 			state, rew, done, info = self._env.step(action)
# 			notdone = not done
# 			reward += (rew) * (notdone)
# 			notdone *= notdone
# 			if done:
# 				break
# 		return state, reward, done,  info