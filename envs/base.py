import gym
import numpy as np
from functools import partial
from envs import mr


class BaseEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, scenario, evaluation='difference', random_images=True):
        self.indices = []
        self.cur_image_idx = None
        self.random_images = random_images
        self.scenario = scenario

        if scenario == 'basic':
            self.actions = self._generate_basic_actions()
        elif scenario == 'rotation':
            self.actions = self._generate_rotation_actions()
        elif scenario == 'hierarchical':
            self.actions = self._generate_hierarchical_actions()
        elif scenario == 'shear':
            self.actions = self._generate_shear_actions()
        else:
            raise NotImplementedError("Unknown scenario '{}'".format(scenario))

        self.max_reward = max((a[2] for a in self.actions))
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.hierarchical_actions = {}

        if scenario == 'hierarchical':
            self.hierarchical_actions[self.action_space.n - 2] = {
                'space': gym.spaces.Discrete(len(self.actions[self.action_space.n - 2][1])),
                'max_reward': max((a[2] for a in self.actions[self.action_space.n - 2][1]))
            }
            self.hierarchical_actions[self.action_space.n - 1] = {
                'space': gym.spaces.Discrete(len(self.actions[self.action_space.n - 1][1])),
                'max_reward': max((a[2] for a in self.actions[self.action_space.n - 1][1]))
            }

        self.evaluation_rules = evaluation

        if evaluation not in ('difference', 'groundtruth'):
            raise Exception("Unknown evaluation '{}'".format(evaluation))

    def reset(self):
        # Sample new image from test set
        if len(self.indices) == 0:
            self.indices = self._initialize_indices()

        self.cur_image_idx = self.indices.pop(0)
        image, _ = self._get_image(self.cur_image_idx)

        return self.pre_transformation(image)

    def _initialize_indices(self):
        indices = list(range(0, len(self.dataset)))

        if self.random_images:
            np.random.shuffle(indices)

        return indices

    def _get_image(self, idx):
        raise NotImplementedError

    @staticmethod
    def _generate_basic_actions():
        r = 1  # Reward for the agent if the transformation leads to a change in the output
        actions = [
            ('blur', mr.blur, r),
            ("fliplr", mr.fliplr, r),
            ("flipud", mr.flipud, r),
            ("grayscale", mr.grayscale, r),
            ("invert", mr.invert, r),
            ("rot-30", partial(mr.rotate, degrees=-30), r),
            ("rot30", partial(mr.rotate, degrees=30), r),
            ("shear-20", partial(mr.shear, degrees=-20), r),
            ("shear20", partial(mr.shear, degrees=20), r),
        ]

        return actions

    @staticmethod
    def _generate_rotation_actions(max_rotation=90, step_size=5):
        actions = []

        degrees = [(deg, 10000.0 / (2 ** i)) for i, deg in enumerate(range(step_size, max_rotation + 1, step_size))]
        degrees += [(-deg, r) for deg, r in degrees]
        degrees.sort(key=lambda x: x[0])  # Cosmetic for nicer debugging

        for deg, r in degrees:
            if deg == 0:
                continue

            f = partial(mr.rotate, degrees=deg)
            actions.append(("rot{:d}".format(deg), f, r))

        return actions

    @staticmethod
    def _generate_shear_actions(max_shear=45, step_size=5):
        actions = []

        degrees = [(deg, 10000.0 / (2 ** i)) for i, deg in enumerate(range(step_size, max_shear + 1, step_size))]
        degrees += [(-deg, r) for deg, r in degrees]
        degrees.sort(key=lambda x: x[0])  # Cosmetic for nicer debugging

        for deg, r in degrees:
            if deg == 0:
                continue

            f = partial(mr.shear, degrees=deg)
            actions.append(("shear{:d}".format(deg), f, r))

        return actions

    def _generate_hierarchical_actions(self, max_rotation=90, max_shear=45, step_size=5):
        basic_actions = [a for a in self._generate_basic_actions() if
                         not (a[0].startswith('rot') or a[0].startswith('shear'))]
        rot_actions = self._generate_rotation_actions(max_rotation=max_rotation, step_size=step_size)
        shear_actions = self._generate_shear_actions(max_shear=max_shear, step_size=step_size)
        actions = basic_actions + [("rotation", rot_actions, 1), ("shear", shear_actions, 1)]
        return actions

    def is_hierarchical_action(self, action_idx):
        return action_idx in self.hierarchical_actions

    def get_action(self, action_idx, parameter_idx=None):
        if self.is_hierarchical_action(action_idx):
            if parameter_idx is None:
                raise Exception("Hierarchical action {:d} requires an additional parameter argument".format(action_idx))

            return self.actions[action_idx][1][parameter_idx][1]
        else:
            return self.actions[action_idx][1]

    def get_action_name(self, action_idx, parameter_idx=None):
        action_name = self.actions[action_idx][0]

        if self.is_hierarchical_action(action_idx):
            param_name = self.actions[action_idx][1][parameter_idx][0]
        else:
            param_name = None

        return action_name, param_name

    def render(self, mode='human', close=False):
        pass

    def step(self, action):
        pass

    def action_names(self):
        return [a[0] for a in self.actions]
