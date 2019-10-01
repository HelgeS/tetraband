import numpy as np


class AbstractAgent(object):
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        raise NotImplementedError()

    def update(self, rewards, done=True):
        raise NotImplementedError()


class DummyAgent(AbstractAgent):
    """Dummy agent that does nothing"""

    def act(self, observation):
        pass

    def update(self, rewards, done=True):
        pass


class RandomAgent(AbstractAgent):
    def act(self, observation):
        act_idx = self.env.action_space.sample()

        if self.env.is_hierarchical_action(action_idx=act_idx):
            parameter_space = self.env.hierarchical_actions[act_idx]['space']
            param_idx = parameter_space.sample()
        else:
            param_idx = None

        return act_idx, param_idx

    def update(self, rewards, done=True):
        pass


class BanditAgent(AbstractAgent):
    def __init__(self, env, feature_extractor, seed=None):
        super(BanditAgent, self).__init__(env)
        self.model = self._create_bandit(env.action_space.n, seed)
        self.feature_history = None
        self.feature_extractor = feature_extractor
        self.sub_bandits = {}

        for act_idx in range(0, self.env.action_space.n):
            if self.env.is_hierarchical_action(action_idx=act_idx):
                parameter_space = self.env.hierarchical_actions[act_idx]['space']
                self.sub_bandits[act_idx] = self._create_bandit(parameter_space.n)

    def _create_bandit(self, num_actions, seed=None):
        # --epsilon: Epsilon-Greedy exploration
        # --cover: Online Cover exploration
        # --nn N: use sigmoidal feedforward network w/ N hidden units
        from vowpalwabbit import pyvw
        cmd = "--nn 16 --epsilon 0.1 --cover 3 --cb_explore {}".format(num_actions)

        if seed:
            cmd += " --random_seed {}".format(seed)

        bandit = pyvw.vw(cmd, quiet=True)
        return bandit

    def act(self, observation):
        features = self.feature_extractor.extract(observation)
        feature_string = "| " + " ".join((str(i) + ":" + str(x) for i, x in enumerate(features, start=1)))

        action_probs = np.array(self.model.predict(feature_string))
        action_probs /= action_probs.sum()  # Normalize, otherwise np.random.choice might throw an error
        action = np.random.choice(range(len(action_probs)), size=1, p=action_probs)[0]

        if self.env.is_hierarchical_action(action_idx=action):
            parameter_probs = np.array(self.sub_bandits[action].predict(feature_string))
            parameter_probs /= parameter_probs.sum()  # Normalize, otherwise np.random.choice might throw an error
            parameter = np.random.choice(range(len(parameter_probs)), size=1, p=parameter_probs)[0]
            p_probs = parameter_probs[parameter]
        else:
            parameter = None
            p_probs = None

        self.feature_history = (feature_string, action, action_probs[action], parameter, p_probs)

        return action, parameter

    def update(self, rewards, done=True):
        action_reward, parameter_reward = rewards

        features, action, action_prob, param, param_prob = self.feature_history

        self._update_bandit(self.model, action, self.env.max_reward, action_reward, features, action_prob)

        if self.env.is_hierarchical_action(action):
            max_reward = self.env.hierarchical_actions[action]['max_reward']
            self._update_bandit(self.sub_bandits[action], param, max_reward, parameter_reward,
                                features, param_prob)

        self.feature_history = None

    def _update_bandit(self, bandit, action, max_reward, reward, features, prob):
        # prob must be included, even if it is static
        vwinput = "{action:d}:{cost:.2f}:{prob:.2f} {features}".format(action=action + 1,
                                                                       cost=max_reward - reward,
                                                                       features=features,
                                                                       prob=prob)
        bandit.learn(vwinput)
