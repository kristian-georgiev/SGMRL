import unittest
from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np
import tensorflow as tf
import pickle
import gym

class DummySpace(object):
    def __init__(self, dim):
        self._dim = dim

    @property
    def shape(self):
        return self._dim

class DummyEnv(object):
    def __init__(self, obs_dim, act_dim):
        self._observation_space = gym.spaces.Box(low=-np.ones(obs_dim), high=np.ones(obs_dim), dtype=np.float32)
        self._action_space = gym.spaces.Box(low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_obs(self, n=None):
        if n is None:
            return np.random.uniform(0, 1, size=self.observation_space.shape)
        else:
            return np.random.uniform(0, 1, size=(n,) + self.observation_space.shape)


class TestPolicy(unittest.TestCase):

    def setUp(self):
        sess = tf.compat.v1.get_default_session()
        if sess is None:
            tf.compat.v1.InteractiveSession()

    def test_output_sym(self):
        with tf.compat.v1.Session() as sess:
            obs_dim = 23
            action_dim = 7
            self.env = DummyEnv(obs_dim, action_dim)
            self.policy = GaussianMLPPolicy(obs_dim,
                                            action_dim,
                                            name='test_policy_output_sym',
                                            hidden_sizes=(64, 64))

            obs_ph_1 = tf.compat.v1.placeholder(dtype=tf.float32, name="obs_ph_1",
                                       shape=(None,) +  self.env.observation_space.shape)
            output_sym_1 = self.policy.distribution_info_sym(obs_ph_1)

            sess.run(tf.compat.v1.global_variables_initializer())

            n_obs = self.env.get_obs(n=100)
            action, agent_infos = self.policy.get_actions(n_obs)
            agent_infos_output_sym = sess.run(output_sym_1, feed_dict={obs_ph_1: n_obs})

            for k in agent_infos.keys():
                self.assertTrue(np.allclose(agent_infos[k], agent_infos_output_sym[k], rtol=1e-5, atol=1e-5))

    def test_get_action(self):

        with tf.compat.v1.Session() as sess:
            obs_dim = 23
            action_dim = 7
            self.env = DummyEnv(obs_dim, action_dim)
            self.policy = GaussianMLPPolicy(obs_dim,
                                            action_dim,
                                            name='test_policy_get_action',
                                            hidden_sizes=(64, 64))

            sess.run(tf.compat.v1.global_variables_initializer())

            obs = self.env.get_obs()
            action, agent_infos = self.policy.get_action(obs)
            actions, agents_infos = self.policy.get_actions(np.expand_dims(obs, 0))
            for k in agent_infos.keys():
                self.assertTrue(np.allclose(agent_infos[k], agents_infos[k], rtol=1e-5, atol=1e-5))

    def testSerialize1(self):
        obs_dim = 23
        action_dim = 7
        self.env = DummyEnv(obs_dim, action_dim)
        self.policy = GaussianMLPPolicy(obs_dim,
                                        action_dim,
                                        name='test_policy_serialize',
                                        hidden_sizes=(64, 64))

        sess = tf.compat.v1.get_default_session()
        sess.run(tf.compat.v1.global_variables_initializer())
        all_param_values = self.policy.get_param_values()

        self.policy.set_params(all_param_values)

    def testSerialize2(self):
        obs_dim = 2
        action_dim = 7
        env = DummyEnv(obs_dim, action_dim)
        policy = GaussianMLPPolicy(obs_dim,
                                        action_dim,
                                        name='test_policy_serialize2',
                                        hidden_sizes=(54, 23))

        sess = tf.compat.v1.get_default_session()
        sess.run(tf.compat.v1.global_variables_initializer())

        obs = env.get_obs()
        _, pre_agent_infos = policy.get_action(obs)
        pkl_str = pickle.dumps(policy)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            policy_unpickled = pickle.loads(pkl_str)
            _, post_agent_infos = policy_unpickled.get_action(obs)
            for key in pre_agent_infos.keys():
                self.assertTrue(np.allclose(pre_agent_infos[key], post_agent_infos[key]))


if __name__ == '__main__':
    unittest.main()
