import tensorflow as tf
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import time
import joblib

matplotlib.use('Agg')

def log_prob_gaussian(x_var, std):
    return -1 / (2 * std) * (x_var) ** 2 - 1 / 2 * tf.math.log(2 * np.pi * std ** 2)


def get_phs(horizon, sufix=''):
    act = tf.compat.v1.placeholder(dtype=tf.float32, shape=(horizon,), name='act_'+sufix)
    obs = tf.compat.v1.placeholder(dtype=tf.float32, shape=(horizon,), name='obs_'+sufix)
    rew = tf.compat.v1.placeholder(dtype=tf.float32, shape=(horizon,), name='rew_'+sufix)
    return act, obs, rew


class FeaturePolicy(object):
    def __init__(self, feature, std=0.1):
        self.feature = feature
        self._params_ph = tf.compat.v1.placeholder(name="theta_ph", shape=(2,), dtype=tf.float32)

        self.params = tf.Variable(name="theta", initial_value=[0.09, 0.05],
                                      dtype=tf.float32)

        self._assign = tf.compat.v1.assign(self.params, self._params_ph)

        self.obs_var = obs_var = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,), name='input')

        self.mean_var = self.get_mean(obs_var)
        self.std = std

    def get_mean(self, obs_var, params=None):
        if params is None:
            return self.feature(self.params[0]) * obs_var ** 2 + self.params[1]
        else:
            return self.feature(params[0]) * obs_var ** 2 + params[1]

    def get_actions(self, observations):
        sess = tf.compat.v1.get_default_session()
        means = sess.run(self.mean_var, feed_dict={self.obs_var:observations})
        rnd = np.random.normal(size=len(observations))
        actions = np.clip(means + rnd * self.std, -2, 2)
        return actions

    def log_prob_sym(self, act, obs_var, params=None):
        return -1/ (2 * self.std) * (act - self.get_mean(obs_var, params)) ** 2 - 1/2 * tf.math.log(2 * np.pi * self.std ** 2)

    def get_params(self):
        sess = tf.compat.v1.get_default_session()
        return sess.run(self.params)

    def set_params(self, params):
        sess = tf.compat.v1.get_default_session()
        sess.run(self._assign, feed_dict={self._params_ph: params})


class Algo(object):

    def __init__(self, policy, horizon, type, init_state_std=1, inner_lr=0.1, lr=0.1, normalize=True, positive=False):
        """

        :param policy:
        :param horizon:
        :param num_traj: Refers to the number of trajectories per meta_task
        :param type:
        :param init_state_std:
        :param inner_lr:
        :param lr:
        """
        assert type in ['dice', 'exploration', 'robust']
        self.policy = policy
        self.horizon = horizon
        self.type = type
        self.init_state_std = init_state_std
        self.inner_lr = inner_lr
        self._lr = lr
        self.meta_tasks = 2
        self._adapted_params = None
        self._pre_policy = None
        self._normalize = normalize
        self._positive = positive
        self._build()

    def _build(self):
        act, obs, rew = get_phs(self.horizon)
        log_prob_traj = self.log_prob_traj(act, obs)
        self._grad_policy = tf.gradients(ys=log_prob_traj, xs=self.policy.params)[0]
        self._hessian = tf.hessians(ys=log_prob_traj, xs=self.policy.params)[0]
        self._inputs = [act, obs, rew]

    def log_prob_traj(self, act_var, obs_var, params=None):
        """
        This assumes that the obs_var has size (Samples, traj_length)
        :param act_var:
        :param obs_var:
        :return:
        """
        log_probs = self.policy.log_prob_sym(act_var, obs_var, params=params)
        log_prob_traj = tf.reduce_sum(input_tensor=log_probs)
        return log_prob_traj

    def switch_to_pre(self):
        self.policy.set_params(self._pre_policy)
        self._pre_policy = None

    def adapt(self, data, num_traj=None):
        gradients = []
        sess = tf.compat.v1.get_default_session()
        if self._pre_policy is None:
            self._pre_policy = self.policy.get_params()
        if num_traj is None:
            num_traj = data['act'].shape[0]

        for i in range(num_traj):
            inputs = [data['act'][i], data['obs'][i], data['rew'][i]]
            feed_dict = dict(zip(self._inputs, inputs))
            grad = sess.run(self._grad_policy, feed_dict=feed_dict)
            gradients.append(grad * np.sum(data['rew'][i]))
        grad = np.mean(gradients, axis=0)
        post_policy = self._pre_policy + self.inner_lr * grad
        self.policy.set_params(post_policy)
        return grad

    def compute_all_gradients(self, data, num_traj=None):
        # self.switch_to_pre()
        sess = tf.compat.v1.get_default_session()
        gradients_post = []
        ret_post = np.sum(data[1]['rew'], axis=-1)
        if self._normalize:
            ret_post = (ret_post - np.mean(ret_post))/np.var(ret_post)

        if self._positive:
            ret_post = (ret_post - np.min(ret_post))/(np.max(ret_post) - np.min(ret_post))


        if num_traj is None:
            num_traj = data[1]['act'].shape[0]

        for i in range(num_traj):
            inputs = [data[1]['act'][i], data[1]['obs'][i], data[1]['rew'][i]]
            feed_dict = dict(zip(self._inputs, inputs))
            grad = sess.run(self._grad_policy, feed_dict=feed_dict)
            gradients_post.append(grad)

        self.switch_to_pre()

        gradients_pre = []
        hessians = []
        ret_pre = np.sum(data[0]['rew'], axis=-1)
        if self._normalize:
            ret_pre = (ret_pre - np.mean(ret_pre))/np.var(ret_pre)

        if self._positive:
            ret_pre = (ret_pre - np.min(ret_pre))/(np.max(ret_pre) - np.min(ret_pre))

        if num_traj is None:
            num_traj = data[0]['act'].shape[0]

        for i in range(num_traj):
            inputs = [data[0]['act'][i], data[0]['obs'][i], data[0]['rew'][i]]
            feed_dict = dict(zip(self._inputs, inputs))
            grad, hess = sess.run([self._grad_policy, self._hessian], feed_dict=feed_dict)
            gradients_pre.append(grad)
            hessians.append(hess)

        opt_data = dict(grad_pre=np.array(gradients_pre),
                        grad_post=np.array(gradients_post),
                        hess=np.array(hessians),
                        ret_pre=np.array(ret_pre),
                        ret_post=np.array(ret_post)
                        )

        return opt_data

    def train(self, opts_data, opt_type):
        total_grad = []

        if opt_type == 'dice':
            opt_infos = dict(grad_J_post=[], grad_J_pre=[])
            for opt_data in opts_data:

                ret_pre = np.expand_dims(np.expand_dims(opt_data['ret_pre'], axis=-1), axis=-1)
                H = np.eye(2) + self.inner_lr * np.mean(opt_data['hess'] * ret_pre, axis=0)
                ret_post = np.expand_dims(opt_data['ret_post'], axis=-1)
                g_r_post = np.mean(opt_data['grad_post'] * ret_post, axis=0)
                grad_J_post = self.inner_lr * np.dot(H, g_r_post)


                n = opt_data['ret_pre'].shape[0]
                ret_pre = np.expand_dims(opt_data['ret_pre'], axis=-1)
                grad_J_pre = self.inner_lr * np.dot(opt_data['grad_pre'].T / np.sqrt(n),
                                              np.dot(opt_data['grad_pre'] / np.sqrt(n) * ret_pre,
                                              g_r_post))

                opt_infos['grad_J_pre'].append(grad_J_pre)
                opt_infos['grad_J_post'].append(grad_J_post)

                total_grad.append(grad_J_post + grad_J_pre)

        elif opt_type == 'exploration':
            opt_infos = dict(grad_J_post=[], grad_J_pre=[])
            for opt_data in opts_data:
                ret_pre = np.expand_dims(np.expand_dims(opt_data['ret_pre'], axis=-1), axis=-1)
                H = np.eye(2) + self.inner_lr * np.mean(opt_data['hess'] * ret_pre, axis=0)
                ret_post = np.expand_dims(opt_data['ret_post'], axis=-1)
                g_r_post = np.mean(opt_data['grad_post'] * ret_post, axis=0)
                grad_J_post = self.inner_lr * np.dot(H, g_r_post)

                grad_J_pre = self.inner_lr * np.sum(opt_data['grad_pre'], axis=0) * np.mean(ret_post)

                opt_infos['grad_J_pre'].append(grad_J_pre)
                opt_infos['grad_J_post'].append(grad_J_post)

                total_grad.append(grad_J_post + grad_J_pre)

        elif opt_type =='robust':
            opt_infos = dict(grad=[])
            for opt_data in opts_data:
                ret_pre = np.expand_dims(opt_data['ret_pre'], axis=-1)
                grad = np.mean(opt_data['grad_pre'] * ret_pre, axis=0)
                opt_infos['grad'].append(grad)

                total_grad.append(grad)

        else:
            raise NotImplementedError

        new_params = self.policy.get_params().copy() + self._lr * np.mean(total_grad, axis=0).copy()
        self.policy.set_params(new_params)
        return opt_infos



def collect_samples(policy, num_samples, traj_len, init_state_std=1, goal=None):
    data = dict(act=list(), obs=list(), adv=list(), rew=list())
    if goal is None:
        goal = np.random.choice([-1, 1])

    s = np.random.uniform(-2, 2, size=(num_samples,))

    for t in range(traj_len):
        a = policy.get_actions(s)
        r = np.exp(-0.1 * (a - goal) ** 2)
        data['obs'].append(s)
        data['act'].append(a)
        data['rew'].append(r)
        s = a.copy()

    data['obs'] = np.array(data['obs']).T
    data['act'] = np.array(data['act']).T
    data['rew'] = np.array(data['rew']).T
    return data


def plot_thetas(thetas, fig, ax, colors):
    x, y = list(zip(*thetas))
    ax.scatter(x, y, color=colors)


def run_1d_experiment(opt_type, num_meta_tasks, horizon, num_samples, num_itr, lr, inner_lr, init_state_std, exp_dir):
    feature = tf.identity

    policy = FeaturePolicy(feature, std=1)
    algo = Algo(policy, horizon, opt_type, inner_lr=inner_lr, lr=lr, normalize=False, positive=False)
    thetas = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print(policy.get_params())
        for itr in range(num_itr):
            opts_data = []
            _grads = []
            for g in [-1., 1] * num_meta_tasks:
                _data = {0: dict(), 1: dict()}
                data = collect_samples(policy, num_samples, horizon, goal=g, init_state_std=init_state_std)
                _data[0].update(copy.deepcopy(data))

                if opts_data != 'robust':
                    t0 = time.time()
                    grad = algo.adapt(data)
                    # print("Time adapt  :", time.time() - t0)
                    _grads.append(grad)
                    data = collect_samples(policy, num_samples, horizon, goal=g, init_state_std=init_state_std)
                    _data[1].update(copy.deepcopy(data))

                t0 = time.time()
                opt_data = algo.compute_all_gradients(_data)
                # print("Time compute gradients  :", time.time() - t0)
                opts_data.append(opt_data)

            t0 = time.time()
            algo.train(opts_data, opt_type=opt_type)
            # print("Time train  :", time.time() - t0)
            theta = policy.get_params()
            thetas.append(copy.copy(theta))
            print("Iteration %d" % itr, "   policy params: ", *policy.get_params())

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = matplotlib.cm.get_cmap('viridis')
    z = range(len(thetas))
    normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    colors = [cmap(normalize(value)) for value in z]
    plot_thetas(thetas, fig, ax, colors)
    ax.plot([0], [0], "*k")
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    fig.savefig(exp_dir + '/convergence_plot.png')
    joblib.dump(thetas, exp_dir + '/thetas.pkl')


