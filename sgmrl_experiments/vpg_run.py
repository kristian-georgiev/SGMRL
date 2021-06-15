import os
import json
from datetime import datetime
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import joblib
import pathmagic

with pathmagic.context():
    from maml_zoo.utils.utils import set_seed
    from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
    from maml_zoo.envs.point_envs.point_env_2d_v2 import MetaPointEnv
    from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
    from maml_zoo.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
    from maml_zoo.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
    from maml_zoo.envs.normalized_env import normalize
    from maml_zoo.meta_algos import VPGSGMRL, VPGMAML
    from maml_zoo.meta_trainer import Trainer
    from maml_zoo.meta_tester import Tester
    from maml_zoo.samplers.maml_sampler import MAMLSampler
    from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
    from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
    from maml_zoo.logger import logger


MAML_ZOO_PATH = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])
ENV_DICT = {'point': MetaPointEnv,
            'cheetah_dir': HalfCheetahRandDirecEnv,
            }

parser = ArgumentParser()
parser.add_argument('--algo', choices=['sgmrl', 'maml'], default='sgmrl')
parser.add_argument('--env', choices=['point', 'cheetah_dir'], default='point')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--n_adapt_steps', type=int, default=1)
parser.add_argument('--dir')
parser.add_argument('--config',
                    default='/sgmrl_configs/sgmrl_full.json')


def main(config):
    set_seed(config['seed'])
    tf.compat.v1.disable_eager_execution()
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_id in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

    baseline = LinearFeatureBaseline()
    env = normalize(ENV_DICT[config['env']]())

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
    )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    Algo = VPGSGMRL if args.algo == 'sgmrl' else VPGMAML
    algo = Algo(
        policy=policy,
        inner_type=config['inner_type'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        inner_lr=config['inner_lr'],
        learning_rate=config['learning_rate']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'])

    tester = Tester(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=80,
        num_inner_grad_steps=config['num_inner_grad_steps'])

    best_itr = trainer.train(tester)
    print(best_itr)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dir is not None:
        path = MAML_ZOO_PATH + '/data/' + args.dir
    else:
        now = str(datetime.now().date()) + '_' + str(datetime.now().time())[:5]
        path = MAML_ZOO_PATH + f'/data/{args.env}_{args.algo}_seed_{args.seed}_n_adapt_steps_{str(args.n_adapt_steps)}_{now}'
    logger.configure(dir=path,
                     format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(MAML_ZOO_PATH + args.config, 'r'))
    config['dir'] = path
    config['env'] = args.env
    config['seed'] = args.seed
    config['num_inner_grad_steps'] = args.n_adapt_steps
    json.dump(config, open(path + '/params.json', 'w'))
    main(config)
