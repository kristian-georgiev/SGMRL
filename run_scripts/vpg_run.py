import os
import json
import numpy as np
import pathmagic

with pathmagic.context():
    from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
    from maml_zoo.envs.point_envs.point_env_2d import MetaPointEnv
    from maml_zoo.envs.normalized_env import normalize
    from maml_zoo.meta_algos import VPGMAML
    from maml_zoo.meta_trainer import Trainer
    from maml_zoo.samplers.maml_sampler import MAMLSampler
    from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
    from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
    from maml_zoo.logger import logger


MAML_ZOO_PATH = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


def main(config):
    baseline = LinearFeatureBaseline()
    env = normalize(MetaPointEnv())

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

    algo = VPGMAML(
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
        num_inner_grad_steps=config['num_inner_grad_steps'],  # This is repeated in
        # MAMLPPO, it's confusing
    )
    trainer.train()


if __name__ == "__main__":
    idx = np.random.randint(0, 1000)
    logger.configure(dir=MAML_ZOO_PATH + '/data/vpg/test_%d' % idx,
                     format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(MAML_ZOO_PATH + "/configs/vpg_maml_config.json", 'r'))
    json.dump(config, open(MAML_ZOO_PATH + '/data/vpg/test_%d/params.json' % idx, 'w'))
    main(config)
