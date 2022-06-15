# Code for _On the Convergence Theory of Debiased Model-Agnostic Meta-Reinforcement Learning_
- [NeurIPS 2021 version](https://proceedings.neurips.cc/paper/2021/hash/18085327b86002fc604c323b9a07f997-Abstract.html)
- [ArXiv version](https://arxiv.org/abs/2002.05135)

(a previous version of the manuscript was entitled _Provably Convergent Policy Gradient Methods for Model-Agnostic Meta-Reinforcement Learning_)

This codebase is based off of the [ProMP repository](https://github.com/jonasrothfuss/ProMP).

To run the setup from the numerical experiments section of our paper, please use `sgmrl_experiments/vpg_run.py`. Our modified implementation is located in `maml_zoo/meta_algos/vpg_sgmrl.py`.

Cite as
```
@article{fallah2021convergence,
  title={On the convergence theory of debiased model-agnostic meta-reinforcement learning},
  author={Fallah, Alireza and Georgiev, Kristian and Mokhtari, Aryan and Ozdaglar, Asuman},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={3096--3107},
  year={2021}
}
```
