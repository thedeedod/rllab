from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
import time

def run_task(v):
    #env = TfEnv(normalize(CartpoleEnv()))
    env = TfEnv(normalize(SwimmerEnv()))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    to_plot = True

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        plot=to_plot
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    start_time = time.time()
    algo.train()
    net_time = time.time() - start_time
    print("\n--- TensorFlow [Plot = %s]: %s seconds ---" % (to_plot, net_time))

run_experiment_lite(
    run_task,
    exp_prefix="second_exp",
    n_parallel=4,
    plot=True
    )
