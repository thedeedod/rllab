from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
import time

def run_task(v):
    env = normalize(SwimmerEnv())

    policy = GaussianMLPPolicy(
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
        max_path_length=500,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        plot=to_plot
    )
    start_time = time.time()
    algo.train()
    net_time = time.time() - start_time
    print("\n--- Theano [Plot = %s]: %s seconds ---" % (to_plot, net_time))


run_experiment_lite(
    run_task,
    exp_prefix="first_exp",
    n_parallel=4,
    plot=True
    )
