from distutils.util import strtobool
import argparse, os, yaml
import torch
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from envs import *
from model import *

# Disable cuDNN multi-threading and non-determinism
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

"""
python train_encoder_PointNav.py --train --file ../cfg/point_nav_encoder_ppo.yaml 
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False,
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-t", "--train", action='store_true', help="train network")
    ap.add_argument("-p", "--play", action='store_true', help="play(test) network")
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False,
                    help="new sigma value if fixed_sigma: True in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), nargs='?', const=True, default=False,
                    help="track experiment with Weights & Biases")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
                    help="wandb project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
                    help="wandb entity/team name")

    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(ap.parse_args())
    config_name = args['file']

    print('Loading config:', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']
        if args['seed'] > 0:
            config['params']['seed'] = args['seed']
            config['params']['config']['env_config']['seed'] = args['seed']

        # Environment and wrapper setup
        rl_device = 'cuda:0'
        env = EnvPointNavVer0_2()
        wrapped_env = RlGamesVecEnvWrapper(env, rl_device)
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env}
        )

        # Runner and custom agent registration
        runner = Runner()
        runner.algo_factory.register_builder(
            'encoder_a2c_discrete',
            lambda **kwargs: EncoderDiscreteA2CAgent(**kwargs)
        )

        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialize wandb if requested
    if args["track"]:
        import wandb
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    if args["play"]:
        # 在测试模式下启用录制
        wrapped_env.enable_render()

    runner.run(args)

    if args["play"]:
        # 测试结束后保存录像
        wrapped_env.stop_and_save(path="outputs/plays/gif", filename="test.gif")


    if args["track"]:
        import wandb
        wandb.finish()