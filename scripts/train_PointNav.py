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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False, 
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
        help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(ap.parse_args())
    config_name = args['file']

    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)

        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']

        if args['seed'] > 0:
            config['params']['seed'] = args['seed']
            config['params']['config']['env_config']['seed'] = args['seed']

        from rl_games.torch_runner import Runner

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.init(object_store_memory=1024*1024*1000)


        rl_device = 'cuda:0'
        env = EnvPointNavVer0_1()
        wrapped_env = RlGamesVecEnvWrapper(env, rl_device)
        vecenv.register(
            "IsaacRlgWrapper", 
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register(
            "rlgpu", 
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})
        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    global_rank = int(os.getenv("RANK", "0"))
    if args["track"] and global_rank == 0:
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

    try:
        import ray
    except ImportError:
        pass
    else:
        ray.shutdown()

    if args["track"] and global_rank == 0:
        wandb.finish()
