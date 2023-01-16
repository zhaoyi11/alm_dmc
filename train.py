import wandb
import time
import random
import hydra
import warnings
warnings.simplefilter("ignore", UserWarning)

from omegaconf import DictConfig

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg: DictConfig):
    if cfg.seed == -1 : cfg.seed = random.randint(0, 1000)
    if cfg.benchmark == 'dmc':
        from workspaces.mujoco_workspace import MujocoWorkspace as W
    else:
        raise NotImplementedError

    domain, task = str(cfg.id).replace('-', '_').split('_', 1)
    if domain in ['dog', 'humanoid']:
        # use 100 latent dim for hard tasks
        cfg.latent_dims = 100
        cfg.expl_duration = 500 # episodes
        cfg.num_train_steps = 5000 * (1000 // cfg.action_repeat) # 5000 episodes
    
    cfg.expl_duration *= 1000 / cfg.action_repeat / cfg.update_every_steps # episodes to steps

    if cfg.wandb_log:
        with wandb.init(project="alm", name=f'{cfg.id}-{str(cfg.seed)}-{int(time.time())}',
                                    group=f'{cfg.id}', 
                                    config=cfg,
                                    monitor_gym=True):
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
    
if __name__ == '__main__':
    main()