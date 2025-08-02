import os
import jax
import time
import hydra
import wandb
import omegaconf
import traceback
import pickle
import socket
import numpy as np
from threading import Thread
import signal
import sys

from common.buffers import DMCCompatibleDictReplayBuffer
from diffusion.dime import DIME
from omegaconf import DictConfig
from models.utils import is_slurm_job
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from models.actor_critic_evaluation_callback import EvalCallback
import gymnasium as gym

class IPCEnvironment(gym.Env):
    """Wrapper environment that communicates with Metaworld via IPC"""
    
    def __init__(self, observation_space, action_space, host='localhost', port=12345):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the Metaworld environment process"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        max_retries = 30
        for i in range(max_retries):
            try:
                self.socket.connect((self.host, self.port))
                self.connected = True
                print(f"Connected to Metaworld environment at {self.host}:{self.port}")
                break
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    print(f"Connection attempt {i+1} failed, retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise Exception("Failed to connect to Metaworld environment")
    
    def _send_data(self, data):
        """Send data via socket"""
        serialized = pickle.dumps(data)
        data_size = len(serialized)
        self.socket.sendall(data_size.to_bytes(4, byteorder='big'))
        self.socket.sendall(serialized)
    
    def _receive_data(self):
        """Receive data via socket"""
        # First, receive the size of the data
        size_bytes = self.socket.recv(4)
        if len(size_bytes) < 4:
            raise ConnectionError("Failed to receive data size")
        data_size = int.from_bytes(size_bytes, byteorder='big')
        
        # Then receive the actual data
        received_data = b''
        while len(received_data) < data_size:
            chunk = self.socket.recv(data_size - len(received_data))
            if not chunk:
                raise ConnectionError("Connection lost while receiving data")
            received_data += chunk
        
        return pickle.loads(received_data)
    
    def reset(self, seed=None):
        """Reset the environment"""
        if not self.connected:
            self.connect()
        
        self._send_data({'action': 'reset', 'seed': seed})
        response = self._receive_data()
        
        if response['status'] == 'success':
            return response['observation'], response['info']
        else:
            raise Exception(f"Reset failed: {response['error']}")
    
    def step(self, action):
        """Take a step in the environment"""
        self._send_data({'action': 'step', 'data': action})
        response = self._receive_data()
        
        if response['status'] == 'success':
            return (response['observation'], response['reward'], 
                   response['terminated'], response['truncated'], response['info'])
        else:
            raise Exception(f"Step failed: {response['error']}")
    
    def close(self):
        """Close the environment connection"""
        if self.socket:
            try:
                self._send_data({'action': 'close'})
                self.socket.close()
            except:
                pass
        self.connected = False
    
    def render(self, mode='human'):
        """Render the environment (not implemented for IPC)"""
        # IPC environments don't support rendering directly
        # The rendering would need to be handled by the remote environment
        pass


def create_dummy_spaces_for_metaworld():
    """Create observation and action spaces matching typical Metaworld environments"""
    import gymnasium as gym
    
    # Typical Metaworld observation space (39-dimensional)
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
    )
    
    # Typical Metaworld action space (4-dimensional for robot arm)
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(4,), dtype=np.float32
    )
    
    return observation_space, action_space


def _create_alg(cfg: DictConfig):
    """Create DIME algorithm with IPC environment wrapper"""
    
    # Create dummy spaces for Metaworld
    obs_space, action_space = create_dummy_spaces_for_metaworld()
    
    # Create IPC environment wrapper
    training_env = IPCEnvironment(obs_space, action_space, port=12345)
    eval_env = IPCEnvironment(obs_space, action_space, port=12346)  # Different port for eval
    
    # Create log directories
    tensorboard_log_dir = f"./logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed={str(cfg.seed)}/"
    eval_log_dir = f"./eval_logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed={str(cfg.seed)}/eval/"
    
    # Create DIME model
    model = DIME(
        "MlpPolicy",  # Use MlpPolicy for vector observations
        env=training_env,
        model_save_path=None,
        save_every_n_steps=int(cfg.tot_time_steps / 100000),
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
        replay_buffer_class=None  # Use default replay buffer
    )
    
    # Create log dir where evaluation results will be saved
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=cfg.seed,
        best_model_save_path=None,
        log_path=eval_log_dir,
        eval_freq=max(300000 // cfg.log_freq, 1),
        n_eval_episodes=5, 
        deterministic=True, 
        render=False
    )
    
    if cfg.wandb["activate"]:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=0)])
    else:
        callback_list = CallbackList([eval_callback])
    
    return model, callback_list


def initialize_and_run(cfg):
    """Initialize and run DIME training"""
    cfg = hydra.utils.instantiate(cfg)
    seed = cfg.seed
    
    # Initialize wandb if activated
    if cfg.wandb["activate"]:
        name = f"seed_{seed}"
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"],
            group=cfg.wandb["group"],
            job_type=cfg.wandb["job_type"],
            name=name,
            config=wandb_config,
            entity=cfg.wandb["entity"],
            sync_tensorboard=True,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')
    
    # Create algorithm
    model, callback_list = _create_alg(cfg)
    
    # Start training
    print("Starting DIME training with IPC communication...")
    model.learn(total_timesteps=cfg.tot_time_steps, progress_bar=True, callback=callback_list)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nReceived shutdown signal, cleaning up...")
    sys.exit(0)


@hydra.main(version_base=None, config_path="configs", config_name="metaworld")
def main(cfg: DictConfig) -> None:
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        starting_time = time.time()
        print("Starting DIME algorithm process...")
        
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
        
        end_time = time.time()
        print(f"Training took: {(end_time - starting_time)/3600} hours")
        
        if cfg.wandb["activate"]:
            wandb.finish()
            
    except Exception as ex:
        print("-- Exception occurred. Traceback:")
        traceback.print_tb(ex.__traceback__)
        print(ex, flush=True)
        print("--------------------------------\n")
        traceback.print_exception(ex)
        
        if cfg.wandb["activate"]:
            wandb.finish()


if __name__ == "__main__":
    main()
