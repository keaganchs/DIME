import socket
import pickle
import numpy as np
import signal
import sys
import time
import threading
from typing import Optional, Dict, Any

import gymnasium as gym
import metaworld

# Import Metaworld
# try:
#     import gymnasium as gym
#     import metaworld
#     METAWORLD_AVAILABLE = True
# except ImportError:
#     METAWORLD_AVAILABLE = False
#     print("Warning: Metaworld not available. Install with: pip install metaworld")


class MetaworldServer:
    """Server that hosts Metaworld environments and communicates via IPC"""
    
    def __init__(self, env_name: str = 'reach-v3', host: str = 'localhost', 
                 train_port: int = 12345, eval_port: int = 12346, seed: int = 0):
        self.env_name = env_name
        self.host = host
        self.train_port = train_port
        self.eval_port = eval_port
        self.seed = seed
        
        # Environment instances
        self.train_env = None
        self.eval_env = None
        
        # Server sockets
        self.train_server = None
        self.eval_server = None
        
        # Client connections
        self.train_client = None
        self.eval_client = None
        
        # Running flags
        self.running = True
        self.train_thread = None
        self.eval_thread = None
        
        # Initialize environments
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize Metaworld environments"""
        # if not METAWORLD_AVAILABLE:
            # raise ImportError("Metaworld is not available. Please install it in this environment.")
        
        print(f"Initializing Metaworld environment: {self.env_name}")
        
        # Create ML1 environment (single task)
        ml1 = metaworld.ML1(self.env_name, seed=self.seed)
        
        # Training environment
        self.train_env = ml1.train_classes[self.env_name]()
        train_task = ml1.train_tasks[0]
        self.train_env.set_task(train_task)
        
        # Evaluation environment (using test tasks if available, otherwise train tasks)
        self.eval_env = ml1.train_classes[self.env_name]()
        eval_task = ml1.test_tasks[0] if len(ml1.test_tasks) > 0 else ml1.train_tasks[0]
        self.eval_env.set_task(eval_task)
        
        print(f"Environment initialized:")
        print(f"  Observation space: {self.train_env.observation_space}")
        print(f"  Action space: {self.train_env.action_space}")
    
    def _create_server_socket(self, port: int) -> socket.socket:
        """Create and configure server socket"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, port))
        server_socket.listen(1)
        return server_socket
    
    def _send_data(self, client_socket: socket.socket, data: Dict[str, Any]):
        """Send data via socket"""
        try:
            serialized = pickle.dumps(data)
            data_size = len(serialized)
            client_socket.sendall(data_size.to_bytes(4, byteorder='big'))
            client_socket.sendall(serialized)
        except Exception as e:
            print(f"Error sending data: {e}")
            raise
    
    def _receive_data(self, client_socket: socket.socket) -> Dict[str, Any]:
        """Receive data via socket"""
        try:
            # First, receive the size of the data
            size_bytes = client_socket.recv(4)
            if len(size_bytes) == 0:
                raise ConnectionError("Client disconnected")
            if len(size_bytes) < 4:
                raise ConnectionError("Failed to receive data size")
            data_size = int.from_bytes(size_bytes, byteorder='big')
            
            # Then receive the actual data
            received_data = b''
            while len(received_data) < data_size:
                chunk = client_socket.recv(data_size - len(received_data))
                if not chunk:
                    raise ConnectionError("Connection lost while receiving data")
                received_data += chunk
            
            return pickle.loads(received_data)
        except Exception as e:
            print(f"Error receiving data: {e}")
            raise
    
    def _handle_client(self, client_socket: socket.socket, env, env_type: str):
        """Handle client requests for a specific environment"""
        print(f"{env_type} client connected")
        
        try:
            while self.running:
                # Receive request
                try:
                    request = self._receive_data(client_socket)
                except (ConnectionError, EOFError, ConnectionResetError):
                    print(f"{env_type} client disconnected during communication")
                    break
                
                action_type = request.get('action')
                
                if action_type == 'reset':
                    # Reset environment
                    seed = request.get('seed')
                    if seed is not None:
                        observation, info = env.reset(seed=seed)
                    else:
                        observation, info = env.reset()
                    
                    response = {
                        'status': 'success',
                        'observation': observation.astype(np.float32),
                        'info': info
                    }
                    
                elif action_type == 'step':
                    # Take step in environment
                    action = np.array(request.get('data'), dtype=np.float32)
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    response = {
                        'status': 'success',
                        'observation': observation.astype(np.float32),
                        'reward': float(reward),
                        'terminated': bool(terminated),
                        'truncated': bool(truncated),
                        'info': info
                    }
                    
                elif action_type == 'close':
                    # Close environment
                    print(f"{env_type} client requested close")
                    break
                    
                else:
                    response = {
                        'status': 'error',
                        'error': f'Unknown action: {action_type}'
                    }
                
                # Send response
                self._send_data(client_socket, response)
                
        except Exception as e:
            print(f"Error handling {env_type} client: {e}")
        finally:
            client_socket.close()
            print(f"{env_type} client disconnected")
    
    def _server_thread(self, server_socket: socket.socket, env, env_type: str):
        """Server thread for handling connections"""
        print(f"Starting {env_type} server on {self.host}:{server_socket.getsockname()[1]}")
        
        try:
            while self.running:
                server_socket.settimeout(1.0)  # Set timeout to check running flag
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"{env_type} connection from {addr}")
                    self._handle_client(client_socket, env, env_type)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error in {env_type} server: {e}")
                        import traceback
                        traceback.print_exc()
        finally:
            server_socket.close()
            print(f"{env_type} server stopped")
    
    def start(self):
        """Start the Metaworld server"""
        print("Starting Metaworld server...")
        
        # Create server sockets
        self.train_server = self._create_server_socket(self.train_port)
        self.eval_server = self._create_server_socket(self.eval_port)
        
        # Start server threads
        self.train_thread = threading.Thread(
            target=self._server_thread,
            args=(self.train_server, self.train_env, "Training"),
            daemon=True
        )
        
        self.eval_thread = threading.Thread(
            target=self._server_thread,
            args=(self.eval_server, self.eval_env, "Evaluation"),
            daemon=True
        )
        
        self.train_thread.start()
        self.eval_thread.start()
        
        print(f"Metaworld server started:")
        print(f"  Training port: {self.train_port}")
        print(f"  Evaluation port: {self.eval_port}")
        print("Waiting for connections...")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server"""
        print("Stopping Metaworld server...")
        self.running = False
        
        # Close server sockets
        if self.train_server:
            self.train_server.close()
        if self.eval_server:
            self.eval_server.close()
        
        # Wait for threads to finish
        if self.train_thread and self.train_thread.is_alive():
            self.train_thread.join(timeout=2)
        if self.eval_thread and self.eval_thread.is_alive():
            self.eval_thread.join(timeout=2)
        
        print("Metaworld server stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nReceived shutdown signal, stopping server...")
    sys.exit(0)


def main():
    """Main function to run the Metaworld server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Metaworld Environment Server')
    parser.add_argument('--env', type=str, default='reach-v3',
                       help='Metaworld environment name (default: reach-v3)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host (default: localhost)')
    parser.add_argument('--train-port', type=int, default=12345,
                       help='Training environment port (default: 12345)')
    parser.add_argument('--eval-port', type=int, default=12346,
                       help='Evaluation environment port (default: 12346)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Environment seed (default: 0)')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = MetaworldServer(
        env_name=args.env,
        host=args.host,
        train_port=args.train_port,
        eval_port=args.eval_port,
        seed=args.seed
    )
    
    try:
        server.start()
    except Exception as e:
        print(f"Server error: {e}")
        server.stop()


if __name__ == "__main__":
    main()
