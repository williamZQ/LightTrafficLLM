
import argparse
import time
import os


class TrainingConfig:
    
    def __init__(self):
        self.device = "Your Cuda ID"
        self.dataset_name = "Your Dataset Path Name"
        self.input_dim = 1
        self.channels = 64
        self.num_nodes = 157
        self.input_length = 12
        self.output_length = 12
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.llm_layers = 3
        self.unfrozen_layers = 2
        self.epochs = 1
        self.print_interval = 50
        self.lora_rank = 8
        self.llm_model_path = "Your LLM Model Path"
        self.weight_decay = 0.0001
        self.save_path = f"./logs/{time.strftime('%Y-%m-%d-%H:%M:%S')}-"
        self.early_stopping_patience = 100
        self.gradient_clip = 5
        self.adj_matrix_path = None  
        
    def _setup_dataset_specific_config(self):
        dataset_configs = {
            "bike_drop": {"num_nodes": 250, "data_path": "./data/bike_drop", "adj_matrix_path": "./data/bike_drop/adj_mx.pkl"},
            "bike_pick": {"num_nodes": 250, "data_path": "./data/bike_pick", "adj_matrix_path": "./data/bike_pick/adj_mx.pkl"},
            "taxi_drop": {"num_nodes": 266, "data_path": "./data/taxi_drop", "adj_matrix_path": "./data/taxi_drop/adj_mx.pkl"},
            "taxi_pick": {"num_nodes": 266, "data_path": "./data/taxi_pick", "adj_matrix_path": "./data/taxi_pick/adj_mx.pkl"},
            "pems_157": {"num_nodes": 157, "data_path": "./data/pems_157", "adj_matrix_path": "./data/pems_157/adj_mx.npz"}
        }
        
        if self.dataset_name in dataset_configs:
            config = dataset_configs[self.dataset_name]
            self.num_nodes = config["num_nodes"]
            self.data_path = config["data_path"]
            self.adj_matrix_path = config["adj_matrix_path"]
        else:
            self.data_path = f"./data/{self.dataset_name}"
            adj_files = ["adj_mx.npz", "adj_mx.pkl", "adj_binary.xlsx"]
            for adj_file in adj_files:
                adj_path = os.path.join(self.data_path, adj_file)
                if os.path.exists(adj_path):
                    self.adj_matrix_path = adj_path
                    break
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="LightTrafficLLM Training Configuration")
        
        parser.add_argument("--device", type=str, default=self.device, 
                          help="Training device (cuda:0, cuda:1, cpu)")

        parser.add_argument("--data", type=str, default=self.dataset_name, 
                          help="Dataset name")
        parser.add_argument("--num_nodes", type=int, default=self.num_nodes, 
                          help="Number of nodes")
        parser.add_argument("--adj_matrix_path", type=str, default=None,
                          help="Adjacency matrix file path")

        parser.add_argument("--input_dim", type=int, default=self.input_dim, 
                          help="Input dimension")
        parser.add_argument("--channels", type=int, default=self.channels, 
                          help="Number of channels")
        parser.add_argument("--input_len", type=int, default=self.input_length, 
                          help="Input sequence length")
        parser.add_argument("--output_len", type=int, default=self.output_length, 
                          help="Output sequence length")
        parser.add_argument("--llm_layer", type=int, default=self.llm_layers, 
                          help="LLM layers")
        parser.add_argument("--U", type=int, default=self.unfrozen_layers, 
                          help="Number of unfrozen attention layers")
        parser.add_argument("--lora_rank", type=int, default=self.lora_rank, 
                          help="LoRA rank")
        parser.add_argument("--llm_path", type=str, default=self.llm_model_path, 
                          help="LLM model path")

        parser.add_argument("--batch_size", type=int, default=self.batch_size, 
                          help="Batch size")
        parser.add_argument("--lrate", type=float, default=self.learning_rate, 
                          help="Learning rate")
        parser.add_argument("--wdecay", type=float, default=self.weight_decay, 
                          help="Weight decay")
        parser.add_argument("--epochs", type=int, default=self.epochs, 
                          help="Number of epochs")
        parser.add_argument("--print_every", type=int, default=self.print_interval, 
                          help="Print interval")

        parser.add_argument("--save", type=str, default=self.save_path, 
                          help="Model save path")
        parser.add_argument("--es_patience", type=int, default=self.early_stopping_patience, 
                          help="Early stopping patience")
        
        args = parser.parse_args()

        self.__dict__.update(vars(args))
        self._setup_dataset_specific_config()
        
        return self
    
    def get_model_save_path(self):
        model_name = self.llm_path.rstrip('/').split('/')[-1]
        return f"{self.save_path}{self.dataset_name}-{model_name}/"
    
    def __str__(self):
        config_str = " Training Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str


def get_training_config():
    config = TrainingConfig()
    return config.parse_arguments()