
import torch
import numpy as np
import util
import os


class DataManager:
    
    def __init__(self, config):

        self.config = config
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.dataloader = None
        self.scaler = None
        
    def load_dataset(self):

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
        
        self.dataloader = util.load_dataset(
            self.data_path, 
            self.batch_size, 
            self.batch_size, 
            self.batch_size
        )
        
        self.scaler = self.dataloader["scaler"]
        return self.dataloader, self.scaler
    
    def get_data_iterator(self, split='train'):

        if self.dataloader is None:
            self.load_dataset()
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("split parameter must be 'train', 'val' or 'test'")
        
        return self.dataloader[f"{split}_loader"].get_iterator()
    
    def get_test_targets(self, device):
        if self.dataloader is None:
            self.load_dataset()
        
        test_targets = torch.Tensor(self.dataloader["y_test"]).to(device)
        return test_targets.transpose(1, 3)[:, 0, :, :]


class DataPreprocessor:

    @staticmethod
    def prepare_batch_data(batch_data, device):

        x, y = batch_data
        
        # Convert to tensor and move to device
        input_tensor = torch.Tensor(x).to(device)
        target_tensor = torch.Tensor(y).to(device)
        
        # Adjust dimensions: (batch, seq_len, nodes, features) -> (batch, features, nodes, seq_len)
        input_tensor = input_tensor.transpose(1, 3)
        target_tensor = target_tensor.transpose(1, 3)
        
        return input_tensor, target_tensor
    
    @staticmethod
    def prepare_test_inputs(batch_data, device):

        x, _ = batch_data

        input_tensor = torch.Tensor(x).to(device)
    
        input_tensor = input_tensor.transpose(1, 3)
        
        return input_tensor


class TrainingDataIterator:
    def __init__(self, data_manager, device):

        self.data_manager = data_manager
        self.device = device
        self.preprocessor = DataPreprocessor()
    
    def iterate_training_batches(self):
        iterator = self.data_manager.get_data_iterator('train')
        
        for batch_idx, batch_data in enumerate(iterator):
            input_data, target_data = self.preprocessor.prepare_batch_data(batch_data, self.device)
            yield batch_idx, input_data, target_data
    
    def iterate_validation_batches(self):
        iterator = self.data_manager.get_data_iterator('val')
        
        for batch_idx, batch_data in enumerate(iterator):
            input_data, target_data = self.preprocessor.prepare_batch_data(batch_data, self.device)
            yield batch_idx, input_data, target_data
    
    def iterate_test_batches(self):
        iterator = self.data_manager.get_data_iterator('test')
        
        for batch_idx, batch_data in enumerate(iterator):
            input_data = self.preprocessor.prepare_test_inputs(batch_data, self.device)
            yield batch_idx, input_data


def create_data_manager(config):
    return DataManager(config)


def create_data_iterator(data_manager, device):
    return TrainingDataIterator(data_manager, device)