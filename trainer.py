
import torch
import numpy as np
import pandas as pd
from model import LightTrafficLLM
from ranger_manager import Ranger
import util


class ModelTrainer:
    def __init__(self, config, scaler, device):

        self.config = config
        self.scaler = scaler
        self.device = device
        self.gradient_clip = config.gradient_clip
        
        self.model = self._initialize_model()
        self.model.to(device)
        
        self.optimizer = Ranger(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.loss_function = util.MAE_torch
        
        print(f"Number of model parameters: {self.model.param_num()}")
        print(self.model)
    
    def _initialize_model(self):
        return LightTrafficLLM(
            input_dim=self.config.input_dim,
            channels=self.config.channels,
            num_nodes=self.config.num_nodes,
            input_len=self.config.input_length,
            output_len=self.config.output_length,
            llm_layer=self.config.llm_layers,
            U=self.config.unfrozen_layers,
            device=self.device,
            lora_rank=self.config.lora_rank,
            llm_path=self.config.llm_model_path
        )
    
    def train_step(self, input_data, target_data):
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(input_data)
        predictions = predictions.transpose(1, 3)
      
        targets = torch.unsqueeze(target_data, dim=1)
  
        denormalized_predictions = self.scaler.inverse_transform(predictions)

        loss = self.loss_function(denormalized_predictions, targets, 0.0)

        loss.backward()

        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()

        metrics = self._compute_metrics(denormalized_predictions, targets)
        
        return loss.item(), metrics
    
    def evaluate_step(self, input_data, target_data):
        self.model.eval()
        
        with torch.no_grad():
    
            predictions = self.model(input_data)
            predictions = predictions.transpose(1, 3)
   
            targets = torch.unsqueeze(target_data, dim=1)

            denormalized_predictions = self.scaler.inverse_transform(predictions)

            loss = self.loss_function(denormalized_predictions, targets, 0.0)
            metrics = self._compute_metrics(denormalized_predictions, targets)
            
            return loss.item(), metrics
    
    def _compute_metrics(self, predictions, targets):
        return {
            'mae': util.MAE_torch(predictions, targets, 0.0).item(),
            'mape': util.MAPE_torch(predictions, targets, 0.0).item(),
            'rmse': util.RMSE_torch(predictions, targets, 0.0).item(),
            'wmape': util.WMAPE_torch(predictions, targets, 0.0).item()
        }
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


class TrainingMetrics:
    
    def __init__(self):
        self.train_losses = []
        self.train_mae = []
        self.train_mape = []
        self.train_rmse = []
        self.train_wmape = []
        
        self.valid_losses = []
        self.valid_mae = []
        self.valid_mape = []
        self.valid_rmse = []
        self.valid_wmape = []
        
        self.train_times = []
        self.valid_times = []
    
    def add_train_metrics(self, loss, mae, mape, rmse, wmape, time_taken):
        self.train_losses.append(loss)
        self.train_mae.append(mae)
        self.train_mape.append(mape)
        self.train_rmse.append(rmse)
        self.train_wmape.append(wmape)
        self.train_times.append(time_taken)
    
    def add_valid_metrics(self, loss, mae, mape, rmse, wmape, time_taken):
        self.valid_losses.append(loss)
        self.valid_mae.append(mae)
        self.valid_mape.append(mape)
        self.valid_rmse.append(rmse)
        self.valid_wmape.append(wmape)
        self.valid_times.append(time_taken)
    
    def get_current_epoch_metrics(self):
        return {
            'train': {
                'loss': np.mean(self.train_losses),
                'mae': np.mean(self.train_mae),
                'mape': np.mean(self.train_mape),
                'rmse': np.mean(self.train_rmse),
                'wmape': np.mean(self.train_wmape)
            },
            'valid': {
                'loss': np.mean(self.valid_losses),
                'mae': np.mean(self.valid_mae),
                'mape': np.mean(self.valid_mape),
                'rmse': np.mean(self.valid_rmse),
                'wmape': np.mean(self.valid_wmape)
            }
        }
    
    def get_pandas_series(self, epoch):
        train_metrics = self.get_current_epoch_metrics()['train']
        valid_metrics = self.get_current_epoch_metrics()['valid']
        
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_mape': train_metrics['mape'],
            'train_wmape': train_metrics['wmape'],
            'valid_loss': valid_metrics['loss'],
            'valid_mae': valid_metrics['mae'],
            'valid_rmse': valid_metrics['rmse'],
            'valid_mape': valid_metrics['mape'],
            'valid_wmape': valid_metrics['wmape']
        }
        
        return pd.Series(metrics_dict)
    
    def clear_epoch_metrics(self):
        self.train_losses.clear()
        self.train_mae.clear()
        self.train_mape.clear()
        self.train_rmse.clear()
        self.train_wmape.clear()
        
        self.valid_losses.clear()
        self.valid_mae.clear()
        self.valid_mape.clear()
        self.valid_rmse.clear()
        self.valid_wmape.clear()