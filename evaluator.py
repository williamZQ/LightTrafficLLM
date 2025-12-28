
import torch
import numpy as np
import pandas as pd
import util


class ModelEvaluator:
    
    def __init__(self, trainer, data_manager, device):
        self.trainer = trainer
        self.data_manager = data_manager
        self.device = device
        self.scaler = data_manager.scaler
    
    def evaluate_on_test_set(self):
        print("Starting model evaluation on test set...")
        
        test_targets = self.data_manager.get_test_targets(self.device)
        
        predictions = self._collect_predictions()
        
        predictions = predictions[:test_targets.size(0), ...]
        
        horizon_metrics = self._compute_horizon_metrics(predictions, test_targets)
        
        average_metrics = self._compute_average_metrics(horizon_metrics)
        
        test_results = self._prepare_test_results(horizon_metrics, average_metrics)
        
        return average_metrics, test_results
    
    def _collect_predictions(self):
 
        predictions_list = []
        
        data_iterator = TrainingDataIterator(self.data_manager, self.device)
        
        for batch_idx, input_data in data_iterator.iterate_test_batches():
            with torch.no_grad():
                batch_predictions = self.trainer.model(input_data)
                batch_predictions = batch_predictions.transpose(1, 3)
                predictions_list.append(batch_predictions.squeeze())
        
        return torch.cat(predictions_list, dim=0)
    
    def _compute_horizon_metrics(self, predictions, targets):
        horizon_metrics = {
            'mae': [],
            'mape': [],
            'rmse': [],
            'wmape': []
        }
        
        for horizon in range(self.trainer.config.output_length):
            pred_horizon = self.scaler.inverse_transform(predictions[:, :, horizon])
            target_horizon = targets[:, :, horizon]
            
            metrics = util.metric(pred_horizon, target_horizon)
            
            horizon_metrics['mae'].append(metrics[0])
            horizon_metrics['mape'].append(metrics[1])
            horizon_metrics['rmse'].append(metrics[2])
            horizon_metrics['wmape'].append(metrics[3])
            
            self._print_horizon_metrics(horizon + 1, metrics)
        
        return horizon_metrics
    
    def _compute_average_metrics(self, horizon_metrics):
        return {
            'mae': np.mean(horizon_metrics['mae']),
            'mape': np.mean(horizon_metrics['mape']),
            'rmse': np.mean(horizon_metrics['rmse']),
            'wmape': np.mean(horizon_metrics['wmape'])
        }
    
    def _prepare_test_results(self, horizon_metrics, average_metrics):
        test_results = []
        
        for horizon in range(len(horizon_metrics['mae'])):
            horizon_result = {
                'horizon': horizon + 1,
                'test_mae': horizon_metrics['mae'][horizon],
                'test_rmse': horizon_metrics['rmse'][horizon],
                'test_mape': horizon_metrics['mape'][horizon],
                'test_wmape': horizon_metrics['wmape'][horizon]
            }
            test_results.append(pd.Series(horizon_result))
        
        average_result = {
            'horizon': 'average',
            'test_mae': average_metrics['mae'],
            'test_rmse': average_metrics['rmse'],
            'test_mape': average_metrics['mape'],
            'test_wmape': average_metrics['wmape']
        }
        test_results.append(pd.Series(average_result))
        
        return test_results
    
    def _print_horizon_metrics(self, horizon, metrics):
        log = "Time step {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(horizon, metrics[0], metrics[2], metrics[1], metrics[3]))
    
    def print_average_metrics(self, average_metrics):
        log = "Average metrics - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(
            average_metrics['mae'], 
            average_metrics['rmse'], 
            average_metrics['mape'], 
            average_metrics['wmape']
        ))


class ResultSaver:

    @staticmethod
    def save_training_results(results, filepath):
        results_df = pd.DataFrame(results)
        results_df.round(8).to_csv(filepath, index=False)
        print(f"Training results saved to: {filepath}")
    
    @staticmethod
    def save_test_results(results, filepath):
        results_df = pd.DataFrame(results)
        results_df.round(8).to_csv(filepath, index=False)
        print(f"Test results saved to: {filepath}")


class EarlyStopping:
    
    def __init__(self, patience=100, min_epochs=200):

        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.best_test_loss = float('inf')
        self.should_stop = False
    
    def check_stop_condition(self, current_loss, current_test_loss, current_epoch):

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_test_loss = current_test_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            
            if self.counter >= self.patience and current_epoch >= self.min_epochs:
                self.should_stop = True
                return True
            
            return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.best_test_loss = float('inf')
        self.should_stop = False


def create_evaluator(trainer, data_manager, device):
    return ModelEvaluator(trainer, data_manager, device)


def create_early_stopping(patience=100, min_epochs=200):
    return EarlyStopping(patience, min_epochs)


from data_loader import TrainingDataIterator