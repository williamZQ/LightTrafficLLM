
import torch
import numpy as np
import pandas as pd
import time
import os
import random
from config import get_training_config
from trainer import ModelTrainer, TrainingMetrics
from data_loader import create_data_manager, create_data_iterator
from evaluator import create_evaluator, create_early_stopping, ResultSaver


class TrainingPipeline:
    
    def __init__(self):
        self.config = None
        self.data_manager = None
        self.trainer = None
        self.evaluator = None
        self.metrics_tracker = None
        self.early_stopping = None
        self.device = None
        
    def setup_environment(self):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'
        
        self._set_random_seed(6666)
        
        torch.cuda.empty_cache()
        
    def _set_random_seed(self, seed):
        random.seed(seed)
        os.environ["PYTHONSEED"] = str(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True  
        torch.manual_seed(seed)
    
    def initialize_components(self):
        self.config = get_training_config()
        
        self.device = torch.device(self.config.device)
        
        self.data_manager = create_data_manager(self.config)
        dataloader, scaler = self.data_manager.load_dataset()
        
        self.trainer = ModelTrainer(self.config, scaler, self.device)
        
        self.evaluator = create_evaluator(self.trainer, self.data_manager, self.device)
        
        self.metrics_tracker = TrainingMetrics()
        
        self.early_stopping = create_early_stopping(
            patience=self.config.early_stopping_patience,
            min_epochs=200
        )
        
        self.save_path = self.config.get_model_save_path()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        print("All components initialized successfully")
        print(self.config)
    
    def run_training(self):
        print("Starting training...")
        
        best_valid_loss = float('inf')
        best_test_loss = float('inf')
        best_epoch = 0
        training_results = []
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)
            
            valid_metrics = self._validate_epoch(epoch)
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            self.metrics_tracker.add_train_metrics(
                train_metrics['loss'], train_metrics['mae'], 
                train_metrics['mape'], train_metrics['rmse'], train_metrics['wmape'], epoch_time
            )
            
            self.metrics_tracker.add_valid_metrics(
                valid_metrics['loss'], valid_metrics['mae'], 
                valid_metrics['mape'], valid_metrics['rmse'], valid_metrics['wmape'], epoch_time
            )
            
            should_save, current_test_loss = self._check_model_saving(
                epoch, valid_metrics['loss'], best_valid_loss
            )
            
            if should_save:
                best_valid_loss = valid_metrics['loss']
                best_test_loss = current_test_loss
                best_epoch = epoch
                self.trainer.save_model(f"{self.save_path}best_model.pth")
            
            training_results.append(self.metrics_tracker.get_pandas_series(epoch))
            ResultSaver.save_training_results(training_results, f"{self.save_path}/train.csv")
            
            if self.early_stopping.check_stop_condition(
                valid_metrics['loss'], current_test_loss, epoch
            ):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            self.metrics_tracker.clear_epoch_metrics()
        
        total_time = time.time() - start_time
        
        self._print_training_summary(best_epoch, best_valid_loss, best_test_loss, total_time)
        
        self._final_testing(best_epoch)
    
    def _train_epoch(self, epoch):
        print(f"Epoch {epoch} training started...")
        
        train_start_time = time.time()
        
        data_iterator = create_data_iterator(self.data_manager, self.device)
        
        epoch_losses = []
        epoch_mae = []
        epoch_mape = []
        epoch_rmse = []
        epoch_wmape = []
        
        for batch_idx, input_data, target_data in data_iterator.iterate_training_batches():
            loss, metrics = self.trainer.train_step(input_data, target_data[:, 0, :, :])
            
            epoch_losses.append(loss)
            epoch_mae.append(metrics['mae'])
            epoch_mape.append(metrics['mape'])
            epoch_rmse.append(metrics['rmse'])
            epoch_wmape.append(metrics['wmape'])
            
            if batch_idx % self.config.print_interval == 0:
                self._print_batch_progress(batch_idx, loss, metrics)
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        print(f"Epoch {epoch} training completed, time: {train_time:.2f} seconds")
        
        return {
            'loss': np.mean(epoch_losses),
            'mae': np.mean(epoch_mae),
            'mape': np.mean(epoch_mape),
            'rmse': np.mean(epoch_rmse),
            'wmape': np.mean(epoch_wmape)
        }
    
    def _validate_epoch(self, epoch):
        print(f"Epoch {epoch} validation started...")
        
        valid_start_time = time.time()
        
        data_iterator = create_data_iterator(self.data_manager, self.device)
        
        valid_losses = []
        valid_mape = []
        valid_rmse = []
        valid_wmape = []
        
        for batch_idx, input_data, target_data in data_iterator.iterate_validation_batches():
            loss, metrics = self.trainer.evaluate_step(input_data, target_data[:, 0, :, :])
            
            valid_losses.append(loss)
            valid_mae.append(metrics['mae'])
            valid_mape.append(metrics['mape'])
            valid_rmse.append(metrics['rmse'])
            valid_wmape.append(metrics['wmape'])
        
        valid_end_time = time.time()
        valid_time = valid_end_time - valid_start_time
        
        print(f"Epoch {epoch} validation completed, time: {valid_time:.2f} seconds")
        
        return {
            'loss': np.mean(valid_losses),
            'mae': np.mean(valid_mae),
            'mape': np.mean(valid_mape),
            'rmse': np.mean(valid_rmse),
            'wmape': np.mean(valid_wmape)
        }
    
    def _check_model_saving(self, epoch, current_valid_loss, best_valid_loss):
        if current_valid_loss < best_valid_loss:
            print("### Found better model ###")
            
            if epoch <= 100:
                # Skip full testing for first 100 epochs
                print(f"Epoch {epoch} validation loss improved: {current_valid_loss:.4f}")
                return True, float('inf')
            else:
                # Perform full testing after 100 epochs
                print(f"Epoch {epoch} validation loss improved: {current_valid_loss:.4f}")
                
                # Evaluate on test set
                average_metrics, _ = self.evaluator.evaluate_on_test_set()
                test_loss = average_metrics['mae']
                
                self.evaluator.print_average_metrics(average_metrics)
                
                return True, test_loss
        
        return False, float('inf')
    
    def _print_batch_progress(self, batch_idx, loss, metrics):
        log = "Batch: {:03d}, Training Loss: {:.4f}, Training MAE: {:.4f}, Training RMSE: {:.4f}, Training MAPE: {:.4f}, Training WMAPE: {:.4f}"
        print(log.format(
            batch_idx, loss, metrics['mae'], metrics['rmse'], metrics['mape'], metrics['wmape']
        ))
    
    def _print_training_summary(self, best_epoch, best_valid_loss, best_test_loss, total_time):
        """Print training summary"""
        print("\n" + "="*50)
        print("Training Completed Summary")
        print("="*50)
        print(f"Best model epoch: {best_epoch}")
        print(f"Best validation loss: {best_valid_loss:.4f}")
        if best_test_loss < float('inf'):
            print(f"Corresponding test loss: {best_test_loss:.4f}")
        print(f"Total training time: {total_time:.2f} seconds")
        print("="*50)
    
    def _final_testing(self, best_epoch):
        print("\nStarting final testing...")
        
        self.trainer.load_model(f"{self.save_path}best_model.pth")
        
        average_metrics, test_results = self.evaluator.evaluate_on_test_set()
        
        ResultSaver.save_test_results(test_results, f"{self.save_path}/test.csv")
        
        print(f"Final testing completed, best model from epoch {best_epoch}")
        self.evaluator.print_average_metrics(average_metrics)


def main():
    pipeline = TrainingPipeline()
    
    pipeline.setup_environment()
    
    pipeline.initialize_components()

    pipeline.run_training()


if __name__ == "__main__":
    main()