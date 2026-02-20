# LOAD MODULES

# Standard library
from typing import Dict, Union, Literal

# Proprietary
from src.utils.metrics import (
    RMSSE_calculation, 
    sMAPE_calculation
)

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_forecasting.models import BaseModel
from lightning import LightningModule

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

class NBEATS_block(nn.Module):
    """
    This is the code for one N-BEATS block.
    It outputs:
    (1) a forecast for each time step in the forecast period);
    (2) and a backcast of the input to facilitate sequential analysis (feed residual to next block).
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int):
        super().__init__()
        self.forecast_length = forecast_length
        # Shared layers in block
        self.fc1 = nn.Linear(backcast_length, hidden_layer_units)
        self.fc2 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc3 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc4 = nn.Linear(hidden_layer_units, hidden_layer_units)
        # Task specific (backcast & forecast) layers in block
        self.fc_backcast = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc_forecast = nn.Linear(hidden_layer_units, hidden_layer_units)
        # Block output layers
        self.fc_backcast_output = nn.Linear(hidden_layer_units, backcast_length)
        self.fc_forecast_output = nn.Linear(hidden_layer_units, forecast_length)

    def forward(self, x: torch.Tensor):
        # Shared
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.01)
        h3 = F.leaky_relu(self.fc3(h2), negative_slope=0.01)
        h4 = F.leaky_relu(self.fc4(h3), negative_slope=0.01)
        # Task specific
        h_backcast = F.leaky_relu(self.fc_backcast(h4), negative_slope=0.01)
        h_forecast = F.leaky_relu(self.fc_forecast(h4), negative_slope=0.01)
        # Outputs - backcast + forecast for each period in forecast_length
        backcast = self.fc_backcast_output(h_backcast)
        forecast = self.fc_forecast_output(h_forecast)

        return backcast, forecast
        
# # Test NBEATS_block
# network = NBEATS_block(5,2,10)
# x = torch.rand(20, 5) # batch_size = 20 x lookback_window_length = 5
# backcast, forecast = network(x)
# print(backcast.shape) # dim = bs x bl
# print(forecast.shape) # dim = bs x fl 
# print("Number of parameters:", num_parameters(network))
# (((5*10)+10)+((10*10)+10)+((10*10)+10)+((10*10)+10)+ # shared
#   2*((10*10)+10)+ # task-specific
#  ((10*5)+5)+ # backcast
#  2*(((10*1)+1)+((10*4)+4))) # forecast

class NBEATS_module(nn.Module):
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int,
                 n_blocks: int,
                 n_blocks_shared: int):
        self.forecast_length = forecast_length  
        super().__init__()
        # Init construction N-BEATS blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = NBEATS_block(backcast_length,
                                 forecast_length,
                                 hidden_layer_units)
            for _ in range(n_blocks_shared):
                self.blocks.append(block)

    def forward(self, backcast: torch.Tensor):
        forecast = torch.zeros_like(backcast[:, :self.forecast_length])
        # Loop through blocks
        for block_id in range(len(self.blocks)):
            b, f = self.blocks[block_id](backcast)
            backcast = backcast - b
            forecast = forecast + f

        return forecast
    
# # Test NBEATS_module
# network = NBEATS_module(5,2,10,3,1)
# x = torch.rand(20, 5) # batch_size = 20 x lookback_window_length = 5
# forecast = network(x)
# print(forecast.shape) # dim = bs x fl
# print("Number of parameters:", num_parameters(network)) 
# # [Num of pars per block]*3

class LitNBEATSS(LightningModule):
    '''
    A LightningModule to operationalize NBEATSS.
    '''
    def __init__(self,
                 # Model hypers
                 backcast_length_multiplier: int,
                 forecast_length: int,
                 hidden_layer_units: int = 256,
                 n_blocks: int = 10,
                 n_blocks_shared: int = 1,
                 ensemble_size: int = 1,
                 zero_mean: bool = True,
                 unit_variance: bool = True,                 
                 # Optim hypers
                 lambda_stability: float = 0.0,
                 enforce_nonnegative_forecast_metric_calculation: bool = False,
                 learning_rate: float = 1e-3,
                 explr_gamma: float = 1.00,
                 ema_decay: float = 0.00,
                 ):
        super().__init__()      
        if (not isinstance(lambda_stability, float)):
            raise ValueError("Invalid argument: lambda_stability must be a float")
        self.save_hyperparameters()
        self.model = nn.ModuleList([
            NBEATS_module(backcast_length=backcast_length_multiplier * forecast_length,
                          forecast_length=forecast_length,
                          hidden_layer_units=hidden_layer_units,
                          n_blocks=n_blocks,
                          n_blocks_shared=n_blocks_shared
                          ) for _ in range(ensemble_size)])
        self.ema_model = nn.ModuleList([
            NBEATS_module(backcast_length=backcast_length_multiplier * forecast_length,
                          forecast_length=forecast_length,
                          hidden_layer_units=hidden_layer_units,
                          n_blocks=n_blocks,
                          n_blocks_shared=n_blocks_shared
                          ) for _ in range(ensemble_size)])
        
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            _, _, # rescaled_forecast, rescaled_forecast_lagged
            loss_accuracy, _, _, # RMSSE, sMAPE
            loss_stability, _, _, # RMSSC, sMAPC
            loss, w_accuracy, w_stability, 
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             False, # evaluate
                             False, # calculate_metrics
                             )
        
        metrics = {"tloss_a": loss_accuracy,
                   "tloss_s": loss_stability,
                   "tloss": loss,
                   "w_accuracy": w_accuracy,
                   "w_stability": w_stability,
                   }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        return loss
            
    def validation_step(self, batch, batch_idx):
        """
        Performs validation using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            rescaled_forecast, _, # rescaled_forecast_lagged
            loss_accuracy, _, _, # RMSSE, sMAPE
            loss_stability, _, _, # RMSSC, sMAPC
            loss, _, _, # w_accuracy, w_stability
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             True, # evaluate
                             False, # calculate_metrics
                             )
        
        metrics = {"vloss_a": loss_accuracy,
                   "vloss_s": loss_stability,
                   "vloss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        return rescaled_forecast
    
    def test_step(self, batch, batch_idx):
        """
        Performs testing using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            rescaled_forecast, rescaled_forecast_lagged,
            _, RMSSE, sMAPE, # loss_accuracy
            _, RMSSC, sMAPC, # loss_stability
            _, _, _, # loss, w_accuracy, w_stability
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             True, # evaluate
                             True, # calculate_metrics
                             )
        
        metrics = {"RMSSE": RMSSE,
                   "sMAPE": sMAPE,
                   "RMSSC": RMSSC,
                   "sMAPC": sMAPC}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        
        rescaled_forecast_all = torch.stack((rescaled_forecast, rescaled_forecast_lagged), dim=-1)

        return rescaled_forecast_all

    def _get_losses(self,
                    batch,
                    lambda_stability,
                    enforce_nonnegative_forecast_metric_calculation,
                    evaluate,
                    calculate_metrics):
        x, _ = batch
        
        # Batch checks - handle batch_size = 1
        if len(x["encoder_cont"].shape) == 2:
            x["encoder_cont"] = x["encoder_cont"].unsqueeze(0)
        if len(x["decoder_cont"].shape) == 2:
            x["decoder_cont"] = x["decoder_cont"].unsqueeze(0)

        # if x["encoder_cont"].shape[0] != x["decoder_cont"].shape[0]:
        #     raise ValueError("Batch sizes of encoder and decoder do not match")
        bs = x["encoder_cont"].shape[0]

        # Batch data
        lookback_window = x["encoder_cont"][:,:,4] # shape = batch_size x lookback_window_length
        lookback_window_lagged = x["encoder_cont"][:,:,5] # shape = batch_size x lookback_window_length
        forecast_period = x["decoder_cont"][:,:,4] # shape = batch_size x forecast_length
        forecast_period_lagged = x["decoder_cont"][:,:,5] # shape = batch_size x forecast_length
        if evaluate:
            mean = x["decoder_cont"][:,:,0] # shape = batch_size x forecast_length
            std = x["decoder_cont"][:,:,1] # shape = batch_size x forecast_length
        
        # Data augementation on-the-fly
        if not evaluate:
            shift_value = torch.rand(bs, device=self.device)
            shift_sign = torch.randint(0, 2, (bs,), device=self.device) * 2 - 1
            shift = shift_sign * shift_value
            scale = torch.rand(bs, device=self.device) + 0.5
            lookback_window = (lookback_window + shift.unsqueeze(1)) * scale.unsqueeze(1)
            lookback_window_lagged = (lookback_window_lagged + shift.unsqueeze(1)) * scale.unsqueeze(1)
            forecast_period = (forecast_period + shift.unsqueeze(1)) * scale.unsqueeze(1)
            forecast_period_lagged = (forecast_period_lagged + shift.unsqueeze(1)) * scale.unsqueeze(1)
                        
        # Scaling factors
        if not calculate_metrics: # Scaling losses 
            # Scaling constants for RMSSE and RMSSC - shape = batch_size
            scaling_constant_sq_loss = torch.mean(torch.diff(lookback_window)**2, -1) + 1e-3 # for numerical stability
            scaling_constant_sq_lagged_loss = torch.mean(torch.diff(lookback_window_lagged)**2, -1) + 1e-3 # for numerical stability
        else: # Scaling metrics
            scaling_constant_abs = x["encoder_cont"][:,-1,2] # shape = batch_size
            scaling_constant_sq = x["encoder_cont"][:,-1,3] # shape = batch_size

        # Obtain forecast for different ensemble members
        model_forecast = torch.zeros((bs, # batch_size
                                      self.hparams.forecast_length, # forecast per period in forecast_length
                                      self.hparams.ensemble_size
                                      ), 
                                      dtype = torch.float,
                                      device=self.device)
        model_forecast_lagged = torch.zeros_like(model_forecast)
        
        for ensemble_id in range(self.hparams.ensemble_size):
            if not evaluate:
                model_forecast_id = self.model[ensemble_id](lookback_window)
                model_forecast_lagged_id = self.model[ensemble_id](lookback_window_lagged)
            else:
                with torch.no_grad():
                    model_forecast_id = self.ema_model[ensemble_id](lookback_window)
                    model_forecast_lagged_id = self.ema_model[ensemble_id](lookback_window_lagged)

            model_forecast[:, :, ensemble_id] = model_forecast_id
            model_forecast_lagged[:, :, ensemble_id] = model_forecast_lagged_id
            
        final_forecast = model_forecast.mean(-1) # take mean over ensemble_ids
        final_forecast_lagged = model_forecast_lagged.mean(-1) # take mean over ensemble_ids

        rescaled_final_forecast = 0
        rescaled_final_forecast_lagged = 0
        if evaluate:
            with torch.no_grad():
                rescaled_final_forecast = final_forecast * std + mean
                if enforce_nonnegative_forecast_metric_calculation:
                    rescaled_final_forecast = torch.clamp(rescaled_final_forecast, 0)
                if calculate_metrics:
                    rescaled_final_forecast_lagged = final_forecast_lagged * std + mean
                    if enforce_nonnegative_forecast_metric_calculation:
                        rescaled_final_forecast_lagged = torch.clamp(rescaled_final_forecast_lagged, 0)

        # Compute accucary losses/metrics
        loss_accuracy = 0
        RMSSE = 0
        sMAPE = 0
        if not calculate_metrics: # accuracy loss
            RMSSE_loss = RMSSE_calculation(final_forecast, forecast_period, scaling_constant_sq_loss)
            RMSSE_lagged_loss = RMSSE_calculation(final_forecast_lagged, forecast_period_lagged, scaling_constant_sq_lagged_loss)
            loss_accuracy = torch.mean(0.5 * (RMSSE_loss + RMSSE_lagged_loss))
        else: # RMSSE and sMAPE metric
            with torch.no_grad():
                rescaled_forecast_period = forecast_period * std + mean
                RMSSE = RMSSE_calculation(rescaled_final_forecast, rescaled_forecast_period, scaling_constant_sq)
                sMAPE = sMAPE_calculation(rescaled_final_forecast, rescaled_forecast_period)
                
        # Compute stability losses/metrics
        loss_stability = 0
        RMSSC = 0
        sMAPC = 0
        if not calculate_metrics: # stability loss
            loss_stability = RMSSE_calculation(final_forecast[:,:-1], final_forecast_lagged[:,1:], scaling_constant_sq_loss)
        else: # RMSSC and sMAPC metric
            with torch.no_grad():
                RMSSC = RMSSE_calculation(rescaled_final_forecast[:,:-1], rescaled_final_forecast_lagged[:,1:], scaling_constant_sq)
                sMAPC = sMAPE_calculation(rescaled_final_forecast[:,:-1], rescaled_final_forecast_lagged[:,1:])

        # Calculate composite loss
        loss = 0
        w_accuracy = 0
        w_stability = 0
        if not calculate_metrics:
            w_stability = self.hparams.lambda_stability
            w_accuracy = 1 - w_stability
            loss = w_accuracy * loss_accuracy + w_stability * loss_stability

        return (rescaled_final_forecast, rescaled_final_forecast_lagged, 
                loss_accuracy, RMSSE, sMAPE,
                loss_stability, RMSSC, sMAPC,
                loss, w_accuracy, w_stability,
                bs)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, amsgrad=False)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.explr_gamma),
            'interval': 'epoch',  # Update the learning rate after each epoch
            'frequency': 1,       # Apply the scheduler every epoch
        }

        return [optimizer], [scheduler]  # Return the optimizer and scheduler as lists
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        ensemble_id_update = batch_idx % self.hparams.ensemble_size
        for idx, model in enumerate(self.model):
            if idx != ensemble_id_update:
                for param in model.parameters():
                    param.grad = None
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        #if self.global_step > 100: almost no impact for ema_decay = .99
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.hparams.ema_decay * ema_param.data + (1 - self.hparams.ema_decay) * model_param.data
