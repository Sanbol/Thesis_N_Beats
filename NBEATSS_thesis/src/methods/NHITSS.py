from src.utils.metrics import RMSSE_calculation, sMAPE_calculation

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


class NHITS_block(nn.Module):
    """
    One N-HiTS block.
    Key differences from N-BEATS block:
    1. Input is downsampled by MaxPool1d with kernel_size = pool_kernel
    2. Output is fewer coefficients that are interpolated back to full length
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int,
                 pool_kernel: int = 1,
                 n_theta_backcast: int = None,
                 n_theta_forecast: int = None):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.pool_kernel = pool_kernel

        # Pooled input length
        self.pooled_length = backcast_length // pool_kernel

        # Number of basis expansion coefficients (interpolation targets)
        # For backcast: output fewer coefficients, interpolate to backcast_length
        # For forecast: output fewer coefficients, interpolate to forecast_length
        self.n_theta_backcast = n_theta_backcast if n_theta_backcast else max(backcast_length // pool_kernel, 1)
        self.n_theta_forecast = n_theta_forecast if n_theta_forecast else max(forecast_length // pool_kernel, 1)

        # MaxPool for downsampling input
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)

        # Shared layers in block (operate on pooled input)
        self.fc1 = nn.Linear(self.pooled_length, hidden_layer_units)
        self.fc2 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc3 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc4 = nn.Linear(hidden_layer_units, hidden_layer_units)

        # Task specific (backcast & forecast) layers in block
        self.fc_backcast = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc_forecast = nn.Linear(hidden_layer_units, hidden_layer_units)

        # Block output layers - output compressed coefficients
        self.fc_backcast_output = nn.Linear(hidden_layer_units, self.n_theta_backcast)
        self.fc_forecast_output = nn.Linear(hidden_layer_units, self.n_theta_forecast)

    def forward(self, x: torch.Tensor):
        # MaxPool1d expects (batch, channels, length)
        x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)

        h1 = F.leaky_relu(self.fc1(x_pooled), negative_slope=0.01)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.01)
        h3 = F.leaky_relu(self.fc3(h2), negative_slope=0.01)
        h4 = F.leaky_relu(self.fc4(h3), negative_slope=0.01)

        h_backcast = F.leaky_relu(self.fc_backcast(h4), negative_slope=0.01)
        h_forecast = F.leaky_relu(self.fc_forecast(h4), negative_slope=0.01)

        theta_backcast = self.fc_backcast_output(h_backcast)
        theta_forecast = self.fc_forecast_output(h_forecast)

        backcast = F.interpolate(
            theta_backcast.unsqueeze(1), size=self.backcast_length, mode='linear', align_corners=False
        ).squeeze(1)

        forecast = F.interpolate(
            theta_forecast.unsqueeze(1), size=self.forecast_length, mode='linear', align_corners=False
        ).squeeze(1)

        return backcast, forecast


class NHITS_module(nn.Module):
    """
    Full N-HiTS module with multiple blocks at different pooling rates.
    Uses doubly-residual architecture (same as N-BEATS).
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int,
                 n_blocks: int,
                 n_blocks_shared: int,
                 pool_kernels: list = None):
        super().__init__()
        self.forecast_length = forecast_length

        if pool_kernels is None:
            pool_kernels = [1, 2, 4]
        # Extend or truncate to match n_blocks
        while len(pool_kernels) < n_blocks:
            pool_kernels.append(pool_kernels[-1] * 2)
        pool_kernels = pool_kernels[:n_blocks]

        # Make sure pool kernels divide backcast_length evenly
        pool_kernels = [min(pk, backcast_length) for pk in pool_kernels]
        # Adjust to nearest divisor of backcast_length
        for i in range(len(pool_kernels)):
            pk = pool_kernels[i]
            while backcast_length % pk != 0 and pk > 1:
                pk -= 1
            pool_kernels[i] = pk

        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            pk = pool_kernels[block_idx]
            block = NHITS_block(
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                hidden_layer_units=hidden_layer_units,
                pool_kernel=pk,
            )
            for _ in range(n_blocks_shared):
                self.blocks.append(block)

    def forward(self, backcast: torch.Tensor):
        forecast = torch.zeros_like(backcast[:, :self.forecast_length])
        # Loop through blocks (doubly residual)
        for block_id in range(len(self.blocks)):
            b, f = self.blocks[block_id](backcast)
            backcast = backcast - b
            forecast = forecast + f
        return forecast


class LitNHITSS(LightningModule):
    '''
    A LightningModule to operationalize N-HiTS-S (N-HiTS with Stability extension).
    Same training pipeline as LitNBEATSS - only the architecture differs.
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
                 pool_kernels: list = None,
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
        
        backcast_length = backcast_length_multiplier * forecast_length
        
        self.model = nn.ModuleList([
            NHITS_module(backcast_length=backcast_length,
                         forecast_length=forecast_length,
                         hidden_layer_units=hidden_layer_units,
                         n_blocks=n_blocks,
                         n_blocks_shared=n_blocks_shared,
                         pool_kernels=pool_kernels,
                         ) for _ in range(ensemble_size)])
        self.ema_model = nn.ModuleList([
            NHITS_module(backcast_length=backcast_length,
                         forecast_length=forecast_length,
                         hidden_layer_units=hidden_layer_units,
                         n_blocks=n_blocks,
                         n_blocks_shared=n_blocks_shared,
                         pool_kernels=pool_kernels,
                         ) for _ in range(ensemble_size)])
        
    def training_step(self, batch, batch_idx):
        (
            _, _,
            loss_accuracy, _, _,
            loss_stability, _, _,
            loss, w_accuracy, w_stability, 
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             False,
                             False,
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
        (
            rescaled_forecast, _,
            loss_accuracy, _, _,
            loss_stability, _, _,
            loss, _, _,
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             True,
                             False,
                             )
        
        metrics = {"vloss_a": loss_accuracy,
                   "vloss_s": loss_stability,
                   "vloss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        return rescaled_forecast
    
    def test_step(self, batch, batch_idx):
        (
            rescaled_forecast, rescaled_forecast_lagged,
            _, RMSSE, sMAPE,
            _, RMSSC, sMAPC,
            _, _, _,
            bs,
        ) = self._get_losses(batch,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecast_metric_calculation,
                             True,
                             True,
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

        bs = x["encoder_cont"].shape[0]

        # Batch data
        lookback_window = x["encoder_cont"][:,:,4]
        lookback_window_lagged = x["encoder_cont"][:,:,5]
        forecast_period = x["decoder_cont"][:,:,4]
        forecast_period_lagged = x["decoder_cont"][:,:,5]
        if evaluate:
            mean = x["decoder_cont"][:,:,0]
            std = x["decoder_cont"][:,:,1]
        
        # Data augmentation on-the-fly
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
        if not calculate_metrics:
            scaling_constant_sq_loss = torch.mean(torch.diff(lookback_window)**2, -1) + 1e-3
            scaling_constant_sq_lagged_loss = torch.mean(torch.diff(lookback_window_lagged)**2, -1) + 1e-3
        else:
            scaling_constant_abs = x["encoder_cont"][:,-1,2]
            scaling_constant_sq = x["encoder_cont"][:,-1,3]

        # Obtain forecast for different ensemble members
        model_forecast = torch.zeros((bs,
                                      self.hparams.forecast_length,
                                      self.hparams.ensemble_size
                                      ), 
                                      dtype=torch.float,
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
            
        final_forecast = model_forecast.mean(-1)
        final_forecast_lagged = model_forecast_lagged.mean(-1)

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

        # Compute accuracy losses/metrics
        loss_accuracy = 0
        RMSSE = 0
        sMAPE = 0
        if not calculate_metrics:
            RMSSE_loss = RMSSE_calculation(final_forecast, forecast_period, scaling_constant_sq_loss)
            RMSSE_lagged_loss = RMSSE_calculation(final_forecast_lagged, forecast_period_lagged, scaling_constant_sq_lagged_loss)
            loss_accuracy = torch.mean(0.5 * (RMSSE_loss + RMSSE_lagged_loss))
        else:
            with torch.no_grad():
                rescaled_forecast_period = forecast_period * std + mean
                RMSSE = RMSSE_calculation(rescaled_final_forecast, rescaled_forecast_period, scaling_constant_sq)
                sMAPE = sMAPE_calculation(rescaled_final_forecast, rescaled_forecast_period)
                
        # Compute stability losses/metrics
        loss_stability = 0
        RMSSC = 0
        sMAPC = 0
        if not calculate_metrics:
            loss_stability = RMSSE_calculation(final_forecast[:,:-1], final_forecast_lagged[:,1:], scaling_constant_sq_loss)
        else:
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
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        ensemble_id_update = batch_idx % self.hparams.ensemble_size
        for idx, model in enumerate(self.model):
            if idx != ensemble_id_update:
                for param in model.parameters():
                    param.grad = None
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.hparams.ema_decay * ema_param.data + (1 - self.hparams.ema_decay) * model_param.data
