# LOAD MODULES

# Standard library

# Proprietary

# Third party
import torch

def RMSSE_calculation(forecast: torch.Tensor, 
                      actual: torch.Tensor,
                      scaling_constant: torch.Tensor):
     fl = forecast.shape[-1]
     RMSSE_instance = torch.sqrt(torch.mean(
          ((actual - forecast)**2 / scaling_constant.unsqueeze(-1).expand(-1, fl)), 
          dim = -1))
     RMSSE_instance = torch.clamp(RMSSE_instance, 0, 5)
     RMSSE = torch.mean(RMSSE_instance)
     return RMSSE

def sMAPE_calculation(forecast: torch.Tensor, 
                      actual: torch.Tensor):
     sMAPE_instance = 200 * torch.mean(torch.abs(actual - forecast) / (torch.abs(actual) + torch.abs(forecast) + 1e-3), dim = -1)
     sMAPE = torch.mean(sMAPE_instance)
     return sMAPE
