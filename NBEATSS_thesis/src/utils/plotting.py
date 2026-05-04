import torch
import numpy as np
import plotly.graph_objects as go

def plot_forecasts(lookback_window, forecasts, forecast_period, sample, epoch):
    """
    Torch.Tensor input shapes:
    lookback_window = batch_size x backcast_length
    forecasts = batch_size x forecast_length
    forecast_period = batch_size x forecast_length
    """

    lookback_periods = lookback_window.shape[1]
    forecast_periods = forecasts.shape[1]

    lookback_window = lookback_window[sample,:].numpy()
    forecasts = forecasts[sample,:].numpy()
    forecast_period = forecast_period[sample,:].numpy()

    fig = go.Figure()

    # Plot lookback window
    fig.add_trace(go.Scatter(x=np.arange(lookback_periods), 
                             y=lookback_window, 
                             mode='lines', name='Lookback Window', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=np.arange(lookback_periods, lookback_periods + forecast_periods), 
                             y=forecasts,
                             mode='lines', name=f'Forecast', line=dict(color='mediumblue')))

    # Plot forecast period
    fig.add_trace(go.Scatter(x=np.arange(lookback_periods, lookback_periods + forecast_periods), 
                             y=forecast_period, mode='lines', name='Forecast Period', line=dict(color='lawngreen', width=3)))

    if epoch is None:
        fig.update_layout(title=f"Best model",
                          xaxis_title='Periods',
                          yaxis_title='Values',
                          showlegend=False)
    else:
        fig.update_layout(title=f"Epoch {epoch}",
                          xaxis_title='Periods',
                          yaxis_title='Values',
                          showlegend=False)
    
    return fig
