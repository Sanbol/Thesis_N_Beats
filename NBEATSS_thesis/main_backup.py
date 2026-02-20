# 1: Evaluation
# For forecast accuracy metric calculation: only forecasts are used so that there is no data leakage.
# For loss calculation, both the forecasts and the forecasts_lagged are used.
# For forecast stability metric calculation: both forecasts and forecasts_lagged are used, however,
# we do not use the first value (one-step ahead forecast) of forecasts_lagged to avoid data leakage.
# For metric calculations, we can opt to zero out negative forecasts.

# 2: Dataset creation
# Prepending the time series in order to make sure that each time series is long enough to match the model config
# can be time consuming (for high backcast_length_multiplier).
# Datasets for specific config are saved in data/processed so that they can be reused.

# 3: Model architecure
# Model is as close as possible to original N-BEATS architecture.

##############
# LOAD MODULES
##############

# Install Python 3.12.11
# pip install torch
# pip install lightning
# pip install plotly
# pip install pandas
# pip install wandb
# pip install pytorch-forecasting==1.1.1

# Standard library
import os

# Proprietary
from src.methods.NBEATSS import LitNBEATSS 
from src.utils.callbacks import LoadModelWarning, WriteForecastsToCSV, PlotTestPredictions

# Third party
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

def main():
    ##########################
    # CHECK CONNECTION W WANDB
    ##########################

    wandb.login()
    project_name = "NBEATSS_thesis"

    ##########################
    # EXPERIMENT CONFIGURATION
    ##########################

    # Dataset
    dataset = "M3" #"M4"
    subset = "Monthly"
    dataset_id = "M3M" #"M4M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model
    load_model = True #True or False
    update_loaded_model_specific_training_and_eval_hparams = True #True or False, only relevant when a model is loaded

    # Model architecture - load existing model or specify model hyperparameters
    if load_model:
        model_id = "dxpmjk9i"
        checkpoint = "last" #"best" or "last"
    else: # hparams below are ignored if model is loaded
        backcast_length_multiplier = 8
        forecast_length = 6
        hidden_layer_units = 32 #512
        n_blocks = 3 #5
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True # model argument as it affects model weights
        unit_variance = True # model argument as it affects model weights

    # Model training and evaluation
    eval_mode = 'test' #'validation' or 'test'
    random_seed = 2956
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6 #10
    batch_size = 32 #512
    num_workers = 0  # Set to 0 to avoid multiprocessing issues on Windows
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams: 
    # model-specific training and evaluation hparams below are ignored if (load_model == True) AND (update_loaded_model_specific_training_and_eval_hparams == False)
        lambda_stability = 0.02  # Added regularization for forecast stability
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-5  # Reduced for fine-tuning (prevents catastrophic forgetting)
        explr_gamma = 0.97  # Learning rate decay: learning_rate*(explr_gamma)**epoch
        ema_decay = 0.99  # Exponential moving average for smoother updates
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 50 #250
    patience = 1e6 #20 #1e6 for specific number of epochs
    max_epochs = 15  # Extended for better convergence with lower learning rate

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium") # if we run on GPU
    save_forecasts = False #False
    plot_forecasts = False #False

    ###################################################################################################
    # Do not change anything below this line - only use for running experiment w config specified above
    ###################################################################################################

    L.seed_everything(random_seed, workers=True)

    # INIT MODEL
    if load_model == True:
        path_to_checkpoint = project_name + "/" + model_id + "/checkpoints/" + checkpoint + ".ckpt"
        NBEATSS = LitNBEATSS.load_from_checkpoint(path_to_checkpoint)
        # Extract model-dependent hyperparameters required for data loading
        forecast_length = NBEATSS.hparams["forecast_length"]
        backcast_length_multiplier = NBEATSS.hparams["backcast_length_multiplier"]
        zero_mean = NBEATSS.hparams["zero_mean"]
        unit_variance = NBEATSS.hparams["unit_variance"]
        if update_loaded_model_specific_training_and_eval_hparams == True:
            # Update hyperparameters that do not affect the model architecture
            NBEATSS.hparams["lambda_stability"]=lambda_stability
            NBEATSS.hparams["enforce_nonnegative_forecast_metric_calculation"]=enforce_nonnegative_forecast_metric_calculation
            NBEATSS.hparams["learning_rate"]=learning_rate
            NBEATSS.hparams["explr_gamma"]=explr_gamma
            NBEATSS.hparams["ema_decay"]=ema_decay
    else:
        NBEATSS = LitNBEATSS(
            # Model hypers
            backcast_length_multiplier=backcast_length_multiplier,
            forecast_length=forecast_length,
            hidden_layer_units=hidden_layer_units,
            n_blocks=n_blocks,
            n_blocks_shared=n_blocks_shared,
            ensemble_size=ensemble_size,
            zero_mean=zero_mean,
            unit_variance=unit_variance,
            # Optim hypers
            lambda_stability=lambda_stability,
            enforce_nonnegative_forecast_metric_calculation=enforce_nonnegative_forecast_metric_calculation,
            learning_rate=learning_rate,
            explr_gamma=explr_gamma,
            ema_decay=ema_decay)
        
    # LOAD DATA(LOADERS)
    if dataset == "M3":
        from src.data.M3 import load_data
    if dataset == "M4":
        from src.data.M4 import load_data

    train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target = None, None, None, None, None
    if eval_mode == "validation":
        train_dataloader, validation_dataloader, validation_dataloader_target, _, _ = load_data(
            subset=subset,
            test_mode_nrows=test_mode_nrows,
            backcast_length_multiplier=backcast_length_multiplier,
            forecast_length=forecast_length,
            validation_periods=validation_periods,
            test_periods=test_periods,
            zero_mean=zero_mean,
            unit_variance=unit_variance,
            forecasting_origin_range_multiplier=int(forecasting_origin_range_multiplier),
            batch_size=batch_size,
            num_workers=num_workers)
    else:
        _, _, _, trainandvalidation_dataloader, test_dataloader_target = load_data(
            subset=subset,
            test_mode_nrows=test_mode_nrows,
            backcast_length_multiplier=backcast_length_multiplier,
            forecast_length=forecast_length,
            validation_periods=validation_periods,
            test_periods=test_periods,
            zero_mean=zero_mean,
            unit_variance=unit_variance,
            forecasting_origin_range_multiplier=int(forecasting_origin_range_multiplier),
            batch_size=batch_size,
            num_workers=num_workers)
        
    # Check output dataloaders
    # x_train, y_train = next(iter(train_dataloader))
    # for batch_idx, (x_train, _) in enumerate(train_dataloader):
    #     print("Batch:", batch_idx, "Input shape:", x_train["encoder_cont"].shape)
    # # Check randomness over batches and epochs
    # for batch_idx, (x_train, _) in enumerate(train_dataloader):
    #     if batch_idx < 2 or batch_idx==91:
    #         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
    #         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
    #         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
    # for batch_idx, (x_train, _) in enumerate(train_dataloader):
    #     if batch_idx < 2 or batch_idx==91:
    #         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
    #         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
    #         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
    # for batch_idx, (x_train, _) in enumerate(train_dataloader):
    #     if batch_idx < 2 or batch_idx==91:
    #         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
    #         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
    #         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
    # for batch_idx, (x_validation, _) in enumerate(validation_dataloader):
    #     print("Batch:", batch_idx, "Input shape:", x_validation["encoder_cont"].shape)

    # CREATE TRAINER
    # Create wandb logger and log config vars and hyperparameters that are not yet automatically logged to wandb
    wandb_logger = WandbLogger(project=project_name, log_model=True)
    # Init callbacks that will be used in Trainer
    modelsummary_callback = ModelSummary(max_depth=3)#-1)
    if save_forecasts == True:
        write_forecasts = WriteForecastsToCSV(wandb_logger=wandb_logger)
    if plot_forecasts == True:
        plot_test_predictions = PlotTestPredictions()
    if load_model == True:
        load_model_warning = LoadModelWarning(model_id=model_id)

    # TRAIN AND EVALUATE MODEL
    print("Start model training and evaluation.")
    if eval_mode == 'validation':
        # Init additional callbacks that will be used in Trainer
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="vloss", mode="min", save_last=True)
        early_stop_callback = EarlyStopping(monitor="vloss", mode="min", patience=int(patience))
        callbacks_list = [
            modelsummary_callback,
            checkpoint_callback,
            early_stop_callback,
        ]
        if save_forecasts == True:
            callbacks_list.append(write_forecasts)
        if plot_forecasts == True:
            callbacks_list.append(plot_test_predictions)
        if load_model == True:
            callbacks_list.append(load_model_warning)
        # Init Trainer
        trainer = Trainer(
            callbacks=callbacks_list,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=max_norm,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            #deterministic=False,
            max_epochs=max_epochs,
            #val_check_interval=100,
            limit_train_batches=batches_per_epoch,
            #log_every_n_steps=1,
        )
        # Fit
        trainer.fit(NBEATSS, train_dataloader, validation_dataloader)
        # Evaluate
        if max_epochs == 0:
            trainer.test(NBEATSS, validation_dataloader_target, verbose=True)
        else: # best vs last --> best for standard hparam tuning, last for additional hparam tuning with fixed number of learning iterations 
            trainer.test(NBEATSS, validation_dataloader_target, "best", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log        
            #trainer.test(NBEATSS, validation_dataloader_target, "last", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log
    elif eval_mode ==  'test':
        # Init additional callbacks that will be used in Trainer
        checkpoint_callback = ModelCheckpoint(save_last=True)
        callbacks_list = [
            modelsummary_callback,
            checkpoint_callback,
        ]
        if save_forecasts == True:
            callbacks_list.append(write_forecasts)
        if plot_forecasts == True:
            callbacks_list.append(plot_test_predictions)
        if load_model == True:
            callbacks_list.append(load_model_warning)
        # Init Trainer
        trainer = Trainer(
            callbacks=callbacks_list,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=max_norm,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            #deterministic=False,
            max_epochs=max_epochs,
            #val_check_interval=100,
            limit_train_batches=batches_per_epoch,
        )
        # Fit
        trainer.fit(NBEATSS, trainandvalidation_dataloader)
        # Evaluate
        if max_epochs == 0:
            trainer.test(NBEATSS, test_dataloader_target, verbose=True)
        else:
            trainer.test(NBEATSS, test_dataloader_target, "last", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log

    # LOG HPARAM (UPDATES)
    if load_model == True:
        wandb_logger.experiment.config.update({
            "dataset": dataset_id,
            "random_seed": random_seed,
            "origin_range": forecasting_origin_range_multiplier,
            "batch_size": batch_size,
            "patience": patience,
            "max_epochs": max_epochs,
            "max_norm": max_norm,
            "eval_mode": eval_mode,
            "n_batches": batches_per_epoch,
        })
        if update_loaded_model_specific_training_and_eval_hparams == True:
            wandb_logger.experiment.config.update({
                "lambda_stability": lambda_stability,
                "enforce_nonnegative_forecast_metric_calculation": enforce_nonnegative_forecast_metric_calculation,
                "learning_rate": learning_rate,
                "explr_gamma": explr_gamma,
                "ema_decay": ema_decay,
            }, allow_val_change=True)
    else:
        wandb_logger.experiment.config["dataset"] = dataset_id
        wandb_logger.experiment.config["random_seed"] = random_seed
        wandb_logger.experiment.config["origin_range"] = forecasting_origin_range_multiplier
        wandb_logger.experiment.config["batch_size"] = batch_size
        wandb_logger.experiment.config["patience"] = patience
        wandb_logger.experiment.config["max_epochs"] = max_epochs
        wandb_logger.experiment.config["max_norm"] = max_norm
        wandb_logger.experiment.config["eval_mode"] = eval_mode
        wandb_logger.experiment.config["n_batches"] = batches_per_epoch

    wandb.finish()

if __name__ == '__main__':
    main()
