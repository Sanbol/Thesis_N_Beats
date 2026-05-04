import os

from src.methods.NHITSS import LitNHITSS
from src.utils.callbacks import LoadModelWarning, WriteForecastsToCSV, PlotTestPredictions

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

def main():
    if os.environ.get("WANDB_MODE") != "offline":
        wandb.login()
    project_name = "NBEATSS_thesis"

    ##########################
    # EXPERIMENT CONFIGURATION
    ##########################

    # Dataset
    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model
    load_model = False
    update_loaded_model_specific_training_and_eval_hparams = False

    # Model architecture - load existing model or specify model hyperparameters
    if load_model:
        model_id = ""
        checkpoint = "last"
    else:
        backcast_length_multiplier = 8
        forecast_length = 6
        hidden_layer_units = 32
        n_blocks = 3
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True
        unit_variance = True

    # Model training and evaluation
    eval_mode = 'test'
    random_seed = 1
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = 0.0
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = 0.0
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 50
    patience = 1e6
    max_epochs = 10

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False

    ###################################################################################################
    # Do not change anything below this line - only use for running experiment w config specified above
    ###################################################################################################

    L.seed_everything(random_seed, workers=True)

    if load_model:
        path_to_checkpoint = project_name + "/" + model_id + "/checkpoints/" + checkpoint + ".ckpt"
        model = LitNHITSS.load_from_checkpoint(path_to_checkpoint)
        forecast_length = model.hparams["forecast_length"]
        backcast_length_multiplier = model.hparams["backcast_length_multiplier"]
        zero_mean = model.hparams["zero_mean"]
        unit_variance = model.hparams["unit_variance"]
        if update_loaded_model_specific_training_and_eval_hparams:
            model.hparams["lambda_stability"]=lambda_stability
            model.hparams["enforce_nonnegative_forecast_metric_calculation"]=enforce_nonnegative_forecast_metric_calculation
            model.hparams["learning_rate"]=learning_rate
            model.hparams["explr_gamma"]=explr_gamma
            model.hparams["ema_decay"]=ema_decay
    else:
        model = LitNHITSS(
            backcast_length_multiplier=backcast_length_multiplier,
            forecast_length=forecast_length,
            hidden_layer_units=hidden_layer_units,
            n_blocks=n_blocks,
            n_blocks_shared=n_blocks_shared,
            ensemble_size=ensemble_size,
            zero_mean=zero_mean,
            unit_variance=unit_variance,
            lambda_stability=lambda_stability,
            enforce_nonnegative_forecast_metric_calculation=enforce_nonnegative_forecast_metric_calculation,
            learning_rate=learning_rate,
            explr_gamma=explr_gamma,
            ema_decay=ema_decay)

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

    wandb_logger = WandbLogger(project=project_name, log_model=True)
    modelsummary_callback = ModelSummary(max_depth=3)
    if save_forecasts:
        write_forecasts = WriteForecastsToCSV(wandb_logger=wandb_logger)
    if plot_forecasts:
        plot_test_predictions = PlotTestPredictions()
    if load_model:
        load_model_warning = LoadModelWarning(model_id=model_id)

    print("Start model training and evaluation.")
    if eval_mode == 'validation':
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="vloss", mode="min", save_last=True)
        early_stop_callback = EarlyStopping(monitor="vloss", mode="min", patience=int(patience))
        callbacks_list = [
            modelsummary_callback,
            checkpoint_callback,
            early_stop_callback,
        ]
        if save_forecasts:
            callbacks_list.append(write_forecasts)
        if plot_forecasts:
            callbacks_list.append(plot_test_predictions)
        if load_model:
            callbacks_list.append(load_model_warning)
        trainer = Trainer(
            callbacks=callbacks_list,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=max_norm,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            max_epochs=max_epochs,
            limit_train_batches=batches_per_epoch,
        )
        trainer.fit(model, train_dataloader, validation_dataloader)
        if max_epochs == 0:
            trainer.test(model, validation_dataloader_target, verbose=True)
        else:
            trainer.test(model, validation_dataloader_target, "best", verbose=True)
    elif eval_mode == 'test':
        checkpoint_callback = ModelCheckpoint(save_last=True)
        callbacks_list = [
            modelsummary_callback,
            checkpoint_callback,
        ]
        if save_forecasts:
            callbacks_list.append(write_forecasts)
        if plot_forecasts:
            callbacks_list.append(plot_test_predictions)
        if load_model:
            callbacks_list.append(load_model_warning)
        trainer = Trainer(
            callbacks=callbacks_list,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=max_norm,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            max_epochs=max_epochs,
            limit_train_batches=batches_per_epoch,
        )
        trainer.fit(model, trainandvalidation_dataloader)
        if max_epochs == 0:
            trainer.test(model, test_dataloader_target, verbose=True)
        else:
            trainer.test(model, test_dataloader_target, "last", verbose=True)

    if load_model:
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
        if update_loaded_model_specific_training_and_eval_hparams:
            wandb_logger.experiment.config.update({
                "lambda_stability": lambda_stability,
                "enforce_nonnegative_forecast_metric_calculation": enforce_nonnegative_forecast_metric_calculation,
                "learning_rate": learning_rate,
                "explr_gamma": explr_gamma,
                "ema_decay": ema_decay,
            }, allow_val_change=True)
    else:
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

    wandb.finish()

if __name__ == '__main__':
    main()
