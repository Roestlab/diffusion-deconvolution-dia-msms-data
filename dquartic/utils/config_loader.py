import json


def load_train_config(config_path: str, **kwargs):
    """
    Load the training configuration from a JSON file and override the parameters with the provided keyword arguments if passed.

    Parameters:
        config_path (str): The path to the JSON configuration file.
        **kwargs: Keyword arguments to override the configuration parameters.

    Returns:
        dict: The loaded configuration parameters with overridden values.
    """
    with open(config_path, "r") as f:
        config_params = json.load(f)

    # Override the config params with the keyword arguments
    if "ms2_data_path" in kwargs:
        if kwargs["ms2_data_path"] is not None:
            config_params["data"]["ms2_data_path"] = kwargs["ms2_data_path"]

    if "ms1_data_path" in kwargs:
        if kwargs["ms1_data_path"] is not None:
            config_params["data"]["ms1_data_path"] = kwargs["ms1_data_path"]

    if "batch_size" in kwargs:
        if kwargs["batch_size"] is not None:
            config_params["model"]["batch_size"] = kwargs["batch_size"]

    if "checkpoint_path" in kwargs:
        if kwargs["checkpoint_path"] is not None:
            config_params["model"]["checkpoint_path"] = kwargs["checkpoint_path"]

    if "use_wandb" in kwargs:
        if kwargs["use_wandb"] is not None:
            config_params["wandb"]["use_wandb"] = kwargs["use_wandb"]

    if "threads" in kwargs:
        if kwargs["threads"] is not None:
            config_params["threads"] = kwargs["threads"]

    return config_params


def generate_train_config(config_path: str):
    """
    Generate a training configuration file.

    Parameters:
        config_path (str): The path to save the generated JSON configuration file.
    """
    full_config = {
        "data": {
            "ms2_data_path": "data/MS2_data.csv",
            "ms1_data_path": "data/MS1_data.csv",
            "normalize": "minmax",
        },
        "model": {
            "checkpoint_path": "best_model.ckpt",
            "num_epochs": 10000,
            "warmup_epochs": 5,
            "batch_size": 1,
            "learning_rate": 0.00001,
            "num_timesteps": 1000,
            "beta_start": 0.001,
            "beta_end": 0.00125,
            "pred_type": "eps",
            "ms1_loss_weight": 0.0,
            "use_model": "Unet1D",
            "CustomTransformer": {
                "input_dim": 40000,
                "hidden_dim": 1024,
                "num_heads": 8,
                "num_layers": 8,
            },
            "Unet1D": {
                "dim": 4,
                "channels": 1,
                "dim_mults": [1, 2, 2, 3, 3, 4, 4],
                "cond_channels": 1,
                "cond_init_dim": 4,
                "has_condition": True,
            },
        },
        "wandb": {
            "use_wandb": True,
            "wandb_project": "dquartic",
            "wandb_name": None,
            "wandb_id": None,
            "wandb_resume": None,
            "wandb_architecture": "DDIM(Unet1D)",
            "wandb_dataset": "MS2",
            "wandb_mode": "offline",
        },
        "threads": 4,
    }

    # Save the configuration to a JSON file
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=4)
