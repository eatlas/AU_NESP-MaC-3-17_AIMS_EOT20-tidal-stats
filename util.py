import yaml

def load_config(config_path: str, required_params: list[str]) -> dict:
    """
    Load configuration parameters from a YAML file and ensure that all expected parameters are present.
    If any required parameter is missing, an error is raised indicating which variable is missing.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration file does not contain a valid YAML mapping.")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file {config_path}: {e}")
    
    missing = [param for param in required_params if param not in config]
    if missing:
        raise ValueError(
            f"Missing configuration parameter(s): {', '.join(missing)}. "
            "Please check the configuration file and refer to the documentation for valid options."
        )
    
    return config

def print_config(config: dict):
    """Print the configuration parameters as a formatted list."""
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()