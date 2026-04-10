from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    db_path: str = "data/inventory.db"
    predictions_db_path: str = "data/predictions.db"

    # Models
    models_dir: str = "models"
    model_registry_path: str = "models/registry.json"

    # Retraining
    retrain_min_improvement: float = 0.01

    # Logging
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }


settings = Settings()
