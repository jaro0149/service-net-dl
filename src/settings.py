from pydantic import Field
from pydantic_settings import BaseSettings


class TrainingSettings(BaseSettings):
    """Configuration settings for model training parameters."""

    n_batch_size: int = Field(default=64, description="Training batch size")
    learning_rate: float = Field(default=0.15, description="Learning rate for training")
    n_epochs: int = Field(default=28, description="Number of training epochs")
    report_every: int = Field(default=1, description="Report metrics every N epochs")

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "TRAINING_"
        case_sensitive = False


class ModelSettings(BaseSettings):
    """Configuration settings for model architecture."""

    n_hidden_units: int = Field(default=128, description="Number of hidden units in the RNN")

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "MODEL_"
        case_sensitive = False


class DatasetSettings(BaseSettings):
    """Configuration settings for dataset splitting and processing."""

    train_ratio: float = Field(default=0.85, description="Ratio of training dataset split")
    test_ratio: float = Field(default=0.15, description="Ratio of test dataset split")

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "DATASET_"
        case_sensitive = False
