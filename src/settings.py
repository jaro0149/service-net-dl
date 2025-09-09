from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainingSettings(BaseSettings):
    """Configuration settings for model training parameters."""

    seed: int = Field(default=2025, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.85, description="Ratio of training dataset split")
    data_dir: str = Field(default="./data/services", description="Directory for storing training and testing datasets")
    shuffle_buffer_size: int = Field(default=128, description="Size of the buffer used for shuffling")
    n_batch_size: int = Field(default=64, description="Training batch size")
    learning_rate: float = Field(default=0.01, description="Learning rate for training")
    n_epochs: int = Field(default=512, description="Number of training epochs")
    report_every: int = Field(default=1, description="Report metrics every N epochs")

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "TRAINING_"
        case_sensitive = False


class LstmModelSettings(BaseSettings):
    """Configuration settings for LSTM-based model architecture."""

    n_hidden_units: int = Field(default=256, description="Number of hidden units in the one LSTM layer")
    n_layers: int = Field(default=1, description="Number of layers in the LSTM")
    dropout_p: float = Field(default=0.5, description="Dropout probability for the LSTM")

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "MODEL_"
        case_sensitive = False


class CharAugmentationSettings(BaseSettings):
    """Configuration settings for character-level text augmentation."""

    keyboard_aug_char_p: float = Field(
        default=0.1,
        description="Probability of keyboard typos augmentation per character",
    )
    ocr_aug_char_p: float = Field(
        default=0.05,
        description="Probability of OCR errors augmentation per character",
    )
    random_char_aug_p: float = Field(
        default=0.05,
        description="Probability of random character substitution per character",
    )
    random_char_aug_min: int = Field(
        default=1,
        description="Minimum number of characters to augment",
    )
    random_char_aug_max: int = Field(
        default=3,
        description="Maximum number of characters to augment",
    )

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "CHAR_AUG_"
        case_sensitive = False


class WordAugmentationSettings(BaseSettings):
    """Configuration settings for word-level text augmentation."""

    synonym_aug_p: float = Field(
        default=0.1,
        description="Probability of synonym replacement per word",
    )
    word_delete_aug_p: float = Field(
        default=0.05,
        description="Probability of word deletion",
    )
    word_swap_aug_p: float = Field(
        default=0.05,
        description="Probability of word swapping",
    )
    word_crop_aug_p: float = Field(
        default=0.05,
        description="Probability of word cropping",
    )
    word_aug_min: int = Field(
        default=2,
        description="Minimum number of words to augment",
    )
    word_aug_max: int = Field(
        default=5,
        description="Maximum number of words to augment",
    )
    synonym_aug_src: str = Field(
        default="wordnet",
        description="Source for synonym augmentation",
    )

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "WORD_AUG_"
        case_sensitive = False


class TextAugmentationSettings(BaseSettings):
    """Configuration settings for text augmentation parameters."""

    char_augmentation: CharAugmentationSettings = Field(default_factory=CharAugmentationSettings)
    word_augmentation: WordAugmentationSettings = Field(default_factory=WordAugmentationSettings)

    char_level_probability: float = Field(
        default=0.5,
        description="Probability of choosing character-level vs word-level augmentation",
    )

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "TEXT_AUG_"
        case_sensitive = False


class MonitorType(Enum):
    """Enum for monitoring metric types."""

    ACCURACY = "accuracy"
    LOSS = "loss"


class EarlyStoppingSettings(BaseSettings):
    """Configuration settings for early stopping functionality."""

    enabled: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, description="Number of epochs to wait after the last improvement")
    min_delta: float = Field(default=0.0, description="Minimum change to qualify as an improvement")
    monitor: MonitorType = Field(default=MonitorType.ACCURACY, description="Metric to monitor (accuracy or loss)")
    start_from_metric: float = Field(
        default=0.6,
        description="Initial metric value to start monitoring improvement from",
    )

    class Config:
        """Configuration class for defining environment settings and options for a particular application."""

        env_prefix = "EARLY_STOPPING_"
        case_sensitive = False
