import logging
import random
from typing import Any

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nltk
from nlpaug import Augmenter
from nlpaug.util import Action

from settings import TextAugmentationSettings

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Text augmentation pipeline with configurable parameters."""

    def __init__(self, settings: TextAugmentationSettings | None = None) -> None:
        """Initialize the TextAugmenter instance.

        This class serves as a tool for performing text augmentation.
        It sets up augmentation settings and prepares augmenters for both character-level and
        word-level modifications.

        :param settings: Configuration for text augmentation.
        """
        self.settings = settings if settings else TextAugmentationSettings()
        self.char_augmenters = self._initialize_char_augmenters()
        self.word_augmenters = self._initialize_word_augmenters()
        nltk.download("averaged_perceptron_tagger_eng")

    def _initialize_char_augmenters(self) -> list[Augmenter]:
        char_settings = self.settings.char_augmentation
        return [
            nac.KeyboardAug(
                aug_char_p=char_settings.keyboard_aug_char_p,
                aug_char_min=char_settings.random_char_aug_min,
                aug_char_max=char_settings.random_char_aug_max,
            ),
            nac.OcrAug(
                aug_char_p=char_settings.ocr_aug_char_p,
                aug_char_min=char_settings.random_char_aug_min,
                aug_char_max=char_settings.random_char_aug_max,
            ),
            nac.RandomCharAug(
                action=Action.SUBSTITUTE,
                aug_char_p=char_settings.random_char_aug_p,
                aug_char_min=char_settings.random_char_aug_min,
                aug_char_max=char_settings.random_char_aug_max,
            ),
        ]

    def _initialize_word_augmenters(self) -> list[Augmenter]:
        word_settings = self.settings.word_augmentation
        return [
            naw.SynonymAug(
                aug_src=word_settings.synonym_aug_src,
                aug_p=word_settings.synonym_aug_p,
                aug_min=word_settings.word_aug_min,
                aug_max=word_settings.word_aug_max,
            ),
            naw.RandomWordAug(
                action=Action.DELETE,
                aug_p=word_settings.word_delete_aug_p,
                aug_min=word_settings.word_aug_min,
                aug_max=word_settings.word_aug_max,
            ),
            naw.RandomWordAug(
                action=Action.SWAP,
                aug_p=word_settings.word_swap_aug_p,
                aug_min=word_settings.word_aug_min,
                aug_max=word_settings.word_aug_max,
            ),
            naw.RandomWordAug(
                action=Action.CROP,
                aug_p=word_settings.word_crop_aug_p,
                aug_min=word_settings.word_aug_min,
                aug_max=word_settings.word_aug_max,
            ),
        ]

    def augment_text(self, text: str) -> str:
        """Augments the provided text by randomly applying either character-level or word-level augmentation.

        Augmentations are applied based on a predefined probability.
        If the text is too short, it defaults to character-level augmentation.
        If augmentation fails for any reason, the original text is returned as a fallback.

        :param text: The input text to be augmented.
        :return: The augmented text.
        """
        try:
            # Randomly choose between character and word level augmentation
            if random.random() < self.settings.char_level_probability:
                # Apply character-level augmentation
                augmenter = random.choice(self.char_augmenters)
                return self._decapsulate_result(augmenter.augment(text))
            # Apply word-level augmentation (skip if text is too short)
            if len(text.split()) > 1:
                augmenter = random.choice(self.word_augmenters)
                result = augmenter.augment(text)
                return self._decapsulate_result(result)
            # For single words, fall back to character augmentation
            augmenter = random.choice(self.char_augmenters)
            result = augmenter.augment(text)
            return self._decapsulate_result(result)
        except Exception as e: # noqa: BLE001
        # If augmentation fails, return the original text
            logger.warning("Augmentation failed for '%s': %s", text, e)
            return text

    @staticmethod
    def _decapsulate_result(result: Any) -> str: # noqa: ANN401
        entry = result[0] if isinstance(result, list) else result
        if isinstance(entry, str):
            return entry
        msg = f"Unexpected result type: {type(entry)}"
        raise ValueError(msg)
