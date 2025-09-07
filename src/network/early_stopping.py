import copy
import logging

from torch.nn import Module

from settings import EarlyStoppingSettings, MonitorType

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to stop training when the validation metric stops improving."""

    def __init__(self, settings: EarlyStoppingSettings | None = None) -> None:
        """Create an instance of the early stopper with zero counters.

        :param settings: Early stopping configuration settings
        """
        if settings is None:
            self.settings = EarlyStoppingSettings()
        else:
            self.settings = settings

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score: float, model: Module) -> bool:
        """Check if training should stop based on the current score.

        :param score: Current validation score
        :param model: Model to save if it's the best so far
        :return: True if training should stop, False otherwise
        """
        if self.settings.monitor == MonitorType.ACCURACY:
            improved = self.best_score is None or score > self.best_score + self.settings.min_delta
        else:
            improved = self.best_score is None or score < self.best_score - self.settings.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            logger.debug("New best %s: %.4f", self.settings.monitor.value, score)
        else:
            self.counter += 1
            logger.debug("No improvement in %s for %d/%d epochs",
                         self.settings.monitor.value, self.counter, self.settings.patience)

            if self.counter >= self.settings.patience:
                self.early_stop = True
                logger.info("Early stopping triggered! Best %s: %.4f", self.settings.monitor.value, self.best_score)

        return self.early_stop
