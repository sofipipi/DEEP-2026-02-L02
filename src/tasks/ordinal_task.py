"""
Tarea ordinal CORAL.

La clase CoralTask ya conecta la perdida y la prediccion con el Trainer.
Las cuatro funciones matematicas son TODO(alumno):

1. labels_to_levels
2. coral_loss
3. effective_number_weights
4. logits_to_ordinal_predictions
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import TrainingConfig
from src.tasks.base import ClassificationTask, LossFunction


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(
        num_classes - 1,
        device=labels.device
    )

    levels = labels.long().unsqueeze(1) > thresholds.unsqueeze(0)

    return levels.float()




def coral_loss(
    logits,
    labels,
    num_classes,
    class_weights=None
):
    levels = labels_to_levels(labels, num_classes)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        levels,
        reduction="none"
    )

    loss = loss.sum(dim=1)

    if class_weights is not None:
        weights = class_weights.to(
            device=logits.device,
            dtype=loss.dtype
        )

        sample_weights = weights[labels.long()]
        loss = loss * sample_weights

    return loss.mean()




def effective_number_weights(
    labels,
    num_classes,
    beta=0.99
):
    labels = np.asarray(labels, dtype=np.int64)

    if not 0 <= beta < 1:
        raise ValueError("beta debe estar entre 0 y 1.")

    counts = np.bincount(
        labels,
        minlength=num_classes
    ).astype(np.float64)

    weights = np.zeros(num_classes, dtype=np.float64)

    present = counts > 0

    if beta == 0:
        weights[present] = 1.0
    else:
        weights[present] = (
            (1.0 - beta) /
            (1.0 - np.power(beta, counts[present]))
        )

    # Normalizar para que las clases presentes tengan
    # un peso promedio igual a 1.
    weights[present] /= weights[present].mean()

    return torch.tensor(weights, dtype=torch.float32)




@torch.no_grad()
def logits_to_ordinal_predictions(
    logits,
    threshold=0.5
):
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).sum(dim=1)

    return predictions.long()




class CoralLoss:
    """Envoltorio invocable de coral_loss con K y pesos fijos por fold."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return coral_loss(logits, labels, self.num_classes, self.class_weights)


class CoralTask(ClassificationTask):
    """
    Clasificación ordinal mediante CORAL.

    Define la pérdida y las predicciones que utilizará
    el Trainer.
    """

    name = "coral"

    def build_criterion(
        self,
        y_train: np.ndarray,
        config: TrainingConfig,
        device: torch.device
    ) -> LossFunction:

        class_weights = None

        if config.use_class_weights:

            class_weights = effective_number_weights(
                y_train,
                self.num_classes,
                beta=config.beta
            ).to(device)

        return CoralLoss(
            self.num_classes,
            class_weights
        )

    def predict(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:

        return logits_to_ordinal_predictions(logits)

