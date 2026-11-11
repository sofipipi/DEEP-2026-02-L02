
"""
Experimento 2: Conv1D + MLP con cabeza ordinal CORAL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import Conv1DBackbone


class CoralLayer(nn.Module):
    """
    Capa ordinal CORAL.

    Genera K-1 logits para una clasificación ordinal
    con K clases.

    Todos los umbrales comparten el mismo vector de pesos.
    Los sesgos se parametrizan para mantener su orden.
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes debe ser al menos 2.")

        self.input_size = input_size
        self.num_classes = num_classes

        # Puntaje latente compartido por todos los umbrales.
        self.score = nn.Linear(
            input_size,
            1,
            bias=False
        )

        # Primer sesgo o umbral.
        self.bias_0 = nn.Parameter(torch.zeros(1))

        # Diferencias entre umbrales.
        # Hay K-2 diferencias para K-1 umbrales.
        self.raw_deltas = nn.Parameter(
            torch.zeros(num_classes - 2)
        )


        

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor de características (B, input_size).

        Returns:
            Logits ordinales de forma (B, num_classes - 1).
        """

        # Puntaje compartido: (B, 1).
        score = self.score(inputs)

        # Softplus garantiza diferencias estrictamente positivas.
        deltas = F.softplus(self.raw_deltas)

        # Construimos los sesgos en orden decreciente.
        remaining_biases = (
            self.bias_0 - torch.cumsum(deltas, dim=0)
        )

        biases = torch.cat([
            self.bias_0,
            remaining_biases
        ])

        # Cada columna corresponde a un umbral ordinal.
        logits = score + biases.unsqueeze(0)

        return logits


class Conv1DCoralNet(nn.Module):
    """
    Modelo Conv1D + MLP con cabeza ordinal CORAL.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Extractor de características compartido.
        self.backbone = Conv1DBackbone(
            num_features=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Cabeza ordinal CORAL.
        self.head = CoralLayer(
            input_size=self.backbone.output_dim,
            num_classes=num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: datos de entrada (B, num_features).

        Returns:
            Logits CORAL de forma (B, num_classes - 1).
        """

        # Extraer características.
        features = self.backbone(inputs)

        # Calcular logits ordinales.
        logits = self.head(features)

        return logits
