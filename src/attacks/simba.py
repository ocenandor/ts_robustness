# Code from https://github.com/EricKolibacz/replicating-simple-blackbox-attack

import random
import numpy as np
import torch
from torch.linalg import vector_norm
from scipy.fftpack import idct
from abc import ABC, abstractmethod
from typing import Union


# Interface defining the structure of the search vector sets
class SearchVectors(ABC):
    """Abstract class for the set of search vectors"""

    @abstractmethod
    def get_random_vector(self) -> torch.Tensor:
        """Get a random new search vector (without replacement)

        Returns:
            torch.Tensor: search vectors
        """


# Descrete cosine transform search vector set
class DCTSearchVectors(SearchVectors):
    """Search Vectors with base derived from the discrete cosine transform.
    Defined by Guo et al. 2018."""

    def __init__(self, size: torch.Size, ratio: float) -> None:
        if size[0] != 3 and size[0] != 1:
            raise ValueError(f"size = (3, w, h) or size = (1, w, h). Passed image has dimensions {size}")
        self.size = size
        self.frequency_dimensions = [
            (i, j, k) for i in range(3) for j in range(int(size[1] * ratio)) for k in range(int(size[2] * ratio))
        ]

    def get_random_vector(self) -> torch.Tensor:
        if len(self.frequency_dimensions) == 0:
            raise IndexError("No Vectors left")
        dimension = self.frequency_dimensions.pop(random.randrange(len(self.frequency_dimensions)))
        frequency_coefficients = np.zeros(self.size)
        frequency_coefficients[dimension] = 1.0
        return torch.from_numpy(self.idct_2d(frequency_coefficients)).float()

    def idct_2d(self, frequency_coefficients: np.array) -> np.array:
        """2 dimension discrete cosine transform (DCT)

        Args:
            frequency_coefficients (np.array): frequency coefficients with shape (h, w, 3)

        Returns:
            np.array: signal in 2D image space
        """
        return idct(
            idct(frequency_coefficients.T, axis=1, norm="ortho").T,
            axis=2,
            norm="ortho",
        )


# Module containing the class for the cartesian search vector set
class CartesianSearchVectors(SearchVectors):
    """Search Vectors with cartesian base."""

    def __init__(self, size: torch.Size) -> None:
        self.size = size
        self.vector_dimensions = list(np.arange(0, size.numel()))

    def get_random_vector(self) -> torch.Tensor:
        if len(self.vector_dimensions) == 0:
            raise IndexError("No Vectors left")
        dimension = self.vector_dimensions.pop(random.randrange(len(self.vector_dimensions)))
        search_vector = torch.zeros((self.size.numel()))
        search_vector[dimension] = 1
        return search_vector.reshape(self.size)


# Simple Black-box Adversarial Attacks (SimBA) proposed by Guo et al.
def simba(signal: torch.Tensor,
          model,
          step_size: float,
          max_iter=500,
          device='cpu') -> torch.Tensor:

    """Adversarial method from the paper Simple Black-box Adversarial Attacks (Guo et al. 2019)
    Named how it is named in the paper

    Args:
        model: the to-be-attacked model (no access on the parameters)
        image (torch.Tensor): input image
        label (int): correct label of the image (ground truth)
        basis (SearchVectors): set of orthogonal search vectors
        step_size (float): the magnitude of the image pertubation in search vector direction
        max_iter (int): the budget of pertubation allowed

    Returns:
        torch.Tensor: pertubation
        int: number of steps to 
    """
    basis = CartesianSearchVectors(signal.size())
    pertubation: torch.Tensor = torch.zeros(signal.shape).to(device)

    probability, prediction = predict(model, signal.to(device))
    label = prediction.item()
    steps, queries, l2_norm = 0, 0, []
    while prediction.item() == label and steps + 1 < max_iter:
        try:
            search_vector = basis.get_random_vector().to(device)
        except IndexError:
            # print("Evaluated all vectors")
            break
        for alpha in [-step_size, step_size]:
            pertubed_image: torch.Tensor = signal + pertubation + alpha * search_vector

            queries += 1
            probability_perturbed, prediction_perturbed = predict(model, pertubed_image.to(device))

            if probability_perturbed <= probability:
                steps += 1

                pertubation += alpha * search_vector
                if queries % 64 == 0:
                    l2_norm.append(vector_norm(pertubation.cpu()))
                probability = probability_perturbed
                prediction = prediction_perturbed

    l2_norm.append(vector_norm(pertubation.cpu()))

    return pertubation, steps


def predict(model, signal: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
    """Simple helper function to predict class (with probability)"""
    logits = model(signal.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    probability, prediction = torch.topk(probabilities, 1)
    return probability, prediction
