import numpy as np
import torch


class UncertaintyHandler:
    def __init__(self, predictions):
        """
        :param predictions: torch.tensor(# of batch size, # of classes, h, w)
        """
        self.predictions = predictions

    def separate_uncertainty(self) -> np.array:
        """
        :return: aleatoric and epistemic uncertainty: np.array(# of classes x h x w)
        """
        if not hasattr(self, "aleatoric") or not hasattr(self, "epistemic"):
            self._separate_uncertainty()
        return self.aleatoric[0], self.epistemic[0]

    def get_entropy(self) -> torch.Tensor:
        """
        :return: entropy: torch.tensor(h x w)
        """
        if not hasattr(self, "entropy"):
            self._entropy()
        return self.entropy[0]

    def get_BALD(self):
        """
        :return: BALD: torch.tensor(h x w)
        """
        if not hasattr(self, "BALD"):
            self._BALD()
        return self.BALD[0]

    def get_BatchBALD(self):
        raise NotImplementedError

    def get_sparse_subset_approximation(self):
        raise NotImplementedError

    def _separate_uncertainty(self):
        assert hasattr(self, "MC_predictions"), "Monte carlo samples are required for separating uncertainty"

        MC_predictions_np = np.array(self.MC_predictions.cpu())
        predicted_prob = np.mean(MC_predictions_np, axis=1)
        prob_centered = MC_predictions_np - predicted_prob

        self.aleatoric = np.mean(MC_predictions_np - MC_predictions_np ** 2, axis=1)
        self.epistemic = np.mean(prob_centered ** 2, axis=1)

    def _entropy(self):
        self.entropy = -(torch.log(self.predictions) * self.predictions).sum(1)

    def _BALD(self):
        assert hasattr(self, "MC_predictions"), "Monte carlo samples are required for calcuating BALD"

        if not hasattr(self, "entropy"):
            self._entropy()
        cond_entropy = -(self.MC_predictions * torch.log(self.MC_predictions))

        self.BALD = self.entropy - cond_entropy.mean(1).sum(1)
