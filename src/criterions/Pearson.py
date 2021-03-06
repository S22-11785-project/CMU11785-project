import torch
from pytorch_forecasting.metrics import MultiHorizonMetric


class Pearson_train(MultiHorizonMetric):
    """
    Quantile loss with Pearson
    
    calculate the correlation coefficient between two 1D series of multiple variables and observations
    x: variables at future time t with quantile q  
    y: observations at future time t
    i,e. corrcoef(x[:, t, q], y[:, t])
    """

    def __init__(
        self,
        quantiles: list[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss with Pearson

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: output from tft, shape:[Batch_size, prediction_length, Quantile]
            target: true value, shape:[Batch_size, prediction_length]
        output:
            losses: Pearson correlation coefficient, shape:[Batch_size, prediction_length, Quantile]
                    for each y_pred in the batch, the losses is the same,
                    i,e. losses[i, ...] = losses[j, ...]
        
        """
        _, T, _ = y_pred.shape
        losses = []
        for t in range(T):
            loss_t = []
            for i, q in enumerate(self.quantiles):
                y1 = y_pred[..., t, i].unsqueeze(0)
                t1 = target[..., t].unsqueeze(0)
                z = torch.cat((y1, t1), dim=0)
                errors = torch.corrcoef(z)[0][1]
                loss_t.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
            loss_t = torch.cat(loss_t, dim=0)
            losses.append(loss_t.unsqueeze(0))
        losses = torch.cat(losses, dim=0)

        return torch.ones_like(y_pred) * losses
    
class Pearson_test(MultiHorizonMetric):
    """
    Quantile loss with Pearson

    reshape two 2D series of multiple variables and observations as two 1D series
    x[B, T, q], y[B, T] -> x[B * T, q], y[B * T]
    then calculate the correlation coefficient between them
    x: variables with quantile q
    y: observations
    i,e. corrcoef(x[B * T, q], y[B * T])
    """

    def __init__(
        self,
        quantiles: list[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss with Pearson

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: output from tft, shape:[Batch_size, prediction_length, Quantile]
            target: true value, shape:[Batch_size, prediction_length]
        output:
            losses: Pearson correlation coefficient, shape:[Batch_size, prediction_length, Quantile]
                    for each Quantile, the losses is the same,
                    i,e. losses[i1,j1, ...] = losses[i2,j2, ...]

        """
        B, T, Q = y_pred.shape
        y1 = y_pred.reshape(1, B*T, Q)
        t1 = target.reshape(1, B*T)
        losses = []
        for i, q in enumerate(self.quantiles):
            z = torch.cat((y1[..., i], t1), dim=0)
            errors = torch.corrcoef(z)[0][1]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=0)

        return torch.ones_like(y_pred) * losses
