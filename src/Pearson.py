from pytorch_forecasting.metrics import MultiHorizonMetric


class Pearson(MultiHorizonMetric):
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
            losses: Pearson correlation coefficient, shape:[prediction_length, Quantile]
        
        """
        _, T, _ = y_pred.shape
        losses = []
        for t in range(T):
            loss_t = []
            for i, q in enumerate(self.quantiles):
                x1 = x[..., t, i].unsqueeze(0)
                y1 = y[..., t].unsqueeze(0)
                z = torch.cat((x1, y1), dim=0)
                errors = torch.corrcoef(z)[0][1]
                loss_t.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
            loss_t = torch.cat(loss_t, dim=0)
            losses.append(loss_t.unsqueeze(0))
        losses = torch.cat(losses, dim=0)

        return losses
