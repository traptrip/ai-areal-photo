import torch


class RegressionLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        # input shape (batch_size, 5)
        batch_size = y_pred.shape[0]

        x_center_true = (y_true[..., 0] + y_true[..., 2]) / 2
        y_center_true = (y_true[..., 1] + y_true[..., 3]) / 2
        x_center_pred = (y_pred[..., 0] + y_pred[..., 2]) / 2
        y_center_pred = (y_pred[..., 1] + y_pred[..., 3]) / 2

        x_metr = torch.abs(x_center_true - x_center_pred)
        y_metr = torch.abs(y_center_true - y_center_pred)
        angle_metr = torch.abs(y_true[..., 4] - y_pred[..., 4])

        metric = 1 - (
            0.7 * 0.5 * (x_metr + y_metr)
            + 0.3 * torch.min(angle_metr, torch.abs(angle_metr - 360))
        )
        # return -metric.sum()

        return -(metric.sum() / (batch_size + 1))
