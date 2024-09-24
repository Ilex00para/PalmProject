import torch
from torch import nn
import numpy as np

def Test_Step(model: nn.Module,
                 x,
                 y,
                 loss_function,
                 device = None) -> tuple:
    model.eval()
    #with detect_anomaly(): #for NaN gradient detection
    with torch.no_grad():
        x, y = x.to(device), y.to(device)

        prediction = model(x)

        #print(f'pred:{prediction}, real: {y}')
        validation_loss = loss_function(prediction, y)

        #metrics
        rmse = np.sqrt(validation_loss.detach().cpu().numpy())
        real_value = y.detach().cpu().numpy().reshape(-1)
        prediction = prediction.detach().cpu().numpy().reshape(-1)

    return validation_loss, (rmse, real_value, prediction)