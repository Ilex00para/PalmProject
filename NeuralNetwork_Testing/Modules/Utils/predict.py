import pandas as pd

def predict(model, dataloader, device):
    real_pred_array = {'Predicted':[],'Actual':[]}

    model.to(device)
    model.eval()
    for i, batch in enumerate(dataloader):

        X, y = batch[0].to(device), batch[1].to(device)

        prediction = model(X)

        real_pred_array['Actual'].append(y.detach().cpu().numpy()[0][0])
        real_pred_array['Predicted'].append(prediction.detach().cpu().numpy()[0][0])

    return pd.DataFrame(real_pred_array)