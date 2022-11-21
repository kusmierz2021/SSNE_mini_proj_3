model.eval()

predictions = []

with torch.no_grad():
    for x, cat_x in data_loader:
        x, cat_x = x.to(device), cat_x.to(device)
        output = model(x, cat_x)
        pred = torch.round(output)
        predictions = predictions + pred.tolist()

predictions = [int(x[0]) for x in predictions]
print(predictions)

