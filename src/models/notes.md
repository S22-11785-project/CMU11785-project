Save PyTorch models:
```
torch.save(model.state_dict(), '<model_path>.pth')
```

Load PyTorch models:
```
model = ModelClass(**args)
model.load_state_dict(torch.load(<model_path>))
```