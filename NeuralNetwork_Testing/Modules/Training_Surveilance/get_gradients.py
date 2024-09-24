import torch

def get_gradients(model:torch.nn.Module) -> dict:
    gradients = {}
    with open('./Gradients.txt', 'w') as f:            
        for name, param in model.named_parameters():
            f.write(f'{name}:\n {param.grad}\n')
            gradients[name] = param.grad
    return gradients