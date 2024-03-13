import torch
import torch.nn as nn

def BIM(model, input, target, epsilon, num_iterations, device='cuda'):
    '''
    model
    input - signal which must be attacked
    target - true label of signal
    epsilon - parameter of attack
    num_iterations - number of iterations for attack
    
    return:
    adversarial_input - attacked signal
    '''
    model.train()
    loss_function = torch.nn.BCELoss()

    if(input.dim() == 1):
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
    input = input.to(device)
    target = target.to(device)

    adversarial_input = input.clone().requires_grad_(True)
    
    for _ in range(num_iterations):
        adversarial_input.requires_grad = True
        predictions = model(adversarial_input)
        loss = loss_function(predictions.flatten(), target) 
        grad_ = torch.autograd.grad(loss, adversarial_input, retain_graph=True)[0]
        adversarial_input = adversarial_input.data + epsilon * torch.sign(grad_)
    
    return adversarial_input
