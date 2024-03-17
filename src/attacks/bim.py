import torch
from torch.linalg import vector_norm

def bim(input, model, epsilon=0.01, max_iter=50, device='cpu'):
    '''
    model
    input - signal which must be attacked
    epsilon - parameter of attack
    max_iter - number of iterations for attack
    
    return:
    adversarial_input - attacked signal
    '''
    
    assert type(input) == torch.Tensor, 'Use tensored input'
    if input.dim() == 1:
        input = input.unsqueeze(0)
    elif input.dim() == 2:
        assert input.shape[0] == 1, 'Only single sample input available'
    
    loss_function = torch.nn.CrossEntropyLoss()
    input = input.to(device)
                     
    model.eval()
    with torch.no_grad():
        target = torch.argmax(model(input), 1)


    adversarial_input = input.clone()
    
    for iter in range(max_iter):
        adversarial_input.requires_grad = True
        predictions = model(adversarial_input)
        perturbed_target = torch.argmax(predictions, 1)
        if perturbed_target != target: # early stopping
            break

        loss = loss_function(predictions, target) 
        grad_ = torch.autograd.grad(loss, adversarial_input, retain_graph=True)[0]
        adversarial_input = adversarial_input.data + epsilon * torch.sign(grad_)
        
    l2norm = (vector_norm(adversarial_input - input) / vector_norm(input)).detach().cpu().numpy()
    return adversarial_input, iter, perturbed_target != target, l2norm