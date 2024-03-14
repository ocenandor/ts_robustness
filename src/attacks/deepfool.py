import numpy as np
import torch


def deepfool(input, model, epsilon=0.02, max_iter=50, device='cpu', *args, num_classes=2):

    """
       :param input: signal of size L x 1
       :param model: modelwork (input: signals, output: values of activation **BEFORE** softmax).
       :param epsilon: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed input
    """

    assert type(input) == torch.Tensor, 'Use tensored input'
    if input.dim() == 1:
        input = input.unsqueeze(0)
    elif input.dim() == 2:
        assert input.shape[0] == 1, 'Only single sample input available'

    input = input.to(device)
                     
    model.eval()
    with torch.no_grad():
        init_predictions = model(input)
        target = torch.argmax(init_predictions)
        sort_ind = init_predictions.flatten().argsort(descending=True)

    adversarial_input = input.clone()
    w = torch.zeros(input.shape)
    r_tot = torch.zeros(input.shape)

    for iter in range(max_iter):
        adversarial_input.requires_grad = True
        predictions = model(adversarial_input)

        perturbed_target = torch.argmax(predictions)
        if perturbed_target != target: # early stopping
            break
        pert = np.inf
        predictions[0, sort_ind[0]].backward(retain_graph=True)
        grad_orig = adversarial_input.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            if adversarial_input.grad is not None:
                adversarial_input.grad.zero_()

            predictions[0, sort_ind[k]].backward(retain_graph=True)
            cur_grad = adversarial_input.grad.data.cpu().numpy().copy()
            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (predictions[0, sort_ind[k]] - predictions[0, sort_ind[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten()) + 1e-12)
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k
        
        r_tot += (pert+1e-4) * w / (np.linalg.norm(w) + 1e-12)
        adversarial_input = input + (1 + epsilon) * r_tot.to(device)
        
    return adversarial_input, iter