# https://github.com/LTS4/DeepFool.git

import copy

import numpy as np
import torch
from torch.autograd import Variable


def deepfool(signal, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param signal: signal of size L x 1
       :param net: network (input: signals, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed signal
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # print("Using GPU")
        signal = signal.cuda()
        net = net.cuda()
    # else:
        # print("Using CPU")


    f_signal = net.forward(Variable(signal[None, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_signal)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = signal.cpu().numpy().shape
    pert_signal = copy.deepcopy(signal)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_signal[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            if x.grad is not None:
                x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_signal = signal + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_signal = signal + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_signal, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_signal