import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

R = 1.3
LABEL_OUTER_SPHERE = 1


def generate_clean_batch(batch_size, d):
    """
    Returns a dataset of adversarial spheres
    @param batch_size of mini-batch
    @param d: dimensionality of the spheres
    """
    x = np.random.multivariate_normal(np.zeros(d), np.eye(d), batch_size)

    x_train = np.zeros((batch_size, d))
    y_train = np.zeros((batch_size, 1))

    x_shuff = np.zeros((batch_size, d))
    y_shuff = np.zeros((batch_size, 1))

    euclidean_norm = np.linalg.norm(x, axis=1)

    # outer sphere, radius R
    for i in range(batch_size // 2):
        x_train[i, :] = R * x[i, :] / euclidean_norm[i]

    # inner sphere, radius 1
    for i in range(batch_size // 2, batch_size):
        x_train[i, :] = x[i, :] / euclidean_norm[i]

    # assign training labels
    y_train[:batch_size // 2] = LABEL_OUTER_SPHERE

    return (x_train, y_train)

def evaluate(model, data_loader, loss_fnct, device):
    """
    Evaluate model in terms of accuracy and loss on data_loader
    @param
    """
    total = 0
    correct = 0
    mb_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            mb_loss += loss_fnct(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = float(correct) / total
        lss = mb_loss / len(data_loader)
    return acc, lss

criterion = nn.CrossEntropyLoss(reduction='sum')

def fgm_l2(model, im, labels, eps, device):
    """
    Performs one iteration of FGM (non-targeted).
    @param
    """
    x_ = Variable(im, requires_grad=True)
    red_ind = list(np.arange(1, len(x_.shape)))
    loss = criterion(model(x_), labels)
    loss.backward()
    loss_grad = x_.grad.data.clone()
    square = torch.max(torch.FloatTensor([1e-12]).to(device),  # to prevent div by zero
                       torch.sum(loss_grad**2, dim=red_ind, keepdim=True))
    normalized_loss_grad = loss_grad / torch.sqrt(square)
    adv_img = x_.detach() + (eps * normalized_loss_grad).to(device)
    return adv_img


def evaluate_pgd_acc(model, data_loader, device, pgd_iters):
    """
    Evaluate accuracy for adversarial examples on the data manifold
    Differs from find_nearest_error in that we don't search for nearest errors,
    we search for all errors on the manifold.
    @param
    """
    total = 0
    correct = 0
    model.eval()
    for inputs, labels in data_loader:
        labels = labels.to(device)
        inputs = inputs.to(device)
        advex = inputs.clone()
        for i in range(pgd_iters):
            advex = fgm_l2(model, advex, labels, 0.1, device)
            # project back on to manifold
            advex /= torch.unsqueeze(advex.norm(p=2, dim=1), dim=1)
            advex[labels == LABEL_OUTER_SPHERE] *= R
        outputs = model(advex)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return float(correct) / total


def find_nearest_error(model, data_loader, device, pgd_iters):
    """
    Find the nearest error using PGD, stop as soon as error found so it will
    have minimum distance.
    @param
    """
    total = 0
    correct = 0
    d_err_min = 10
    model.eval()
    for i, (inputs, labels) in enumerate(data_loader):
        labels = labels.to(device)
        inputs = inputs.to(device)
        advex = inputs.clone()
        # search for nearest error d(E)
        for j in range(pgd_iters):
            advex = fgm_l2(model, advex, labels, 0.01, device)
            # project back on to manifold
            advex /= torch.unsqueeze(advex.norm(p=2, dim=1), dim=1)
            outputs = model(advex)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_idx = predicted == labels
            correct += (correct_idx).sum().item()
            if (correct_idx == 0).sum() > 0:
                # compute nearest error on inner sphere
                d_err = (inputs[correct_idx == 0] - advex[correct_idx == 0]
                         ).norm(p=2, dim=1).min().item()
                if d_err < d_err_min:
                    print('itr %d got d = %f' % (j, d_err))
                    d_err_min = d_err
                break
        print('%d of %d' % (i, len(data_loader)))
    return d_err_min
