import os
import argparse
import numpy as np
from numpy.linalg import det, inv
import pandas as pd

# for linear model
import torch
import torch.nn as nn
import torch.utils.data

from ReLUNetwork import ReLUNetwork

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Adversarial Spheres Training')
    # model settings
    parser.add_argument('--batch_norm', help='do batch normalization \
                        (with m=0, track=False)', action="store_true")
    parser.add_argument('--layers', help='num layers in model', default=2,
                        type=int)
    # training hyper-parameters
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--bs', help='examples per mini-batch', type=int,
                        default=50)
    parser.add_argument('--epochs', help='number of epochs to train for',
                        type=int, default=200)
    parser.add_argument('--seed', help='random seed', type=int, default=1)

    args = parser.parse_args()

    IDX_LSS = 0
    IDX_ACC = 1
    N = 500  # This is the dataset size.
    d = 2
    R = 1.3
    N_CLASSES = 2
    num_units = 1000
    do_batch_norm = True if args.batch_norm else False
    train_stats = np.zeros((5, args.epochs, 2))

    loss_fnct = nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=0)
    batch_softmax = torch.nn.Softmax(dim=1)
    device = torch.device('cuda:0')

    for seed in range(5):
        np.random.seed(seed)
        torch.manual_seed(seed)

        x = np.random.multivariate_normal(np.zeros(d), np.eye(d), N * 2)
        x = x / np.expand_dims(np.linalg.norm(x, axis=1), 1)

        # Training set
        train_x = np.zeros((2 * N, 2))  # DATA, FEATURES
        train_y = np.zeros(2 * N)       # DATA, Int label

        # class 1
        train_x[:N, 0] = x[:N, 0].copy() #+ np.random.randn(N) / 30
        train_x[:N, 1] = x[:N, 1].copy() #+ np.random.randn(N) / 30
        train_y[:N] = 0

        # class 2
        train_x[N:, 0] = x[N:, 0].copy() * R #+ np.random.randn(N) / 30
        train_x[N:, 1] = x[N:, 1].copy() * R #+ np.random.randn(N) / 30
        train_y[N:] = 1

        X_test = torch.tensor(train_x, dtype=torch.float)
        Y_test = torch.tensor(train_y, dtype=torch.long)

        # create datasets
        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
        test_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=100, shuffle=False)
        rng_state = np.random.get_state()
        np.random.shuffle(train_x)
        np.random.set_state(rng_state)
        np.random.shuffle(train_y)
        is_shuffled = True

        X_train = torch.tensor(train_x, dtype=torch.float)
        Y_train = torch.tensor(train_y, dtype=torch.long)

        # create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=N, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.bs, shuffle=True)

        model = ReLUNetwork(train_x.shape[1], args.layers, num_units,
                            N_CLASSES, do_batch_norm=do_batch_norm).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        total_step = len(train_loader)

        for epoch in range(args.epochs):
            model.train()
            train_mb_loss = 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fnct(outputs, labels)
                train_mb_loss += loss.item()
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_stats[seed, epoch, IDX_LSS] = train_mb_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                for inputs, labels in train_loader_noshuffle:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_stats[seed, epoch, IDX_ACC] = float(correct) / total
                print('Epoch [{}/{}], Loss: {:.3f}, Train Acc: {:.4f}'
                      .format(epoch, args.epochs, train_stats[seed, epoch, 0],
                              train_stats[seed, epoch, 1]))
        filename = 'df_ep%d_bn%s_lr%.e' % (args.epochs, do_batch_norm, args.lr)
        print(filename)
        df_lss = pd.DataFrame(train_stats[:, :, 0]).melt()
        df_acc = pd.DataFrame(train_stats[:, :, 1]).melt()
        df_lss.to_pickle(filename + '_lss')
        df_acc.to_pickle(filename + '_acc')
