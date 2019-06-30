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
from utils import find_nearest_error, generate_clean_batch, evaluate, evaluate_pgd_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Adversarial Spheres Training')
    parser.add_argument('--sheet_id', help='Google sheet id')
    parser.add_argument('--sheet', help='Google sheet tab name')
    # model settings
    parser.add_argument('--batch_norm', help='do batch normalization \
                        (with m=0, track=False)', action="store_true")
    parser.add_argument('--layers', help='number of hidden layers', default=2,
                        type=int)
    # training hyper-parameters
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--bs', help='mini-batch size', type=int,
                        default=50)
    parser.add_argument('--epochs', help='number of epochs to train for',
                        type=int, default=100)
    parser.add_argument('--pgd_iters', help='number of pgd iterations',
                        type=int, default=1000)
    parser.add_argument('--d', help='input dimension', type=int, default=10)

    args = parser.parse_args()

    N = 10000  # train and val on 10k pts
    test_N = 100000  # test on 100k pts
    train_x, train_y = generate_clean_batch(N, args.d)
    val_x, val_y = generate_clean_batch(N, args.d)
    test_x, test_y = generate_clean_batch(test_N, args.d)

    train_y = train_y.reshape(-1)
    val_y = val_y.reshape(-1)
    test_y = test_y.reshape(-1)

    IDX_LSS = 0
    IDX_ACC = 1
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

        # Train
        X_train = torch.tensor(train_x, dtype=torch.float)
        Y_train = torch.tensor(train_y, dtype=torch.long)

        train_batch_size = 50
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=1000, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

        # Val
        X_val = torch.tensor(val_x, dtype=torch.float)
        Y_val = torch.tensor(val_y, dtype=torch.long)

        val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
        val_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=1000, shuffle=False)

        # Val inner sphere only
        X_val_inner = torch.tensor(val_x[N // 2:], dtype=torch.float)
        Y_val_inner = torch.tensor(val_y[N // 2:], dtype=torch.long)
        val_inner_dataset = torch.utils.data.TensorDataset(X_val_inner, Y_val_inner)
        val_inner_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=val_inner_dataset, batch_size=N // 2, shuffle=False)

        # Test
        X_test = torch.tensor(test_x, dtype=torch.float)
        Y_test = torch.tensor(test_y, dtype=torch.long)

        # create datasets
        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
        test_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=N // 10, shuffle=False)  # 100k / 10 = 10k

        # Tst inner sphere only
        X_tst_inner = torch.tensor(test_x[test_N // 2:], dtype=torch.float)
        Y_tst_inner = torch.tensor(test_y[test_N // 2:], dtype=torch.long)
        tst_inner_dataset = torch.utils.data.TensorDataset(X_tst_inner, Y_tst_inner)
        tst_inner_loader_noshuffle = torch.utils.data.DataLoader(
            dataset=tst_inner_dataset, batch_size=N // 20, shuffle=False)

        model = ReLUNetwork(train_x.shape[1], args.layers, num_units,
                            N_CLASSES, do_batch_norm=do_batch_norm).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        dataset_type = ['trn', 'val']
        lss = {}    # cross-entropy loss
        acc = {}    # prediction accuracy
        lss['trnmb'] = []  # one entry per minibatch

        for dst in dataset_type:
            lss[dst] = []  # one entry per epoch
            acc[dst] = []  # one entry per epoch

        total_step = len(train_loader)
        ########################## begin training loop #########################
        for epoch in range(args.epochs):
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fnct(outputs, labels)
                lss['trnmb'].append(loss.item()) # record training loss
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #train_stats[seed, epoch, IDX_LSS] = train_mb_loss / len(train_loader)
            # record train loss (avg value over mini-batches)
            lss['trn'].append(np.mean(lss['trnmb'][-total_step:]))
            train_acc, __ = evaluate(
                model, train_loader_noshuffle, loss_fnct, device)
            val_acc, val_lss = evaluate(
                model, val_loader_noshuffle, loss_fnct, device)
            lss['val'].append(val_lss)

            print('Epoch [{}/{}], Loss: {:.5f} (Train), {:.5f} (Val); \
                  Acc: {:.5f} (Train), {:.5f} (Val)'.format(
                      epoch, args.epochs, lss['trn'][-1], lss['val'][-1],
                      train_acc, val_acc))
        ########################## end training loop ###########################

        tst_acc, tst_lss = evaluate(
            model, test_loader_noshuffle, loss_fnct, device)

        pgd_acc = evaluate_pgd_acc(
            model, val_loader_noshuffle, device, args.pgd_iters)
            #model, test_loader_noshuffle, device, args.pgd_iters)

        d_err_min = find_nearest_error(
            model, val_inner_loader_noshuffle, device, args.pgd_iters)  # for faster eval
            #model, tst_inner_loader_noshuffle, device, args.pgd_iters)


        print(d_err_min)

        if args.sheet_id is not None:
            import pickle
            from googleapiclient.discovery import build
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            # Google Sheets API
            SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
            creds = None
            # The file token.pickle stores the user's access and refresh tokens, and
            # is created automatically when the authorization flow completes for the
            # first time.
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server()
                # Save the credentials for the next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            service = build('sheets', 'v4', credentials=creds)
            seed2col = ['C', 'D', 'E', 'F', 'G']
            values = [
                [lss['trn'][-1]],
                [lss['val'][-1]],
                [train_acc],
                [val_acc],
                [tst_acc],
                [pgd_acc],
                [d_err_min]
            ]
            body = {
                'values': values
            }
            #SKIP_SIZE = 31
            if args.d == 10:
                ROW_OFFSET = 3
            elif args.d == 20:
                ROW_OFFSET = 11
            elif args.d == 30:
                ROW_OFFSET = 19
            #SHEET = 'Auto'
            #if batchnorm:
                #ROW_OFFSET += SKIP_SIZE
            #ROW_OFFSET += (seed - 1) * (len(values) + 1)
            result = service.spreadsheets().values().update(
                spreadsheetId=args.sheet_id, range="%s!%s%d:%s%d" %
                (args.sheet, seed2col[seed], ROW_OFFSET, seed2col[seed],
                 ROW_OFFSET + len(values)),
                valueInputOption='USER_ENTERED', body=body).execute()
            print('{0} cells updated.'.format(result.get('updatedCells')))
            print(result)
            #j += 1
        '''
        filename = 'df_ep%d_bn%s_lr%.e' % (args.epochs, do_batch_norm, args.lr)
        print(filename)
        df_lss = pd.DataFrame(train_stats[:, :, 0]).melt()
        df_acc = pd.DataFrame(train_stats[:, :, 1]).melt()
        df_lss.to_pickle(filename + '_lss')
        df_acc.to_pickle(filename + '_acc')
        '''
