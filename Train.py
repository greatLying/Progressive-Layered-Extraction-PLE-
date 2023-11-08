

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import csv
from Models import Expert, Tower, CGC, PLE

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()  # 拉成一维矩阵
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def data_preparation():
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    train_df = pd.read_csv('F:\census-income.data.gz', delimiter=',', header=None, index_col=None, names=column_names)
    test_df = pd.read_csv('F:\census-income.test.gz', delimiter=',', header=None, index_col=None, names=column_names)

    label_columns = ['income_50k', 'marital_stat']

    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_transformed = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    test_transformed = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)
    train_labels = train_df[label_columns]
    test_labels = test_df[label_columns]

    test_transformed['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    train_income = to_categorical((train_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    other_income = to_categorical((test_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((test_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    validation_indices = test_transformed.sample(frac=0.5, replace=False, random_state=seed).index
    test_indices = list(set(test_transformed.index) - set(validation_indices))
    validation_data = test_transformed.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = test_transformed.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = train_transformed
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def getTensorDataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x.astype(np.float32))
    tensor_y = torch.Tensor(my_y)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


def test(loader):
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            y1, y2 = y[:, 0], y[:, 1]
            yhat_1, yhat_2 = yhat[0], yhat[1]

            loss1, loss2 = loss_fn(yhat_1, y1.view(-1, 1)), loss_fn(yhat_2, y2.view(-1, 1))
            loss = loss1 + loss2

            t1_hat = yhat_1.cpu().numpy()
            t2_hat = yhat_2.cpu().numpy()

            t1_pred += list(t1_hat)
            t2_pred += list(t2_hat)
            t1_target += list(y1.cpu().numpy())
            t2_target += list(y2.cpu().numpy())

    auc_1 = roc_auc_score(t1_target, t1_pred)
    auc_2 = roc_auc_score(t2_target, t2_pred)
    return auc_1, auc_2


random.seed(3)
np.random.seed(3)
seed = 3
batch_size = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()

train_label_tmp = np.column_stack((np.argmax(train_label[0], axis=1), np.argmax(train_label[1], axis=1)))
train_loader = DataLoader(dataset=getTensorDataset(train_data.to_numpy(), train_label_tmp), batch_size=batch_size,
                          shuffle=True)

validation_label_tmp = np.column_stack((np.argmax(validation_label[0], axis=1), np.argmax(validation_label[1], axis=1)))
val_loader = DataLoader(dataset=getTensorDataset(validation_data.to_numpy(), validation_label_tmp),
                        batch_size=batch_size)

test_label_tmp = np.column_stack((np.argmax(test_label[0], axis=1), np.argmax(test_label[1], axis=1)))
test_loader = DataLoader(dataset=getTensorDataset(test_data.to_numpy(), test_label_tmp), batch_size=batch_size)

model = PLE(num_CGC_layers=4, input_size=499, num_specific_experts=4, num_shared_experts=4, experts_out=32, experts_hidden=32,
            towers_hidden=8)
model = model.to(device)
lr = 1e-4
n_epochs = 100
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
losses = []
val_loss = []

with open("PLE_results.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Train Loss", "Val Task1 AUC", "Val Task2 AUC"])
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        print("Epoch: {}/{}".format(epoch, n_epochs))
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            y1, y2 = y[:, 0], y[:, 1]
            y_1, y_2 = y_hat[0], y_hat[1]

            loss1 = loss_fn(y_1, y1.view(-1, 1))
            loss2 = loss_fn(y_2, y2.view(-1, 1))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        losses.append(np.mean(epoch_loss))
        auc1, auc2 = test(val_loader)
        print(
            'train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.3f}'.format(np.mean(epoch_loss), auc1, auc2))
        csvwriter.writerow([epoch, np.mean(epoch_loss), auc1, auc2])
    auc1, auc2 = test(test_loader)
    print('test task1 auc: {:.3f}, test task2 auc: {:.3f}'.format(auc1, auc2))
    


# In[ ]:





# In[ ]:




