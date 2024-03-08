import os
import random
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from RawNet2 import RawNet
from fastai.vision import *
from fastai.basics import *
from torchaudio import load as audio_load

device = 0
torch.cuda.set_device(device)


def get_fnames(data_root):
    data_root_p = Path(data_root / 'Auto_Tuned')
    data_root_n = Path(data_root / 'Original')
    fnames_p = list(map(str, data_root_p.rglob("*.wav")))
    fnames_n = list(map(str, data_root_n.rglob("*.wav")))
    fnames = list(zip(fnames_p, fnames_n))
    random.seed(54)
    random.shuffle(fnames)
    random.seed()
    fnames_p, fnames_n = zip(*fnames)
    fnames_p, fnames_n = list(fnames_p), list(fnames_n)

    fnames_train = fnames_p + fnames_n

    return fnames_train


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.label_map = {'positive': 1, 'negative': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, sr = audio_load(self.data[idx])
        if Path(self.data[idx]).parent.parent.name[0] == 'A':
            label = 'positive'
        else:
            label = 'negative'
        label_idx = self.label_map[label]
        return audio, label_idx


def train_epoch(train_loader, model, lr, optim, device, criterion):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 50 == 0:
            sys.stdout.write('\r \t {}'.format((running_loss / num_total)))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def evaluate_accuracy(dev_loader, model, device, criterion):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            running_loss += (batch_loss.item() * batch_size)

        running_loss /= num_total
        valid_accuracy = (num_correct / num_total) * 100

    return running_loss, valid_accuracy


def main():

    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('model_dir')
    

    args = ap.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = 100

    # Create custom datasets for training and validation
    train_dataset = CustomDataset(get_fnames(data_dir), transform=None)

    val_split = 0.1  # 10% for validation
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size

    # Create training and validation subsets
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                             generator=torch.Generator().manual_seed(42))

    # Create custom DataLoaders for training and validation
    batch_size = 64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # RawNet()
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    model = RawNet(parser1['model'], device)
    model = model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('Number of parameters: ', nb_params)

    best_loss = 1000000000000
    lr = 0.0001
    weight_decay = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, lr, optimizer, device, criterion)
        valid_loss, valid_accuracy = evaluate_accuracy(val_loader, model, device, criterion)

        print('\nepoch:{} - train_loss:{} - valid_loss:{} - train_accuracy:{:.2f} - valid_accuracy{:.2f}'.format(
            epoch,
            running_loss, valid_loss, train_accuracy, valid_accuracy))

        if valid_loss < best_loss:
            print('best model find at epoch', epoch)
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_epoch_{}.pth'.format(epoch)))

            best_loss = min(valid_loss, best_loss)




if __name__ == "__main__":
    main()
