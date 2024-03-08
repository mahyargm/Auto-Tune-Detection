from fastai.vision import *
from fastai.basics import *

import torch
import argparse
from RawNet2 import RawNet
from torchaudio import load as audio_load
from torch.utils.data import Dataset, DataLoader





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

    fnames_test = fnames_p + fnames_n

    return fnames_test

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


def evaluate_model(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        tp+= sum([(a == 1) and (b == 1) for a, b in zip(batch_pred, batch_y)]).item()
        tn+= sum([(a == 0) and (b == 0) for a, b in zip(batch_pred, batch_y)]).item()
        fp+= sum([(a == 1) and (b == 0) for a, b in zip(batch_pred, batch_y)]).item()
        fn+= sum([(a == 0) and (b == 1) for a, b in zip(batch_pred, batch_y)]).item()

        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    
    accuracy = 100 * ((tp + tn) / num_total)
    precision = 100 * (tp / (tp+fp))
    recall = 100 * (tp / (tp+fn))
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")


def main():

    device = 0
    torch.cuda.set_device(device)

    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('model_path')
    

    args = ap.parse_args()
    
    model_path = args.model_path
    data_dir = Path(args.data_dir)
    # Create custom datasets for training and validation
    test_dataset = CustomDataset(get_fnames(data_dir), transform=None)


    # RawNet()
    dir_yaml =os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    model = RawNet(parser1['model'], device)
    model =(model).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('Number of parameters: ', nb_params)

    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    print('Model loaded : {}'.format(model_path))

    data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
    evaluate_model(data_loader, model, device)

if __name__=='__main__':
    main()

