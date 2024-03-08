from fastai.vision import *
from fastai.basics import *
from fastai.vision.all import *
from fastai.vision.augment import *
from torchvision.models import resnet18, efficientnet_v2_s, resnext50_32x4d, resnet50
import torch.nn as nn
from triplet_loss import TripletLoss
import argparse

def create_getter(data_root):
    def get_fnames(i):
        data_root_p = Path(data_root / 'Auto_Tuned')
        data_root_n = Path(data_root / 'Original')
        fnames_p = get_image_files(data_root_p)
        fnames_n = get_image_files(data_root_n)
        fnames = list(zip(fnames_p, fnames_n))
        random.seed(54)
        random.shuffle(fnames)
        random.seed()
        fnames_p, fnames_n = zip(*fnames)
        fnames_p, fnames_n = list(fnames_p), list(fnames_n)

        fnames_train = fnames_p + fnames_n


        return fnames_train
    return get_fnames



def label_func(x):
    if x.parent.parent.name[0] == 'A': return 'positive'
    else: return 'negative'

def dataloadercreator(label_func, preprocess, data_root, batch_size):
    # Function to create the dataloader
    data = DataBlock(
        blocks=(ImageBlock, CategoryBlock()), 
        get_items= create_getter(data_root),
        splitter = RandomSplitter(valid_pct=1/10., seed=54),
        get_y=label_func)
    
    data.transform = preprocess

    dls = data.dataloaders('', batch_size=batch_size, num_classes=2)
    return dls


class branch(nn.Module):
    def __init__(self, output_embedding_length = 512, backbone = 'ResNet50', device = 0):
        super(branch, self).__init__()
        self.preprocess = lambda x:x
        if backbone == 'ResNet18':
            self.model = resnet18(pretrained=False)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(512, output_embedding_length, bias=True, device=device)
            )
        if backbone == 'ResNet50':
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(2048, output_embedding_length, bias=True, device=device)
            )
        if backbone == 'EfficientNet':
            self.model = efficientnet_v2_s(pretrained=False)
            self.model.classifier[1] = nn.Linear(1280, 512, device=device)
        if backbone == 'ResNext':
            self.model = resnext50_32x4d(pretrained=False)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(2048, output_embedding_length, bias=True, device=device)
            )
            
    def forward(self, x):
        logits = self.model(x)
        return F.normalize(logits, p=2, dim=-1)

def main():
    device = 0
    output_embedding_length = 512
    num_epochs = 100
    lr = 0.0001
    batch_size = 64

    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('model_name') # ["ResNet50", "ResNet18", "EfficientNet", "ResNext"]
    

    args = ap.parse_args()
    
    model_name = args.model_name
    data_dir = Path(args.data_dir)

    model = branch(output_embedding_length = output_embedding_length, backbone=model_name, device=device)
    dls=dataloadercreator(label_func, model.preprocess, data_dir, batch_size)
    print(len(dls.train))
    learn = Learner(dls, model, loss_func=TripletLoss(device), cbs=[Recorder, CSVLogger])
    learn.fit_one_cycle(num_epochs, lr, cbs=SaveModelCallback(fname=model_name))

if __name__=='__main__':
    main()