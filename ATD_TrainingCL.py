from fastai.vision import *
from fastai.basics import *
from fastai.vision.all import *
from fastai.vision.augment import *
from torchvision.models import resnet18, efficientnet_v2_s, resnext50_32x4d, resnet50
import torch
import torch.nn as nn
from triplet_loss import TripletLoss
import torch.optim as optim
import argparse

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

class CLModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(CLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class CLDataset():
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_vector, label = self.data[idx]
        if self.transform:
            feature_vector = self.transform(feature_vector)
        return feature_vector, label
    
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


def fv_extractor(dataloader, learner, have_valid=False):
    if have_valid:
        fvs_train = []
        for index, (images,labels) in enumerate(dataloader.train):
            fvbatch = learner.model(images.data).detach()
            for index2, label in enumerate(labels.cpu().numpy()):
                fv = fvbatch[index2].data.cpu().numpy()
                fvs_train.append((fv, label))
        fvs_valid = []
        for index, (images,labels) in enumerate(dataloader.valid):
            fvbatch = learner.model(images.data).detach()
            for index2, label in enumerate(labels.cpu().numpy()):
                fv = fvbatch[index2].data.cpu().numpy()
                fvs_valid.append((fv, label))
        return fvs_train, fvs_valid
    else:
        fvs_train = []
        for index, (images,labels) in enumerate(dataloader):
            fvbatch = learner.model(images.data).detach()
            for index2, label in enumerate(labels.cpu().numpy()):
                fv = fvbatch[index2].data.cpu().numpy()
                fvs_train.append((fv, label))
        return fvs_train

def train_classifier(model, train_loader, valid_loader, model_name, lr, num_epochs = 100, device = 0):

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_validation_loss = float('inf')
    best_model_state_dict = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for feature_vector, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(feature_vector)
            loss = criterion(outputs, labels.view(-1, 1).float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
       
        average_loss = total_loss / len(train_loader)

        
        # Validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for feature_vector, labels in valid_loader:
                outputs = model(feature_vector)
                validation_loss += criterion(outputs, labels.view(-1, 1).float()).item()

        average_validation_loss = validation_loss / len(valid_loader)
        if epoch%10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f} - Validation Loss: {average_validation_loss:.4f}")

        # Save the best model based on validation loss
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_model_state_dict = model.state_dict()
            print(epoch, ' saved!')

    print("Training finished.")


    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(model.state_dict(), "models/CL_{}.pth".format(model_name))


def main():
    device = 0
    output_embedding_length = 512
    batch_size = 64
    hidden_size1 = 256  # Number of units in the first hidden layer (Classification)
    hidden_size2 = 64  # Number of units in the second hidden layer (Classification)
    num_epochs = 500
    lr = 0.00001
    
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('model_name') # ["ResNet50", "ResNet18", "EfficientNet", "ResNext"]
    

    args = ap.parse_args()
    
    model_name = args.model_name

    data_dir = Path(args.data_dir)

    model = branch(output_embedding_length = output_embedding_length, backbone=model_name, device=device)
    dls=dataloadercreator(label_func, model.preprocess, data_dir, batch_size)
    learn = Learner(dls, model, loss_func=TripletLoss(device), cbs=[Recorder, CSVLogger])

    learn.load(model_name)
    learn.validate()
    fvs_train,fvs_valid = fv_extractor(dls, learn, True)

    # Data preparation for the final classification
    data_train = [(torch.tensor(feature_vector).to(device), torch.tensor(label, dtype=torch.int64).to(device)) for feature_vector, label in fvs_train]
    data_valid = [(torch.tensor(feature_vector).to(device), torch.tensor(label, dtype = torch.int64).to(device)) for feature_vector, label in fvs_valid]

    # Create Classification datasets for training and validation
    train_dataset = CLDataset(data_train, transform=None)
    valid_dataset = CLDataset(data_valid, transform=None)

    # Create Classification DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Classification Model
    model = CLModel(output_embedding_length, hidden_size1, hidden_size2).to(device)
    train_classifier(model, train_loader, valid_loader, model_name, lr, num_epochs, device)

if __name__=='__main__':
    main()