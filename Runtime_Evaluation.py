import librosa
from pathlib import Path
import numpy as np
import librosa.display
from fastai.vision import *
from fastai.basics import *
from fastai.vision.all import *
from fastai.vision.augment import *
import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_v2_s, resnet18, resnext50_32x4d
from torchvision import transforms
from matplotlib import cm
from PIL import Image
import sys
sys.path.append('./Dataset_Creation_Tools/vocal_remover/')
from inference import isolate_vocals
import argparse
import shutil
from RawNet2 import RawNet
from B02_models.backend import AASIST
from B02_models.model import Residual_block
from torchaudio.transforms import MelSpectrogram
class SVDDModel(nn.Module):
    def __init__(self, device, frontend=None):
        super(SVDDModel, self).__init__()

        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(128 * 5, 23 * 29) # match the output shape of the rawnet encoder

        self.backend = AASIST()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.encoder(x.unsqueeze(0))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        x = self.backend(x)
        x1 = self.sigmoid(x[1])
        return x[0], x1

class branch(nn.Module):
    def __init__(self, output_embedding_length=512, backbone='ResNet50', device=0):
        super(branch, self).__init__()
        self.preprocess = lambda x: x
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

def mel_spectrogram_tensor(audio, sr):
    msp = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=1024)
    output_dB = librosa.power_to_db(msp, ref=np.max)

    output_dB_flipped = np.flipud(output_dB)

    colormap = cm.get_cmap('magma')
    normalized_output = (output_dB_flipped - np.min(output_dB)) / (np.max(output_dB_flipped) - np.min(output_dB))

    colormap_values = colormap(normalized_output)[:,:,:3]
    colormap_values_uint8 = (255 * colormap_values).astype(np.uint8)
    colormap_img = Image.fromarray(colormap_values_uint8)

    transform = transforms.ToTensor()
    return transform(colormap_img).unsqueeze(0)  # Add batch dimension

def realtime_validation(model_branch, model_cl, vocal_segment_tensor, device):
    with torch.no_grad():
        start = time.time()
        feature_vector = model_branch(vocal_segment_tensor.to(device))
        fs_time = (time.time()-start)
        start = time.time()
        output = model_cl(feature_vector.squeeze(0))
        cl_time = (time.time()-start)
    return output, fs_time, cl_time

def baseline_validation(model, vocal_segment_tensor, device):
    with torch.no_grad():
        start = time.time()
        output = model(vocal_segment_tensor.to(device))
        inftime = (time.time()-start)
    return output, inftime

def main():
    device = 0

    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('music_dir') # Path to 10 seconf audio files
    ap.add_argument('model_name') # ["ResNet50", "ResNet18", "EfficientNet", "ResNext"]

    args = ap.parse_args()

    music_dir = Path(args.music_dir)
    model_name = args.model_name
    assert model_name in ["ResNet50", "ResNet18", "EfficientNet", "ResNext", "B01", "B02"], "Model name should be in ['ResNet50', 'ResNet18', 'EfficientNet', 'ResNext', 'B01', 'B02']"
    # Load models
    if model_name=='B01':
        dir_yaml =os.path.splitext('model_config_RawNet')[0] + '.yaml'
        with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
        model = RawNet(parser1['model'], device)
        model.load_state_dict(torch.load(f'models/RawNet.pth'))
        model =(model).to(device)
        model.eval()

    elif model_name=='B02':
        model = SVDDModel(frontend=None, device=device)
        model.to(device)
        model.eval()

    else:
        model_branch = branch(output_embedding_length=512, backbone=model_name, device=device)
        model_branch.load_state_dict(torch.load(f'models/{model_name}.pth'))
        model_branch.to(device)
        model_branch.eval()
        model_cl = CLModel(512, hidden_size1=256, hidden_size2=64).to(device)
        model_cl.load_state_dict(torch.load(f'models/CL_{model_name}.pth'))
        model_cl.to(device)
        model_cl.eval()

    total_segments = 0
    fs_times = 0 
    cl_times = 0
    sum_time = 0
    for music_path in list(map(str, music_dir.rglob("*.wav"))): 
        # Vocal separation
        y, sr = librosa.load(music_path, sr=None, mono=False)
        vocal, sr = isolate_vocals(y, sr)
        

        if model_name=='B01':
            output, inftime = baseline_validation(model, torch.from_numpy(vocal).unsqueeze(0).unsqueeze(0), device)
            sum_time += inftime
        

        elif model_name=='B02':
            melspec = MelSpectrogram(
                sample_rate=44100,
                n_mels=128,
                n_fft=2048,
                win_length=2048,
                hop_length=1024,
            ).to(device)
            vocal_segment_tensor = melspec(torch.from_numpy(vocal).unsqueeze(0).to(device))
            output, inftime = baseline_validation(model, vocal_segment_tensor, device)
            sum_time += inftime

        else:
            vocal_segment_tensor = mel_spectrogram_tensor(audio=vocal, sr=sr)
            output, fs_time, cl_time = realtime_validation(model_branch, model_cl, vocal_segment_tensor, device)
            fs_times += fs_time 
            cl_times += cl_time
        total_segments += 1

    if model_name in ["B01","B02"]:
        print(f'Total time: {(sum_time / total_segments) * 1000:.3f} ms')
    else:
        print(f'Feature extractor time: {(fs_times / total_segments) * 1000:.3f} ms')
        print(f'Classifier time: {(cl_times / total_segments) * 1000:.3f} ms')
        print(f'Total time: {((cl_times / total_segments) * 1000) + ((fs_times / total_segments) * 1000):.3f} ms')
if __name__ == '__main__':
    main()
