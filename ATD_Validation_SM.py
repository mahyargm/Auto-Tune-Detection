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

def mel_spectrogram(audio, sr, path):
    msp = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=1024)
    output_dB = librosa.power_to_db(msp, ref=np.max)

    output_dB_flipped = np.flipud(output_dB)

    colormap = cm.get_cmap('magma')
    normalized_output = (output_dB_flipped - np.min(output_dB)) / (np.max(output_dB_flipped) - np.min(output_dB))

    colormap_values = colormap(normalized_output)
    colormap_values_uint8 = (255 * colormap_values).astype(np.uint8)
    colormap_img = Image.fromarray(colormap_values_uint8)
    colormap_img.save(path)

def Validation(temp_dir, model_name, segment_num, device):
    model = branch(output_embedding_length = 512, backbone=model_name, device=device)
    model.load_state_dict(torch.load(r'models/{}.pth'.format(model_name)))
    model.to(device)
    model.eval()
    
    # Feature extraction from spectrograms
    features = []
    for c in range(segment_num):
        image_path = temp_dir / 'vocal_{}.png'.format(c+1)
        vocal = Image.open(str(image_path)).convert('RGB')  # Open image using PIL
        vocal = transforms.ToTensor()(vocal).unsqueeze(0).to(device)
        features.append(model(vocal))
    
    # Final classification
    model = CLModel(512, hidden_size1 = 256, hidden_size2 = 64).to(device)
    model.load_state_dict(torch.load('models/CL_{}.pth'.format(model_name)))
    model.to(device)
    model.eval()

    predictions_binary = []
    predictions = []
    with torch.no_grad():
        for feature_vector in features:
            output = model(feature_vector.squeeze(0))
            predictions_binary.append((output >= 0.5))
            predictions.append(output)
    
    print('Number of Auto-Tuned segments: {}/{}'.format(predictions_binary.count(True), len(predictions_binary)))
    print('The average likelihood: {}'.format(sum(predictions) / len(predictions)))


def main():

    device = 0

    # Create a temporary directory
    output_dir = Path(r'./temp')
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    output_dir.mkdir()

    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('music_path')
    ap.add_argument('model_name') # ["ResNet50", "ResNet18", "EfficientNet", "ResNext"]

    args = ap.parse_args()
    
    music_path = args.music_path
    model_name = args.model_name

    # Vocal separation
    y, sr = librosa.load(music_path, sr=None, mono=False)
    vocal, sr = isolate_vocals(y, sr)

    # Preprocessing
    segment_duration = 10
    counter_outputs = 0
    samples_per_segment = int(segment_duration * sr)

    for i in range(0, len(vocal), samples_per_segment):
        if (i+samples_per_segment)>len(vocal): continue
        vocal_segment = vocal[i:i+samples_per_segment]
        energy = np.sum(vocal_segment**2)
        if energy < 10: continue
        counter_outputs += 1
        vocal_output_path = output_dir / 'vocal_{}.png'.format(counter_outputs)

        mel_spectrogram(audio = vocal_segment, sr=sr, path=vocal_output_path)
    if not counter_outputs: 
        print('No singing voice segments have been found!')
        os.rmdir(output_dir)
        exit()
    Validation(output_dir, model_name, counter_outputs, device)
    shutil.rmtree(output_dir)


if __name__=='__main__':
    main()


