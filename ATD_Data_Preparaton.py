#!/usr/bin/python3
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import os
from PIL import Image
from matplotlib import cm

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

def main():
    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('output_dir')

    args = ap.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    segment_duration = 10  # Duration in seconds

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('Auto_Tuned_Vocal_is.wav'):
                counter_segments = 0
                filepath = Path(os.path.join(root, file))

                v, sr = librosa.load(str(filepath), sr=None, mono=True)
                samples_per_segment = int(segment_duration * sr)

                for i in range(0, len(v), samples_per_segment):

                    if (i+samples_per_segment)>len(v): continue
                    
                    v_segment = v[i:i+samples_per_segment]

                    energy = np.sum(v_segment**2)

                    if energy < 10: continue

                    counter_segments += 1
                    if filepath.parent.parent.name == 'Training':
                        vocal_output_path = output_dir / filepath.parent.parent.name / 'Auto_Tuned' / filepath.parent.name / 'is_{}.png'.format(counter_segments)
                    else:
                        vocal_output_path = output_dir / filepath.parent.parent.parent.name / filepath.parent.parent.name / 'Auto_Tuned'  / filepath.parent.name / 'is_{}.png'.format(counter_segments)
                    vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                    mel_spectrogram(v_segment, sr, str(vocal_output_path))

            if file.endswith('Auto_Tuned_Vocal.wav'):
                counter_segments = 0
                filepath = Path(os.path.join(root, file))

                v, sr = librosa.load(str(filepath), sr=None, mono=True)

                for i in range(0, len(v), samples_per_segment):

                    if (i+samples_per_segment)>len(v): continue
                    
                    v_segment = v[i:i+samples_per_segment]

                    energy = np.sum(v_segment**2)

                    if energy < 10: continue

                    counter_segments += 1
                    if filepath.parent.parent.name == 'Training':
                        vocal_output_path = output_dir / filepath.parent.parent.name / 'Auto_Tuned' / filepath.parent.name / '{}.png'.format(counter_segments)
                    else:
                        vocal_output_path = output_dir / filepath.parent.parent.parent.name / filepath.parent.parent.name / 'Auto_Tuned'  / filepath.parent.name / '{}.png'.format(counter_segments)
                    vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                    mel_spectrogram(v_segment, sr, str(vocal_output_path))   

            if file.endswith('Original_Vocal.wav'):
                counter_segments = 0
                filepath = Path(os.path.join(root, file))

                v, sr = librosa.load(str(filepath), sr=None, mono=True)

                for i in range(0, len(v), samples_per_segment):

                    if (i+samples_per_segment)>len(v): continue
                    
                    v_segment = v[i:i+samples_per_segment]

                    energy = np.sum(v_segment**2)

                    if energy < 10: continue

                    counter_segments += 1
                    if filepath.parent.parent.name == 'Training':
                        vocal_output_path = output_dir / filepath.parent.parent.name / 'Original' / filepath.parent.name / '{}.png'.format(counter_segments)
                    else:
                        vocal_output_path = output_dir / filepath.parent.parent.parent.name / filepath.parent.parent.name / 'Original'  / filepath.parent.name / '{}.png'.format(counter_segments)
                    vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                    mel_spectrogram(v_segment, sr, str(vocal_output_path))

            if file.endswith('Original_Vocal_is.wav'):
                counter_segments = 0
                filepath = Path(os.path.join(root, file))

                v, sr = librosa.load(str(filepath), sr=None, mono=True)

                for i in range(0, len(v), samples_per_segment):

                    if (i+samples_per_segment)>len(v): continue
                    
                    v_segment = v[i:i+samples_per_segment]

                    energy = np.sum(v_segment**2)

                    if energy < 10: continue

                    counter_segments += 1
                    if filepath.parent.parent.name == 'Training':
                        vocal_output_path = output_dir / filepath.parent.parent.name / 'Original' / filepath.parent.name / 'is_{}.png'.format(counter_segments)
                    else:
                        vocal_output_path = output_dir / filepath.parent.parent.parent.name / filepath.parent.parent.name / 'Original' / filepath.parent.name / 'is_{}.png'.format(counter_segments)           
                    
                    vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                    mel_spectrogram(v_segment, sr, str(vocal_output_path))          
            if file.endswith('st_Auto_Tuned.wav'):
                filepath = Path(os.path.join(root, file))
                v, sr = librosa.load(str(filepath), sr=None, mono=True)

                vocal_output_path = output_dir / filepath.parent.parent.parent.name / 'Auto_Tuned' / filepath.parent.parent.name / (filepath.parent.name + '.png')
                vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                mel_spectrogram(v, sr, str(vocal_output_path))

            if file.endswith('st_Original.wav'):
                filepath = Path(os.path.join(root, file))
                v, sr = librosa.load(str(filepath), sr=None, mono=True)

                vocal_output_path = output_dir / filepath.parent.parent.parent.name / 'Original'  / filepath.parent.parent.name / (filepath.parent.name + '.png')
                vocal_output_path.parent.mkdir(parents=True, exist_ok=True)
                mel_spectrogram(v, sr, str(vocal_output_path))



    
if __name__=='__main__':
    main()
    