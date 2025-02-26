"""
This script creates the ATDDv2 dataset with timestamped ground truths from the original dataset.
"""
import soundfile as sf
import json
from pathlib import Path
import os
import numpy as np
import random
import shutil
from collections import defaultdict
import librosa
import psola
random.seed(8)


def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)


def autotune(audio, sr, correction_function):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = 512
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)


    # Apply the adjustment strategy to the pitch.
    corrected_f0 = correction_function(f0)

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('output_dir')
    args = ap.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    sr = 16000
    gts = {}
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('Original_Song.wav'): # Modify this line to match the original song file names
                filepath = Path(os.path.join(root, file))
                AT_filepath = Path(os.path.join(root, 'Auto_Tuned_Song.wav'))
                audio_len = len(sf.SoundFile(filepath))
                counter_segments = 0
                for i in range(0, audio_len, 10*sr):
                    if (i+(10*sr))>audio_len: continue
                    x, sr = sf.read(filepath, start = i, stop = i+(10*sr))
                    if len(x.shape) == 2:
                        x = x.sum(axis=1) / 2
                    
                    if sr!=16000: 
                        print('Resampling to 16000') 
                        x = librosa.resample(x, sr, 16000)
                        sr = 16000
                         
                    energy = np.sum(x**2)
                    if energy < 10: continue
                    counter_segments += 1
                    Original_output_path = output_dir / filepath.parent.parent.name / 'Original' / filepath.parent.name / f'{counter_segments}.wav'
                    Original_output_path.parent.mkdir(parents=True, exist_ok=True)
                    # sf.write(str(Original_output_path), x, sr)
                    d = int(random.uniform(0.2,0.8) * sr * 10)  # duration (AT)
                    s = random.randint(0, (10*sr)-d)  # start (AT)
                    e = s+d # end (AT)
                    AT_x, sr = sf.read(AT_filepath, start = i+s, stop = i+e)
                    if len(AT_x.shape) == 2:
                        AT_x = AT_x.sum(axis=1) / 2
                    x[s:e] = AT_x
                    Autotuned_output_path = output_dir / filepath.parent.parent.name / 'Autotuned' / filepath.parent.name / f'{counter_segments}.wav'
                    Autotuned_output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(Autotuned_output_path), x, sr)

                    gts[str(Autotuned_output_path)] = {
                        'Start': s/sr.,
                        'End': e/sr.,
                        'Duration': d/sr.
                    }

    with open(output_dir / 'data.json', "w") as json_file:
        json.dump(gts, json_file, indent=4)

if __name__ == '__main__':
    main()