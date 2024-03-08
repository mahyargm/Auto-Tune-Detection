#!/usr/bin/python3
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import psola
import os


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
    args = ap.parse_args()
    
    dataset_dir = Path(args.dataset_dir)


    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'):


                filepath = Path(os.path.join(root, file))


                # Load the audio file.
                y, sr = librosa.load(str(filepath), sr=None, mono=False)

                # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
                if y.ndim > 1:
                    y = y[0, :]

                # Perform the auto-tuning.
                pitch_corrected_y = autotune(y, sr, closest_pitch)
                print(filepath.parent)

                # Write the corrected audio to an output file.
                output_path = Path(filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix))
                sf.write(str(output_path), pitch_corrected_y, sr)


    
if __name__=='__main__':
    main()
    