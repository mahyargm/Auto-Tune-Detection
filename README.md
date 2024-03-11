# Auto-Tune Detector (ATD)

This tool utilizes deep learning and spectrogram-based techniques to identify Auto-Tuned vocals in music recordings. [Preprint paper link](http://arxiv.org/abs/2403.05380).


## Installation:

```bash
pip install requirements.txt
```

## Data Preparation:

1. Download the dataset from [this link](https://doi.org/10.5281/zenodo.10695885).
2. Prepare the data:
- For Auto-Tune Detector (ATD) training and validation:

```bash
python ATD_Data_Preparaton.py <dataset_directory> <Processed_dataset_dir> 
```

   Example:
```bash
python ATD_Data_Preparaton.py ./Dataset/ ./ATD_dataset 
```

   - For baseline training and validation:

```bash
python baseline_data_preparaton.py <dataset_directory> <Processed_dataset_dir>
```

   Example:
```bash
python baseline_data_preparaton.py ./Dataset/ ./baseline_dataset 
```

## Training:

- Train the ATD networks (choose backbone_name from {'ResNet18', 'ResNet50' 'ResNext', 'EfficientNet'}):

```bash
python ATD_Training.py <training_dataset_directory> <backbone_name>
python ATD_TrainingCL.py <training_dataset_directory> <backbone_name> 
```

   Example:
```bash
python ATD_Training.py ./ATD_dataset/Training EfficientNet 
python ATD_TrainingCL.py ./ATD_dataset/Training EfficientNet 
```

- Train the baseline:

```bash
python baseline_training.py <training_dataset_directory> <models_directory> 
```

   Example:
```bash
python baseline_training.py ./baseline_dataset/Training ./models 
```

## Testing:
Download the pretrained models from [this link](https://drive.google.com/file/d/1SpqwZgKKY5zIflhD-e5QfT0ZG1ZIGuv1/view?usp=sharing) and put them in the 'models' directory.

- To test the ATD models:

```bash
python ATD_Validation.py <test_dataset_directory> <backbone_name>
```

   Example:
```bash
python ATD_Validation.py ./ATD_dataset/Test/Simple EfficientNet 
```

- To test the baseline:

```bash
python baseline_validation.py <test_dataset_directory> <saved_model_path>
```

   Example:
```bash
python baseline_validation.py ./ATD_dataset/Test/Simple ./models/RawNet.pth 
```

## Single Music Testing for ATD:

```bash
python ATD_validation_SM.py <song_path> <backbone_name>
```

   Example:
```bash
python ATD_validation_SM.py ./Song.wav EfficientNet
```

## Contact

For any question regarding this repository, please contact:
- m.goharimoghaddam@unibs.it

## Acknowledgements

- [vocal-remover](https://github.com/tsurumeso/vocal-remover) by [Tsurumeso](https://github.com/tsurumeso)
- [ASVspoof 2021 Baseline CM & Evaluation Package](https://github.com/asvspoof-challenge/2021) by [ASVspoof-challenge](https://github.com/asvspoof-challenge)
- [triplet-loss-pytorch](https://github.com/alfonmedela/triplet-loss-pytorch) by [Alfonso Medela](https://github.com/alfonmedela)

## Citation

If you use this code in your research please use the following citation:

```bibtex
@misc{gohari2024spectrogrambased,
      title={Spectrogram-Based Detection of Auto-Tuned Vocals in Music Recordings}, 
      author={Mahyar Gohari and Paolo Bestagini and Sergio Benini and Nicola Adami},
      year={2024},
      eprint={2403.05380},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
