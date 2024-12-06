Libraries needed:

os
torch
torchvision
sklearn
pathlib
matplotlib
seaborn
shutil
librosa
numpy

Three scripts are provided.

convertToMelSpectrogram.py is a utility script to convert audio (16 kHz, mono) to 
Mel-spectrograms. Ensure that audio is divided into classes by subfolders. To run, use:

python convertToMelSpectrogram.py

split_data.py is a utility script to split the dataset into train and test categories. Ensure that spectrograms
are divided into classes by subfolders. To run, use:

python split_data.py

train_classifier.py is the main script to train and evaluate CNN models. Ensure 
that the dataset is split into folders appropriately as outputted by split_data.py. To run, use:

python train_classifier.py
