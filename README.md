# ASR Experiment 2
This repository contains all the source code required for running the experiments related to research question 2 of group 14 for the course (Automatic) Speech Recognition (LET-REMA-LCEX10).

The files in this repository are sourced from https://github.com/dimitriStoidis/GenGAN. Credit goes to Dimitri Stoidis and his team.

Author: Carla Vintila - s1071616

## Project Structure
- ```generate_json.py``` – Prepares ```.json``` metadata for speaker IDs and gender labels.

- ```GenderNet.py``` – Trains a CNN-based gender classifier using K-Fold CV.

- ```InferenceGenderNet.py``` – Performs ensemble evaluation of trained GenderNet models.

- ```GenGAN.py``` – Applies GenGAN to anonymize voice data.

- ```networks.py, modules.py``` – Contain model architectures and utilities.

## Requirements
Can be installed with ```pip install -r requirements.txt```.

## Downloading the dataset

The datasets can be downloaded from [here](https://www.openslr.org/94/). We use the ```mls_polish_opus.tar.gz``` and ```mls_dutch_opus.tar.gz``` datasets.

## Pre-processing
Speaker metadata and gender labels can be found in the ```json``` folder. If required, they can be generated again using the ```generate_json.py``` file from the metadata files associated with the MLS datasets.

## Training GenderNet classifier
Training the model is done in ```GenderNet.py```. Arguments are listed inside the file. Ensure you update the config inside ```GenderNet.py``` for Dutch/Polish datasets.

## Running GenGAN for voice anonymization
Instructions on how to run the pre-trained GenGAN model on the audio files can be found in the [Demo section](https://github.com/dimitriStoidis/GenGAN?tab=readme-ov-file#demo) of the parent repository. 
To transform speech with GenGAN, run ```GenGAN.py``` with the correct audio paths inside the argument parser. Make sure you have ```.opus``` audio in the input directory and that the GenGAN models (```multi_speaker.pt, netG_epoch_25.pt```) are in the correct ```--path_to_models``` folder.

## Evaluation
To assess GenGAN’s privacy protection, run ```InferenceGenderNet.py```on the **transformed audio**. Update the config for Polish/Dutch ```audio_list``` and the appropriate model checkpoints. 
