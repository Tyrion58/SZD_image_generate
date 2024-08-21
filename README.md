# Predicting Structure Zone Diagrams for Thin Film Synthesis using AC-GAN

This repository contains the implementation of an Auxiliary Classifier Generative Adversarial Network (AC-GAN) for generating Structure Zone Diagrams (SZDs) used in thin film synthesis. 
The project replicates the methodology described in the paper 
"[Predicting Structure Zone Diagrams for Thin Film Synthesis by Generative Machine Learning](https://www.researchgate.net/publication/340232305_Predicting_structure_zone_diagrams_for_thin_film_synthesis_by_generative_machine_learning)."

## 1. Installation and Setup
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Tyrion58/SZD_image_generate
```

### Environment Setup
To simplify the installation process, you can use the provided `env.yml` file to recreate the conda environment required for this project:

```bash
conda env create -n SZD -f env.yml
```

Activate the environment with:

```bash
conda activate SZD
```

## 2. Training the Model
Before starting the training process, register an account on [Weights & Biases (wandb)](https://wandb.ai) and obtain your personal API key. Log in to wandb with:

```bash
wandb login
```

Enter your API key when prompted. To begin training, execute:

```bash
python train.py
```

### Customizing Training Parameters
If you need to customize the training parameters, use the help command to view the available options:

```bash
python train.py -h
```

This command provides detailed explanations for each parameter, allowing you to adjust them according to your needs. For example:

```bash
D:\SZD_cgan\SZD_image_generate>python train.py -h
usage: train.py [-h] [--model_name MODEL_NAME] [--data_name DATA_NAME] [--data_path DATA_PATH]
                [--image_subpath IMAGE_SUBPATH] [--checkpoint_dir CHECKPOINT_DIR] [--epochs EPOCHS] [-b BATCH_SIZE]
                [--lr LR] [--beta1 BETA1] [--image_size IMAGE_SIZE] [--channels CHANNELS] [--netD NETD] [--netG NETG]
                [--train TRAIN]

Train

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model architecture
  --data_name DATA_NAME
                        The name of the dataset
  --data_path DATA_PATH
                        Path to the dataset
  --image_subpath IMAGE_SUBPATH
                        Path to image subdirectory
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save the model checkpoints
  --epochs EPOCHS       Number of total epochs to run (Default: 30)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size of the dataset (Default: 32)
  --lr LR               Learning rate (Default: 0.0002)
  --beta1 BETA1         beta1 of Adam optimizer (Default: 0.5)
  --image_size IMAGE_SIZE
                        Image size (Default: 64)
  --channels CHANNELS   Number of image channels (Default: 1)
  --netD NETD           Path to Discriminator checkpoint
  --netG NETG           Path to Generator checkpoint
  --train TRAIN         Whether to load the pretrained model
```

During training, you can monitor the progress in real-time by following the URL provided in the command line.

## 3. Model Evaluation
To evaluate the performance of your trained model, run the following command from the root directory:

```bash
python eval.py --model_dir YOURMODEL
```

As with `train.py`, you can use the help command to view and modify evaluation options:

```bash
python eval.py -h
```

Upon completion, the evaluation results will be saved in an `eval_image` folder in the root directory, named with the current timestamp.
