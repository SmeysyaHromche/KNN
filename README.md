# KNN - OCR on historical dataset
### Authors: Jan Čech, Matěj Čech, Myron Kukhta
### Date: 10.05.2026

## Project Overview

This project is a student research work focused on OCR (Optical Character Recognition) for historical handwritten documents. The main goal of the project is to investigate the applicability of the Transformer Decoder architecture with a self-attention mechanism for converting handwritten text into machine-readable format.

The model is trained on a dataset provided by the university, consisting of more than 200,000 images of historical handwritten texts in German, French, and Spanish.

The model architecture consists of three main components:

* **Image Backbone** — a Swin Transformer v2 model used for extracting visual image features (with optional usage of VGG and ConvNeXt);
* **Visual Adapter** — a module responsible for stabilizing visual tokens and projecting them into the vocabulary space;
* **Transformer Attention Decoder** — a self-attention-based decoder that generates textual sequences.

The project is primarily implemented using PyTorch and related Python ecosystem tools for training and researching machine learning models.

### Enviroment

To simplify environment setup, the project includes an `env.sh` script located in the root directory. Its primary purpose is to automate the creation and management of a Python virtual environment for running and developing the project.

The script supports several operating modes:

```bash
Usage: ./env.sh [OPTION]

Options:
  -c, --create         Create environment if it does not exist
  -r, --recreate       Recreate environment from scratch
  -d, --delete         Delete environment
  -a, --activate       Activate environment (must be sourced)
  -e, --extra NAME     Install optional dependency group into existing environment
                       Example: ./env.sh --extra dev

  -t, --torch TYPE     Install PyTorch into existing environment
                       TYPE: cpu | cuda
                       Example: ./env.sh --torch cuda

  -h, --help           Show this help message
```

### Data Preparation

The project uses a data storage format inspired by the datasets provided by the faculty. Images are expected to be stored in an LMDB database, while separate annotation files must be prepared for the `train`, `test`, and `validation` splits using the following format:

```text
id_img label
```

where:

* `id_img` — the image key stored in the LMDB database;
* `label` — the corresponding text transcription.

After preparing the `.txt` annotation files, the `scripts/file_to_lmdb.py` script should be executed. This script converts the annotation files into an LMDB-based format optimized for fast parsing and efficient `DataLoader` performance (after as LMDB metadata file).

Detailed information about available arguments can be obtained with:

```bash
python scripts/file_to_lmdb.py --help
```

### Model Training

After preparing the environment and datasets, the training process can be launched using a configuration file in JSON format (`learnconfig.json` in the root of project).

As a result of the training process, the model checkpoints and training logs will be saved to the directory specified in the `train.output_model_dir` configuration parameter.

Example training configuration file:

```json
{
    "data":{
        "path_to_trn_meta_db": ".data/lines_split_train.lmdb",
        "path_to_vld_meta_db": ".data/lines_split_valid.lmdb",
        "path_to_tst_meta_db": "",
        "path_to_db": ".data/lines_48-1.15.lmdb",
        "path_to_vocabulary_file": "src/common/vocabulary.txt",
        "image_target_height": 32,
        "batch_size": 64,
        "num_workers_train": 1,
        "num_workers_validation": 1
    },
    "train":{
        "num_of_epochs": 10,
        "save_model_per_epoch": true,
        "output_model_dir": "src/learn/output",
        "optimizer_lr": 1e-4,
        "device": "cuda",

        "unfreeze_swin_epoch": 2,
        "unfreeze_swin_norms_epoch": -1,
        "swin_optimizer_lr": 2e-5,

        "unfreeze_vgg_epoch": 3,
        "vgg_optimizer_lr": 5e-5,

        "unfreeze_convnext_epoch": 3,
        "convnext_optimizer_lr": 5e-5
    },
    "model":{
        "feature_extractor": "convnext",

        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_seq_len": 1024,

        "is_pretrain_swin": true,
        "is_pretrain_vgg": true,
        "is_pretrain_convnext": true
    }
}
```

Configuration Description:

#### `data`

Dataset loading and preprocessing parameters.

| Parameter                 | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `path_to_trn_meta_db`     | Path to the LMDB metadata file used for training.               |
| `path_to_vld_meta_db`     | Path to the LMDB metadata file used for validation.             |
| `path_to_tst_meta_db`     | Path to the LMDB metadata file used for testing.                |
| `path_to_db`              | Path to the main LMDB image database.                           |
| `path_to_vocabulary_file` | Path to the vocabulary file used for tokenization and decoding.  Use provided `src/common/vocabulary.txt` |
| `image_target_height`     | Target image height after preprocessing.                        |
| `batch_size`              | Number of samples processed in a single batch.                  |
| `num_workers_train`       | Number of worker processes used by the training `DataLoader`.   |
| `num_workers_validation`  | Number of worker processes used by the validation `DataLoader`. |

#### `train`

Training process parameters.

| Parameter                   | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| `num_of_epochs`             | Total number of training epochs.                        |
| `save_model_per_epoch`      | Enables checkpoint saving after each epoch.             |
| `output_model_dir`          | Directory where model checkpoints are stored.           |
| `optimizer_lr`              | Base learning rate for the optimizer.                   |
| `device`                    | Device used for training (`cpu` or `cuda`).             |
| `unfreeze_swin_epoch`       | Epoch at which the Swin backbone becomes trainable.     |
| `unfreeze_swin_norms_epoch` | Epoch at which Swin normalization layers are unfrozen.  |
| `swin_optimizer_lr`         | Learning rate used for the Swin backbone.               |
| `unfreeze_vgg_epoch`        | Epoch at which the VGG backbone becomes trainable.      |
| `vgg_optimizer_lr`          | Learning rate used for the VGG backbone.                |
| `unfreeze_convnext_epoch`   | Epoch at which the ConvNeXt backbone becomes trainable. |
| `convnext_optimizer_lr`     | Learning rate used for the ConvNeXt backbone.           |

#### `model`

Model architecture parameters.

| Parameter              | Description                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `feature_extractor`    | Backbone architecture used for visual feature extraction (`swin`, `vgg`, `convnext`). |
| `d_model`              | Transformer embedding dimension.                                                            |
| `nhead`                | Number of attention heads in the Transformer decoder.                                       |
| `num_layers`           | Number of Transformer decoder layers.                                                       |
| `dim_feedforward`      | Hidden dimension of the feedforward network inside Transformer blocks.                      |
| `dropout`              | Dropout probability used in the Transformer layers.                                         |
| `max_seq_len`          | Maximum generated sequence length.                                                          |
| `is_pretrain_swin`     | Enables pretrained weights for the Swin backbone.                                           |
| `is_pretrain_vgg`      | Enables pretrained weights for the VGG backbone.                                            |
| `is_pretrain_convnext` | Enables pretrained weights for the ConvNeXt backbone.                                       |


### Running the Training Process

Example command:

```bash
python3 -m src.learn.train
```

**Important:** The application always loads its configuration from the `learnconfig.json` file located in the project root directory.


### Model Evaluation

After preparing the environment and datasets, the evaluation process can be launched using a configuration file in JSON format (`evalconfig.json` in the root of project).

As a result of the evaluation process, the metrics results will be in CLI output and logs of prediction will be saved to the directory specified in the `output.output_model_dir` configuration parameter.

Example training configuration file:

```json
{
    "data": {
        "dataset": "ocr",
        "path_to_tst_meta_db": "/home/xkukht01/knn_data/knn_ocr/read/valid.lmdb",    
        "path_to_db": "/home/xkukht01/knn_data/knn_ocr/read/lines_48-1.15.lmdb",
        "path_to_vocabulary_file": "src/common/vocabulary.txt",
        "image_target_height": 32,
        "batch_size": 64
    },
    "model": {
        "path_to_model": "/home/xkukht01/Dev/KNN/out/swin/epoch_20.pt",
        "device": "cpu",
        "is_pretrain_swin": true,
        "max_seq_len": 1024,
        "img_pad_value": 1.0
    },
    "output": {
        "path_to_output_file": "src/evaluation/model_evaluation.txt"
    }
}
```

Configuration Description:

#### `data`

Parameters related to dataset loading and preprocessing.

| Parameter                 | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `dataset`                 | Dataset type identifier. Use or `iam` (from kaggle) or `ocr` (provided by faculty )                                       |
| `path_to_tst_meta_db`     | Path to the LMDB database used for validation or testing.       |
| `path_to_db`              | Path to the main LMDB dataset used during training.             |
| `path_to_vocabulary_file` | Path to the vocabulary file used for tokenization and decoding. Use provided `src/common/vocabulary.txt` |
| `image_target_height`     | Target image height after preprocessing.                        |
| `batch_size`              | Number of samples processed in a single batch.                  |

#### `model`

Model and inference/training configuration.

| Parameter          | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `path_to_model`    | Path to a pretrained model checkpoint.                      |
| `device`           | Device used for execution (`cpu` or local cuda name device by default `cuda`).                |
| `is_pretrain_swin` | Enables loading pretrained weights for the Swin backbone.   |
| `max_seq_len`      | Maximum output sequence length generated by the decoder.    |
| `img_pad_value`    | Padding value applied to input images during preprocessing. |

#### `output`

Output configuration.

| Parameter             | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `path_to_output_file` | Path to the file where evaluation results or logs will be saved. |



### Running the Evaluation Process

Example command:

```bash
python3 -m src.evaluation.evaluator
```

**Important:** The application always loads its configuration from the `evalconfig.json` file located in the project root directory.