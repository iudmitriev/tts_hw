# ASR homework

## Installation guide

#### Using docker
The best way to use this project is using docker

```shell 
docker build -t src_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	src_image python -m unittest 
```
Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize


#### Using poetry
Alternatively, you can install all dependencies using [poetry](https://python-poetry.org/). 
```shell 
poetry install
```
This command will create virtual enviroment. You can either enter it using
```shell 
poetry shell
```
or start all commands with poetry
```shell 
poetry run python train.py -c config.json
```

Note:
If you prefer this way and want to use CUDA with this project, you need to [install](https://developer.nvidia.com/cuda-11-8-0-download-archive) it separately. The supported version is 11.8.

## Best model
#### Description
The best result was achieved using DeepSpeech2 model and librispeech-4-gram language model

#### Download guide
To download the best model weights and config, you need to run download_best_model.sh
```shell 
sh download_best_model.sh

# If you are using poetry, you should still start this line with poetry run
# poetry run sh download_best_model.sh
```

#### Training
To train this model independently, you should run
```shell 
python train.py -c src/configs/config.json
```

#### Scores
The best model achives the following scores on Librespeech dataset
|  Dataset   |  CER   |  WER  |
| ---------- | ------ | ----- |
| test-clean |  4.36  | 11.23 |
| test-other |  14.57 | 29.44 |

#### Testing
To verify score on Librespeech-test-other dataset you should download model and run
```shell 
python test.py
```

## Credits
This homework was done by Ivan Dmitriev

This repository is based on a fork
of [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
