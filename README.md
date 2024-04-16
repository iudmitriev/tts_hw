# TTS homework

## Installation guide

### Creating virtual enviroment

#### Using poetry

The best way to use this project is using [poetry](https://python-poetry.org/).
After installing poetry, run

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
If you want to use CUDA with this project, you need to [install](https://developer.nvidia.com/cuda-11-8-0-download-archive) it separately. The supported version is 11.8.

#### Using docker

Alternatively, you can use docker

```shell
docker build -t src_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	src_image 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

### Downloading dataset and checkpoint

To download ljspeech dataset, you should run

```shell
chmod +x scripts/download_dataset.sh
poetry run sh scripts/download_dataset.sh
```

To run model, you should calculate pitch and energy for all files in dataset by running

```shell
poetry run python scripts/extract_pitch_and_energy.py
```

To download model chekpoint, run

```shell
chmod +x scripts/download_best_model.sh
poetry run sh scripts/download_best_model.sh
```

## Best model

#### Description

The model is a implementation of FastSpeech2

#### Download results

To download the generated model files

```shell
poetry run sh scripts/download_results.sh
```

#### Training

To train this model independently, you should run

```shell
poetry run python train.py
```

The config for this model is file /src/config.yaml

#### Testing

To generate speech for specific text, you should change add it to test_texts.json and run

```shell
poetry run python test.py
```

The resulting voice recording will appear in /results/

## Credits

This homework was done by Ivan Dmitriev

This repository is based on a fork
of [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
