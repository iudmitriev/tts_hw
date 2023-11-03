import collections
import warnings

import numpy as np
import torch
import hydra
import logging

from omegaconf.dictconfig import DictConfig

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
import src.text_encoder as module_text_encoder
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(version_base=None, config_path="src", config_name="config")
def main(config: DictConfig):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    # text_encoder
    if "text_encoder" not in config:
        text_encoder = module_text_encoder.CTCCharTextEncoder()
    else:
        text_encoder = hydra.utils.instantiate(config["text_encoder"])

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    model = hydra.utils.instantiate(config["arch"], n_class=len(text_encoder))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = hydra.utils.instantiate(config["loss"]).to(device)

    metrics = [
        hydra.utils.instantiate(metric, text_encoder=text_encoder)
        for metric_name, metric in config["metrics"].items()
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = hydra.utils.instantiate(config["optimizer"], trainable_params)
    lr_scheduler = hydra.utils.instantiate(config["lr_scheduler"], optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    main()
