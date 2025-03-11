# pylint: disable=[E1101,W0621,E0401]

import copy
import os
import warnings
import logging

import omegaconf
from omegaconf import OmegaConf

from conerf.utils.config import config_parser, load_config
from conerf.utils.utils import setup_seed
from utils import create_trainer # pylint: disable=E0611

warnings.filterwarnings("ignore", category=UserWarning)


def run_cmd(cmd: str):
    os.system(cmd)

    return True


def train(config: OmegaConf):
    trainer = create_trainer(config)
    trainer.update_meta_data()
    trainer.train()
    # print(f"total iteration: {trainer.iteration}")


if __name__ == "__main__":
    args = config_parser()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )

    # parse YAML config to OmegaConf
    config = load_config(args.config)
    config["config_file_path"] = args.config

    assert config.dataset.scene != "" or args.scene != ""

    setup_seed(config.seed)

    if args.val != -1:
        config.dataset.val_interval = args.val

    scenes = []
    if args.scene != "":  # Overwrite scenes in config file.
        scenes.append(args.scene)
    else:
        if (
            type(config.dataset.scene) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
        ):
            scene_list = list(config.dataset.scene)
            for sc in config.dataset.scene:
                scenes.append(sc)
        else:
            scenes.append(config.dataset.scene)

    for scene in scenes:
        data_dir = os.path.join(config.dataset.root_dir, scene)
        assert os.path.exists(data_dir), f"Dataset does not exist: {data_dir}!"

        local_config = copy.deepcopy(config)
        local_config.expname = (
            f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene}"
        )
        local_config.expname = local_config.expname + "_" + args.suffix
        local_config.dataset.scene = scene
        local_config.dataset.model_folder = args.model_folder
        local_config.dataset.init_ply_type = args.init_ply_type
        local_config.dataset.load_specified_images = args.load_specified_images

        train(local_config)
