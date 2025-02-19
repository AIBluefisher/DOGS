from omegaconf import OmegaConf

from conerf.base.model_base import ModelBase
from conerf.trainers.bags_gaussian_trainer import BagsTrainer
from conerf.trainers.deblur_gaussian_trainer import DeblurGaussianSplatTrainer
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer
from conerf.trainers.scaffold_gs_trainer import ScaffoldGSTrainer


def create_trainer(
    config: OmegaConf,
    prefetch_dataset=True,
    trainset=None,
    valset=None,
    model: ModelBase = None
):
    """Factory function for training neural network trainers."""
    if config.neural_field_type == "gs":
        if not config.geometry.get("deblur", False):
            trainer = GaussianSplatTrainer(
                config, prefetch_dataset, trainset, valset, model)
        else:
            if config.deblur.method == "deblur_3dgs":
                print('DeblurGaussianSplatTrainer!')
                trainer = DeblurGaussianSplatTrainer(
                    config, prefetch_dataset, trainset, valset, model)
            elif config.deblur.method == "bags":
                print('BagTrainer!')
                trainer = BagsTrainer(
                    config, prefetch_dataset, trainset, valset, model)
            else:
                raise NotImplementedError

    elif config.neural_field_type == "scaffold_gs":
        trainer = ScaffoldGSTrainer(
            config, prefetch_dataset, trainset, valset, model)

    else:
        raise NotImplementedError

    return trainer
