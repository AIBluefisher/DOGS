import argparse
from omegaconf import OmegaConf


def strs2ints(strs):
    strs = strs.split(',')
    ints = []
    for num in strs:
        ints.append(int(num))
    print(f'ints: {ints}')
    return ints


def calc_milestones(max_step, muls, divs):
    # muls, divs = strs2ints(muls), strs2ints(divs)
    milestones = "["
    for mul, div in zip(muls, divs):
        milestones += str(max_step * mul // div)
        milestones += ","
    real_milestones = milestones[:-1]
    real_milestones += "]"
    return real_milestones


OmegaConf.register_new_resolver(
    'calc_exp_lr_decay_rate',
    lambda factor, n: factor**(1./n)
)
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('divi', lambda a, b: a // b)
OmegaConf.register_new_resolver(
    'calc_milestones',
    lambda max_step, muls, divs: calc_milestones(
        max_step, muls, divs)  # pylint: disable=W0108
)


def config_parser():
    parser = argparse.ArgumentParser()

    ##################################### Base configs ########################################
    parser.add_argument("--config",
                        type=str,
                        default="",
                        help="absolute path of config file")
    parser.add_argument("--suffix",
                        type=str,
                        default="",
                        help="suffix for training folder")
    parser.add_argument("--scene",
                        type=str,
                        default="",
                        help="name for the trained scene")
    parser.add_argument("--expname",
                        type=str,
                        default="",
                        help="experiment name")
    parser.add_argument("--model_folder",
                        type=str,
                        default="sparse",  # ['sparse', 'zero_gs']
                        help="folder that contain colmap model output")
    parser.add_argument("--init_ply_type",
                        type=str,
                        default="sparse",  # ['sparse', 'dense']
                        help="use dense or sparse point cloud to initialize 3DGS")
    parser.add_argument("--load_specified_images",
                        action="store_true",
                        help="Only load the specified images to train.")
    parser.add_argument("--val",
                        type=int,
                        default=-1,
                        help="Validation interval")

    ##################################### Block Training ########################################
    parser.add_argument("--block_id",
                        type=int,
                        default=0,
                        help="block id")
    parser.add_argument("--block_data_path",
                        type=str,
                        default="",
                        help="directory that stores the block data")
    parser.add_argument("--train_local",
                        action="store_true",
                        help="train local blocks")

    ##################################### registration ########################################
    parser.add_argument("--position_embedding_type",
                        type=str,
                        default="sine",
                        help="which kind of positional embedding to use in transformer")
    parser.add_argument("--position_embedding_dim",
                        type=int,
                        default=256,
                        help="dimensionality of position embeddings")
    parser.add_argument("--position_embedding_scaling",
                        type=float,
                        default=1.0,
                        help="position embedding scale factor")
    parser.add_argument("--num_downsample",
                        type=int,
                        default=6,
                        help="how many layers used to downsample points")
    parser.add_argument("--robust_loss",
                        action="store_true",
                        help="whether to use robust loss function")

    #################################### composite inr blocks #################################
    parser.add_argument("--enable_composite",
                        action="store_true",
                        help="whether to composite implicit neural representation blocks.")

    args = parser.parse_args()

    return args


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)

    return conf
