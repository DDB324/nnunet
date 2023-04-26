import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET 参数指定了使用 IPv4 地址族，SOCK_STREAM 参数指定了使用 TCP 协议。
    s.bind(("", 0))  # 双引号中的空字符串表示本地地址，0 表示让系统自动选择一个可用的端口号，并将其绑定到该地址上。在这种情况下，该函数返回绑定的实际端口号。
    port = s.getsockname()[1]  # 返回当前套接字绑定的本地地址和端口。它返回的是一个元组，元组的第一个元素是本地地址，第二个元素是绑定的端口号。
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks 加载nnunet类
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                 trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    # issubclass() 函数来判断 nnunet_trainer 是否是 nnUNetTrainer 类或其子类
    # 如果判断结果为 False，即 nnunet_trainer 不是 nnUNetTrainer 类或其子类，assert 语句就会抛出 AssertionError 异常，
    # 同时输出 'The requested nnunet trainer class must inherit from nnUNetTrainer' 的错误提示信息。
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from nnUNetTrainer'

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                  f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


# 初始化分布式数据并行的进程组
# 使用 PyTorch 提供的 dist 模块，将在当前进程中初始化 world_size 个进程，并将它们分组。
# 每个进程都会被分配一个 rank 值，从 0 到 world_size-1，指定它在进程组中的位置。这里使用的是 NCCL 后端
def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


# rank代表每个进程的编号
# world_size代表整个训练过程的进程数
# dataset_name_or_id、configuration、fold、tr、p、use_compressed、disable_checkpointing、c、val、pretrained_weights、npz参数和run_training函数中的参数含义相同。
def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, world_size):
    # setup_ddp函数配置分布式训练环境
    setup_ddp(rank, world_size)
    # 根据当前进程编号设置使用的GPU设备
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           use_compressed)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],  # 接受一个字符串或整数作为参数，用于指定数据集的名称或ID。
                 configuration: str, fold: Union[int, str],  # 接受一个字符串作为参数，用于指定模型的配置文件。
                 trainer_class_name: str = 'nnUNetTrainer',  # 接受一个字符串作为参数，用于指定训练器的类名。默认值为'nnUNetTrainer'
                 plans_identifier: str = 'nnUNetPlans',  # 接受一个字符串作为参数，用于指定计划的标识符。默认值为'nnUNetPlans'。
                 pretrained_weights: Optional[str] = None,  # 接受一个可选的字符串作为参数，用于指定预训练权重的路径。
                 num_gpus: int = 1,  # 接受一个整数作为参数，用于指定要使用的GPU数量。默认值为1。
                 use_compressed_data: bool = False,  # 接受一个布尔值作为参数，用于指定是否使用压缩数据进行训练。默认值为False。
                 export_validation_probabilities: bool = False,  # 接受一个布尔值作为参数，用于指定是否导出最终验证集上的softmax预测结果。默认值为False。
                 continue_training: bool = False,  # 接受一个布尔值作为参数，用于指定是否继续训练之前的模型。默认值为False。
                 only_run_validation: bool = False,  # 接受一个布尔值作为参数，用于指定是否仅运行验证（而不进行训练）。默认值为False。
                 disable_checkpointing: bool = False,  # 接受一个布尔值作为参数，用于指定是否禁用检查点。默认值为False。
                 device: torch.device = torch.device(
                     'cuda')):  # 接受一个torch.device对象作为参数，用于指定要使用的设备（如CPU或GPU）。默认值为torch.device('cuda')，即使用GPU。

    # fold有'all'或者0-4作为参数
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    # 这段代码的作用是检查num_gpus参数是否大于1。
    if num_gpus > 1:
        # 如果num_gpus大于1，则表示需要使用多个GPU进行训练。在这种情况下，该代码将检查设备类型是否为cuda，因为多GPU训练仅实现了cuda设备。
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        # os.environ是Python的一个内置模块，它可以用来访问和设置环境变量。
        # 设置环境变量MASTER_ADDR和MASTER_PORT
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():  # 返回一个包含所有当前环境变量名的列表。
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        # mp.spawn() 方法用于启动分布式数据并行训练（Distributed Data Parallel, DDP），通过将多个 GPU 连接在一起进行训练，以加快训练速度。
        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    # gpu<=1的情况
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
                continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse  # argparse模块解析命令行参数
    parser = argparse.ArgumentParser()  # 创建一个空的ArgumentParser对象，并将其赋值给parser变量,这个对象将用于定义和解析命令行参数
    # add_argument()方法向parser对象添加你需要的参数
    # 数据集
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    # 训练配置
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    # 设置交叉验证的测试集
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading "
                             "compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    # action：指定要执行的动作。在这里，store_true表示如果--npz选项被指定，则设置一个名为npz的变量为True。
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to '
                             'predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()  # 添加了所有需要的参数，就可以使用parse_args()方法从命令行中解析这些参数

    # 断言语句，用于确保args.device变量的值为'cpu'、'cuda'或者'mps'中的一个，否则会抛出一个AssertionError。
    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not ' \
                                                  f'tested/supported. Got: {args.device}. '
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing,
                 device=device)


if __name__ == '__main__':
    run_training_entry()
