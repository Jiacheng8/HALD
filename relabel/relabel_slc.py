import argparse
import os
import numpy as np
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils_fkd import *
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='FKD Soft Label Generation w/ Mix Augmentation')
    parser.add_argument('--syn-data-path', required=True, type=str,
                        help='the path to the syn data which is being processed in this relabeling process')
    parser.add_argument('--online', action='store_true',
                        help='use online model')
    parser.add_argument('--model-choice', nargs='+', 
                        help='A list containing the choices of the compare model')
    parser.add_argument('--model-weight', nargs='+', 
                        help='A list containing the choices of the compare model')
    parser.add_argument('--img-mode', type=str,default="F",
                        help='whether to use the evaluation mode or not')
    parser.add_argument('--model-pool-dir', type=str, default=None,
                        help='required when pretrained model type is offline, the directory of the models when using offline mode')
    parser.add_argument('--fkd-path',required=True, type=str,
                        help='the path to save the fkd soft labels')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--dataset-name', default='cifar100', type=str,
                        help='dataset name')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    # FKD soft label generation args
    parser.add_argument('--SLC', default=100, type=int)
    parser.add_argument("--temperature", type=int, default=20,
                        help="soft max temperature")
    parser.add_argument("--min-scale-crops", type=float, default=0.08,
                        help="argument in RandomResizedCrop")
    parser.add_argument("--max-scale-crops", type=float, default=1.,
                        help="argument in RandomResizedCrop")
    parser.add_argument('--mode', default='fkd_save', type=str, metavar='N',)
    parser.add_argument('--fkd-seed', default=42, type=int, metavar='N')
    parser.add_argument('--mix-type', default = None, type=str, choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--workers-per-gpu', default=2, type=int,)
    

    
    return parser.parse_args()


# set up the mean, std and ncls for the dataset
def process_dataset(args):
    if args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.input_size = 32
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.input_size = 224
        args.n_bits = 14
        args.ratio = 90
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.jitter = 4
        args.input_size = 64
    else:
        raise ValueError('dataset not supported')


def worker_entry(rank, gpu_id, start_epoch, end_epoch, args):
    args = copy.deepcopy(args)
    args.gpu = gpu_id
    args.rank = rank
    args.start_epoch = start_epoch
    args.end_epoch = end_epoch
    for models in args.teacher_model_lis:
        models.cuda(args.gpu)
    run_worker(args)


def main():
    args = parse_args()
    process_dataset(args)
    
    # BSSL mode settings
    if args.img_mode == 'rded':
        args.eval_mode = 'T'
    else:
        args.eval_mode = 'F'
        
    # Print important arguments
    if args.eval_mode == 'T':
        print("🧊  BSSL Mode: \033[93mOFF\033[0m → 🔓 Using raw teacher outputs")
    else:
        print("🔥  BSSL Mode: \033[92mON\033[0m  → 🧠 Using Batch-Specific Soft Labelling")
    print(f"🧠  Total Loaded Teacher Models: \033[96m{len(args.model_choice)}\033[0m")
    print(f"📁  Loading Synthetic Data from: \033[94m{args.syn_data_path}\033[0m")
    total_gpus = torch.cuda.device_count()
    workers_per_gpu = args.workers_per_gpu  # or any m you want
    total_workers = total_gpus * workers_per_gpu
    ipc = int(count_img_files(args.syn_data_path) / args.ncls)
    print(f"🖼️  Each class has \033[92m{ipc}\033[0m images.")
    epochs = args.SLC // ipc
    total_epochs = epochs
    print(f"🚀 Launching {total_workers} workers over {total_gpus} GPUs for {args.SLC} SLC.")
    epochs_per_worker = total_epochs // total_workers
    processes = []
    
    # set up the fkd path

    
    args.fkd_path = args.fkd_path + f'_bs{args.batch_size}_slc{args.SLC}_ipc{ipc}'

    # load teacher models
    teacher_model_lis = []
    for model_name in args.model_choice:
        model = load_online_model(model_name, args) if args.online else load_model(args, model_name)
        # BSSL settings
        if args.eval_mode == 'T':
            model.eval()
        else:
            print(f"🧊  Setting model \033[93m{model_name}\033[0m to \033[92mtrain\033[0m mode for BSSL.")
            model.train()
        # # freeze all layers  
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        teacher_model_lis.append(model)


    normalize = transforms.Normalize(mean=args.mean_norm,
                                     std=args.std_norm)
    
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops,
                                               args.max_scale_crops),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    # processing the index list for each epoch
    epoch_index_info = {}
    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    dataset_len = len(train_dataset)
    for epoch in range(epochs):
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(args.fkd_seed + epoch)
        indices = torch.randperm(dataset_len, generator=epoch_generator).tolist()
        epoch_index_info[str(epoch)] = indices
    if not os.path.exists(args.fkd_path):
        os.makedirs(args.fkd_path)
    torch.save(epoch_index_info, os.path.join(args.fkd_path, 'sampler_indices_dict.pt'))

    # load pre-defined indices to the trainloader
    sampler = EpochBatchSampler(train_dataset) 
    sampler.indices_epoch = {int(k): v for k, v in epoch_index_info.items()}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)
    
    args.train_loader = train_loader
    args.sampler = sampler
    args.teacher_model_lis = teacher_model_lis
    args.epoch_index_info = epoch_index_info
    
    for rank in range(total_workers):
        gpu_id = rank % total_gpus
        start_epoch = rank * epochs_per_worker
        if rank == total_workers - 1:
            end_epoch = total_epochs
        else:
            end_epoch = start_epoch + epochs_per_worker

        print(f"🧵  \033[96mWorker {rank:>2}\033[0m on 🖥️  GPU \033[92m{gpu_id}\033[0m → 📆 Epochs \033[94m{start_epoch} ~ {end_epoch - 1}\033[0m")

        p = mp.Process(target=worker_entry, args=(rank, gpu_id, start_epoch, end_epoch, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def run_worker(args):
    args.epoch_batch = []
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), 
                    position=args.rank, 
                    desc=f"[GPU {args.gpu}] Worker {args.rank}", 
                    leave=False):
        args.sampler.set_epoch(epoch)
        dir_path = args.fkd_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save(args.train_loader, args.teacher_model_lis, dir_path, args, epoch)
        epoch_config_path = os.path.join(dir_path, 'epoch_{}.tar'.format(epoch))
        torch.save(args.epoch_batch, epoch_config_path)
        args.epoch_batch = []


@torch.no_grad()
def save(train_loader, model_lis, dir_path, args, epoch):
    if args.model_weight is None:
        weights = [1.0 / len(model_lis)] * len(model_lis)
    else:
        w = np.array([float(w) for w in args.model_weight])
        temperature = 10
        w = w / temperature
        weights = np.exp(w) / np.sum(np.exp(w))
    
    for batch_idx, (images, target, flip_status, coords_status, index) in enumerate(train_loader):
        images = images.cuda(args.gpu)
        images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)

        total_output = []
        for idx, model in enumerate(model_lis):
            output = model(images) * weights[idx]
            total_output.append(output)

        output = torch.stack(total_output, 0).sum(0)
        
        '''
        Applying SoftMax to the Embeddings.
        Comment below if you don't want to apply softmax in soft labels generation stage.
        '''
        soft_labels = F.softmax(output/args.temperature, dim=1)
            
        # '''
        # Prune irrelevant classes
        # '''
        # mix_index = mix_index.cuda(args.gpu)
        # reserved_index = retrieve_index_prior(args, target, mix_index)        
        # reserved_values = torch.gather(soft_labels, dim=1, index=reserved_index)
        
        '''
        Applying bit-wise quantization to the pruned soft labels.
        '''
        # quantized = float_to_int_n_torch(reserved_values, args.n_bits)
        # packed_bytes = pack_tensor(quantized, args.n_bits)
        
        # Simulating the process of decoding and compute mean error
        # packed_bytes = soft_labels.cpu()
        
        # # Simulating the process of decoding and compute mean error
        # unpacked_int = unpack_tensor(packed_bytes, soft_labels.shape, args.n_bits)
        # restored_soft_labels = int_n_to_float_torch(unpacked_int, args.n_bits).cuda(args.gpu)
        
        # # Compare with original
        # error = (soft_labels - restored_soft_labels).abs().float()
        # mean_error = error.mean().item()
        # print(f"Mean error: {mean_error:.6f}")
        soft_labels = soft_labels.half()
        
        batch_config = [coords_status.half(), flip_status, mix_index.cpu(), mix_lam, mix_bbox, soft_labels.cpu()]
        # batch_config = [packed_bytes.cpu()]
        # batch_config = [coords_status, flip_status, mix_index.cpu(), mix_lam, mix_bbox, reserved_values.cpu()]
        # batch_config = [reserved_values.cpu()]

        # batch_config = [packed_bytes.cpu()]
        # batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
        # torch.save(batch_config, batch_config_path)
        args.epoch_batch.append(batch_config)


if __name__ == '__main__':
    main()