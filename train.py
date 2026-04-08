import shutil
from typing import Dict
import torch.backends.cudnn as cudnn
cudnn.benchmark=False
import torch

import numpy as np
import time
import os
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed
from Test_TestSet import Test_TestSet
from pathlib import Path
import csv
import multiprocessing
import train_helpers


class CUDAPrefetcher:
    def __init__(self, loader_iterable, device):
        self.loader = iter(loader_iterable)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.next = None
        self.preload()

    def preload(self):
        import numpy as np
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            moved = {}
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                    if t.dtype == torch.uint8 and t.dim() == 4:
                        t = t.pin_memory().to(self.device, non_blocking=True).to(torch.float32)
                        t = t.contiguous(memory_format=torch.channels_last)
                    elif t.dtype == torch.float32:
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    else:
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    moved[k] = t
                elif torch.is_tensor(v):
                    t = v
                    if t.device.type == 'cpu':
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    else:
                        t = t.to(self.device, non_blocking=True)
                    if t.dim() == 4 and t.dtype == torch.float32:
                        t = t.contiguous(memory_format=torch.channels_last)
                    moved[k] = t
                else:
                    moved[k] = v
            self.next = moved

    def __iter__(self): return self
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is None: raise StopIteration
        batch = self.next
        self.preload()
        return batch
    
def collate_to_numpy(batch):
    """Collate a batch into NumPy arrays for the CUDA prefetcher."""
    import numpy as np
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ('ref', 'p0', 'judge', 'mos'):
            arrs = []
            for v in vals:
                if torch.is_tensor(v):
                    arrs.append(v.numpy())
                elif isinstance(v, np.ndarray):
                    arrs.append(v)
                else:
                    arrs.append(np.asarray(v))
            out[k] = np.stack(arrs, axis=0)
        else:
            out[k] = np.array(vals)
    return out
def _format_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"
def clear_ssd_cache(cache_root: str, *, remove_root: bool=False, dry_run: bool=False) -> Dict[str, int | str]:
    """Recursively clear the SSD cache and return deletion statistics."""
    if not cache_root:
        return {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0, "human_freed": "0 B"}

    root = Path(cache_root)
    if not root.exists() or not root.is_dir():
        return {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0, "human_freed": "0 B"}

    files = []
    dirs  = []

    if remove_root:
        for p in root.rglob("*"):
            (files if p.is_file() else dirs).append(p)
        dirs.append(root)
    else:
        for p in root.rglob("*"):
            (files if p.is_file() else dirs).append(p)

    bytes_freed = 0
    for f in files:
        try:
            bytes_freed += f.stat().st_size
        except Exception:
            pass

    files_deleted = len(files)
    dirs_deleted  = len(dirs)

    if not dry_run:
        if remove_root:
            shutil.rmtree(root, ignore_errors=True)
        else:
            for child in root.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except Exception as e:
                    pass

    return {
        "files_deleted": files_deleted,
        "dirs_deleted": dirs_deleted,
        "bytes_freed": bytes_freed,
        "human_freed": _format_bytes(bytes_freed),
    }
os.environ['PYTHONWARNINGS'] = 'ignore'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='', help='datasets to train on')
    parser.add_argument('--testcsv', type=str, default='', help='datasets to test on')
    parser.add_argument('--different_testset', '-dt', action='store_true', default=False, help='use different testset than trainset. If so, provide 2 src_root, root_refPatches, root_distPatches for train and testset respectively.')
    parser.add_argument('--use_folds', action='store_true', help='use k-folds for testing')
    
    parser.add_argument('--src_root', nargs='+', help='root folder containing ref and dist folders')
    parser.add_argument('--cache_root', type=str, default="", help='root folder for caching viewpoints on SSD. Leave empty to disable caching.')
    parser.add_argument('--root_refPatches', type=str, help='reference patches relative location')
    parser.add_argument('--root_distPatches', type=str, help='distorted patches relative location')
    parser.add_argument('--name', type=str, help='directory name for training')
    parser.add_argument('--target', type=str, help='type of ground truth scores: mos or judges')

    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU', default=True)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

    parser.add_argument('--nThreads', type=int, default=10, help='number of threads to use in data loader') 
    
    parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
    parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
    parser.add_argument('--npatches', type=int, default=150, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='# initial learning rate')
    
    parser.add_argument('--testset_freq', type=int, default=2, help='frequency of evaluating the testset')
    parser.add_argument('--display_freq', type=int, default=50000, help='frequency (in instances) of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=50000, help='frequency (in instances) of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency (in instances) of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
    parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
    parser.add_argument('--use_html', action='store_true', help='save off html pages')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')

    parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
    parser.add_argument('--train_plot', action='store_true', help='plot saving')

    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg
    
    opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    load_size = 64

    visualizer = Visualizer(opt)

    if opt.use_folds:
        num_folds = 5
    for fold in range(num_folds if opt.use_folds else 1):
        if(opt.use_folds):
            print('--- Starting fold k%d ---'%fold)
        trainer = lpips.Trainer()
        trainer.initialize(
            model=opt.model,
            net=opt.net,
            use_gpu=True,
            is_train=True,
            lr=opt.lr,
            pnet_rand=opt.from_scratch,
            pnet_tune=opt.train_trunk,
            gpu_ids=[0]
        )
        print("Model on:", next(trainer.net.parameters()).device)
        if(opt.use_folds):
            opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name,'fold_k'+str(fold))
        else: 
            opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
        if(not os.path.exists(opt.save_dir)):
            os.mkdir(opt.save_dir)
        elif opt.use_folds:
            print('fold %d already exists, skipping...' % fold)
            continue  # skip existing fold

        Testset = opt.testcsv
        Testset_name, ext = os.path.splitext(Testset)
        
        if(opt.use_folds):
            Testset = Testset_name + '_k' + str(fold) + ext
        data_loader_testSet = dl.CreateDataLoader(Testset,dataset_mode='2afc', Nbpatches= opt.npatches, batch_size=opt.batch_size,
                                                pin_memory=False, drop_last=False, prefetch_factor=None, nThreads=0,
                                                src_root=opt.src_root[1] if opt.different_testset else opt.src_root[0], 
                                                root_refPatches=opt.root_refPatches, 
                                                root_distPatches=opt.root_distPatches, 
                                                cache_root=opt.cache_root,
                                                target = opt.target) 
        test_TestSet = Test_TestSet(opt)
        total_steps = 0
        
        start_time = time.time()
        print('Start training with the following options:')
        for k, v in sorted(vars(opt).items()):  
            print('%s: %s' % (str(k), str(v)))
        print('Total number of patches: %d, batch size: %d, input images per batch: %d' % (opt.npatches, opt.batch_size, opt.nInputImg))
        print('Total number of epochs: %d, learning rate: %.6f' % (opt.nepoch + opt.nepoch_decay, opt.lr))



        for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
                trainSet = opt.datasets
                if(opt.use_folds):
                    trainSet_name, ext = os.path.splitext(trainSet)
                    trainSet = trainSet_name + '_k' + str(fold) + ext
                    
                data_loader = dl.CreateDataLoader(trainSet,dataset_mode='2afc', trainset=True, Nbpatches=opt.npatches, 
                                            load_size = load_size, batch_size=opt.batch_size, serial_batches=True, nThreads=opt.nThreads, 
                                            pin_memory=True, persistent_workers=True, prefetch_factor=2,  # prefetch_factor=2,
                                            src_root=opt.src_root[0], 
                                            root_refPatches=opt.root_refPatches, 
                                            root_distPatches=opt.root_distPatches, 
                                            cache_root=opt.cache_root,
                                            target=opt.target)
                dataset = data_loader.load_data()
                dataset_size = len(data_loader)
                D = len(dataset)
        
                num_batches = len(dataset)
                num_samples = len(dataset.dataset)
                print(f'Epoch {epoch}, batches: {num_batches}, samples: {num_samples}, bs={opt.batch_size}, workers={opt.nThreads}')

                device = torch.device('cuda:0')
                prefetch = CUDAPrefetcher(dataset, device)

                epoch_start_time = time.time()
                nb_batches = 0 
                Loss_trainset = 0 
                for i, data in enumerate(prefetch): 
                    iter_start_time = time.time()
                    total_steps += opt.batch_size
                    epoch_iter = total_steps - dataset_size * (epoch - 1)

                    trainer.set_input(data)
                    if i == 0:
                        try:
                            pdev = next(trainer.net.parameters()).device
                        except StopIteration:
                            pdev = "no-params"
                        print(f"[DEV] torch.cuda.is_available={torch.cuda.is_available()} | "
                            f"net={pdev} | ref={trainer.ref.device} | p0={trainer.p0.device} | "
                            f"amp={trainer.use_amp} dtype={trainer.amp_dtype}")
                        print("ref dtype/range:", trainer.ref.dtype, float(trainer.ref.min()), float(trainer.ref.max()))
                        assert torch.cuda.is_available()
                        assert str(pdev).startswith("cuda")
                        assert str(trainer.ref.device).startswith("cuda")
                        assert str(trainer.p0.device).startswith("cuda")
                    if i%50 == 0:
                        print('Epoch %d, Batch %d / %d, Total Steps %d' % (epoch, i, dataset_size, total_steps))
                    trainer.optimize_parameters()

                    errors = trainer.get_current_errors()
                    Loss_trainset += errors['loss_total']
                    nb_batches += 1 

                if epoch % opt.save_epoch_freq == 0:
                    print('saving the model at the end of epoch %d, iters %d' %
                        (epoch, total_steps))
                    trainer.save(opt.save_dir, 'latest')
                    trainer.save(opt.save_dir, epoch)
                    
                    print('nb batch %.1f'%nb_batches)
                    Loss_trainset = Loss_trainset/nb_batches
                    print('Epoch Loss %.6f'%Loss_trainset)
                    resPerEpoch = dict([('Trainset_Totalloss', Loss_trainset)])
                    
                    for key in resPerEpoch.keys():
                        visualizer.plot_current_errors_save(epoch, float(0), opt, resPerEpoch, keys=[key,], name=key, to_plot=opt.train_plot)


                if epoch % opt.testset_freq == 0:
                    ld = data_loader_testSet
                    if hasattr(ld, "load_data"):
                        tmp = ld.load_data()
                        if hasattr(tmp, "__iter__"):
                            ld = tmp
                        elif hasattr(tmp, "dataloader"):
                            ld = tmp.dataloader
                    else:
                        ld = getattr(ld, "dataloader", ld)

                    res_testset = trainer.Testset_DSIS(ld)
                    print(f"[TestSet] SROCC={res_testset['SROCC']:.4f}")

                    with torch.no_grad():
                        if "mos_pred" in res_testset and "mos_true" in res_testset:
                            pred = torch.from_numpy(res_testset["mos_pred"]).to(trainer.device)
                            true = torch.from_numpy(res_testset["mos_true"]).to(trainer.device)
                            test_loss = float(trainer.rankLoss(pred, true).mean().item())
                        else:
                            # Fall back to the aggregated legacy loss when raw predictions are unavailable.
                            test_loss = float(res_testset.get("loss", float("nan")))
                    print(f"[TestSet] loss={test_loss:.6f}")

                    res_plot = {
                        "SROCC": float(res_testset["SROCC"]),
                        "loss":  test_loss,                 
                    }
                    keys_to_plot = ["SROCC", "loss"]

                    test_TestSet.plot_TestSet_save(
                        epoch=epoch,
                        res=res_plot,
                        keys=keys_to_plot,
                        name="TestSet",                     
                        to_plot=opt.train_plot,
                        what_to_plot="TestSet_Res",
                    )

                    info = (
                        f"{opt.nepoch},{opt.nepoch_decay},{opt.npatches},{opt.nInputImg},"
                        f"{opt.lr},{epoch},{Loss_trainset},{test_loss},{res_testset['SROCC']}\n"
                    )
                else:
                    info = (
                        f"{opt.nepoch},{opt.nepoch_decay},{opt.npatches},{opt.nInputImg},"
                        f"{opt.lr},{epoch},{Loss_trainset}\n"
                    )

                print('End of epoch %d / %d \t Time Taken: %d sec' %
                    (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

                if epoch > opt.nepoch:
                    trainer.update_learning_rate(opt.nepoch_decay)

        print( 'End of %d epochs. Time taken: %d sec' %(opt.nepoch + opt.nepoch_decay,  time.time() -  start_time))
        print( 'Clearing the cache ...')
        if hasattr(opt, "cache_root") and opt.cache_root:
            stats = clear_ssd_cache(opt.cache_root, remove_root=False, dry_run=False)
            print(f"[cache] removed {stats['files_deleted']} files, {stats['dirs_deleted']} dirs | freed {stats['human_freed']}")

    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
