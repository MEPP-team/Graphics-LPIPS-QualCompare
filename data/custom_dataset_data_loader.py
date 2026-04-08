import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os

def CreateDataset(
        dataroots,
        dataset_mode='2afc',
        load_size=64, 
        trainset=False , 
        Nbpatches=205, 
        src_root=None, 
        root_refPatches=None, 
        root_distPatches=None, 
        target=None,
        cache_root=None
    ):
    dataset = None
    # Our dataset is baaset on the DSIS protocol (not 2afc). I adapted the code to suit DSIS. However, I did not change the function name.
    if dataset_mode=='2afc': # human judgements
        from data.dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from data.dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized." % dataset_mode)

    dataset.initialize(
        dataroots=dataroots,
        load_size=load_size,
        Trainset=trainset,
        maxNbPatches=Nbpatches,
        src_root=src_root,
        root_refPatches=root_refPatches,
        root_distPatches=root_distPatches,
        target=target,
        cache_root=cache_root
    )
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(
            self, 
            data_csvfile, 
            trainset=False, 
            Nbpatches=205, 
            dataset_mode='2afc',
            load_size=64,
            batch_size=600,
            serial_batches=True,
            nThreads=30,
            pin_memory=True,
            src_root=None,
            root_refPatches=None,
            root_distPatches=None,
            cache_root=None,
            target=None,
            **dl_kwargs
        ):
        BaseDataLoader.initialize(self)
        if(not isinstance(data_csvfile,list)):
            data_csvfile = [data_csvfile,]

        self.dataset = CreateDataset(
            data_csvfile,
            dataset_mode=dataset_mode,
            load_size=load_size,
            trainset=trainset,
            Nbpatches=Nbpatches,
            src_root=src_root,
            root_refPatches=root_refPatches,
            root_distPatches=root_distPatches,
            target=target,
            cache_root=cache_root
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            pin_memory=pin_memory,
            num_workers=int(nThreads),
            drop_last=True,
            **dl_kwargs
            )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
