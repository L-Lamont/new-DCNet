import logging

from PIL import ImageFile, Image, ImageDraw
import numpy as np
from fastai.vision.all import *

class DataStore():
    def __init__(self, data):
        self.data = data

    def get_x(self, idx):
        return PILImage(self.data[idx][0])

    def get_y(self, idx):
        return TensorHeatMap(self.data[idx][1])


class DataStoreHighRes():
    def __init__(self, data):
        self.data = data

    def get_x(self, idx):
        return PILImage(self.data[1][idx])

    def get_y(self, idx):
        return TensorHeatMap(self.data[0][idx])


class TensorHeatMap(TensorBase):                                                
    _show_args = {'interpolation': 'nearest'}                                   
                                                                                
    def show(self, ctx=None, alpha=1.0, mask_alpha=1.0, **kwargs):              
        for m, (_,_,c,_) in zip(self,classes):                                  
            nodes = [0.0, 1.0]                                                  
            cs = [[0,0,0,0],c]                                                  
            cmap = LinearSegmentedColormap.from_list("cmap", list(zip(nodes, cs)))
            show_image(m, ctx=ctx, cmap=cmap, alpha=mask_alpha, **{**self._show_args, **kwargs})


def load_labels(args, dist):
    """Loads labels"""

    holdout_path = os.path.join(args.data, 'Model/holdout_lbls.txt')

    with open(holdout_path, 'r') as filehandle:
        holdout_lbls = json.load(filehandle)

    if dist.rank == 0:
        logging.info('Loaded labels')

    return holdout_lbls


def get_dataset(args, dist, image_path, holdout):
    """Builds and returns dataset"""

    img2msk_path = os.path.join(args.data, image_path)

    with open(img2msk_path, 'rb') as f:
        img2msk = pickle.load(f)

    if holdout or args.high_res == 0:
        datastore = DataStore(img2msk)
    else:
        datastore = DataStoreHighRes(img2msk)
    
    if not holdout:
        if args.high_res == 0:
            idxs = range(len(img2msk))
        else:
            idxs = range(len(img2msk[0]))

        splits = RandomSplitter(seed=args.seed)(idxs)
        dset = Datasets(
            items=idxs,
            tfms=[[datastore.get_x, ToTensor, IntToFloatTensor], [datastore.get_y]],
            splits=splits
        )
    else:
        idxs = range(len(load_labels(args, dist)))
    
        dset = Datasets(
            items=idxs,
            tfms=[[datastore.get_x, ToTensor, IntToFloatTensor], [datastore.get_y]]
        )

    if dist.rank == 0:
        logging.info('Built dataset (holdout={})'.format(holdout))
        logging.info('len(dataset): {}'.format(len(dset)))


    return dset


def get_dataloader(args, dist, image_path, holdout=False):
    """Builds and returns the dataloader"""

    if not holdout:
        batch_tfms = [
            *aug_transforms(
                max_zoom = 1.0,
                max_warp = 0,
                max_rotate = 45,
                flip_vert = True
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    else:
        batch_tfms = [
            Normalize.from_stats(*imagenet_stats)
        ]

    dsets = get_dataset(args, dist, image_path, holdout)

    dls = dsets.dataloaders(
        bs = args.batch_size,
        after_batch = batch_tfms,
        pin_memory = True
    )

    xb, yb = dls.one_batch()

    if dist.rank == 0:
        logging.info('Created dataloader (holdout={})'.format(holdout))
    
    return dls
