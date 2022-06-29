import os
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastai.torch_core import TensorBase

class TensorMultiPoint(TensorBase):
    _show_args = dict(s=100, marker='.')

    @classmethod
    def create(cls, t, img_size=None)->None:
        "Convert an array or a list of points `t` to a `Tensor`"
        return cls(tensor(t).float(), img_size=img_size)

    def show(self, ctx=None, cmap=None, classes=None, **kwargs):
        if 'figsize' in kwargs: del kwargs['figsize']
        for (cls_idx,_,color,_) in classes:
            pts = np.array([(y,x) for idx,x,y in self if idx==cls_idx])
            if len(pts) == 0: continue
            ctx.scatter(pts[:, 0], pts[:, 1], color=color, **{**self._show_args, **kwargs})
        return ctx


def max_px(box, targ):
    m = np.zeros_like(targ)
    y,x,h,w = box
    m[:, x:x+w,y:y+h] = 1
    ind = np.unravel_index(np.argmax(targ*m, axis=None), targ.shape)
    return ind


def extract_predictions(lbl, pred, score_thresh=0.1, min_area=4):
    pscore = (pred > score_thresh).astype(np.uint8)
    contours,hierarchy = cv2.findContours(pscore.max(0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_area]
    pred_points = [max_px(pbox, pred) for pbox in pboxes]
    
    lbl_points = list(zip(*(lbl==1.0).nonzero()))
    
    return pred_points, lbl_points


def show_legend(classes):
    patches = []
    for idx, label, color, radius in classes:
        patches.append(mpatches.Patch(color=color, label=label))

    plt.legend(
        handles=patches,
        bbox_to_anchor=(1,1),
        loc='upper left'
    )


def save_predictions(args, classes, inps, preds, lbls, out, ho_lbls):
    """Loops through predictions and saves the target vs prediction image"""

    for idx in range(len(ho_lbls[:])):
        inp, _ = out[idx]
        pred, lbl = preds[idx], lbls[idx]
        pred_points, lbl_points = extract_predictions(
                lbl, 
                pred, 
                score_thresh=0.1,
                min_area=4
        )

        fig, ax = plt.subplots(ncols=2, figsize=(14,14))

        target = classes2name(lbl_points, classes)
        prediction = classes2name(pred_points, classes)

        fname = ho_lbls[idx]['External ID']

        t_title = "{} Target: {}".format(fname, target)
        p_title = "{} Prediction: {}".format(fname, prediction)

        ax[0].set_title(t_title)
        ctx = inp.show(ax[0])
        TensorMultiPoint(lbl_points).show(ctx=ctx, classes=classes)  

        ax[1].set_title(p_title)
        ctx = inp.show(ax[1])
        TensorMultiPoint(pred_points).show(ctx=ctx, classes=classes)   

        show_legend(classes)
        plt.savefig(os.path.join(args.output_images, 'p_vs_t_{}'.format(idx)))


def classes2name(points, classes):
    res = [0] * len(classes)
    for i in points:
        res[i[0]] += 1

    return res
