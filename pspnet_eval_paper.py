import numpy as np
from clusternet import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.cityscapes_dataset import Cityscapes

from utils.IoU_paper import AverageMeter, intersectionAndUnion, intersectionAndUnionGPU
import logging

# Set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Dataset
dataset_test = Cityscapes(split='test',
                           data_root='cityscapes_dataset',
                           data_list='cityscapes_dataset/list/cityscapes//val_set.txt')

# Dataloader
test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

# Criterion
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

# Load Model
with torch.no_grad():
    model = PSPNet(layers=50, bins=(2, 3, 6, 8), dropout=0.1, classes=35, zoom_factor=8, use_ppm=False, pretrained=True, criterion=criterion).to(device)
    model_path = '30-10-2022, 13:09:40/clusternet.pth'
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval() #SET MODEL TO EVAL MODE (FIXED PARAMETERS)

    if __name__ == '__main__':

        figure_count = 1

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        intersection_meter.reset()
        union_meter.reset()
        target_meter.reset()
        logger = get_logger()

        for img, label in test_dataloader:
            
            #CALCULATE NETWORK OUTPUT FOR INPUT IMG (FORWARD PASS)

            output = model(img.to(device))
            output = output.squeeze(0)
            soft_agg = torch.exp(output)
            soft_agg = torch.sum(soft_agg, 0)

            pred = output
            for x in range(35):
                soft_seg = torch.exp(pred[x, :, :])
                pred_seg = soft_seg/soft_agg
                pred_seg[pred_seg <= 0.5] = 0
                pred_seg[pred_seg > 0.5] = x

                pred[x, :, :] = pred_seg
            
            pred = torch.sum(pred, 0)

            label = label.to("cuda")
            label = label.squeeze(0)

            intersection, union, target = intersectionAndUnionGPU(pred, label, 35)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = torch.mean(iou_class)
        mAcc = torch.mean(acc_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


        logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

        iou_class = torch.Tensor.cpu(iou_class)
        iou_class = iou_class.detach().numpy()
        acc_class = torch.Tensor.cpu(acc_class)
        acc_class = acc_class.detach().numpy()

        np.savetxt('IoU.csv', iou_class, delimiter=' ')
        np.savetxt('Acc.csv', acc_class, delimiter=' ')