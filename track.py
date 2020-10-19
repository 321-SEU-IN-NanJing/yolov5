import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, bbox_iou)
from utils.torch_utils import select_device, load_classifier, time_synchronized


old_tracklets = []
baseid = 0
end_duration = 10
class_map = {cls: idx for idx, cls in enumerate(['car', 'bus', 'van', 'others'])}
rev_class_map = ['car', 'bus', 'van', 'others']

def detect(model, img, im0s, names, device):
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    dets = []
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # save results
            for *xyxy, conf, cls in det:
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                attributs = {'classes': cls.item(), 'boxes': [i.item() for i in xyxy], 'conf': conf.item()}
                dets.append(attributs)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))
    return dets


def ious(a, b):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    # print(a.shape, b.shape)
    temps = torch.ones((1, b.size(0)))
    for line in a:
        temps = torch.cat((temps, bbox_iou(line, b, DIoU=True).reshape(1, b.size(0))), dim=0)
    return temps.numpy()[1:]


def distance2d(a, b):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    temp1 = torch.cat(((a[:, 0:1] + a[:, 2:3])/2., (a[:, 1:2] + a[:, 3:4])/2.), dim=1)
    # assert temp1.size(1) == 2
    temp2 = torch.cat(((b[:, 0:1] + b[:, 2:3])/2., (b[:, 1:2] + b[:, 3:4])/2.), dim=1)
    temp1 = temp1.repeat([1, b.size(0)]).reshape([a.size(0), b.size(0), 2])
    temp2 = temp2.repeat([a.size(0), 1]).reshape([a.size(0), b.size(0), 2])
    temp3 = torch.square_(temp1 - temp2)
    temp4 = torch.sqrt_(temp3[..., 0] + temp3[..., 1])
    return temp4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    source, imgsz = opt.source, opt.img_size
    # Initialize
    device = select_device(opt.device)
    # Load model
    # model = torch.load(opt.weights, map_location=device)['model']
    model = attempt_load(opt.weights, map_location=device)
    model.to(device).eval()
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    with torch.no_grad():
        _ = model(img.float())

    for _, img, im0s, __ in dataset:
        with torch.no_grad():
            detecs = detect(model, img, im0s, names, device)
        if not len(old_tracklets):  # if no tracklet in last frame ,add new into it
            # print(detecs)
            for det in detecs:
                baseid += 1
                det['id'] = baseid
                det['statue'] = 'active'
                det['end_num'] = end_duration
                old_tracklets.append(det)
        else:
            if not len(detecs):  # if no detec in new frame, but tracklet exist, the same method of unmatched old_tracks
                for old_track in old_tracklets:
                    old_track['statue'] = 'hang on'
                    old_track['end_num'] -= 1
                    if old_track['end_num'] == -1:
                        old_track['statue'] = 'throw away'
                old_tracklets = [old_track for old_track in old_tracklets if old_track['statue'] != 'throw away']
            else:  # matching
                detec_boxes = [det_dict['boxes'] for det_dict in detecs]
                old_tracklets_boxes = [track_dict['boxes'] for track_dict in old_tracklets]
                cost_mat = 1 - ious(old_tracklets_boxes, detec_boxes)
                # cost_mat = distance2d(old_tracklets_boxes, detec_boxes)
                o_line_x, o_line_y = linear_sum_assignment(cost_mat)

                # gate control
                select_cost = [True for i in range(len(o_line_x))]
                for idx, line in enumerate(o_line_x):
                    # print(cost_mat[line, line_y[idx]])
                    if cost_mat[line, o_line_y[idx]] > 1:
                        select_cost[idx] = False
                line_x = o_line_x[select_cost]
                line_y = o_line_y[select_cost]

                # M X N , K matches
                # for match_one, idx in enumerate(line_x):
                # print("++++++++++++++ ", old_tracklets)
                for idx, line in enumerate(line_x):
                    old_tracklets[line]['boxes'] = detec_boxes[line_y[idx]]
                    old_tracklets[line]['conf'] = detecs[line_y[idx]]['conf']
                    old_tracklets[line]['classes'] = detecs[line_y[idx]]['classes']
                    old_tracklets[line]['statue'] = 'active'
                    old_tracklets[line]['end_num'] = end_duration

                # for unmatched old_tracks, entering throw away process
                old_tracklets = np.array(old_tracklets)
                selctive_idx = np.array([i for i in range(len(old_tracklets_boxes))])
                a = np.ma.array(selctive_idx, mask=False)
                a.mask[line_x] = True
                a = a.compressed()
                for i in old_tracklets[a]:
                    i['statue'] = 'hang on'
                    i['end_num'] -= 1
                    if i['end_num'] == -1:
                        i['statue'] = 'throw away'
                old_tracklets = old_tracklets.tolist()
                old_tracklets = [old_track for old_track in old_tracklets if old_track['statue'] != 'throw away']

                # for unmatched new_detecs
                selctive_idx = np.array([i for i in range(len(detec_boxes))])
                a = np.ma.array(selctive_idx, mask=False)
                a.mask[line_y] = True
                a = a.compressed()
                print("++++++++++++++1 ", old_tracklets)
                for det in np.array(detecs)[a]:
                    baseid += 1
                    det['id'] = baseid
                    det['statue'] = 'active'
                    det['end_num'] = end_duration
                    old_tracklets.append(det)
                # print("++++++++++++++2 ", old_tracklets)
        # draw
        if opt.view_img:  # Add bbox to image
            # print(old_tracklets)
            for objects in old_tracklets:
                # print('+++++', objects)
                if objects['statue'] == 'hang on':
                    # print('hang on not dis')
                    continue
                elif objects['statue'] == 'throw away':
                    raise AssertionError
                id = objects['id']
                xyxy = objects['boxes']
                conf = objects['conf']
                cls = objects['classes']
                label = '%d %.2f' % (id, conf)
                # print(objects['statue'])
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=2)
        cv2.imshow('tracking', im0s)
        cv2.waitKey(1)

