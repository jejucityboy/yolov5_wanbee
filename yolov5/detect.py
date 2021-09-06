import argparse
import os
import platform
import shutil
import time as t
from pathlib import Path
import winsound

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, ap_per_class)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import datetime
from playsound import playsound
import requests



def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    pcount = 0
    fcount = 0

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = t.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                im0 = cv2.resize(im0, dsize=(1280, 960))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                score = 0
                person_x1 = 0
                person_x2 = 0
                person_y1 = 0
                person_y2 = 0

                # if len(det): print(det)

                for obj in reversed(det):
                    if names[int(obj[-1])] == "person":
                        print("person")
                        person_x1, person_y1, person_x2, person_y2 = obj[:4].type(torch.int)

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format\

                    # x1, y1, x2, y2 = [int(x) for x in xyxy]
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])

                    bbox=[(x1, y1), (x2, y2)]

                    w = x2 - x1
                    h = y2 - y1
                    center_x = x1 + w/2
                    center_y = y1 + h/2

                    if save_img or view_img :  # Add bbox to image

                        if names[int(cls)] == "helmet" :
                            print("helmet")
                            if person_x1 < center_x < person_x2 and person_y1 < center_y < person_y2 :
                                print("good")
                                # print(bbox)
                                score = score + 1

                        if names[int(cls)] == "safety belt" :
                            print("belt")
                            if person_x1 < center_x < person_x2 and person_y1 < center_y < person_y2 :
                                print("good")
                                # print(bbox)
                                score = score + 1

                        if names[int(cls)] == "safety vest" :
                            print("vest")
                            if person_x1 < center_x < person_x2 and person_y1 < center_y < person_y2 :
                                print("good")
                                # print(bbox)
                                score = score + 1

                        if names[int(cls)] == "head" :
                            print("head")
                            if person_x1 < center_x < person_x2 and person_y1 < center_y < person_y2 :
                                # print(bbox)
                                score = score - 1

                        if names[int(cls)] == "cap" :
                            print("cap")
                            if person_x1 < center_x < person_x2 and person_y1 < center_y < person_y2 :
                                # print(bbox)
                                score = score - 1

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                if score >= 2 :
                    cv2.putText(im0, "DETECTING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                    pcount = pcount + 1
                    print("pcount : ",  pcount)
                    if pcount > 50 :
                        cv2.putText(im0, "PASS", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

                        fcount = 0
                        if pcount == 51:
                            playsound('C:/practice/hi.mp3')
                            # cv2.imwrite("C:/practice/save/pass/" + str(now) + ".jpg", im0)
                            #
                            # # 보내고자하는 파일을 'rb'(바이너리 리드)방식 열고
                            # files = open("C:/practice/save/pass/" + str(now) + ".jpg", 'rb')
                            #
                            # date = datetime.datetime.now().strftime("%Y-%m-%d")
                            # time = datetime.datetime.now().strftime("%H:%M:%S")
                            # # 파이썬 딕셔너리 형식으로 file 설정
                            # upload = {'file': files}
                            # # String 포맷
                            #
                            # obj = {'date': date, 'time': time}
                            #
                            # # request.post방식으로 파일전송.
                            # res = requests.post('http://15.164.212.166:8080/success_image',
                            #                      files=upload,
                            #                      data=obj)
                            #
                            # print(res.text)

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)

                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                if score < 2:
                    cv2.putText(im0, "DETECTING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                    fcount = fcount + 1
                    print("fcount : ", fcount)
                    if fcount > 50 :
                        cv2.putText(im0, "FAILED", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                        pcount = 0
                        if fcount == 51 :
                            playsound('C:/practice/failed.mp3')
                            # cv2.imwrite("C:/practice/save/failed/" + str(now) + ".jpg", im0)
                            #
                            # # 보내고자하는 파일을 'rb'(바이너리 리드)방식 열고
                            # files = open("C:/practice/save/failed/"+ str(now) + ".jpg", 'rb')
                            #
                            # date = datetime.datetime.now().strftime("%Y-%m-%d")
                            # time = datetime.datetime.now().strftime("%H:%M:%S")
                            # # 파이썬 딕셔너리 형식으로 file 설정
                            # upload = {'file': files}
                            # # String 포맷
                            #
                            # obj = {'date': date, 'time': time}
                            #
                            # # request.post방식으로 파일전송.
                            # res = requests.post('http://15.164.212.166:8080/upload',
                            #                     files=upload,
                            #                     data=obj)
                            #
                            # print(res.text)

            else :
                # print("None")
                pcount = 0
                fcount = 0
                im0 = cv2.resize(im0, dsize=(1280, 960))
                cv2.putText(im0, "NONE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (t.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp16_wanbee20201122/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
