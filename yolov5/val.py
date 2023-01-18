# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm
import pandas as pd
### order of columns: img_name, l3/l2pred
##### pdffk = pd.read_csv("test_dumps_orig_list2_out.txt",sep='\t',header=None)
# pdffk = pd.read_csv("test_dumps_orig_list2_outl3.txt",sep='\t',header=None)
# pdffk = pd.read_csv("test_dumps_orig_list2_outl2.txt",sep='\t',header=None)
# pdffk = pd.read_csv("test_dumps_orig_list2_outl2_new.txt",sep='\t',header=None)
# pdffk = pd.read_csv("test_dumps_orig_list2_outl1.txt",sep='\t',header=None)
pdffk = pd.read_csv("test_dumps_orig_list2_outl1_new.txt",sep='\t',header=None)
fnamespdf = list(pdffk[0])
fnamespdf_cut = ['_'.join(fi.split('_')[:-1]) for fi in fnamespdf]

### order of columns: img_name, l3gt_str, l3pred_str
# pdfik = pd.read_csv("test_dumps_orig_gtvalid_wpred_list2.txt",sep='\t',header=None)
### order of columns: img_name, l2pred_idx, l2gt_idx
# pdfik = pd.read_csv("test_dumps_orig_list2_outl3.txt",sep='\t',header=None)

# pdfik = pd.read_csv("test_dumps_orig_gtvalid_list2_outl2.txt",sep='\t',header=None)
pdfik = pd.read_csv("test_dumps_orig_gtvalid2_list2_outl2.txt",sep='\t',header=None)

# pdfik = pd.read_csv("test_dumps_orig_gtvalid_list2_outl1.txt",sep='\t',header=None)
# pdfik = pd.read_csv("test_dumps_orig_list2_outl1.txt",sep='\t',header=None)
gtfnamespdf = list(pdfik[0])

l3_new_index_mapper = list(pd.read_csv("l3_new_index_mapper.csv")['l3_new_index'])
l3_class_indexes = {'autorickshaw_Atul_Atul': 1, 'autorickshaw_autorickshaw_autorickshaw': 2, 'autorickshaw_Bajaj_Bajaj': 3, 'autorickshaw_Mahindra_Mahindra': 4, 'autorickshaw_Piaggio_Piaggio': 5, 'autorickshaw_TVS_TVS': 6, 'bus_bus_bus': 7, 'car_Audi_A3': 8, 'car_Audi_Q3': 9, 'car_Bmw_2-Series-220d': 10, 'car_Bmw_3-Series': 11, 'car_Bmw_X1': 12, 'car_car_car': 13, 'car_Chevrolet_Aveo': 14, 'car_Chevrolet_Beat': 15, 'car_Chevrolet_Cruze': 16, 'car_Chevrolet_Enjoy': 17, 'car_Chevrolet_Spark': 18, 'car_Chevrolet_Tavera': 19, 'car_Fiat_Linea': 20, 'car_Fiat_PuntoEvo': 21, 'car_Force_TraxToofan': 22, 'car_Ford_Aspire': 23, 'car_Ford_Ecosport': 24, 'car_Ford_EcoSportTitanium': 25, 'car_Ford_Everest': 26, 'car_Ford_Fiesta': 27, 'car_Ford_Figo': 28, 'car_Ford_Ikon': 29, 'car_Hindustan_Ambassador': 30, 'car_Honda_Accord': 31, 'car_Honda_Amaze': 32, 'car_Honda_Brio': 33, 'car_Honda_Brv': 34, 'car_Honda_City': 35, 'car_Honda_Civic': 36, 'car_Honda_Cr-V': 37, 'car_Honda_Jazz': 38, 'car_Honda_Wr-V': 39, 'car_Hyundai_Accent': 40, 'car_Hyundai_Aura': 41, 'car_Hyundai_Creta': 42, 'car_Hyundai_Eon': 43, 'car_Hyundai_I10': 44, 'car_Hyundai_I20': 45, 'car_Hyundai_Santro': 46, 'car_Hyundai_Verna': 47, 'car_Hyundai_Xcent': 48, 'car_Jeep_Compass': 49, 'car_Jeep_Wrangler': 50, 
'car_Mahindra_Bolero': 51, 'car_Mahindra_Reva': 52, 'car_Mahindra_Scorpio': 53, 'car_Mahindra_TUV300': 54, 'car_Mahindra_Verito': 55, 'car_Mahindra_XUV500': 56, 'car_Mahindra_Xylo': 57, 'car_MarutiSuzuki_1000': 58, 'car_MarutiSuzuki_Alto800': 59, 'car_MarutiSuzuki_AltoK10': 60, 'car_MarutiSuzuki_Baleno': 61, 'car_MarutiSuzuki_Celerio': 62, 'car_MarutiSuzuki_Ciaz': 63, 'car_MarutiSuzuki_Dzire': 64, 'car_MarutiSuzuki_Eeco': 65, 'car_MarutiSuzuki_Ertiga': 66, 'car_MarutiSuzuki_Esteem2000': 67, 'car_MarutiSuzuki_Ignis': 68, 'car_MarutiSuzuki_Omni': 69, 'car_MarutiSuzuki_Ritz': 70, 'car_MarutiSuzuki_S-Cross': 71, 'car_MarutiSuzuki_Swift': 72, 'car_MarutiSuzuki_SX4': 73, 'car_MarutiSuzuki_VitaraBrezza': 74, 'car_MarutiSuzuki_WagonR': 75, 'car_MarutiSuzuki_Zen': 76, 'car_Mercedes-Benz_A-Class': 77, 'car_Mercedes-Benz_AmgGt4-DoorCoupe': 78, 'car_Mercedes-Benz_C-Class': 79, 'car_Mercedes-Benz_E-Class': 80, 'car_Mercedes-Benz_G-Class': 81, 'car_Mercedes-Benz_Gla-Class': 82, 'car_Mercedes-Benz_Gls': 83, 'car_Mercedes-Benz_S-Class': 84, 'car_Mitsubishi_Lancer': 85, 'car_Nissan_Micra': 86, 'car_Nissan_Sunny': 87, 'car_Nissan_Terrano': 88, 'car_Renault_Duster': 89, 'car_Renault_Kwid': 90, 'car_Renault_Lodgy': 91, 'car_Renault_Logan': 92, 'car_Renault_Scala': 93, 'car_Skoda_Fabia': 94, 'car_Skoda_Octavia': 95, 'car_Skoda_Rapid': 96, 'car_Skoda_Superb': 97, 'car_TataMotors_Hexa': 98, 'car_TataMotors_Indica': 99, 'car_TataMotors_Indigo': 100,
'car_TataMotors_Nano': 101, 'car_TataMotors_Nexon': 102, 'car_TataMotors_Safari': 103, 'car_TataMotors_Sumo': 104, 'car_TataMotors_Tiago': 105, 'car_TataMotors_Tigor': 106, 'car_TataMotors_Zest': 107, 'car_Toyota_Corolla': 108, 'car_Toyota_Etios': 109, 'car_Toyota_EtiosLiva': 110, 'car_Toyota_Fortuner': 111, 'car_Toyota_Innova': 112, 'car_Toyota_Qualis': 113, 'car_Volkswagen_Ameo': 114, 'car_Volkswagen_Jetta': 115, 'car_Volkswagen_Polo': 116, 'car_Volkswagen_Vento': 117, 'car_Volvo_Xc40': 118, 'car_Volvo_Xc60': 119, 'mini-bus_mini-bus_mini-bus': 120, 'motorcycle_Bajaj_Avenger': 121, 'motorcycle_Bajaj_CT100': 122, 'motorcycle_Bajaj_Discover': 123, 'motorcycle_Bajaj_Discover100': 124, 'motorcycle_Bajaj_Discover110': 125, 'motorcycle_Bajaj_Discover125': 126, 'motorcycle_Bajaj_Discover135': 127, 'motorcycle_Bajaj_Platina': 128, 'motorcycle_Bajaj_Pulsar150': 129, 'motorcycle_Bajaj_Pulsar180': 130, 'motorcycle_Bajaj_Pulsar200': 131, 'motorcycle_Bajaj_Pulsar220F': 132, 'motorcycle_Bajaj_PulsarNS200': 133, 'motorcycle_Bajaj_PulsarRS200': 134, 'motorcycle_Bajaj_V12': 135, 'motorcycle_Bajaj_V15': 136, 'motorcycle_Hero_Glamour': 137, 'motorcycle_Hero_HFDeluxe': 138, 'motorcycle_Hero_Hunk': 139, 'motorcycle_Hero_Passion': 140, 'motorcycle_Hero_PassionPlus': 141, 'motorcycle_Hero_PassionPro': 142, 'motorcycle_Hero_Splendor': 143, 'motorcycle_Hero_SuperSplendor': 144, 'motorcycle_Hero_XPulse200': 145, 'motorcycle_HeroHonda_CBZ': 146, 'motorcycle_HeroHonda_SplendorNXG': 147, 'motorcycle_Honda_CBHornet160R': 148, 'motorcycle_Honda_CBTwister': 149, 'motorcycle_Honda_Karizma': 150, 'motorcycle_Honda_KarizmaZMR': 151, 'motorcycle_Honda_Shine': 152, 'motorcycle_Honda_SP125': 153, 'motorcycle_Honda_StunnerCBF': 154, 'motorcycle_Honda_Unicorn': 155, 'motorcycle_KTM_Duke': 156, 'motorcycle_Mahindra_Centuro': 157, 'motorcycle_motorcycle_motorcycle': 158, 'motorcycle_RoyalEnfield_Bullet350': 159, 'motorcycle_RoyalEnfield_Bullet500': 160, 'motorcycle_RoyalEnfield_Classic350': 161, 'motorcycle_RoyalEnfield_Classic500': 162, 'motorcycle_RoyalEnfield_ContinentalGT650': 163, 'motorcycle_RoyalEnfield_Interceptor650': 164, 'motorcycle_RoyalEnfield_Meteor350': 165, 'motorcycle_RoyalEnfield_Thunderbird350': 166, 'motorcycle_RoyalEnfield_Thunderbird350X': 167, 'motorcycle_Suzuki_Gixxer': 168, 'motorcycle_Suzuki_Samurai': 169, 'motorcycle_Suzuki_Slingshot': 170, 'motorcycle_TVS_ApacheRR310': 171, 'motorcycle_TVS_ApacheRTR160': 172, 'motorcycle_TVS_ApacheRTR200': 173, 'motorcycle_TVS_Excel100': 174, 'motorcycle_TVS_ExcelHeavyDuty': 175, 'motorcycle_TVS_Sport': 176, 'motorcycle_TVS_StarCityPlus': 177, 'motorcycle_TVS_Victor': 178, 'motorcycle_TVS_XL100': 179, 'motorcycle_Yamaha_Crux': 180, 'motorcycle_Yamaha_Fazer': 181, 'motorcycle_Yamaha_FZ25': 182, 'motorcycle_Yamaha_FZS-FI': 183, 'motorcycle_Yamaha_FZ-V3': 184, 'motorcycle_Yamaha_Libero': 185, 'motorcycle_Yamaha_R15': 186, 'motorcycle_Yamaha_RX100': 187, 'scooter_Bajaj_Chetak': 188, 'scooter_Hero_Duet': 189, 'scooter_Hero_Maestro': 190, 'scooter_Hero_Pleasure': 191, 'scooter_Honda_Activa': 192, 'scooter_Honda_Aviator': 193, 'scooter_Honda_Dio': 194, 'scooter_Honda_Grazia': 195, 'scooter_Mahindra_Gusto': 196, 'scooter_scooter_scooter': 197, 'scooter_Suzuki_Access': 198, 'scooter_Suzuki_Burgman': 199, 'scooter_Suzuki_Swish': 200, 'scooter_TVS_Jupiter': 201, 'scooter_TVS_Ntorq': 202, 'scooter_TVS_Pep': 203, 'scooter_TVS_Streak': 204,
'scooter_TVS_Wego': 205, 'scooter_TVS_Zest': 206, 'scooter_Vespa_VXL125': 207, 'scooter_Vespa_ZX125': 208, 'scooter_Yamaha_Fascino125': 209, 'scooter_Yamaha_RayZR': 210, 'truck_AshokLeyland_AshokLeyland': 211, 'truck_BharatBenz_BharatBenz': 212, 'truck_Eicher_Eicher': 213, 'truck_Mahindra_Mahindra': 214, 'truck_SML_SML': 215, 'truck_Tata_Tata': 216, 'truck_truck_truck': 217}
l3_class_indexes_names = list(l3_class_indexes.keys())
l2_names_order = []
l1_names_order = []
l3_l2_mapper = []
l3_l1_mapper = []
pname = ''
p1name = ''
for l3i in l3_class_indexes_names:
    if(pname!='_'.join(l3i.split('_')[:2])):
        pname = '_'.join(l3i.split('_')[:2])
        l2_names_order.append(pname)
    l3_l2_mapper.append(len(l2_names_order)-1)
    
    if(p1name!=l3i.split('_')[0]):
        p1name = l3i.split('_')[0]
        l1_names_order.append(p1name)
    l3_l1_mapper.append(len(l1_names_order)-1)
print("length of l2 unqiue names: ", len(l2_names_order))
print("length of l1 unqiue names: ", len(l1_names_order))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    matches_iou0 = np.array([])
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            if(i==0):
                matches_iou0 = matches.copy()
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), matches_iou0
#     return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    if(not os.path.exists('/scratch/prafful/test_dumps')):
        os.makedirs('/scratch/prafful/test_dumps')
    if(not os.path.exists('/scratch/prafful/test_dumps_orig')):
        os.makedirs('/scratch/prafful/test_dumps_orig')
    if(not os.path.exists('/scratch/prafful/orig_gt_object_images')):
        os.makedirs('/scratch/prafful/orig_gt_object_images')
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    l3_class_indexes_names = {k: v for k, v in enumerate(list(l3_class_indexes.keys()))}
    l2_names_order_indexes = {k: v for k, v in enumerate(l2_names_order)}
    l1_names_order_indexes = {k: v for k, v in enumerate(l1_names_order)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    l3gt_foryoloobject = []
    l1conf_foryoloobject = []
    yoloobject_img = []
    l3pred_foryoloobject_oldname = []
    l3pred_foryoloobject_newlabel = []
    l3pred_foryoloobject_newname = []
    yoloobject_img_real = []
    predbox_foryoloobject = []
    gtbox_foryoloobject = []

    l3pred_foryoloobject = []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    ijk=0
    lmnop=[]
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        ### loading l3_labels
        targets_l3 = targets.clone()
        targets_l2 = targets.clone()
        targets_l1 = targets.clone()
        for ip, pi in enumerate(paths):
            pi_l3_labels = list(pd.read_csv(pi.replace('_L1/','_L3/').replace('images','labels_l3_backup')[:-4]+'.txt',sep=' ',header=None)[0])
            for ti, pil3 in zip(np.where((targets_l3[:, 0] == ip).cpu().numpy())[0] , pi_l3_labels):
                targets_l3[ti,1] = l3_new_index_mapper[pil3]-1
                targets_l2[ti,1] = l3_l2_mapper[l3_new_index_mapper[pil3]-1]
                targets_l1[ti,1] = l3_l1_mapper[l3_new_index_mapper[pil3]-1]

        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
            targets_l3 = targets_l3.to(device)
            targets_l2 = targets_l2.to(device)
            targets_l1 = targets_l1.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        targets_l3[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        targets_l2[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        targets_l1[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
#             ### for only yolo-l3
#             labels = targets_l3[targets_l3[:, 0] == si, 1:]
            
#             labels = targets[targets[:, 0] == si, 1:]
#             labels_l3 = targets_l3[targets_l3[:, 0] == si, 1:]
#             labels = targets_l2[targets_l2[:, 0] == si, 1:]
            labels = targets_l1[targets_l1[:, 0] == si, 1:]
#             orig_img_temp = cv2.imread(paths[si])
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
#                     if plots:
#                         confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            ### crop image or grab the HRN l3 output
            predi = predn[:, :4].cpu().float().numpy()
#             imik = im[si].cpu().float().numpy()
#             imik = imik.transpose(1, 2, 0)
#             if np.max(imik) <= 1:
#                 imik *= 255
            for pis in range(predi.shape[0]):
#                 image_prefix = "mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis)
#                 if(image_prefix in fnamespdf_cut):
#                     findex = fnamespdf_cut.index(image_prefix)
#                     l3pred_hrn = pdffk[1][findex]
#                     l3pred_foryoloobject.append(l3_class_indexes_names[int(l3pred_hrn)])

#                     yoloobject_img.append(image_prefix)
#                     yoloobject_img_real.append('/'.join(paths[si].split('/')[-3:]))
#                     ### conf. of yolo
#                     l1conf_foryoloobject.append(pred[pis, 4].cpu().numpy())
#                     predibx = list(predn[pis,:4].cpu().numpy())
#                     predbox_foryoloobject.append(predibx)
                
# #                 pred[pis, 5] = l3_new_index_mapper[int(pred[pis, 5].cpu().numpy())]-1
#                 yoloobject_img.append("mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis))
#                 l3pred_foryoloobject.append(int(pred[pis, 5].cpu().numpy()))
#                 l3pred_foryoloobject_oldname.append(names[l3pred_foryoloobject[-1]])
#                 l3pred_foryoloobject_newlabel.append(l3_new_index_mapper[l3pred_foryoloobject[-1]]-1)
#                 l3pred_foryoloobject_newname.append(l3_class_indexes_names[l3pred_foryoloobject_newlabel[-1]])
#         cv2.imwrite("/scratch/prafful/test_dumps_orig/mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis)+"_"+str(ijk)+".png",orig_img_temp[int(predi[pis,1]):int(predi[pis,3]),int(predi[pis,0]):int(predi[pis,2])]) #.astype('uint8')[:,:,::-1]
                try:
                    search_fname = "mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis)+"_"+str(ijk)+".png"
#                     if(search_fname in gtfnamespdf):
#                         findex = gtfnamespdf.index(search_fname)
# #                         ### for l3gt/pred values
# #                         pred[pis, 5] = l3_class_indexes[str(pdfik[2][findex])]-1
#                         ### for l2/l1 pred/gt values
#                         pred[pis, 5] = pdfik[1][findex]
# #                         print(pdfik[1][findex])
#                     else:
#                         findex = fnamespdf.index(search_fname)
#                         pred[pis, 5] = pdffk[1][findex]
                    findex = fnamespdf.index(search_fname)
                    pred[pis, 5] = pdffk[1][findex]
                    predn[pis, 5] = pred[pis, 5]
                except:
                    lmnop.append(1)
#                 cv2.imwrite("/scratch/prafful/test_dumps/mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis)+"_"+str(ijk)+".png",imik[int(predi[pis,1]):int(predi[pis,3]),int(predi[pis,0]):int(predi[pis,2])].astype('uint8')[:,:,::-1])
                ijk+=1


            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                
#                 correct = process_batch(predn, labelsn, iouv)
                correct, correct_matches_iou0 = process_batch(predn, labelsn, iouv)

                #if plots:
                #    confusion_matrix.process_batch(predn, labelsn)
                
#             ###### Qualitative analysis
#             for cii in range(correct_matches_iou0.shape[0]):
#                 pis = int(correct_matches_iou0[cii,1])
#                 yoloobject_img.append("mainbatchi"+str(batch_i)+"_imgbatchi"+str(si)+"_predi"+str(pis))
#                 yoloobject_img_real.append('/'.join(paths[si].split('/')[-3:]))
# #                 l3gt_foryoloobject.append(int(labels_l3[int(correct_matches_iou0[cii,0]),0].cpu().numpy()))
#                 l3gt_foryoloobject.append(l3_class_indexes_names[int(labels_l3[int(correct_matches_iou0[cii,0]),0].cpu().numpy())])
                
#                 findex = fnamespdf_cut.index(yoloobject_img[-1])
#                 l3pred_hrn = pdffk[1][findex]
                
# #                 l3pred_foryoloobject.append(int(l3pred_hrn))
#                 l3pred_foryoloobject.append(l3_class_indexes_names[int(l3pred_hrn)])
#                 labelibx = list(labelsn[int(correct_matches_iou0[cii,0]),1:].cpu().numpy())
#                 predibx = list(predn[int(correct_matches_iou0[cii,1]),:4].cpu().numpy())
#                 predbox_foryoloobject.append(predibx)
#                 gtbox_foryoloobject.append(labelibx)
#                 cv2.imwrite('/scratch/prafful/yoloL1out_predcropped_object_images/'+yoloobject_img[-1]+'.png',orig_img_temp[int(predibx[1]):int(predibx[3]),int(predibx[0]):int(predibx[2])])
                
#                 cv2.imwrite('/scratch/prafful/yoloL1out_gtcropped_object_images/'+yoloobject_img[-1]+'.png',orig_img_temp[int(labelibx[1]):int(labelibx[3]),int(labelibx[0]):int(labelibx[2])])
                
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
#             callbacks.run('on_val_image_end', pred, predn, path, l3_class_indexes_names, im[si])
#             callbacks.run('on_val_image_end', pred, predn, path, l2_names_order_indexes, im[si])
            callbacks.run('on_val_image_end', pred, predn, path, l1_names_order_indexes, im[si])

        # Plot images
        #if plots and batch_i < 3:
        #    plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
        #    plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end')
    print("Length of lmnop --> Check for errrorrrr here in try catch",len(lmnop))
#     pd.DataFrame({'yoloobject_img':yoloobject_img,'l3pred_foryoloobject':l3pred_foryoloobject,'l3pred_foryoloobject_oldname':l3pred_foryoloobject_oldname,'l3pred_foryoloobject_newlabel':l3pred_foryoloobject_newlabel,'l3pred_foryoloobject_newname':l3pred_foryoloobject_newname}).to_csv("/scratch/prafful/yolo-l3_model_preds.csv")
#     pd.DataFrame({'yoloobject_img_name':yoloobject_img,'l3gt_foryoloobject':l3gt_foryoloobject,'l3hrnpred_foryoloobject':l3pred_foryoloobject}).to_csv("/scratch/prafful/l3gtwpred_foryoloobjects.csv")


#     test_df_logs = pd.DataFrame({'yoloobject_img_main':yoloobject_img_real,'yoloobject_img_name':yoloobject_img,'l3gt_foryoloobject':l3gt_foryoloobject,'l3hrnpred_foryoloobject':l3pred_foryoloobject})
#     gtboxes_df = pd.DataFrame(np.array(gtbox_foryoloobject),columns=['x1gt','y1gt','x2gt','y2gt'])
#     predboxes_df = pd.DataFrame(np.array(predbox_foryoloobject),columns=['x1pred','y1pred','x2pred','y2pred'])
#     test_df_logs = pd.concat([test_df_logs, gtboxes_df], axis=1)
#     test_df_logs = pd.concat([test_df_logs, predboxes_df], axis=1)
# #     test_df_logs.to_csv("fgvd_l3_test_logs.csv",index=False)
    
    ### last used
#     test_df_logs = pd.DataFrame({'yoloobject_img_main':yoloobject_img_real,'yoloobject_img_name':yoloobject_img,'l1conf_foryoloobject':l1conf_foryoloobject,'l3hrnpred_foryoloobject':l3pred_foryoloobject})
#     predboxes_df = pd.DataFrame(np.array(predbox_foryoloobject),columns=['x1pred','y1pred','x2pred','y2pred'])
#     test_df_logs = pd.concat([test_df_logs, predboxes_df], axis=1)
#     test_df_logs.to_csv("fgvd-pred_l3_test_logs.csv",index=False)
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
#         tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=l3_class_indexes_names)
#         tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=l2_names_order_indexes)
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=l1_names_order_indexes)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        print(len(stats) , stats[0].any())

    print(mp, mr, map50)
    print(ap.shape, ap_class.shape)
    ### save the stats to derive L1 and L2 mAPs 
#     with open('v5l_with_hrn_fgd_l3_test_stats.pkl', 'wb') as file:
#     with open('v5l_fgd_l3_test_stats.pkl', 'wb') as file:
#         pickle.dump(stats, file)
    
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(emojis(f'WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️'))

#     # Print results per class
#     if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
#         for i, c in enumerate(ap_class):
#             LOGGER.info(pf % (l3_class_indexes_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

#     # Plots
#     if plots:
#         confusion_matrix.plot(save_dir=save_dir, names=list(l3_class_indexes_names.values()))
#         callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(emojis(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️'))
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
