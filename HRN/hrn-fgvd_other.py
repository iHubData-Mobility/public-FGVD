
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb

wandb.login()

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # Post-activation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        # Pre-activation
        # self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.ReLU() if relu else None
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        #                       stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # x = self.conv(x)
        return x
    

class HIFD2(nn.Module):
    def __init__(self, model, feature_size, dataset):
        super(HIFD2, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        if dataset == 'CUB':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 13),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 38),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 200),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 200)
            )
        elif dataset == 'FGVD':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 7),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 57),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 217),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 217)
            )

    def forward(self, x):
        x = self.features(x)
        x_order = self.conv_block1(x)
        x_family = self.conv_block2(x)
        x_species = self.conv_block3(x)

        x_order_fc = self.pooling(x_order)
        x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)
        x_order_fc = self.fc1(x_order_fc)
        x_family_fc = self.pooling(x_family)
        x_family_fc = x_family_fc.view(x_family_fc.size(0), -1)
        x_family_fc = self.fc2(x_family_fc)
        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc3(x_species_fc)

        y_order_sig = self.classifier_1(self.relu(x_order_fc))
        y_family_sig = self.classifier_2(self.relu(x_family_fc + x_order_fc))
        y_species_sig = self.classifier_3(self.relu(x_species_fc + x_family_fc + x_order_fc))
        y_species_sof = self.classifier_3_1(self.relu(x_species_fc + x_family_fc + x_order_fc))

        return y_order_sig, y_family_sig, y_species_sof, y_species_sig

from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn

class TreeLoss(nn.Module):
    def __init__(self, hierarchy, total_nodes, levels, device):
        super(TreeLoss, self).__init__()
        self.stateSpace = self.generateStateSpace(hierarchy, total_nodes, levels).to(device)

    def forward(self, fs, labels, device):
        # print("shape of statespace",len(self.stateSpace),len(self.stateSpace[1]))
        index = torch.mm(self.stateSpace, fs.T)
        # print("index row:",len(index)," col:", len(index[1]))
        # print("fs shape",fs.shape[0],fs.shape[1])
        joint = torch.exp(index)
        # print(joint)
        z = torch.sum(joint, dim=0)
        # print("z sum = ",z)
        loss = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        
        for i in range(len(labels)):
            # print("labels = ",labels[i])
            # print("Greater than 0",labels[i] > 0)
            # print(torch.where(self.stateSpace[:, labels[i]] > 0)[0])
            """tensor([194], device='cuda:0')"""
            marginal = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, labels[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
            wandb.log({"Tree loss": torch.mean(loss)})
        return torch.mean(loss)

    def inference(self, fs, device):
        with torch.no_grad():
            index = torch.mm(self.stateSpace, fs.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float64).to(device)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    pMargin[i, j] = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, j] > 0)[0]))
            return pMargin

    def generateStateSpace(self, hierarchy, total_nodes, levels):
        stateSpace = torch.zeros(total_nodes + 1, total_nodes)
        recorded = torch.zeros(total_nodes)
        i = 1

        if levels == 2:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[0]] = 1
                i += 1

        elif levels == 3:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                if recorded[path[2]] == 0:
                    stateSpace[i, path[1]] = 1
                    stateSpace[i, path[2]] = 1
                    recorded[path[2]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[2]] = 1
                stateSpace[i, path[0]] = 1
                i += 1

        
        if i == total_nodes + 1:
            return stateSpace
        else:
            print('Invalid StateSpace!!!')

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from os.path import join
from PIL import Image
import random
import math
import os
import networkx as nx
import numpy as np

class FGVDDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=0.0):
        super(FGVDDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1], [6, 6, 1], [7, 7, 2], [8, 8, 3], [9, 8, 3], [10, 9, 3], [11, 9, 3], [12, 9, 3], [13, 10, 3], [14, 11, 3], [15, 11, 3], [16, 11, 3], [17, 11, 3], [18, 11, 3], [19, 11, 3], [20, 12, 3], [21, 12, 3], [22, 13, 3], [23, 14, 3], [24, 14, 3], [25, 14, 3], [26, 14, 3], [27, 14, 3], [28, 14, 3], [29, 14, 3], [30, 15, 3], [31, 16, 3], [32, 16, 3], [33, 16, 3], [34, 16, 3], [35, 16, 3], [36, 16, 3], [37, 16, 3], [38, 16, 3], [39, 16, 3], [40, 17, 3], [41, 17, 3], [42, 17, 3], [43, 17, 3], [44, 17, 3], [45, 17, 3], [46, 17, 3], [47, 17, 3], [48, 17, 3], [49, 18, 3], [50, 18, 3], [51, 19, 3], [52, 19, 3], [53, 19, 3], [54, 19, 3], [55, 19, 3], [56, 19, 3], [57, 19, 3], [58, 20, 3], [59, 20, 3], [60, 20, 3], [61, 20, 3], [62, 20, 3], [63, 20, 3], [64, 20, 3], [65, 20, 3], [66, 20, 3], [67, 20, 3], [68, 20, 3], [69, 20, 3], [70, 20, 3], [71, 20, 3], [72, 20, 3], [73, 20, 3], [74, 20, 3], [75, 20, 3], [76, 20, 3], [77, 21, 3], [78, 21, 3], [79, 21, 3], [80, 21, 3], [81, 21, 3], [82, 21, 3], [83, 21, 3], [84, 21, 3], [85, 22, 3], [86, 23, 3], [87, 23, 3], [88, 23, 3], [89, 24, 3], [90, 24, 3], [91, 24, 3], [92, 24, 3], [93, 24, 3], [94, 25, 3], [95, 25, 3], [96, 25, 3], [97, 25, 3], [98, 26, 3], [99, 26, 3], [100, 26, 3], [101, 26, 3], [102, 26, 3], [103, 26, 3], [104, 26, 3], [105, 26, 3], [106, 26, 3], [107, 26, 3], [108, 27, 3], [109, 27, 3], [110, 27, 3], [111, 27, 3], [112, 27, 3], [113, 27, 3], [114, 28, 3], [115, 28, 3], [116, 28, 3], [117, 28, 3], [118, 29, 3], [119, 29, 3], [120, 30, 4], [121, 31, 5], [122, 31, 5], [123, 31, 5], [124, 31, 5], [125, 31, 5], [126, 31, 5], [127, 31, 5], [128, 31, 5], [129, 31, 5], [130, 31, 5], [131, 31, 5], [132, 31, 5], [133, 31, 5], [134, 31, 5], [135, 31, 5], [136, 31, 5], [137, 32, 5], [138, 32, 5], [139, 32, 5], [140, 32, 5], [141, 32, 5], [142, 32, 5], [143, 32, 5], [144, 32, 5], [145, 32, 5], [146, 33, 5], [147, 33, 5], [148, 34, 5], [149, 34, 5], [150, 34, 5], [151, 34, 5], [152, 34, 5], [153, 34, 5], [154, 34, 5], [155, 34, 5], [156, 35, 5], [157, 36, 5], [158, 37, 5], [159, 38, 5], [160, 38, 5], [161, 38, 5], [162, 38, 5], [163, 38, 5], [164, 38, 5], [165, 38, 5], [166, 38, 5], [167, 38, 5], [168, 39, 5], [169, 39, 5], [170, 39, 5], [171, 40, 5], [172, 40, 5], [173, 40, 5], [174, 40, 5], [175, 40, 5], [176, 40, 5], [177, 40, 5], [178, 40, 5], [179, 40, 5], [180, 41, 5], [181, 41, 5], [182, 41, 5], [183, 41, 5], [184, 41, 5], [185, 41, 5], [186, 41, 5], [187, 41, 5], [188, 42, 6], [189, 43, 6], [190, 43, 6], [191, 43, 6], [192, 44, 6], [193, 44, 6], [194, 44, 6], [195, 44, 6], [196, 45, 6], [197, 46, 6], [198, 47, 6], [199, 47, 6], [200, 47, 6], [201, 48, 6], [202, 48, 6], [203, 48, 6], [204, 48, 6], [205, 48, 6], [206, 48, 6], [207, 49, 6], [208, 49, 6], [209, 50, 6], [210, 50, 6], [211, 51, 7], [212, 52, 7], [213, 53, 7], [214, 54, 7], [215, 55, 7], [216, 56, 7], [217, 57, 7]]
        self.map = {'autorickshaw_Atul_Atul': 1, 'autorickshaw_autorickshaw_autorickshaw': 2, 'autorickshaw_Bajaj_Bajaj': 3, 'autorickshaw_Mahindra_Mahindra': 4, 'autorickshaw_Piaggio_Piaggio': 5, 'autorickshaw_TVS_TVS': 6, 'bus_bus_bus': 7, 'car_Audi_A3': 8, 'car_Audi_Q3': 9, 'car_Bmw_2-Series-220d': 10, 'car_Bmw_3-Series': 11, 'car_Bmw_X1': 12, 'car_car_car': 13, 'car_Chevrolet_Aveo': 14, 'car_Chevrolet_Beat': 15, 'car_Chevrolet_Cruze': 16, 'car_Chevrolet_Enjoy': 17, 'car_Chevrolet_Spark': 18, 'car_Chevrolet_Tavera': 19, 'car_Fiat_Linea': 20, 'car_Fiat_PuntoEvo': 21, 'car_Force_TraxToofan': 22, 'car_Ford_Aspire': 23, 'car_Ford_Ecosport': 24, 'car_Ford_EcoSportTitanium': 25, 'car_Ford_Everest': 26, 'car_Ford_Fiesta': 27, 'car_Ford_Figo': 28, 'car_Ford_Ikon': 29, 'car_Hindustan_Ambassador': 30, 'car_Honda_Accord': 31, 'car_Honda_Amaze': 32, 'car_Honda_Brio': 33, 'car_Honda_Brv': 34, 'car_Honda_City': 35, 'car_Honda_Civic': 36, 'car_Honda_Cr-V': 37, 'car_Honda_Jazz': 38, 'car_Honda_Wr-V': 39, 'car_Hyundai_Accent': 40, 'car_Hyundai_Aura': 41, 'car_Hyundai_Creta': 42, 'car_Hyundai_Eon': 43, 'car_Hyundai_I10': 44, 'car_Hyundai_I20': 45, 'car_Hyundai_Santro': 46, 'car_Hyundai_Verna': 47, 'car_Hyundai_Xcent': 48, 'car_Jeep_Compass': 49, 'car_Jeep_Wrangler': 50, 
                    'car_Mahindra_Bolero': 51, 'car_Mahindra_Reva': 52, 'car_Mahindra_Scorpio': 53, 'car_Mahindra_TUV300': 54, 'car_Mahindra_Verito': 55, 'car_Mahindra_XUV500': 56, 'car_Mahindra_Xylo': 57, 'car_MarutiSuzuki_1000': 58, 'car_MarutiSuzuki_Alto800': 59, 'car_MarutiSuzuki_AltoK10': 60, 'car_MarutiSuzuki_Baleno': 61, 'car_MarutiSuzuki_Celerio': 62, 'car_MarutiSuzuki_Ciaz': 63, 'car_MarutiSuzuki_Dzire': 64, 'car_MarutiSuzuki_Eeco': 65, 'car_MarutiSuzuki_Ertiga': 66, 'car_MarutiSuzuki_Esteem2000': 67, 'car_MarutiSuzuki_Ignis': 68, 'car_MarutiSuzuki_Omni': 69, 'car_MarutiSuzuki_Ritz': 70, 'car_MarutiSuzuki_S-Cross': 71, 'car_MarutiSuzuki_Swift': 72, 'car_MarutiSuzuki_SX4': 73, 'car_MarutiSuzuki_VitaraBrezza': 74, 'car_MarutiSuzuki_WagonR': 75, 'car_MarutiSuzuki_Zen': 76, 'car_Mercedes-Benz_A-Class': 77, 'car_Mercedes-Benz_AmgGt4-DoorCoupe': 78, 'car_Mercedes-Benz_C-Class': 79, 'car_Mercedes-Benz_E-Class': 80, 'car_Mercedes-Benz_G-Class': 81, 'car_Mercedes-Benz_Gla-Class': 82, 'car_Mercedes-Benz_Gls': 83, 'car_Mercedes-Benz_S-Class': 84, 'car_Mitsubishi_Lancer': 85, 'car_Nissan_Micra': 86, 'car_Nissan_Sunny': 87, 'car_Nissan_Terrano': 88, 'car_Renault_Duster': 89, 'car_Renault_Kwid': 90, 'car_Renault_Lodgy': 91, 'car_Renault_Logan': 92, 'car_Renault_Scala': 93, 'car_Skoda_Fabia': 94, 'car_Skoda_Octavia': 95, 'car_Skoda_Rapid': 96, 'car_Skoda_Superb': 97, 'car_TataMotors_Hexa': 98, 'car_TataMotors_Indica': 99, 'car_TataMotors_Indigo': 100,
                    'car_TataMotors_Nano': 101, 'car_TataMotors_Nexon': 102, 'car_TataMotors_Safari': 103, 'car_TataMotors_Sumo': 104, 'car_TataMotors_Tiago': 105, 'car_TataMotors_Tigor': 106, 'car_TataMotors_Zest': 107, 'car_Toyota_Corolla': 108, 'car_Toyota_Etios': 109, 'car_Toyota_EtiosLiva': 110, 'car_Toyota_Fortuner': 111, 'car_Toyota_Innova': 112, 'car_Toyota_Qualis': 113, 'car_Volkswagen_Ameo': 114, 'car_Volkswagen_Jetta': 115, 'car_Volkswagen_Polo': 116, 'car_Volkswagen_Vento': 117, 'car_Volvo_Xc40': 118, 'car_Volvo_Xc60': 119, 'mini-bus_mini-bus_mini-bus': 120, 'motorcycle_Bajaj_Avenger': 121, 'motorcycle_Bajaj_CT100': 122, 'motorcycle_Bajaj_Discover': 123, 'motorcycle_Bajaj_Discover100': 124, 'motorcycle_Bajaj_Discover110': 125, 'motorcycle_Bajaj_Discover125': 126, 'motorcycle_Bajaj_Discover135': 127, 'motorcycle_Bajaj_Platina': 128, 'motorcycle_Bajaj_Pulsar150': 129, 'motorcycle_Bajaj_Pulsar180': 130, 'motorcycle_Bajaj_Pulsar200': 131, 'motorcycle_Bajaj_Pulsar220F': 132, 'motorcycle_Bajaj_PulsarNS200': 133, 'motorcycle_Bajaj_PulsarRS200': 134, 'motorcycle_Bajaj_V12': 135, 'motorcycle_Bajaj_V15': 136, 'motorcycle_Hero_Glamour': 137, 'motorcycle_Hero_HFDeluxe': 138, 'motorcycle_Hero_Hunk': 139, 'motorcycle_Hero_Passion': 140, 'motorcycle_Hero_PassionPlus': 141, 'motorcycle_Hero_PassionPro': 142, 'motorcycle_Hero_Splendor': 143, 'motorcycle_Hero_SuperSplendor': 144, 'motorcycle_Hero_XPulse200': 145, 'motorcycle_HeroHonda_CBZ': 146, 'motorcycle_HeroHonda_SplendorNXG': 147, 'motorcycle_Honda_CBHornet160R': 148, 'motorcycle_Honda_CBTwister': 149, 'motorcycle_Honda_Karizma': 150, 'motorcycle_Honda_KarizmaZMR': 151, 'motorcycle_Honda_Shine': 152, 'motorcycle_Honda_SP125': 153, 'motorcycle_Honda_StunnerCBF': 154, 'motorcycle_Honda_Unicorn': 155, 'motorcycle_KTM_Duke': 156, 'motorcycle_Mahindra_Centuro': 157, 'motorcycle_motorcycle_motorcycle': 158, 'motorcycle_RoyalEnfield_Bullet350': 159, 'motorcycle_RoyalEnfield_Bullet500': 160, 'motorcycle_RoyalEnfield_Classic350': 161, 'motorcycle_RoyalEnfield_Classic500': 162, 'motorcycle_RoyalEnfield_ContinentalGT650': 163, 'motorcycle_RoyalEnfield_Interceptor650': 164, 'motorcycle_RoyalEnfield_Meteor350': 165, 'motorcycle_RoyalEnfield_Thunderbird350': 166, 'motorcycle_RoyalEnfield_Thunderbird350X': 167, 'motorcycle_Suzuki_Gixxer': 168, 'motorcycle_Suzuki_Samurai': 169, 'motorcycle_Suzuki_Slingshot': 170, 'motorcycle_TVS_ApacheRR310': 171, 'motorcycle_TVS_ApacheRTR160': 172, 'motorcycle_TVS_ApacheRTR200': 173, 'motorcycle_TVS_Excel100': 174, 'motorcycle_TVS_ExcelHeavyDuty': 175, 'motorcycle_TVS_Sport': 176, 'motorcycle_TVS_StarCityPlus': 177, 'motorcycle_TVS_Victor': 178, 'motorcycle_TVS_XL100': 179, 'motorcycle_Yamaha_Crux': 180, 'motorcycle_Yamaha_Fazer': 181, 'motorcycle_Yamaha_FZ25': 182, 'motorcycle_Yamaha_FZS-FI': 183, 'motorcycle_Yamaha_FZ-V3': 184, 'motorcycle_Yamaha_Libero': 185, 'motorcycle_Yamaha_R15': 186, 'motorcycle_Yamaha_RX100': 187, 'scooter_Bajaj_Chetak': 188, 'scooter_Hero_Duet': 189, 'scooter_Hero_Maestro': 190, 'scooter_Hero_Pleasure': 191, 'scooter_Honda_Activa': 192, 'scooter_Honda_Aviator': 193, 'scooter_Honda_Dio': 194, 'scooter_Honda_Grazia': 195, 'scooter_Mahindra_Gusto': 196, 'scooter_scooter_scooter': 197, 'scooter_Suzuki_Access': 198, 'scooter_Suzuki_Burgman': 199, 'scooter_Suzuki_Swish': 200, 'scooter_TVS_Jupiter': 201, 'scooter_TVS_Ntorq': 202, 'scooter_TVS_Pep': 203, 'scooter_TVS_Streak': 204,
                    'scooter_TVS_Wego': 205, 'scooter_TVS_Zest': 206, 'scooter_Vespa_VXL125': 207, 'scooter_Vespa_ZX125': 208, 'scooter_Yamaha_Fascino125': 209, 'scooter_Yamaha_RayZR': 210, 'truck_AshokLeyland_AshokLeyland': 211, 'truck_BharatBenz_BharatBenz': 212, 'truck_Eicher_Eicher': 213, 'truck_Mahindra_Mahindra': 214, 'truck_SML_SML': 215, 'truck_Tata_Tata': 216, 'truck_truck_truck': 217}
        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                lists = l.strip().strip('\n').split('\t')
                imagename = lists[0]
                # print("image name : ", imagename)
                classname = " ".join(i for i in lists[1:])
                # print("classname",classname)
                name_list.append(imagename)
                class_label = self.map[classname]
                # print("class_label",class_label)
                family_label_list.append(self.trees[class_label-1][1] + 7)
                species_label_list.append(class_label + 64)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)
        # print("self.image_filenames, self.labels")
        # print(self.image_filenames,"-", self.labels)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, family_label_list, species_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(species_label_list[i]) not in class_imgs.keys():
                class_imgs[str(species_label_list[i])] = {'images': [], 'family': []}
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(species_label_list[i])]['family'].append(family_label_list[i])
            else:
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            # print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')
        # print("max of labels",max(labels))

        return images, labels

import torch
from torch.nn.modules.activation import Softmax
# from utils import *
import copy
import time
from sklearn.metrics import confusion_matrix, average_precision_score




def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device, devices, save_name):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    wandb.watch(net, log="all", log_freq=10)
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        net = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            optimizer.zero_grad()

            if len(devices) > 1:
                xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            else:
                xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            
            # print("len xc1_sig=", len(xc1_sig))
            # print("len xc2_sig=", len(xc2_sig))
            # print("len xc3_sig=", len(xc3_sig))
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'FGVD':
                leaf_labels = torch.nonzero(targets > 63, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'FGVD':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 64
                select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss
            
                
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
            with torch.no_grad():
                _, order_predicted = torch.max(xc1_sig.data, 1)
                order_total += order_targets.size(0)
                order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

                _, family_predicted = torch.max(xc2_sig.data, 1)
                family_total += family_targets.size(0)
                family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

                if leaf_labels.shape[0] > 0:
                    select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                    select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc3.data, 1)
                    _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()
        
        if lr_adjt == 'Step':
            scheduler.step()
        
        ephsilon = 1e-6

        train_order_acc = 100.*order_correct/(order_total + ephsilon)
        train_family_acc = 100.*family_correct/(family_total + ephsilon)
        train_species_acc_soft = 100.*species_correct_soft/(species_total + ephsilon)
        train_species_acc_sig = 100.*species_correct_sig/(species_total + ephsilon)
        train_loss = train_loss/(idx+1)
        epoch_end = time.time()

        print('Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_order_acc, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss, (epoch_end - epoch_start)))
        if epoch%5 == 0:
            checkpoint_dir = "/ssd_scratch/prafful/hrn_save_ckp_22-7/"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = checkpoint_dir + save_name+ "_"+ str(epoch) + ".pt"
            # ckp = checkpoint_dir + save_name + "_"+ str(epoch) +'_model.pt'
            # torch.save(net, ckp) 
            checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            
            print("Saved Successfully at",epoch)
        test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, testloader, CELoss, tree, device, dataset)
        
        wandb.log({"test_order_acc": test_order_acc})
        wandb.log({"test_family_acc": test_family_acc})
        wandb.log({"test_species_acc_soft": test_species_acc_soft})
        wandb.log({"test_species_acc_sig": test_species_acc_sig})
        wandb.log({"test_loss": test_loss})

        wandb.log({"train_order_acc": train_order_acc})
        wandb.log({"train_family_acc": train_family_acc})
        wandb.log({"train_species_acc_soft": train_species_acc_soft})
        wandb.log({"train_species_acc_sig": train_species_acc_sig})
        wandb.log({"train_loss": train_loss})

        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            best_model = net
            best_optimizer = optimizer
            best_scheduler = scheduler
            net.cpu()
            os.makedirs("/ssd_scratch/prafful/hrn_save_model/best/", exist_ok=True)
            torch.save(net, "/ssd_scratch/prafful/hrn_save_model/best/"+ save_name+ "_model" +'.pth')
            # torch.save(net, './models_'+dataset+'/model_'+save_name+'.pth')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))

    # Saving best model
    
    # os.makedirs("/ssd_scratch/prafful/hrn_save_model/best/", exist_ok=True)
    # torch.save(best_model, "/ssd_scratch/prafful/hrn_save_model/best/"+ save_name+ "_"+ best_epoch +'_model.pt')
    checkpoint = {'epoch': best_epoch, 'state_dict': best_model.state_dict(), 'optimizer': best_optimizer.state_dict(),'scheduler': best_scheduler.state_dict()}
    torch.save(checkpoint, "/ssd_scratch/prafful/hrn_save_model/best/"+ save_name+ "_"+ str(best_epoch) +'_ckp.pt')
    print("Saving the best model")

    


def test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
            elif dataset == 'FGVD':
                leaf_labels = torch.nonzero(targets > 63, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 64
            select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
            ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
            loss = ce_loss_species + tree_loss

            test_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, family_predicted = torch.max(xc2_sig.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            _, species_predicted_soft = torch.max(xc3.data, 1)
            _, species_predicted_sig = torch.max(xc3_sig.data, 1)
            species_total += select_leaf_labels.size(0)
            species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
            species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        ephsilon = 1e-6
        test_order_acc = 100.* order_correct/(order_total + ephsilon)
        test_family_acc = 100.* family_correct/(family_total + ephsilon)
        test_species_acc_soft = 100.* species_correct_soft/(species_total + ephsilon)
        test_species_acc_sig = 100.* species_correct_sig/(species_total + ephsilon)
        test_loss = test_loss/(idx+1)
        epoch_end = time.time()
        print('test_order_acc = %.5f,test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
             (test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))

    return test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss
    

def test_AP(model, dataset, test_set, test_data_loader, device):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            if dataset == 'CUB' or dataset == 'FGVD':
                y_order_sig, y_family_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, y_family_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            else:
                y_order_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            
            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))     

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import networkx as nx


trees_FGVD = [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1], [6, 6, 1], [7, 7, 2], [8, 8, 3], [9, 8, 3], [10, 9, 3], [11, 9, 3], [12, 9, 3], [13, 10, 3], [14, 11, 3], [15, 11, 3], [16, 11, 3], [17, 11, 3], [18, 11, 3], [19, 11, 3], [20, 12, 3], [21, 12, 3], [22, 13, 3], [23, 14, 3], [24, 14, 3], [25, 14, 3], [26, 14, 3], [27, 14, 3], [28, 14, 3], [29, 14, 3], [30, 15, 3], [31, 16, 3], [32, 16, 3], [33, 16, 3], [34, 16, 3], [35, 16, 3], [36, 16, 3], [37, 16, 3], [38, 16, 3], [39, 16, 3], [40, 17, 3], [41, 17, 3], [42, 17, 3], [43, 17, 3], [44, 17, 3], [45, 17, 3], [46, 17, 3], [47, 17, 3], [48, 17, 3], [49, 18, 3], [50, 18, 3], [51, 19, 3], [52, 19, 3], [53, 19, 3], [54, 19, 3], [55, 19, 3], [56, 19, 3], [57, 19, 3], [58, 20, 3], [59, 20, 3], [60, 20, 3], [61, 20, 3], [62, 20, 3], [63, 20, 3], [64, 20, 3], [65, 20, 3], [66, 20, 3], [67, 20, 3], [68, 20, 3], [69, 20, 3], [70, 20, 3], [71, 20, 3], [72, 20, 3], [73, 20, 3], [74, 20, 3], [75, 20, 3], [76, 20, 3], [77, 21, 3], [78, 21, 3], [79, 21, 3], [80, 21, 3], [81, 21, 3], [82, 21, 3], [83, 21, 3], [84, 21, 3], [85, 22, 3], [86, 23, 3], [87, 23, 3], [88, 23, 3], [89, 24, 3], [90, 24, 3], [91, 24, 3], [92, 24, 3], [93, 24, 3], [94, 25, 3], [95, 25, 3], [96, 25, 3], [97, 25, 3], [98, 26, 3], [99, 26, 3], [100, 26, 3], [101, 26, 3], [102, 26, 3], [103, 26, 3], [104, 26, 3], [105, 26, 3], [106, 26, 3], [107, 26, 3], [108, 27, 3], [109, 27, 3], [110, 27, 3], [111, 27, 3], [112, 27, 3], [113, 27, 3], [114, 28, 3], [115, 28, 3], [116, 28, 3], [117, 28, 3], [118, 29, 3], [119, 29, 3], [120, 30, 4], [121, 31, 5], [122, 31, 5], [123, 31, 5], [124, 31, 5], [125, 31, 5], [126, 31, 5], [127, 31, 5], [128, 31, 5], [129, 31, 5], [130, 31, 5], [131, 31, 5], [132, 31, 5], [133, 31, 5], [134, 31, 5], [135, 31, 5], [136, 31, 5], [137, 32, 5], [138, 32, 5], [139, 32, 5], [140, 32, 5], [141, 32, 5], [142, 32, 5], [143, 32, 5], [144, 32, 5], [145, 32, 5], [146, 33, 5], [147, 33, 5], [148, 34, 5], [149, 34, 5], [150, 34, 5], [151, 34, 5], [152, 34, 5], [153, 34, 5], [154, 34, 5], [155, 34, 5], [156, 35, 5], [157, 36, 5], [158, 37, 5], [159, 38, 5], [160, 38, 5], [161, 38, 5], [162, 38, 5], [163, 38, 5], [164, 38, 5], [165, 38, 5], [166, 38, 5], [167, 38, 5], [168, 39, 5], [169, 39, 5], [170, 39, 5], [171, 40, 5], [172, 40, 5], [173, 40, 5], [174, 40, 5], [175, 40, 5], [176, 40, 5], [177, 40, 5], [178, 40, 5], [179, 40, 5], [180, 41, 5], [181, 41, 5], [182, 41, 5], [183, 41, 5], [184, 41, 5], [185, 41, 5], [186, 41, 5], [187, 41, 5], [188, 42, 6], [189, 43, 6], [190, 43, 6], [191, 43, 6], [192, 44, 6], [193, 44, 6], [194, 44, 6], [195, 44, 6], [196, 45, 6], [197, 46, 6], [198, 47, 6], [199, 47, 6], [200, 47, 6], [201, 48, 6], [202, 48, 6], [203, 48, 6], [204, 48, 6], [205, 48, 6], [206, 48, 6], [207, 49, 6], [208, 49, 6], [209, 50, 6], [210, 50, 6], [211, 51, 7], [212, 52, 7], [213, 53, 7], [214, 54, 7], [215, 55, 7], [216, 56, 7], [217, 57, 7]]

trees_family_to_order_FGVD = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 2], [8, 3], [8, 3], [9, 3], [9, 3], [9, 3], [10, 3], [11, 3], [11, 3], [11, 3], [11, 3], [11, 3], [11, 3], [12, 3], [12, 3], [13, 3], [14, 3], [14, 3], [14, 3], [14, 3], [14, 3], [14, 3], [14, 3], [15, 3], [16, 3], [16, 3], [16, 3], [16, 3], [16, 3], [16, 3], [16, 3], [16, 3], [16, 3], [17, 3], [17, 3], [17, 3], [17, 3], [17, 3], [17, 3], [17, 3], [17, 3], [17, 3], [18, 3], [18, 3], [19, 3], [19, 3], [19, 3], [19, 3], [19, 3], [19, 3], [19, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [20, 3], [21, 3], [21, 3], [21, 3], [21, 3], [21, 3], [21, 3], [21, 3], [21, 3], [22, 3], [23, 3], [23, 3], [23, 3], [24, 3], [24, 3], [24, 3], [24, 3], [24, 3], [25, 3], [25, 3], [25, 3], [25, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [26, 3], [27, 3], [27, 3], [27, 3], [27, 3], [27, 3], [27, 3], [28, 3], [28, 3], [28, 3], [28, 3], [29, 3], [29, 3], [30, 4], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [31, 5], [32, 5], [32, 5], [32, 5], [32, 5], [32, 5], [32, 5], [32, 5], [32, 5], [32, 5], [33, 5], [33, 5], [34, 5], [34, 5], [34, 5], [34, 5], [34, 5], [34, 5], [34, 5], [34, 5], [35, 5], [36, 5], [37, 5], [38, 5], [38, 5], [38, 5], [38, 5], [38, 5], [38, 5], [38, 5], [38, 5], [38, 5], [39, 5], [39, 5], [39, 5], [40, 5], [40, 5], [40, 5], [40, 5], [40, 5], [40, 5], [40, 5], [40, 5], [40, 5], [41, 5], [41, 5], [41, 5], [41, 5], [41, 5], [41, 5], [41, 5], [41, 5], [42, 6], [43, 6], [43, 6], [43, 6], [44, 6], [44, 6], [44, 6], [44, 6], [45, 6], [46, 6], [47, 6], [47, 6], [47, 6], [48, 6], [48, 6], [48, 6], [48, 6], [48, 6], [48, 6], [49, 6], [49, 6], [50, 6], [50, 6], [51, 7], [52, 7], [53, 7], [54, 7], [55, 7], [56, 7], [57, 7]]


def get_order_family_target(targets, device, dataset):

    order_target_list = []
    family_target_list = []
    target_list_sig = []
    # print("in function get_order_family_target")

    for i in range(targets.size(0)):
        if dataset == 'CUB':
            if targets[i] < 51 and targets[i] > 12: 
                order_target_list.append(trees_family_to_order_CUB[targets[i]-13][1]-1)
                family_target_list.append(int(targets[i]-13))
            elif targets[i] > 50:
                order_target_list.append(trees_CUB[targets[i]-51][1]-1)
                family_target_list.append(trees_CUB[targets[i]-51][2]-1)

        elif dataset == 'FGVD':
            if targets[i] < 64 and targets[i] > 6: 
                order_target_list.append(trees_family_to_order_FGVD[targets[i]-7][1]-1)
                family_target_list.append(int(targets[i]-7))
            elif targets[i] > 63:
                # print("targets[i] = ",targets[i])
                # print("targets[i]-64 = ",targets[i]-64)
                # print("")
                order_target_list.append(trees_FGVD[targets[i]-64][2]-1)
                family_target_list.append(trees_FGVD[targets[i]-64][1]-1)
    
        target_list_sig.append(int(targets[i]))
    
    order_target_list = torch.from_numpy(np.array(order_target_list)).to(device)  
    family_target_list = torch.from_numpy(np.array(family_target_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    return order_target_list, family_target_list, target_list_sig


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws

from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, models
import torch.hub
import argparse
from torch.optim import lr_scheduler

# from RFM import HIFD2

# from tree_loss import TreeLoss
# from dataset import CubDataset, CubDataset2, AirDataset, AirDataset2
# from train_test import test, test_AP
# from train_test import train


# def arg_parse():
#     parser = argparse.ArgumentParser(description='PyTorch Deployment')
#     parser.add_argument('--worker', default=4, type=int, help='number of workers')
#     parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth', help='Path of trained model')
#     parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
#     parser.add_argument('--proportion', type=float, help='Proportion of species label')  
#     parser.add_argument('--epoch', type=int, help='Epochs')
#     parser.add_argument('--batch', type=int, help='batch size')      
#     parser.add_argument('--dataset', type=str, default='CUB', help='dataset name')
#     parser.add_argument('--img_size', type=str, default='448', help='image size')
#     parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual')
#     parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')

#     args = parser.parse_args()

#     if args.proportion == 0.1: 
#         args.epoch = 100
#         args.batch = 8
#         args.lr_adjt = 'Step'
    
#     return args


if __name__ == '__main__':
    # Hyper-parameters
    nb_epoch = 10
    batch_size = 32
    num_workers = 8
    proportion = 0.0
    dataset = "FGVD"
    img_size = 448
    device_no = ['0','1']
    lr_adjt = "Cos"
    model = "./resnet50-19c8e357.pth"
    wandb.init(project="1st Trained on cuda")
    wandb.config = {"learning_rate": 0.02, "epochs": nb_epoch, "batch_size": batch_size,"dataset":dataset}


    # args = arg_parse()
    print('==> proportion: ', proportion)
    print('==> epoch: ', nb_epoch)
    print('==> batch: ', batch_size)
    print('==> dataset: ', dataset)
    print('==> img_size: ', img_size)
    print('==> device: ', device_no)
    print('==> lr_adjt: ', lr_adjt)

    




    # Preprocess
    transform_train = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    
    if dataset == 'FGVD':
        # train_data_dir = '/content/drive/MyDrive/Fgvd_all_dataset/Final_backup/cropped/crop_train'
        # test_data_dir = '/content/drive/MyDrive/Fgvd_all_dataset/Final_backup/cropped/crop_test'
        # train_list = '/content/drive/MyDrive/Fgvd_all_dataset/Final_backup/cropped/train_list.txt'
        # test_list = '/content/drive/MyDrive/Fgvd_all_dataset/Final_backup/cropped/test_list.txt'
        train_data_dir = "./cropped/crop_train"
        test_data_dir = "./cropped/crop_test"
        train_list = "./cropped/train_list.txt"
        test_list = "./cropped/test_list.txt"

        trees = [[64, 0, 7], [65, 0, 8], [66, 0, 9], [67, 0, 10], [68, 0, 11], [69, 0, 12], [70, 1, 13], [71, 2, 14], [72, 2, 14], [73, 2, 15], [74, 2, 15], [75, 2, 15], [76, 2, 16], [77, 2, 17], [78, 2, 17], [79, 2, 17], [80, 2, 17], [81, 2, 17], [82, 2, 17], [83, 2, 18], [84, 2, 18], [85, 2, 19], [86, 2, 20], [87, 2, 20], [88, 2, 20], [89, 2, 20], [90, 2, 20], [91, 2, 20], [92, 2, 20], [93, 2, 21], [94, 2, 22], [95, 2, 22], [96, 2, 22], [97, 2, 22], [98, 2, 22], [99, 2, 22], [100, 2, 22], [101, 2, 22], [102, 2, 22], [103, 2, 23], [104, 2, 23], [105, 2, 23], [106, 2, 23], [107, 2, 23], [108, 2, 23], [109, 2, 23], [110, 2, 23], [111, 2, 23], [112, 2, 24], [113, 2, 24], [114, 2, 25], [115, 2, 25], [116, 2, 25], [117, 2, 25], [118, 2, 25], [119, 2, 25], [120, 2, 25], [121, 2, 26], [122, 2, 26], [123, 2, 26], [124, 2, 26], [125, 2, 26], [126, 2, 26], [127, 2, 26], [128, 2, 26], [129, 2, 26], [130, 2, 26], [131, 2, 26], [132, 2, 26], [133, 2, 26], [134, 2, 26], [135, 2, 26], [136, 2, 26], [137, 2, 26], [138, 2, 26], [139, 2, 26], [140, 2, 27], [141, 2, 27], [142, 2, 27], [143, 2, 27], [144, 2, 27], [145, 2, 27], [146, 2, 27], [147, 2, 27], [148, 2, 28], [149, 2, 29], [150, 2, 29], [151, 2, 29], [152, 2, 30], [153, 2, 30], [154, 2, 30], [155, 2, 30], [156, 2, 30], [157, 2, 31], [158, 2, 31], [159, 2, 31], [160, 2, 31], [161, 2, 32], [162, 2, 32], [163, 2, 32], [164, 2, 32], [165, 2, 32], [166, 2, 32], [167, 2, 32], [168, 2, 32], [169, 2, 32], [170, 2, 32], [171, 2, 33], [172, 2, 33], [173, 2, 33], [174, 2, 33], [175, 2, 33], [176, 2, 33], [177, 2, 34], [178, 2, 34], [179, 2, 34], [180, 2, 34], [181, 2, 35], [182, 2, 35], [183, 3, 36], [184, 4, 37], [185, 4, 37], [186, 4, 37], [187, 4, 37], [188, 4, 37], [189, 4, 37], [190, 4, 37], [191, 4, 37], [192, 4, 37], [193, 4, 37], [194, 4, 37], [195, 4, 37], [196, 4, 37], [197, 4, 37], [198, 4, 37], [199, 4, 37], [200, 4, 38], [201, 4, 38], [202, 4, 38], [203, 4, 38], [204, 4, 38], [205, 4, 38], [206, 4, 38], [207, 4, 38], [208, 4, 38], [209, 4, 39], [210, 4, 39], [211, 4, 40], [212, 4, 40], [213, 4, 40], [214, 4, 40], [215, 4, 40], [216, 4, 40], [217, 4, 40], [218, 4, 40], [219, 4, 41], [220, 4, 42], [221, 4, 43], [222, 4, 44], [223, 4, 44], [224, 4, 44], [225, 4, 44], [226, 4, 44], [227, 4, 44], [228, 4, 44], [229, 4, 44], [230, 4, 44], [231, 4, 45], [232, 4, 45], [233, 4, 45], [234, 4, 46], [235, 4, 46], [236, 4, 46], [237, 4, 46], [238, 4, 46], [239, 4, 46], [240, 4, 46], [241, 4, 46], [242, 4, 46], [243, 4, 47], [244, 4, 47], [245, 4, 47], [246, 4, 47], [247, 4, 47], [248, 4, 47], [249, 4, 47], [250, 4, 47], [251, 5, 48], [252, 5, 49], [253, 5, 49], [254, 5, 49], [255, 5, 50], [256, 5, 50], [257, 5, 50], [258, 5, 50], [259, 5, 51], [260, 5, 52], [261, 5, 53], [262, 5, 53], [263, 5, 53], [264, 5, 54], [265, 5, 54], [266, 5, 54], [267, 5, 54], [268, 5, 54], [269, 5, 54], [270, 5, 55], [271, 5, 55], [272, 5, 56], [273, 5, 56], [274, 6, 57], [275, 6, 58], [276, 6, 59], [277, 6, 60], [278, 6, 61], [279, 6, 62], [280, 6, 63]]
        
        levels = 3
        total_nodes = 281
        trainset = FGVDDataset(train_data_dir, train_list, transform_train, re_level='class', proportion=proportion)
        # Uncomment this line for testing OA results
        testset = FGVDDataset(test_data_dir, test_list, transform_test, re_level='class', proportion=0)
        # Uncomment this line for testing Average PRC results
        # testset = AirDataset2(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # GPU
    device = torch.device("cuda:" + device_no[0])
    # device = torch.device("cuda:" + str(0))
    
    # RFM from scrach
    backbone = models.resnet50(pretrained=False)
    backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
    # backbone = models.resnext101_32x8d(pretrained=False)
    # backbone.load_state_dict(torch.load('./pre-trained/resnext101_32x8d-8ba56ff5.pth'))
    net = HIFD2(backbone, 1024, dataset)

    # RFM from trained model
    # net = torch.load(args.model)

    net.to(device)
    wandb.watch(net)

    # Loss functions
    CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, total_nodes, levels, device)
    
    if proportion > 0.1:       # for p > 0.1
        optimizer = optim.SGD([
            {'params': net.classifier_1.parameters(), 'lr': 0.002},
            {'params': net.classifier_2.parameters(), 'lr': 0.002},
            {'params': net.classifier_3.parameters(), 'lr': 0.002},
            {'params': net.classifier_3_1.parameters(), 'lr': 0.002},
            {'params': net.fc1.parameters(), 'lr': 0.002},
            {'params': net.fc2.parameters(), 'lr': 0.002},
            {'params': net.fc3.parameters(), 'lr': 0.002},
            {'params': net.conv_block1.parameters(), 'lr': 0.002},
            {'params': net.conv_block2.parameters(), 'lr': 0.002},
            {'params': net.conv_block3.parameters(), 'lr': 0.002},
            {'params': net.features.parameters(), 'lr': 0.0002}
        ],
            momentum=0.9, weight_decay=5e-4)
    
    else:     # for p = 0.1
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    
    save_name = dataset+'_'+str(nb_epoch)+'_'+str(img_size)+'_p'+str(proportion)+'_bz'+str(batch_size)+'_ResNet-50_'+'_'+lr_adjt
    train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device, device_no, save_name)

    # Evaluate OA
    # test(net, testloader, CELoss, tree, device, args.dataset)

    # Evaluate Average PRC
    # test_AP(net, args.dataset, testset, testloader, device)
