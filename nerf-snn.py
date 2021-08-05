from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
import copy
import torch.nn as nn
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
import spikingjelly.clock_driven.ann2snn.examples.utils as SJutils
from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
import matplotlib.pyplot as plt
from datasets.shapenet import build_shapenet
from shapenet_test import test_time_optimize, report_result, test

def main(log_dir=None):
    argparser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    argparser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')    
    argparser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    args = argparser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    #############################################
    #转成SNN，代码来自cnn_mnist.py
    model_name='NerfSNN'
    onnxparser = parser(name=model_name,
                        log_dir=log_dir + '/parser',
                        kernel='onnx')
    snn = onnxparser.to_snn(model)

    # 保存转换好的SNN模型
    # Save SNN model
    torch.save(snn, os.path.join(log_dir,'snn-'+model_name+'.pkl'))
    fig = plt.figure('simulator')
    #############################################

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state = checkpoint['meta_model_state_dict']

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        model.load_state_dict(meta_state)
        optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        test_time_optimize(args, model, optim, tto_imgs, tto_poses, hwf, bound)
        scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)

        create_360_video(args, model, hwf, bound, device, idx+1, savedir)
        
        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        test_psnrs.append(scene_psnr)
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")













if __name__ == '__main__':
    main('./cnn_mnist')