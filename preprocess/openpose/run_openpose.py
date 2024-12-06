import pdb

# import config
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
# from preprocess.openpose.annotator.util import resize_image, HWC3
# from preprocess.openpose.annotator.openpose import OpenposeDetector # IDM-VTON/preprocess
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector 

import argparse
from PIL import Image
import torch
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class OpenPose:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.preprocessor = OpenposeDetector()

    def __call__(self, input_image, resolution=384):
        torch.cuda.set_device(self.gpu_id)
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            assert (H == 512 and W == 384), 'Incorrect input image shape'
            # Bắt đầu đo thời gian chạy openpose
            optime = time.time()
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints = {"pose_keypoints_2d": candidate}
            # In ra thời gian chạy openpose
            print(f'Elapsed time openpose: {time.time() - optime}')

            with open(json_path, "w") as f:
                json.dump(keypoints, f)
            
            #print(candidate)
            output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
            cv2.imwrite(out_path, output_image)

        return keypoints


# if __name__ == '__main__':
#     start = time.time()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, default='Girl_384x512.jpg', help='Path to the image')
#     parser.add_argument('--output_dir', type=str, default='output/skeleton/Girl_384x512.jpg', help='Path to the image')
#     parser.add_argument('--keypoints_dir', type=str, default='output/keypoint/Girl_384x512.json', help='Path to the json')
#     # parser.add_argument

#     model = OpenPose(0)
#     # model('./images/bad_model.jpg')
#     global arg
#     arg = parser.parse_args()
#     global json_path, out_path
#     json_path = arg.keypoints_dir
#     out_path = arg.output_dir

#     model(arg.input_dir)
#     print(f'Elapsed time: {time.time() - start}')
