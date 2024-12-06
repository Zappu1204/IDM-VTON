import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch
import time
import argparse


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=['CPUExecutionProvider'])
        

    def __call__(self, input_image,output_dir,face_mask_dir):
        #torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image,output_dir,face_mask_dir)
        return parsed_image, face_mask
    
    
# def main():
#     start = time.time()
#     print("1")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, default='/afh/projects/project-open-ai-instance-sweden-69256885-4e00-4535-ac1a-e03d02348588/shared/Users/HaiBX.B22AT105/IDM-VTON/preprocess/humanparsing/Messi_highreso.jpg', help='Path to the image')
#     parser.add_argument('--output_dir', type=str, default='/afh/projects/project-open-ai-instance-sweden-69256885-4e00-4535-ac1a-e03d02348588/shared/Users/HaiBX.B22AT105/IDM-VTON/preprocess/humanparsing/output/output_image', help='Path to the image')
#     parser.add_argument('--face_mask_dir', type=str, default='/afh/projects/project-open-ai-instance-sweden-69256885-4e00-4535-ac1a-e03d02348588/shared/Users/HaiBX.B22AT105/IDM-VTON/preprocess/humanparsing/output/face_mask', help='Path to the json')
#     # parser.add_argument

#     model = Parsing(0)
#     print("2")
#     # model('./images/bad_model.jpg')
#     global arg
#     arg = parser.parse_args()
#     global face_mask_path, out_path
#     face_mask_path = arg.face_mask_dir
#     out_path = arg.output_dir
#     print("3")

#     output_image1 , face_mask1 = model(arg.input_dir,out_path,face_mask_path)
#     print("4")
#     print(f'Elapsed time: {time.time() - start}')
    
# if __name__ == '__main__':
#     main()