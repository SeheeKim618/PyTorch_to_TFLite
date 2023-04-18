import torch
import os
import sys
sys.path.append('..')
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn import functional as F
from model import get_pose_net
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

model_path = '/home/seheekim/Efficinet3Dpose/output/model_eff/snapshot_24.pth.tar'
img_size = (256, 256)
batch_size = 1
onnx_model_path = 'model_final.onnx'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_pose_net('EFF', 21).to(device)
loaded_state_dict = torch.load(model_path,map_location=device)['network']
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.","") 
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

x = torch.rand((batch_size, 3, 256, 256)).cuda(0)

model = model.to(device)
y = model(x).to(device)

torch.onnx.export(
    model,
    x, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
) 
## onnx == 1.11.0