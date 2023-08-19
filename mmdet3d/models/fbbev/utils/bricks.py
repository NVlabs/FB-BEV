import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2


def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,normalize=False):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    max_ = tensor.flatten(1).max(-1).values[:, None, None]
    min_ = tensor.flatten(1).min(-1).values[:, None, None]
    tensor = (tensor-min_)/(max_-min_)
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=normalize).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


import functools
import time
from collections import defaultdict
time_maps = defaultdict(lambda :0.)
count_maps = defaultdict(lambda :0.)
def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            res = fn(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            time_maps['%s : %s'%(name, fn.__name__) ] += elapsed
            count_maps['%s : %s'%(name, fn.__name__) ] +=1
            print("%s : %s takes up %f "% (name, fn.__name__,time_maps['%s : %s'%(name, fn.__name__) ] /count_maps['%s : %s'%(name, fn.__name__) ] ))
            return res
        return wrapper
    return middle