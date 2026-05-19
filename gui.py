
import shutil
try:
    import dearpygui.dearpygui as dpg
except:
    print("No dpg running")
    dpg = None
from torchvision.utils import save_image
import numpy as np
import random
import os, sys
import torch

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

from gui_utils.base import GUIBase

class GUI(GUIBase):
    def __init__(self,
                 
                 ):
        

        # Initialize DPG      
        super().__init__()

         
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()

    # print('Runing from ... ',os.environ["SLURM_PROCID"])
    # exit()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--test_iterations", type=int, default=4000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000, 15999, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--view-test', action='store_true', default=False)
    
    parser.add_argument("--downsample", type=int, default=1)
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(True)
    print("Experiment: " + args.expname)
    hyp = hp.extract(args)
    initial_name = args.expname     
    name = f'{initial_name}'
    
    gui = GUI(
        
    )
    gui.run()
    del gui
    torch.cuda.empty_cache()
