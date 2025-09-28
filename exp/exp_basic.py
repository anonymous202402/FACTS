import os
import torch
from models import MTS_31, MTS_31F


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MTS_31': MTS_31,
            'MTS_31F': MTS_31F,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        
        self.model = None
        self.model_img = None
        
        # 构建模型
        models = self._build_model()
        if isinstance(models, tuple) and len(models) == 2:
            self.model, self.model_img = models
            self.model.to(self.device)
            if self.model_img is not None:
                self.model_img.to(self.device)
        else:
            self.model = models
            if self.model is not None:
                self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None 

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
