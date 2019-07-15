# coding:utf-8

import warnings
import torch as t


class  DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'AlexNet'

    train_data_root = './data/train/'
    test_data_root = './data/test/'
    load_model_path = None #'./checkpoints'

    batch_size = 32
    use_gpu = False
    num_thread = 4
    print_freq = 20

    debug_file = '/tmp/debug'
    result_file = 'result.cvs'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()