#coding:utf-8

import torch as t
import  time

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))


    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints' + self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        t.save(self.state_dict(), name)

        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(BasicModule):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    flat = Flat()

    print t.arange(0,18).reshape((3, 3, 2))
    print flat.forward(t.arange(0,18).reshape((3, 3, 2)))

    flat.save()

    # flat.load('../checkpoints/')
