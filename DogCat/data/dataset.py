import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T

# show image
# import torchvision as tv
# from torchvision.transforms import ToPILImage
# show = ToPILImage()

class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        print  'DogCat'
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test data/test/cat.1000.jpg
        # train data/train/dog.10001.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split(',')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # varify:train 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            print("imgs train at first size %d" % len(imgs))
            self.imgs = imgs[:int(0.7*imgs_num)]
            print("imgs train post size %d" % len(self.imgs))
        else:
            self.imgs = imgs[int(0.7*imgs_num):]


        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([T.Scale(224),
                                            T.CenterCrop(224),
                                            T.ToTensor(),
                                            normalize])

            else:
                self.transforms = T.Compose([T.Scale(256),
                                            T.RandomSizedCrop(224),
                                            T.RandomHorizontalFlip(),
                                            T.ToTensor(),
                                            normalize])


    def __getitem__(self, item):

        img_path = self.imgs[item]

        if self.test:
            label = int(self.imgs[item].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label


    def __len__(self):
        return len(self.imgs)



if __name__ == "__main__":
    train_dataset = DogCat('./train/', train=True)
    trainloader = data.DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=2
                )

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100)).show()
    # show(images).show()

    print 'test end!!!'


