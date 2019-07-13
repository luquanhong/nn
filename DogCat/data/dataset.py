from torch.utils import data


class DogCat(data.Dataset):

    def __init__(self):
        print  'DogCat'