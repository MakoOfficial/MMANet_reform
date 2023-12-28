import os
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data.dataset import T_co
from torchvision import transforms
import random

seed=1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class resize:
    """resize the pic, and remain the ratio,use 0 padding """

    def __init__(self, reshape_size=224):
        self.reshape_size = reshape_size
        pass

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)
        w, h = int(w / long * self.reshape_size), int(h / long * self.reshape_size)
        img = img.resize((w, h))
        delta_w, delta_h = self.reshape_size - w, self.reshape_size - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        return img


class MMANetDataset(data.Dataset):
    def __init__(self, df, data_dir):  # __init__是初始化该类的一些基础参数
        self.data_dir = data_dir
        self.idList = os.listdir(self.data_dir)  # 目录里的所有文件
        self.df = df    # load the dataframe from cvd file
        self.zscore = []
        self.male = []
        self.ids = []
        # print(type(self.df['id'][0]))
        for i in range(len(self.idList)):
            zscore, male = self.get_label(i)
            self.zscore.append(zscore)
            self.male.append(male)
            # self.ids.append(int(self.idList[i]))
        self.male = torch.stack(self.male).type(torch.FloatTensor)
        self.zscore = torch.stack(self.zscore).type(torch.FloatTensor)
        # print(self.ids)
        self.ids = torch.IntTensor(self.ids)

    def __len__(self):
        return len(self.idList)

    def get_label(self, index):
        image_index = self.idList[index]
        # print(f"image_id: {image_index}")
        image_id = image_index.split('.')[0]
        self.ids.append(int(image_id))
        row = self.df[self.df['id'] == int(image_id)]
        zscroe = np.array(row['zscore'])
        male = np.array(row['male'].astype('float32'))
        return torch.Tensor(zscroe), torch.Tensor(male)

    def __getitem__(self, index):
        return self.ids[index], self.zscore[index], self.male[index]

    def __repr__(self):
        repr = "(MMANetDataset,\n"
        repr += "  len = %s,\n" % str(self.__len__())
        repr += ")"
        return repr


class Kfold_MMANet_Dataset(data.Dataset):

    def __init__(self, ids, zscore, male, data_dir, transforms):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.ids = ids
        self.zscore = zscore
        self.male = male
        self.pic = []
        self.read_pic()

    def __len__(self):
        return self.zscore.shape[0]

    def read_pic(self):
        length = self.ids.shape[0]
        for i in range(length):
            image_index = self.ids[i].item()
            filename = str(image_index) + ".png"
            img_path = os.path.join(self.data_dir, filename)
            img = Image.open(img_path)
            img = np.array(img.convert("RGB"))
            self.pic.append(self.transforms(image=img)['image'])

        self.pic = torch.stack(self.pic).type(torch.FloatTensor)

    def __getitem__(self, index) -> T_co:
        return (self.pic[index], self.male[index]), self.zscore[index]

    def __repr__(self):
        repr = "(DatasetsForKFold,\n"
        repr += "  len = %s,\n" % str(self.__len__())
        ori, canny, age, male = self.__getitem__(0)
        repr += f"the first line :age {age.item()}, male {male.item()}"
        repr += ")"
        return repr