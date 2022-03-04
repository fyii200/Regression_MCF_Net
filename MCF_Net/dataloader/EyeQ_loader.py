# encoding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
from sklearn import preprocessing
import pandas as pd

# scaled_labels range from 0 to 1, used in the classification layer; labels are one-hot encoded (3 outputs for each image)
def load_eyeQ_excel(data_dir, csv_file, n_class=3):
    image_names = []
    labels = []
    scaled_labels= []
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(n_class)))
    df_tmp = pd.read_csv(csv_file)
    img_num = len(df_tmp)

    for idx in range(img_num):
        image_name = df_tmp["image"][idx]
        image_names.append(os.path.join(data_dir, image_name[:-5] + '.jpeg'))
        
        label = lb.transform([int(df_tmp["quality"][idx])])
        labels.append(label)
        
        scaled_label = int(df_tmp["quality"][idx]) / 2
        scaled_labels.append(scaled_label)
        
    labels = np.array(labels)
    labels = labels.reshape( (len(labels), 3) )
    scaled_labels = torch.FloatTensor(scaled_labels).reshape((-1,1)) # make sure shape is (batch_size, 1)

    return image_names, labels, scaled_labels

# USE this data generator when running 'train_test.py'
class DatasetGenerator(Dataset):
    def __init__(self, csv_file, data_dir, transform1=None, transform2=None, n_class=3, set_name='train'):
        
        image_names, labels, scaled_labels = load_eyeQ_excel(data_dir, csv_file, n_class=3)
        self.image_names = image_names
        self.labels = labels
        self.scaled_labels = scaled_labels
        self.n_class = n_class
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train':
            label = self.labels[index]
            scaled_label = self.scaled_labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), torch.FloatTensor(label), torch.FloatTensor(scaled_label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)


# USE this to generate test data during inference (when running 'test_only.py')
class DatasetGenerator_inference(Dataset):
    def __init__(self, data_dir, transform1=None, transform2=None, set_name='train'):

        self.image_names = [os.path.join(data_dir,I) for I in os.listdir(data_dir)]
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train':
            label = self.labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)    
