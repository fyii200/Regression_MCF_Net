import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from utils.trainer import train_step, validation_step, save_output
from dataloader.EyeQ_loader import DatasetGenerator_inference

import pandas as pd
from networks.densenet_mcf import dense121_mcs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='EyeQ_dense121')
parser.add_argument('--model_dir', type=str, default='MCF_Net/best_regression_model')
parser.add_argument('--pre_model', type=str, default='DenseNet121_v3_v1')
parser.add_argument('--test_images_dir', type=str, default='images')
parser.add_argument('--result_name', type=str, default='csv_result')

args = parser.parse_args()

model = dense121_mcs()

if args.pre_model is not None:
    loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'), map_location=device)
    model.load_state_dict(loaded_model['state_dict'])

model.to(device)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

data_test = DatasetGenerator_inference(data_dir=args.test_images_dir, transform1=transform_list_val1,
                                       transform2=transformList2, set_name='test')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


# test
outPRED_mcs = torch.FloatTensor()
model.eval()
iters_per_epoch = len(test_loader)
bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
bar.check_tty = False
for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
    # uncomment the following 3 lines if CUDA is enabled (GPU)
    imagesA = imagesA.cuda()
    imagesB = imagesB.cuda()
    imagesC = imagesC.cuda()

    begin_time = time.time()
    _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
    outPRED_mcs = torch.cat((outPRED_mcs.cuda(), result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                           batch_time=batch_time * (iters_per_epoch - epochID) / 60)
    bar.next()
bar.finish()

# NORMALISE quality scores between 0 and 1
minimum = outPRED_mcs.min()
maximum = outPRED_mcs.max()
outPRED_mcs = (outPRED_mcs - minimum) / (maximum - minimum)

result= {'name':[], 'quality':[]}

for ind in range(len(data_test)):
    result['name'].append( data_test.image_names[ind][len(args.test_images_dir)+1:] )
    result['quality'].append( float(outPRED_mcs.cpu().numpy()[ind]) )
    
result = pd.DataFrame(result)
result.to_csv(args.result_name)                                                             
                              
                    


