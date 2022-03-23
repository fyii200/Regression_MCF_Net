import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import train_step, validation_step, save_output
from utils.metric import compute_metric

import pandas as pd
from networks.densenet_mcf import dense121_mcs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

# Setting parameters
parser = argparse.ArgumentParser(description='EyeQ_dense121')
parser.add_argument('--save_model', type=str, default='DenseNet121_v3_v1')
parser.add_argument('--save_dir', type=str, default='./MCF_Net')
parser.add_argument('--result_name', type=str, default='csv_result')
parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])
parser.add_argument('--crop_size', type=int, default=224)

# Optimization options
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)

args = parser.parse_args()

# Images Labels
train_images_dir = 'images/train'
label_train_file = 'csv_data/EyeQ/EyeQ_train.csv'

# Image Labels: original test set (n=16,249) is further split into a validation set (30%, n=4,875) and a test set (70%, n=12,015). Split validation and test set share the same image directory ('test_images_dir')
test_images_dir = 'images/test'     
label_val_file = 'csv_data/EyeQ/EyeQ_split_val.csv'
label_test_file = 'csv_data/EyeQ/EyeQ_split_test.csv'

best_metric = np.inf
best_iter = 0

model = dense121_mcs(n_class=args.n_classes)
model.to(device)

# Use Mean Absolute Error (MAE) as loss function
MAE_criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

transform_list1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, +180)),
    ])

transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

data_train = DatasetGenerator(csv_file=label_train_file, data_dir=train_images_dir, transform1=transform_list1,
                              transform2=transformList2, n_class=args.n_classes, set_name='train')
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

data_val = DatasetGenerator(csv_file=label_val_file, data_dir=test_images_dir, transform1=transform_list_val1,
                            transform2=transformList2, n_class=args.n_classes, set_name='train')
val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)


data_test = DatasetGenerator(csv_file=label_test_file, data_dir=test_images_dir, transform1=transform_list_val1,
                             transform2=transformList2, n_class=args.n_classes, set_name='test')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)


# Training and validation:
# Run for a fixed number of epoch. Model with the lowest validation MAE is saved as best model
for epoch in range(0, args.epochs):
    _ = train_step(train_loader, model, epoch, optimizer, MAE_criterion, args)
    validation_loss = validation_step(val_loader, model, MAE_criterion)
    print('Current Loss: {}| Best Loss: {} at epoch: {}'.format(validation_loss, best_metric, best_iter))

    # save best model
    if best_metric > validation_loss:
        best_metric = validation_loss
        best_iter = epoch
        model_save_file = os.path.join(args.save_dir, args.save_model + '.tar')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save({'state_dict': model.state_dict(), 'best_loss': best_metric}, model_save_file)
        print('Model saved to %s' % model_save_file)


# Testing
outPRED_mcs = torch.FloatTensor().cuda()
model.eval()
iters_per_epoch = len(test_loader)
bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
bar.check_tty = False
for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
    imagesA = imagesA.cuda()
    imagesB = imagesB.cuda()
    imagesC = imagesC.cuda()

    begin_time = time.time()
    _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
    outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                           batch_time=batch_time * (iters_per_epoch - epochID) / 60)
    bar.next()
bar.finish()

# NORMALISE quality scores between 0 (best) and 1 (worst)
minimum = outPRED_mcs.min()
maximum = outPRED_mcs.max()
outPRED_mcs = (outPRED_mcs - minimum) / (maximum - minimum)

# save NORMALISED quality scores
save_output(label_test_file, outPRED_mcs, args, save_file=args.result_name)

# MAE loss on the test set (desired quality output = continuous)
df_tmp = pd.read_csv(save_file_name)
pred_quality = torch.tensor(df_tmp['quality'])

dt_gt = pd.read_csv(label_test_file)
GT = torch.tensor(dt_gt['quality'] / 2) # scale ground-truth quality score to 0-1

test_MAE = MAE_criterion(pred_quality, GT)

# Print MAE
print('Test MAE = {:0.4f} \n'.format(test_MAE) )


