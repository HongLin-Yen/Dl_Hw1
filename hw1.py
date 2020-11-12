from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class Car_train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):

        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_list = []
        for idx in range(len(self.labels_frame)):
            if self.labels_frame.iloc[ idx, 1] not in self.label_list:
                self.label_list.append(self.labels_frame.iloc[ idx, 1])
    
    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.labels_frame.iloc[idx, 0]).zfill(6) + '.jpg')
        image = Image.open(img_name).convert("RGB")
        car_name = self.labels_frame.iloc[ idx, 1]
        label = self.label_list.index(car_name)
         # print(label, car_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image':image, 'label':label}

        return sample

class Car_test_Dataset(Dataset):
    def __init__(self, root_dir, transform = None):

        self.test_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.length = len(self.test_names)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.test_names[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def names(self):
        return self.test_names

os.environ["CUDA_VISIBLE_DEVICES"]="9"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#super paremeters
LR = 0.01
Epoch = 20
batch = 64
class_num = 196

training = True

PATH = './car_classify-1.pth'

#transforms of images
data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(size = 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

if __name__ == "__main__":

    #load data
    car_dataset = Car_train_Dataset('training_labels.csv', 'training_data/training_data/', transform = data_transform)
    #test_dataset = Car_test_Dataset('training_data/', transform = data_transform)
    test_dataset = Car_test_Dataset('testing_data/testing_data/', transform = data_transform)

    trainloader = DataLoader(car_dataset, batch_size = batch, shuffle = True, num_workers = 4, drop_last = True)
    testloader = DataLoader(test_dataset, batch_size = batch, shuffle = False, num_workers = 4)

    print('load finished')
    """
    print(len(car_dataset.label_list))
    
    for i, data in enumerate(trainloader):
        inputs = data['image']
        labels = data['label']

        print(inputs.size())
        #img = transforms.ToPILImage()(inputs[5].squeeze_(0))
        # img.show()
        for e in labels:
            print(car_dataset.label_list[e])

        break
    """
    #setup network and transfer to device
    net = models.resnet50(pretrained = True)
    fc_inputs = net.fc.in_features
    net.fc = nn.Linear(fc_inputs, class_num)

    net = net.to(device)

   # print(net)

    criterion = nn.CrossEntropyLoss() #classifcation task loss
    optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9)
    
    if training:
        for epoch in range(Epoch): #training
            running_loss = 0.0
            acc=0
            #j = 0
            for i, data in enumerate(trainloader):
                #collect data and transfer to device
                inputs = data['image']
                labels = data['label']

                inputs, labels = inputs.to(device), labels.to(device)

                #clear gradiant
                optimizer.zero_grad()
                
                #forward and backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                #calculate loss rate and accuracy
                _,pre = torch.max(outputs.data, 1)
                """
                if j == 0:
                    print(pre[:3], labels[:3])
                    j += 1

                    for e in pre[:3]:
                        print(car_dataset.label_list[e])

                    for e in labels[:3]:
                        print(car_dataset.label_list[e])
                """
                for ii in range(batch):
                    if pre[ii] == labels[ii] :
                        acc+=1
                running_loss += loss.item()

                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
                    print(acc/( batch*20))
                    acc = 0
            
            #adjust learning rate
            if epoch >= 15:
                LR = 0.0001
            elif epoch >= 5:
                LR = 0.001
            """
            #testing per epoch
            with torch.no_grad():
                answer = []
                
                for i, img in enumerate(testloader):
                    img = img.to(device)

                    outputs = net(img)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.to("cpu")

                    # print(predicted)
            """

        print('Finished Training')

        #save model
        torch.save(net.state_dict(), PATH)
    else: #not training
        net.load_state_dict(torch.load(PATH))    
    
    print("testing")
    with torch.no_grad(): #testing
        answer = []

        for i, img in enumerate(testloader):
            img = img.to(device)

            outputs = net(img)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to("cpu")

            #print(predicted)

            ans_list = predicted.numpy()
            answer.extend(ans_list)

        #print(answer[0:4])

    #create propriate data style
    label_ans = [car_dataset.label_list[e] for e in answer]
    id_ans = test_dataset.test_names
    for i in range(len(id_ans)):
        id_ans[i] = id_ans[i][:-4]

    #print(id_ans[:4], label_ans[:4])

    #print(car_dataset.label_list.index('AM General Hummer SUV 2000'))

    #output result
    ans_dict = {"id": id_ans, "label": label_ans}
    select_df = pd.DataFrame(ans_dict)
    select_df.to_csv('answer.csv', index = False)
    
    #print(select_df)
