import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from scipy import interpolate
import numpy as np
from time import time
import matplotlib.pyplot as plt
from constants import *
import os
import glob
import cv2
import PIL
import random
from cnn_model import CNNModel

# Load in data as np arrays 
recordings_dir = os.getcwd() + '/data/archive/csv/'
csv_list = glob.glob(recordings_dir + '*.csv')

random.shuffle(csv_list)
data = np.zeros((len(csv_list), 21, 1000))
label = np.zeros(len(csv_list), dtype='int')

for i in range(len(csv_list)):
    temp_arr = np.genfromtxt(csv_list[i], delimiter = ',')
    temp_arr = cv2.resize(temp_arr, (1000, 21))
    data[i] = temp_arr
    lang = [lang for lang in PROCESS_LANGUAGE_LIST if lang in csv_list[i]][0]
    label[i] = ETHNICITY_LABEL_DICT[lang]


test_split = int(0.2 * len(label))
train_x = data[:-test_split]
train_y = label[:-test_split]
test_x = data[-test_split:]
test_y = label[-test_split:]

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
num_epochs = 40
epoch_loss = np.zeros(num_epochs)
time0 = time()
for epoch in range(num_epochs):
    running_loss = 0
    for i in range(len(train_x)):
        x, y = torch.from_numpy(train_x[i]).unsqueeze(0).unsqueeze(0), torch.from_numpy(np.array(train_y[i])).unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()
        outputs = model(x.float())
        loss = criterion(outputs, torch.max(y, 1)[0])

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print ('Epoch : %d/%d,  Loss: %.8f' %(epoch+1, num_epochs, running_loss))
    epoch_loss[epoch] = running_loss
print("\nTraining Time (in minutes) =",(time()-time0)/60)

torch.save(model, 'torch_classifier.pt')

fig, ax = plt.subplots()
plt.plot(epoch_loss)
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.title('Model Train Loss by Epochs')
fig.set_size_inches(9, 6)
ax.legend()
plt.savefig('train_loss.png', dpi=400)

correct_count, all_count = 0, 0
for i in range(len(test_x)):
    x, y = torch.from_numpy(test_x[i]).unsqueeze(0).unsqueeze(0), torch.from_numpy(np.array(test_y[i])).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logps = model(x.float())

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = y.numpy()[0][0]
    if(true_label == pred_label): correct_count += 1
    all_count += 1

print("Number Of Sound Files Tested =", all_count)
print("\nModel Accuracy = {:.2f}%".format(correct_count/all_count * 100))