import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # percentage bar for accuracy, loss and validation.

TRAIN_DIR = 'test'
TEST_DIR = 'test'
IMG_SIZE = 240
LR = 1e-3

MODEL_NAME = 'braintumor-{}-{}.model'.format(LR, '2conv-basic')
def label_img(img):
  word_label = img.split(' ')[-2]
    # conversion to one-hot array [no brain tumor,yes brain tumor]
                        
  if word_label == 'no': return [1,0]
                                 
  elif word_label == 'ye': return [0,1]


def create_train_data():
  training_data = []
  for img in tqdm(os.listdir(TRAIN_DIR)):
    label = label_img(img)
    path = os.path.join(TRAIN_DIR,img)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    training_data.append([np.array(img),np.array(label)])
  shuffle(training_data)
  np.save('train_data.npy', training_data)
  return training_data

def process_test_data():
  testing_data = []
  for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR,img)
    img_num = img.split('.')[0]
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img), img_num])   
  shuffle(testing_data)
  np.save('/content/test_data.npy', testing_data)
  return testing_data

train_data = create_train_data()