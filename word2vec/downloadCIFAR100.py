#!/usr/bin/env python
from six.moves.urllib import request
import cPickle as pickle
import os
import cv2
import subprocess
import numpy as np

picBase = "/mnt/s3pic/cifar10/"

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def checkExistance(path):
  if os.path.exists(path):
    return True
  else:
    return False

def makeDirectory(path):
  if not checkExistance(path):
    os.makedirs(path)

def downloadPic(url, name):
  cmd = "wget " + url + " -O " + name + " -q"
  subprocess.call(cmd, shell=True)

def extractExtension(name):
  return re.findall(r"^.*(\..*)$", name)[0]

def moveFile(path, name):
  cmd = "mv " + name + " " + path
  subprocess.call(cmd, shell=True)

def cifar100Extract():
  makeDirectory(picBase)
  makeDirectory(picBase + "train")
  makeDirectory(picBase + "test")
  makeDirectory(picBase + "label")
  if not os.path.exists("cifar-100-python"):
    request.urlretrieve(
     "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
     "./cifar-100-python.tar.gz"
    )
    cmd = "tar -xvzf cifar-100-python.tar.gz"
    subprocess.call(cmd, shell=True) 
    cmd = "rm -r cifar-100-python.tar.gz"
    subprocess.call(cmd, shell=True) 
  def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
  train = unpickle("cifar-100-python/train")
  test = unpickle("cifar-100-python/test")
  tag = unpickle("cifar-100-python/meta")
  tagWith_ = ['aquarium_fish',
   'lawn_mower',
   'maple_tree',
   'oak_tree',
   'palm_tree',
   'pickup_truck',
   'pine_tree',
   'sweet_pepper',
   'willow_tree']
  tagAlter = ["fish",
    "lawnmower",
    "maple",
    "oak",
    "palm",
    "truck",
    "pine",
    "paprika",
    "willow"]
  index = [tag["fine_label_names"].index(with_) for with_ in tagWith_]
  count = 0
  for i in index:
    tag["fine_label_names"][i] = tagAlter[count]
    count = count + 1
  y_train = {}
  y_test = {}
  x_test = test['data']
  x_test = x_test.reshape(len(x_test),3,32,32)
  x_train = train['data']
  x_train= x_train.reshape(len(x_train),3,32,32)
#  for x in zip(x_test, test["filenames"], test["fine_labels"]):
#    cv2.imwrite(picBase + "test/" + x[1], x[0].transpose(1,2,0)[:,:,::-1].copy())
#    y_test[x[1]] = x[2]
#  for x in zip(x_train, train["filenames"], train["fine_labels"]):
#    cv2.imwrite(picBase + "train/" + x[1], x[0].transpose(1,2,0)[:,:,::-1].copy())
#    y_train[x[1]] = x[2]
  
#  save_object(y_test, picBase + "label/y_test.pkl")
#  save_object(y_train, picBase + "label/y_train.pkl")
  return (train, test, tag)
"""
In [38]: tag.keys()
Out[38]: ['fine_label_names', 'coarse_label_names']
In [40]: train.keys()
Out[40]: ['data', 'batch_label', 'fine_labels', 'coarse_labels', 'filenames'
In [41]: len(train["data"])
Out[41]: 50000
In [42]: len(train["data"][0])
Out[42]: 3072 // it means 32*32*3
"""

if __name__ == '__main__':
 # x_train, y_train, x_test, y_test, tag = cifar100Extract()
  train, test, tag = cifar100Extract()
