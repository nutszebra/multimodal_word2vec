#!/usr/bin/env python
from six.moves.urllib import request
import os
import subprocess

def cifar100Extract():
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
  train, test, tag = cifar100Extract()
