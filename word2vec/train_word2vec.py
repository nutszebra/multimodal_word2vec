#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections
import time

import numpy as np
import six.moves.cPickle as pickle
import random

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.optimizers as O

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--unit', '-u', default=100, type=int,
                    help='number of units')
parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='number of epochs to learn')
parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                    default='skipgram',
                    help='model type ("skipgram", "cbow")')
parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                    default='hsm',
                    help='output model type ("hsm": hierarchical softmax, '
                    '"ns": negative sampling, "original": no approximation)')
parser.add_argument('--nsnv', '-v', default=5, type=int,
                    help='the number of negative sampling about vision')
parser.add_argument('--l1', '-l', default=0.0001, 
                    help='hyper parameter for L1 norm')

args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('Window: {}'.format(args.window))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format(args.model))
print('Output type: {}'.format(args.out_type))
print('the number of negative sampling about vision: {}'.format(args.nsnv))
print('hyper parameter for L1 norm: {}'.format(args.l1))
print('')


def continuous_bow(dataset, position):
    h = None

    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    for offset in range(-w, w + 1):
        if offset == 0:
            continue
        d = xp.asarray(dataset[position + offset])
        x = chainer.Variable(d)
        e = model.embed(x)
        h = h + e if h is not None else e

    d = xp.asarray(dataset[position])
    t = chainer.Variable(d)
    return loss_func(h, t)


def skip_gram(dataset, position) :
    d = xp.asarray(dataset[position])
    t = chainer.Variable(d)

    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
    loss = None
    for offset in range(-w, w + 1):
        if offset == 0:
            continue
        d = xp.asarray(dataset[position + offset])
        x = chainer.Variable(d)
        e = model.embed(x)

        loss_i = loss_func(e, t)
        loss = loss_i if loss is None else loss + loss_i

    return loss

def negative_sampling_vision(dataset, position, visual, visual_negative, negative_info, r=0.5):
  d = xp.asarray(dataset[position])
  t = chainer.Variable(d)
  e = model.embed(t)
  c = model.M(e)
  weight = F.split_axis(c,len(c.data),axis=0)
  loss = None
  for info, i in zip(negative_info, range(len(negative_info))):
    if info:
      tmpLoss = None
      v = chainer.Variable(np.array([visual[i]],dtype=np.float32))
      cos_positive = F.sum(v * weight[i]) / chainer.Variable(np.array(np.linalg.norm(weight[i].data) * np.linalg.norm(v.data), dtype=np.float32))
      for v_n in visual_negative[i]:
        v_n = chainer.Variable(np.array([v_n],dtype=np.float32))
        cos_negative =  F.sum(v_n * weight[i]) / chainer.Variable(np.array(np.linalg.norm(weight[i].data) * np.linalg.norm(v_n.data),dtype=np.float32))
        tmpLoss = cos_negative if tmpLoss is None else tmpLoss + cos_negative
      tmpLoss = F.relu(chainer.Variable(np.array(r,dtype=np.float32)) + tmpLoss - cos_positive)
      loss = tmpLoss if loss is None else loss + tmpLoss
    else:
      continue
  return loss


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []
with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])

n_vocab = len(word2index)

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(dataset))

if args.model == 'skipgram':
    train_model = skip_gram
elif args.model == 'cbow':
    train_model = continuous_bow
else:
    raise Exception('Unknown model type: {}'.format(args.model))

model = chainer.FunctionSet(
    embed=F.EmbedID(n_vocab, args.unit),
    M=F.Linear(args.unit, 6272)
)

if args.out_type == 'hsm':
    HSM = F.BinaryHierarchicalSoftmax
    tree = HSM.create_huffman_tree(counts)
    model.l = HSM(args.unit, tree)
    loss_func = model.l
elif args.out_type == 'ns':
    cs = [counts[w] for w in range(len(counts))]
    model.l = F.NegativeSampling(args.unit, cs, 20)
    loss_func = model.l
elif args.out_type == 'original':
    model.l = F.Linear(args.unit, n_vocab)
    loss_func = lambda h, t: F.softmax_cross_entropy(model.l(h), t)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.gpu >= 0:
    model.to_gpu()

dataset = np.array(dataset, dtype=np.int32)

optimizer = O.Adam()
optimizer.setup(model)

begin_time = time.time()
cur_at = begin_time
word_count = 0
skip = (len(dataset) - args.window * 2) // args.batchsize
next_count = 100000

def load_object(path):
  with open(path, 'r') as f:
    answer = pickle.load(f)
  return answer

def toRowVector(vec): 
  dim = vec.shape 
  rowDim = 1 
  for d in dim: 
    rowDim = rowDim * d
  return vec.reshape(rowDim)

visual_train = load_object("/mnt/s3pic/cifar10/smallTrain_y/pca/32/train_32.pkl")
visual_tag = load_object("/mnt/s3pic/cifar10/smallTrain_y/pca/32/tag.pkl")
visual_y = load_object("/mnt/s3pic/cifar10/smallTrain_y/train_y.pkl")
visual_words = load_object("/mnt/s3pic/cifar10/smallTrain_y/words.pkl")

visual_X = np.array(visual_train, dtype=np.float32)
index2visual = {}

count = 0
for name in visual_tag:
  filename = name.split("/")[-1]
  pic_tag = visual_words[visual_y[filename]]
  if pic_tag in word2index:
    if word2index[pic_tag] in index2visual:
      index2visual[word2index[pic_tag]].append(toRowVector(visual_X[count]))
    else:
      index2visual[word2index[pic_tag]] = [toRowVector(visual_X[count])]
  count = count + 1
zeroVector = index2visual[index2visual.keys()[0]][0] * 0

negative_dict ={}

for key in index2visual:
  negative_dict[key] = []
  for keykey in index2visual:
    if key == keykey:
      pass
    else:
      for i in xrange(len(index2visual[keykey])):
        negative_dict[key].append((keykey, i))


for epoch in range(args.epoch):
    accum_loss = 0
    accum_loss_visual = 0
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(skip)
    for i in indexes:
        if word_count >= next_count:
            now = time.time()
            duration = now - cur_at
            throuput = 100000. / (now - cur_at)
            print('{} words, {:.2f} sec, {:.2f} words/sec'.format(
                word_count, duration, throuput))
            next_count += 100000
            cur_at = now

        position = np.array(
            range(0, args.batchsize)) * skip + (args.window + i)
        visual = np.array([random.sample(index2visual[key], 1)[0] if key in index2visual.keys() else zeroVector for key in dataset[position]], dtype=np.float32)
        visual_negative = []
        negative_info = []
        for v, i in zip(visual, range(len(visual))):
          if np.linalg.norm(zeroVector) == np.linalg.norm(v):
            visual_negative.append(np.array([zeroVector for i in xrange(0,args.nsnv)], dtype=np.float32))
            negative_info.append(False)
          else:
            negative_sample = random.sample(negative_dict[dataset[position][i]], args.nsnv)
            visual_negative.append(np.array([index2visual[i[0]][i[1]] for i in negative_sample], dtype=np.float32))
            negative_info.append(True)
        visual_negative = np.array(visual_negative, dtype=np.float32)
        loss = train_model(dataset, position)
        loss_visual = negative_sampling_vision(dataset, position, visual, visual_negative, negative_info)
        if loss_visual is None:
          pass
        else:
          l1_norm = chainer.Variable(np.array(args.l1 * np.linalg.norm(model.M.W),dtype=np.float32))
          loss = loss + loss_visual + l1_norm
        accum_loss += loss.data
        accum_loss_visual = accum_loss_visual if loss_visual is None else accum_loss_visual + loss_visual.data + l1_norm.data
        word_count += args.batchsize

        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

    print(accum_loss)
    print(accum_loss_visual)

model.to_cpu()
with open('model.pickle', 'wb') as f:
    obj = (model, index2word, word2index)
    pickle.dump(obj, f)
