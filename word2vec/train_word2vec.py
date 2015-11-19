#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--unit', '-u', default=300, type=int,
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
                    default='ns',
                    help='output model type ("hsm": hierarchical softmax, '
                    '"ns": negative sampling, "original": no approximation)')
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
print('')


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__(
            embed=F.EmbedID(n_vocab, args.unit),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        h = None
        for c in context:
            e = self.embed(c)
            h = h + e if h is not None else e

        return self.loss_func(h, x)


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        loss = None
        for c in context:
            e = self.embed(c)

            loss_i = self.loss_func(e, x)
            loss = loss_i if loss is None else loss + loss_i

        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            W=L.Linear(n_in, n_out),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.W(x), t)


#def calculate_loss(model, dataset, offset):
def calculate_loss(model, dataset, position):
    # use random window size in the same way as the original word2vec
    # implementation.
    w = np.random.randint(args.window - 1) + 1
"""
In [21]: np.random.randint(5 - 1) + 1
Out[21]: 4

In [22]: np.random.randint(5 - 1) + 1
Out[22]: 2
"""
    context = []
    for offset in range(-w, w + 1):
        if offset == 0:
            continue
        c_data = xp.asarray(dataset[position + offset])
        c = chainer.Variable(c_data)
        context.append(c)
    x_data = xp.asarray(dataset[position])
    x = chainer.Variable(x_data)
"""
dataset = [9, 0, 8, 6, 2, 7, 5, 3, 4, 1]
position=[3, 7]
w = 2
offset in [-2, -1, 0, 1, 2]
if offset = -1
n [59]: xp.asarray(dataset[position + -1])
Out[59]: array([8, 5])

final result of context is:
In [69]: context
Out[69]: [array([0, 7]), array([8, 5]), array([2, 4]), array([7, 1])]

x is:
array([6, 3])
"""
    return model(x, context)


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
#count how many times word, variable in here, appeared
#If the word of "metropolis" appears in ptb.train.txt only once, count is going to be 1
            counts[word2index[word]] += 1
#append index number of words
            dataset.append(word2index[word])
"""if document like this:
apple apple orange orange water apple

word2index = {"apple": 1, "orange": 2, "water": 3}
index2word = {1: "apple", 2: "orange", 3: "water"}
counts = {1: 3, 2: 2, 3: 1}
dataset = [1 1 2 2 3 1]
"""

n_vocab = len(word2index)

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(dataset))

if args.out_type == 'hsm':
    HSM = L.BinaryHierarchicalSoftmax
    tree = HSM.create_huffman_tree(counts)
    loss_func = HSM(args.unit, tree)
elif args.out_type == 'ns':
"""
the structure of cs is same as the variable counts
just it's converted from collenctions to array
"""
    cs = [counts[w] for w in range(len(counts))]
    loss_func = L.NegativeSampling(args.unit, cs, 20)
#args.unit is dimension of words
elif args.out_type == 'original':
    loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.model == 'skipgram':
    model = SkipGram(n_vocab, args.unit, loss_func)
elif args.model == 'cbow':
    model = ContinuousBoW(n_vocab, args.unit, loss_func)
else:
    raise Exception('Unknown model type: {}'.format(args.model))

if args.gpu >= 0:
    model.to_gpu()

dataset = np.array(dataset, dtype=np.int32)

optimizer = O.Adam()
optimizer.setup(model)

begin_time = time.time()
cur_at = begin_time
word_count = 0
"""
len(dataset) = 10000
args.window = 5
args.batchsize = 100
skip = (10000 - 5 * 2) // 100 = 99
"""
skip = (len(dataset) - args.window * 2) // args.batchsize
next_count = 100000
for epoch in range(args.epoch):
    accum_loss = 0
    print('epoch: {0}'.format(epoch))
"""
In [8]: np.random.permutation(10)
Out[8]: array([3, 1, 6, 5, 8, 4, 0, 2, 7, 9])
"""
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

"""
len(dataset) = 10000
args.window = 5
args.batchsize = 100
skip = (10000 - 5 * 2) // 100 = 99
indexes = [3 4 13 ... 92] #len(indexes) = 99
i = 3
position = np.array(range(0, 100)) * 99 + (5 + 3)
         = array([   8,  107,  206,  305,  404,  503,  602,  701,  800,  899,  998,
           1097, 1196, 1295, 1394, 1493, 1592, 1691, 1790, 1889, 1988, 2087,
           2186, 2285, 2384, 2483, 2582, 2681, 2780, 2879, 2978, 3077, 3176,
           3275, 3374, 3473, 3572, 3671, 3770, 3869, 3968, 4067, 4166, 4265,
           4364, 4463, 4562, 4661, 4760, 4859, 4958, 5057, 5156, 5255, 5354,
           5453, 5552, 5651, 5750, 5849, 5948, 6047, 6146, 6245, 6344, 6443,
           6542, 6641, 6740, 6839, 6938, 7037, 7136, 7235, 7334, 7433, 7532,
           7631, 7730, 7829, 7928, 8027, 8126, 8225, 8324, 8423, 8522, 8621,
           8720, 8819, 8918, 9017, 9116, 9215, 9314, 9413, 9512, 9611, 9710,
           9809])
"""
        position = np.array(
            range(0, args.batchsize)) * skip + (args.window + i)
        loss = calculate_loss(model, dataset, position)
        accum_loss += loss.data
        word_count += args.batchsize

        model.zerograds()
        loss.backward()
        optimizer.update()

    print(accum_loss)

with open('word2vec.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), args.unit))
    w = model.embed.W.data
    for i in range(w.shape[0]):
        v = ' '.join(['%f' % v for v in w[i]])
        f.write('%s %s\n' % (index2word[i], v))
