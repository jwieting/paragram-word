from utils import getWordmap
from params import params
from utils import getData
from utils import train
from paragram_word_model import paragram_word_model
import lasagne
import random
import numpy as np
import argparse
import sys

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def str2learner(v):
    if v is None:
        return lasagne.updates.adagrad
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not a learner.')

random.seed(1)
np.random.seed(1)

params = params()

parser = argparse.ArgumentParser()
parser.add_argument("-LW", help="Lambda for word embeddings (normal training).", type=float)
parser.add_argument("-outfile", help="Output file name.")
parser.add_argument("-batchsize", help="Size of batch.", type=int)
parser.add_argument("-dim", help="Size of input.", type=int)
parser.add_argument("-wordfile", help="Word embedding file.")
parser.add_argument("-save", help="Whether to pickle the model.")
parser.add_argument("-train", help="Training data file.")
parser.add_argument("-margin", help="Margin in objective function.", type=float)
parser.add_argument("-samplingtype", help="Type of sampling used.")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training.")
parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
parser.add_argument("-learner", help="Either AdaGrad or Adam.")
parser.add_argument("-num_examples", help="Number of examples to use in training. If not set, will use all examples.", type=int)

args = parser.parse_args()

params.LW = args.LW
params.outfile = args.outfile
params.batchsize = args.batchsize
params.dim = args.dim
params.wordfile = args.wordfile
params.save = str2bool(args.save)
params.train = args.train
params.margin = args.margin
params.type = args.samplingtype
params.epochs = args.epochs
params.evaluate = str2bool(args.evaluate)
params.learner = str2learner(args.learner)
params.learner = lasagne.updates.adagrad

(words, We) = getWordmap(params.wordfile)
examples = getData(params.train, words)

if args.num_examples:
    examples = examples[0:args.num_examples]

print "Number of training examples: ", len(examples)
print sys.argv

model = paragram_word_model(We, params)

train(model,examples,words,params)