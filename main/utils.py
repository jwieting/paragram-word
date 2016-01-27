import numpy as np
from tree import tree
import evaluate
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import cPickle
import time
import sys

def lookup(We,words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return We[words[w],:]
    else:
        return We[words['UUUNKKK'],:]

def lookupIDX(words,w):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        return words['UUUNKKK']

def lookup_with_unk(We,words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return We[words[w],:],False
    else:
        return We[words['UUUNKKK'],:],True

def getData(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) >= 2:
                e = (tree(i[0]), tree(i[1]))
                examples.append(e)
    return examples

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def getPairRand(d,idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def getPairMixScore(d,idx,maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return getPairRand(d,idx)

def getPairsFast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = getPairRand(d,i)
            p2 = getPairRand(d,i)
        if type == "MIX":
            p1 = getPairMixScore(d,i,T[arr[2*i]])
            p2 = getPairMixScore(d,i,T[arr[2*i+1]])
        pairs.append((p1,p2))
    return pairs

def getpairs(model, batch, params):
    g1 = []; g2 = []
    for i in batch:
        g1.append(i[0].embeddings[0])
        g2.append(i[1].embeddings[0])

    embg1 = model.feedforward_function(g1)
    embg2 = model.feedforward_function(g2)

    for idx,i in enumerate(batch):
        i[0].representation = embg1[idx,:]
        i[1].representation = embg2[idx,:]

    pairs = getPairsFast(batch, params.type)
    p1 = []; p2 = []
    for i in pairs:
        p1.append(i[0].embeddings[0])
        p2.append(i[1].embeddings[0])

    return (g1,g2,p1,p2)

def saveParams(model, fname):
    f = file(fname, 'wb')
    cPickle.dump(model.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def train(model,data,words,params):
    start_time = time.time()
    counter = 0
    try:
        for eidx in xrange(params.epochs):
            kf = get_minibatches_idx(len(data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:
                uidx += 1
                batch = [data[t] for t in train_index]
                for i in batch:
                    i[0].populate_embeddings(words)
                    i[1].populate_embeddings(words)

                (g1x,g2x,p1x,p2x) = getpairs(model, batch, params)
                cost = model.train_function(g1x, g2x, p1x, p2x)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                if(checkIfQuarter(uidx,len(kf))):
                    if(params.save):
                        counter += 1
                        model.saveParams(params.outfile+str(counter)+'.pickle')
                    if(params.evaluate):
                        evaluate.evaluate_all(model,words)
                        sys.stdout.flush()

                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    i[0].unpopulate_embeddings()
                    i[1].unpopulate_embeddings()

                #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

            if(params.save):
                counter += 1
                saveParams(params.outfile+str(counter)+'.pickle')

            if(params.evaluate):
                evaluate.evaluate_all(model,words)

            print 'Epoch ', (eidx+1), 'Cost ', cost

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    print "total time:", (end_time - start_time)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                            minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def checkIfQuarter(idx,n):
    if idx==round(n/4.) or idx==round(n/2.) or idx==round(3*n/4.):
        return True
    return False