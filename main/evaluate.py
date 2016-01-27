from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import utils

def read_data(file):
    file = open(file,'r')
    lines = file.readlines()
    lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        i=i.lower()
        if(len(i) > 0):
            i=i.split()
            ex = (i[0],i[1],float(i[2]))
            examples.append(ex)
    return examples

def getCorrelation(examples, We, words):
    gold = []
    pred = []
    num_unks = 0
    for i in examples:
        (v1,t1) = utils.lookup_with_unk(We,words,i[0])
        (v2,t2) = utils.lookup_with_unk(We,words,i[1])
        pred.append(-1*cosine(v1,v2)+1)
        if t1:
            num_unks += 1
        if t2:
            num_unks += 1
        gold.append(i[2])
    return (spearmanr(pred,gold)[0], num_unks)

def evaluateWordSim(We, words):
    ws353ex = read_data('../data/wordsim353.txt')
    ws353sim = read_data('../data/wordsim-sim.txt')
    ws353rel = read_data('../data/wordsim-rel.txt')
    simlex = read_data('../data/SimLex-999.txt')
    (c1,u1) = getCorrelation(ws353ex,We,words)
    (c2,u2) = getCorrelation(ws353sim,We,words)
    (c3,u3) = getCorrelation(ws353rel,We,words)
    (c4,u4) = getCorrelation(simlex,We,words)
    return ([c1,c2,c3,c4],[u1,u2,u3,u4])

def evaluate_all(model,words):
    (corr, unk) = evaluateWordSim(model.all_params[0].get_value(),words)
    s="{0} {1} {2} {3} ws353 ws-sim ws-rel sl999".format(corr[0], corr[1], corr[2], corr[3])
    print s

if __name__ == "__main__":
    (words, We) = utils.getWordmap('../data/glove_small.txt')
    print evaluateWordSim(We, words)