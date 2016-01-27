import theano
from theano import tensor as T
from theano import config
import lasagne
import numpy as np

class paragram_word_model(object):

    def __init__(self, We_initial, params):

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        g1batchindices = T.ivector(); g2batchindices = T.ivector()
        p1batchindices = T.ivector(); p2batchindices = T.ivector()

        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)

        embg1 = lasagne.layers.get_output(l_emb, {l_in:g1batchindices})
        embg2 = lasagne.layers.get_output(l_emb, {l_in:g2batchindices})
        embp1 = lasagne.layers.get_output(l_emb, {l_in:p1batchindices})
        embp2 = lasagne.layers.get_output(l_emb, {l_in:p2batchindices})

        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2

        self.all_params = [We]

        word_reg = 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
        cost = T.mean(cost) + word_reg

        self.feedforward_function = theano.function([g1batchindices], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices,
                                              p1batchindices, p2batchindices], cost)

        prediction = g1g2

        self.scoring_function = theano.function([g1batchindices, g2batchindices],prediction)

        self.train_function = None
        updates = params.learner(cost, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices,
                                               p1batchindices, p2batchindices], cost, updates=updates)