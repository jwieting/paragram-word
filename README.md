# paragram-word

Code to train Paragram word embeddings from the appendix of "From Paraphrase Database to Compositional Paraphrase Model and Back".

The code is written in python and requires numpy, scipy, theano and the lasagne library.

To get started, run setup.sh which will download the required files. Then run demo.sh to start training a model. Check main/train.py for command line options.

If you use our code for your work please cite:

@article{wieting2015ppdb,
title={From Paraphrase Database to Compositional Paraphrase Model and Back},
author={John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu and Dan Roth},
journal={Transactions of the ACL (TACL)},
year={2015}}