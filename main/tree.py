import utils

class tree(object):

    def __init__(self, phrase):
        self.phrase = phrase
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            self.embeddings.append(utils.lookupIDX(words,i))

    def unpopulate_embeddings(self):
        self.embeddings = []