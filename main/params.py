class params(object):

    def __init__(self):
        self.LW = 1e-5
        self.eta = 0.05
    
    def __str__(self):
        t = "LW", self.LW, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)