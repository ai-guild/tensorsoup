import re

class Dictionary(object):
    def __init__(self, i2w):
        self.i2w = i2w

    def index(self, word):
        if word in self.i2w:
            return self.i2w.index(word)
        elif re.search('cand(\d+)', word):
            match = re.search('cand(\d+)', word)
            if match:
                return int(match.group(1))

        else:
            return self.i2w.index('UNK')
                

    # check if a word is worthy
    def is_worthy(self, w):
        return len(w) > 0 and len(w) < 30 and 'http' not in w and 'www' not in w
