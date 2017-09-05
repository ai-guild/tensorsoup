class DataFeed(object):
     
    def __init__(self, dformat, data=None):

        # current iteration
        self.offset = 0

        # data format
        self.dformat = dformat

        # num of examples
        self.n = 0

        # default batch size
        self.B = 2

        # if data available
        if data:
            self.bind(data)


    ''' 
        bind data to feed

    '''
    def bind(self, data):
        # start over
        self.offset = 0
        # get num of examples
        #  get a value from dict
        self.n = len(data[self.dformat[0]])
        # bind data to instance
        self.data = data


    '''
        get num of examples

    '''
    def getN(self):
        return self.n

    
    '''
        get batch next to offset

    '''
    def batch(self, batch_size):
        # fetch batch next to offset
        s, e = self.offset, self.offset + batch_size

        # update offset
        self.offset += batch_size

        # select items from data format
        return [ self.data[k][s:e] for k in self.dformat ]


    '''
        get next batch

    '''
    def next_batch(self, batch_size=1):
        # check limit
        if self.offset + batch_size > self.n:
            # star over
            self.offset = 0
        # return batch
        return self.batch(batch_size)
