from tensorflow.contrib.layers import xavier_initializer as xinit


def init(f):

    def_init = xinit
    def wrapper(*args, **kwargs):
        def_init = xinit
        return f(*args, **kwargs)

    return wrapper
