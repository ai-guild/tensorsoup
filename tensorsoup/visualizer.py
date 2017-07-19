import tensorflow as tf


class Visualizer(object):

    def __init__(self, logdir='./log/', interval=1):

        self.logdir = logdir
        self.writer = tf.summary.FileWriter(self.logdir)
        self.interval = interval

    def attach_graph(self, graph):
        self.writer.add_graph(graph)

    def attach_scalars(self, model):
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('accuracy', model.accuracy)
        self.merge()

    def merge(self):
        self.summary_op = tf.summary.merge_all()

    def log(self, summary, i):
        self.writer.add_summary(summary, i)

    def variable_summaries(self, var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
