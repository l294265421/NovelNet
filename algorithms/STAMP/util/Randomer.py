import tensorflow as tf
# in tf 2.* take use of the tf 1.* api
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

class Randomer(object):
    stddev = None

    @staticmethod
    def random_normal(wshape):
        return tf.random_normal(wshape, stddev=Randomer.stddev)

    @staticmethod
    def set_stddev(sd):
        Randomer.stddev = sd