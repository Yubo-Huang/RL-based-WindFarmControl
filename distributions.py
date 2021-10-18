import tensorflow as tf
import numpy as np
import tf_util as U
from tensorflow.python.ops import math_ops
from multiagent.multi_discrete import MultiDiscrete
from tensorflow.python.ops import nn

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class ClipDiagGaussianPdType(PdType):
    def __init__(self, action_spac):
        self.size = len(action_spac)
        self.min = action_spac[0][0] 
        self.max = action_spac[0][1]
    def pdclass(self):
        return ClipDiagGaussianPd
    def pdfromflat(self, flat):
        return ClipDiagGaussianPd(self.min, self.max, flat)
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class ClipDiagGaussianPd(Pd):
    def __init__(self, min, max, flat):
        self.min = min
        self.max = max
        self.flat = flat
        mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat        
    def mode(self):
        return self.mean
    def logp(self, x):
        return - 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=1) \
               - 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) \
               - U.sum(self.logstd, axis=1)
    def kl(self, other):
        assert isinstance(other, ClipDiagGaussianPd)
        return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=1)
    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), 1)
    def sample(self):
        action = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        action_clip = tf.clip_by_value(action, -1.1, 1.1)
        action_clip = self.min + (action_clip + 1.1) * (self.max - self.min)/2.2
        return action_clip

def make_FFpdtype(ac_space):
    if isinstance(ac_space, list):
        return ClipDiagGaussianPdType(ac_space)