"""
nnbody.py -- The main model definition for the nnbody solver,.
"""
import tensorflow as tf
import os
import math
LEARNING_RATE=1e-2
TRACK_VARS = False

def fc_layer(input_, num_neurons, activation=tf.nn.relu, name="fc"):
    with tf.variable_scope(name):
        num_channels = input_.get_shape().as_list()[-1]
        W = variable([num_channels, num_neurons],num_neurons) 
        b = variable([num_neurons],num_neurons) 
        net = tf.matmul(input_, W) + b

    return  activation(net)


    
def variable(shape, f, name="variable"):
    """
    Creates a tensor of SHAPE drawn from
    a random uniform distribution in 0 +/- 1/sqrt(f)
    """
    v =  tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
    #v = tf.Variable(tf.constant(-0.00001, shape=shape, name=name))
    if TRACK_VARS: variable_summaries(var, name)
    return v


class NNBody:
    """
    The nnbody model.
    """
    def __init__(self, sess, training_data, data_queue, n):
        """
        Initialzies the model with a SESS,
        a DATA_QUEUE, and a number of bodys N.
        """
        self.sess = sess
        self.n = n
        scope_name = "NNBody"
        with tf.device('/cpu:0'):
            with tf.variable_scope(scope_name):
                with tf.variable_scope("data_pipeline"):
                    data_pairs  = training_data
                    inputs = data_pairs[0]
                    desireds = data_pairs[1]
                
                self.output = self.create_model(inputs)
                self.training_op = self.create_training_method(self.output, desireds)

                with tf.variable_scope("requeue"):
                    self.requeue_op = data_queue.enqueue_many(training_data)

                with tf.variable_scope("persitence_initialization"):
                    vars_in_scope = tf.get_collection(
                        tf.GraphKeys.VARIABLES, scope=scope_name)
                    self.saver = tf.train.Saver(vars_in_scope)
                    self.init = tf.variables_initializer(vars_in_scope)


    def initialize(self, model_path, restore=False):
        tm = os.path.join(model_path, "model.cpkt")
        self.sess.run(self.init)
        if restore:
            self.saver.restore(self.sess, tm)

    def save(self, model_path):
        tm = os.path.join(model_path, "model.cpkt")
        self.saver.save(self.sess, tm )

    def get_training_ops(self):
        """
        Gets the training operations
        """
        return [[self.training_op, self.requeue_op]], self.loss


    def create_model(self, inputs):
        """
        Creates the feedforward neural network definiton
        """
        with tf.variable_scope("model"):
            num_inputs = self.n*2*2 + 1 # {P, V} (In R^2), Time
            num_outputs = self.n*2*2 # {P, V} (in R^2)

            head = fc_layer(inputs, 400)
            head = fc_layer(inputs, 300)
            head = fc_layer(inputs, 300)
            head = fc_layer(inputs, 300)
            head = fc_layer(inputs, num_outputs, activation=tf.identity)
            
            return head

        return output


    def create_training_method(self, outputs, desireds):
        """
        Creates the training method for the neural network, by minimizing some 
        expected loss.
        """
        with tf.variable_scope("training"):
            diff = outputs - desireds
            loss = tf.reduce_mean(tf.nn.l2_loss(diff)) 
            # todo add l2 loss
            self.loss =loss
            return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


