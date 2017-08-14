"""
training.py -- The main file for training the nn-body neural network on 
               simulation data.
"""
import tensorflow as tf
import numpy as np
import argparse
import threading
import math
import tqdm
import os

from simulation import nbody, RK4
from nnbody import NNBody

BATCH_SIZE=64
BOOTSTRAP_SIZE=40000
H = 0.01 # RK4 step size
TMAX = 0.2 # Time max

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_options():
    """
    Parses the command line arguments for the trainer
    """
    parser = argparse.ArgumentParser(description='Single agent data generator.')

    parser.add_argument('N', type=int,
                        help='The number of bodies')

    parser.add_argument('model_path', type=str,
                        help='The nodel path')

    parser.add_argument('max_iters', type=int,
                    help='The number of iterations to train the model for.')

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='The batch size')

    parser.add_argument('--bootstrap_size', type=int, default=BOOTSTRAP_SIZE,
                        help='the number of dawtapoints to collect before training')

    parser.add_argument('--restore', action='store_true',
                        help='whether or not to resume training')


    args = parser.parse_args()
    return args



def _flatten(tens):
    """
    Flattens a tensor in numpy
    """
    shape = tens.shape
    dim = np.prod(shape) 
    return np.reshape(tens, [dim])


def build_data_queue(n, bootstrap_size, batch_size):
    with tf.device('/cpu:0'):
        with tf.variable_scope("data_pipeline"):
            data_queue = tf.RandomShuffleQueue(
                capacity=1e6,
                min_after_dequeue=bootstrap_size,
                dtypes=[tf.float32, tf.float32],
                shapes=([n*2*2 +1], [n*2*2]))

            tpv0 = tf.placeholder("float32", shape=n*2*2 +1)
            pvfinal = tf.placeholder("float32", shape=n*2*2)
            enqueue = data_queue.enqueue((tpv0, pvfinal))
            dequeue = data_queue.dequeue_many(batch_size)

    return data_queue, (tpv0, pvfinal, enqueue), dequeue

def build_data_generator(sess, n, device='/gpu:1'):
    """
    A method for generating data given a threading coordinator.
    """

    with tf.device(device):
        with tf.variable_scope("data_generator"):
                with tf.variable_scope("RK4"):

                    PV = tf.placeholder(tf.float32, [n*2,2])
                    M = tf.placeholder(tf.float32, [n])
                    G = tf.constant(1.0)
                    nbody_dif_eq = nbody(M, G=G, eps_radius=1e-4)
                    t, pv = RK4(nbody_dif_eq, PV, TMAX, h=H)

        local_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='data_generator')
        init = tf.variables_initializer(local_vars)
        sess.run(local_vars)

    return (PV, M, G), (t, pv)



def run_data_generator(sess,  coord, rk4in, rk4out, data_feed, n):
    """
    Runs the data generator as a thread 
    """
    (PV, M, G), (tint, pv), (tpv0, pvfinal, enqueue ) = rk4in, rk4out, data_feed
    # Fix the mass from the beginning of training.
    mass = np.random.random(n)*10000
    while not coord.should_stop():
        # Draw random inital conditions
        # Position can be in the unit hypercube, velocity always has initial value zero.
        P0V0 = (np.random.random((n*2,2)) - 0.5)*np.array([[1,1] if i < n else [0,0] for i in range(n*2)])

        # Get Simulation data
        pv_hist = [P0V0]
        tt = 20
        for t in (range(tt)):
            pv_hist += sess.run(pv[1:], {PV: pv_hist[-1], M: mass, G:0.000001})
        pv_hist= np.array(pv_hist)

        tint = np.linspace(0, TMAX*tt, math.ceil(tt*TMAX/H))

        # Eqnueue it.
        flattened_versions = [[tint[i], _flatten(pvinst)] for i, pvinst in enumerate(pv_hist)]
        for i, (t, flatpv) in enumerate(flattened_versions):
            for j, (tend, targetpv) in enumerate(flattened_versions[i:]):
                delta_t = tend- t
                inputs, desireds = np.concatenate([np.array([delta_t]), flatpv]), targetpv
                sess.run(enqueue, {tpv0: inputs, pvfinal: desireds})



def train(sess, coord, max_iters, model, model_path):
    """
    Trains the model untill error is low enough.
    """
    tboard_dir = os.path.join(model_path, 'logs'/)
    ensure_dir(tboard_dir)
    train_writer = tf.summary.FileWriter(tboard_dir,sess.graph)
    for it in range(max_iters):
        if coord.should_stop():
            break

        # Train the model.
        training_ops, loss = model.get_training_ops(), model.loss
        _, summaries, computed_loss = sess.run(
            [training_ops, model.merged, loss])

        train_writer.add_summary(summary, it)

        if it% 100 == 0: print(it, computed_loss)
        if it % 10000 == 0:
            model.save(model_path)

    coord.request_stop()


def main(opts):
    num_bodies = opts.N

    # Set up tensorflow session
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config = config)

    print("Hi!")
    ensure_dir(opts.model_path)

    # Make the data pipeline.
    print("Building data queue")
    data_queue, data_feed, dequeue = build_data_queue(
        num_bodies,
        opts.bootstrap_size,
        opts.batch_size)

    print("Building RK4 data generator.")
    # Make the generator
    rk4in, rk4out = build_data_generator(sess, num_bodies)

    print("Building model.")
    # Make the NNBody neural network
    model = NNBody(sess, dequeue, data_queue, num_bodies)
    model.initialize(opts.model_path, opts.restore)

    # Main thread: create a coordinator.
    coord = tf.train.Coordinator()

    threads = [
        threading.Thread(
            target=run_data_generator,
            args=(sess, coord, rk4in, rk4out, data_feed, num_bodies)),
        threading.Thread(
            target=train,
            args=(sess, coord, opts.max_iters, model, opts.model_path))]

    try:
        for t in threads:
            t.start()
        coord.join(threads)
    except KeyboardInterrupt:
        print("Stopping training.")
        coord.request_stop()





if __name__ == "__main__":
    opts = get_options()
    main(opts)