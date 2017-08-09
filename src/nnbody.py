"""
nnbody.py -- The main model definition for the nnbody solver,.
"""
import tensorflow as tf

BATCH_SIZE=32
LEARNING_RATE=1e-4

class NNBody:
	"""
	The nnbody model.
	"""
	def __init__(self, sess, data_queue):
		self.sess = sess

		with tf.variable_scope("NNBody"):
			with tf.variable_scope("data_pipeline"):
				data_pairs  = data_queue.dequeue_many(BATCH_SIZE)
				inputs = data_pairs[:,0]
				desireds = data_pairs[:,1]
			
			self.output = self.create_model(inputs)
			self.optimize_op = self.create_training_method(self.output, desireds)

			with tf.variable_scope("requeue"):
				self.training_op = data_queue.enqueue(data_queue)

	def create_model(self, inputs):
		"""
		Creates the feedforward neural network definiton
		"""
		with tf.variable_scope("model"):
			output = inputs

			pass

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

			return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


