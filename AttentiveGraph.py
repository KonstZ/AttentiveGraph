
import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import initializations
from keras import regularizers

import tensorflow as tf

class AttentiveGraph(Layer):
	'''
		Graph Neural Network with Attention
		Inputs nodes(with features) and sparse connectivity graph (pairs of connected nodes indexes)
		Produces nodes output features by iteratively combining nodes features with their graph neighbours
		Features from multiple nodes are weighted by softmax attention
		# Example
			nodes = Input((MAX_NODES, NODE_INPUT_FEATURES))
			connections = Input((MAX_CONNECTIONS, 2), dtype='int32')
			ag = AttentiveGraph(GRAPH_STATES, GRAPH_ITERATIONS)([nodes, connections])
			result = TimeDistributed(Dense(NUM_CLASESS, activation='softmax'))(ag)
			model = Model([nodes, connections], result)
	
		# Params:
			nchannels - number of output (and internal) features
			num_iterations - number of iteration over graph
			bidirectional - connections are considered bidirectional
			activation - activation for output (and internal) features
			dropout	- dropout to use between iterations
			regularizer - regularizer for all weights
	'''

	def build(self, input_shape):
		objects_shape = input_shape[0]
		object_channels = objects_shape[-1]
		o2s_shape = (object_channels, self.nchannels)
		s2s_shape = (self.nchannels, self.nchannels)

		# attention weights
		self.state_attention_W = self.add_weight(s2s_shape, 
			initializer=self.inner_init,
            regularizer=self.regularizer,
			name='{}_state_attention_W'.format(self.name))
		self.linked_state_attention_W = self.add_weight(s2s_shape, 
			initializer=self.inner_init,
            regularizer=self.regularizer,
			name='{}_linked_state_attention_W'.format(self.name))
		self.attention_b = self.add_weight((self.nchannels,), 
			initializer='zero',
			name='{}_attention_b'.format(self.name))
		
		# state update weights
		self.object_state_W = self.add_weight(o2s_shape, 
			initializer=self.init,
            regularizer=self.regularizer,
			name='{}_object_state_W'.format(self.name))
		self.linked_state_W = self.add_weight(s2s_shape, 
			initializer=self.inner_init,
            regularizer=self.regularizer,
			name='{}_linked_state_W'.format(self.name))	
		self.state_b = self.add_weight((self.nchannels,), 
			initializer='zero',
			name='{}_state_b'.format(self.name))

	def call(self, x, mask=None):
		objects = x[0]	
		if self.bidirectional:
			connectivity_from = K.concatenate([x[1][::,::,0], x[1][::,::,1]])
			connectivity_to = K.concatenate([x[1][::,::,1], x[1][::,::,0]])
		else:
			connectivity_from = x[1][::,::,0]
			connectivity_to = x[1][::,::,1]

		# make pairs (batch_index, objects_index) to index batch of states with tf.scatter/gather_nd
		batch_index = tf.cumsum(K.ones_like(connectivity_from), axis=0) - 1
		connectivity_from = tf.stack([batch_index, connectivity_from], axis=2)
		connectivity_to = tf.stack([batch_index, connectivity_to], axis=2)

		states = self.activation(K.dot(objects, self.object_state_W) + self.state_b)

		def step(states):
			if self.dropout > 0:		
				states = K.in_train_phase(K.dropout(states, self.dropout), states)

			states_aW = K.dot(states, self.state_attention_W)
			linked_states_aW = K.dot(states, self.linked_state_attention_W)

			states_a = tf.gather_nd(states_aW, connectivity_from)
			linked_states_a = tf.gather_nd(linked_states_aW, connectivity_to)

			attention = K.exp(states_a + linked_states_a + self.attention_b)
			self_attention = K.ones_like(states)
			attention_norm = tf.scatter_nd(connectivity_from, attention, tf.shape(states))
			attention_norm += self_attention
			self_attention = self_attention / attention_norm
			attention /= tf.gather_nd(attention_norm, connectivity_from)

			attented_states = attention * tf.gather_nd(states, connectivity_to)
			linked_gated_states = tf.scatter_nd(connectivity_from, attented_states, tf.shape(states))

			result = self.activation(states * self_attention + K.dot(linked_gated_states, self.linked_state_W) + self.state_b)
			return result

		for i in xrange(self.num_iterations):
			states = step(states)
		
		return states

	def compute_mask(self, input, mask):
		return mask[0]

	def get_output_shape_for(self, input_shape):
		return input_shape[0][:-1] + (self.nchannels,)

	def get_config(self):
		config = {'nchannels': self.nchannels,
				'num_iterations': self.num_iterations,
				'bidirectional': self.bidirectional,
				'activation': self.activation.__name__,
				'dropout': self.dropout,
				'regularizer': self.regularizer.get_config() if self.regularizer else None}
		base_config = super(AttentiveGraph, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def __init__(self, nchannels, num_iterations, bidirectional = True,
			activation='tanh', dropout = 0,	regularizer=None, **kwargs):
		self.nchannels = nchannels
		self.num_iterations = num_iterations
		self.bidirectional = bidirectional
		self.init = initializations.get('glorot_normal')
		self.inner_init = initializations.get('orthogonal')
		self.activation = activations.get(activation)
		self.dropout = dropout
		if self.dropout:
			self.uses_learning_phase = True
		self.regularizer = regularizers.get(regularizer)
		super(AttentiveGraph, self).__init__(**kwargs)


