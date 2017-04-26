# AttentiveGraphGraph 
Neural Network with Attention
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
