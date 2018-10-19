import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

class LSTM():  
    def __init__(self, num_layers, hidden_units, time_steps, batch_size, vocabulary, rosetta):
        self.d_out = tf.placeholder_with_default(0.5, [])
        self.x = tf.placeholder(tf.int32, [batch_size, time_steps])
        self.y = tf.placeholder(tf.int32, [batch_size, time_steps])
        
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.v_size = len(vocabulary)
        self.rosetta = rosetta
        with tf.variable_scope('ms', reuse= tf.AUTO_REUSE):
            
            lstm_layers = [tf.contrib.rnn.BasicLSTMCell(self.hidden_units)] * self.num_layers
            dropout_lstm = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob= self.d_out) for lstm in lstm_layers]
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(dropout_lstm)
            
            init_state = stacked_lstm.zero_state(self.batch_size, tf.float32)
            init = tf.contrib.layers.xavier_initializer()
            
            emb = tf.get_variable('emb', shape= [self.v_size, self.hidden_units], initializer= init)
            embx = tf.nn.embedding_lookup(emb, self.x)
            
            output, state = tf.nn.dynamic_rnn(stacked_lstm, embx, initial_state= init_state)
            output = tf.reshape(output, [-1, self.hidden_units])
            
            fc_one = tf.layers.dense(output, self.v_size, kernel_initializer= init)
            self.resh_fc = tf.reshape(fc_one, [self.batch_size, self.time_steps, self.v_size])
            
            resh_logits = tf.cast(tf.argmax(fc_one, 1), tf.int32)
            num_correct = tf.equal(resh_logits, tf.reshape(self.y, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32))
            
            loss = tf.contrib.seq2seq.sequence_loss(self.resh_fc, self.y, tf.ones([self.batch_size, self.time_steps]), average_across_timesteps= False)
            l_two_reg = tf.contrib.layers.l2_regularizer(0.005)
            t_vars = tf.trainable_variables()
            reg_penalty = tf.contrib.layers.apply_regularization(l_two_reg, weights_list= t_vars)
            reg_loss = reg_penalty + loss
            
            self.cost = tf.reduce_sum(reg_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate= 0.01)
            
            t_vars_grads = tf.gradients(self.cost, t_vars)
            grads, _ = tf.clip_by_global_norm(t_vars_grads, 15)
            self.train = optimizer.apply_gradients(zip(grads, t_vars))
                        
            sess.run(tf.global_variables_initializer())
    
    def batch(self, data, batch_iter, batches_in_epoch):
        batchx = []
        batchy = []
        for bb in range (self.batch_size):
            batch_marker = (batch_iter * self.time_steps * self.batch_size) + (bb * self.time_steps)
            batchx.append(data[batch_marker: batch_marker + self.time_steps])
            batchy.append(data[batch_marker + 1: batch_marker + self.time_steps + 1])
        
        return batchx, batchy

    def nums_words(self, nums):
        word_out = []
        for term in range (len(nums)):
            index = nums[term]
            word_out.append(self.vocabulary[index])
        
        return word_out

    def sample(self, gen_words):
        num_seed = list(np.random.randint(0, len(self.vocabulary) - 1, [self.time_steps]))
        num_words = [num_seed]
        lgs = sess.run(self.resh_fc, feed_dict= {self.d_out: 1.0, self.x: num_words})
        lgs_wi = np.argmax(lgs[0][-1])
        num_words[0].append(lgs_wi)
        for word in range(gen_words):
            data = [num_words[0][(-1 * self.time_steps):]]
            lgs = sess.run(self.resh_fc, feed_dict= {self.d_out: 1.0, self.x: data})
            lgs_wi = np.argmax(lgs[0][-1])
            num_words[0].append(lgs_wi)
        
        return self.nums_words(num_words[0])
        
batch_size = 16
time_steps = 32
hidden_units = 256
num_layers = 2
epochs = 128
batches_in_epoch = len(data) // batch_size

lstm = LSTM(num_layers, hidden_units, time_steps, batch_size, vocabulary, rosetta)
sample_lstm = LSTM(num_layers, hidden_units, time_steps, 1, vocabulary, rosetta)

saver = tf.train.Saver()
samples = []

for epoch_iter in range (epochs):
    for batch_iter in range (batches_in_epoch - time_steps):
        batch_x, batch_y = lstm.batch(data, batch_size, batch_iter)
        sess.run(lstm.train, feed_dict= {lstm.d_out: 0.5, lstm.x: batch_x, lstm.y: batch_y})
        if batch_iter % 100 == 0:
            loss = sess.run(lstm.cost, feed_dict= {lstm.d_out: 1.0, lstm.x: batch_x, lstm.y: batch_y})
            accuracy = sess.run(lstm.accuracy, feed_dict= {lstm.d_out: 1.0, lstm.x: batch_x, lstm.y: batch_y})
            print('Epoch: %s, Batch %s / %s, Perplexity: %s, Accuracy: %s' %(epoch_iter, batch_iter, batches_in_epoch, loss, accuracy))
        if batch_iter % 10000 == 0:
            saver.save(sess, '/Users/Masterchief7269/Desktop/Programming Header/Python/Models/PTB_LSTM/ptblstm')
            sample = sample_lstm.sample(100)
            samples.append(''.join(sample))
