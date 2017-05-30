import time
import tensorflow as tf
import numpy as np

from random import shuffle

from load_data import load_all_data, normalize_sent, capitalization

class LSTMTagger:
    def __init__(self, session, in_size, out_size, lstm_size, word_embeddings, starter_learning_rate):
        # Basic parameters
        self.session = session
        self.in_size = in_size
        self.out_size = out_size
        self.lstm_size = lstm_size


        # Embeddings
        self.cap_embeddings = tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32)
        self.word_embeddings = tf.Variable(word_embeddings, dtype=tf.float32)


        # Input
        self.input_tokens = tf.placeholder(tf.int32, shape=(None,None)) # Shape: batch_size x sequence_length
        self.input_caps = tf.placeholder(tf.int32, shape=(None,None))
        self.input_token_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.input_tokens)
        self.input_cap_embeddings = tf.nn.embedding_lookup(self.cap_embeddings, self.input_caps)

        self.cap_weights = tf.Variable(tf.random_uniform((3,self.lstm_size), minval=-0.1, maxval=0.1), dtype=tf.float32)
        self.flat_tokens_with_caps = tf.reshape(self.input_token_embeddings, (-1,self.lstm_size)) + tf.matmul(tf.reshape(self.input_cap_embeddings, (-1,3)), self.cap_weights)
        self.lstm_input = tf.reshape(self.flat_tokens_with_caps, tf.shape(self.input_token_embeddings))

    
        # Bidirectional RNN (LSTM) layer
        self.forward_cell  = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.lstm_size), state_keep_prob=0.75, input_keep_prob=0.75)
        self.backward_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.lstm_size), state_keep_prob=0.75, input_keep_prob=0.75)
        (self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.forward_cell, self.backward_cell, self.lstm_input, dtype=tf.float32)
        self.flat_output_fw = tf.reshape(self.output_fw, (-1, self.lstm_size))
        self.flat_output_bw = tf.reshape(self.output_bw, (-1, self.lstm_size))


        # Output layer
        self.out_fw_weight = tf.Variable(tf.random_uniform((self.lstm_size,self.out_size), minval=-0.1, maxval=0.1), dtype=tf.float32)
        self.out_bw_weight = tf.Variable(tf.random_uniform((self.lstm_size,self.out_size), minval=-0.1, maxval=0.1), dtype=tf.float32)
        self.out_bias = tf.Variable(tf.random_uniform((self.out_size,), minval=-0.1, maxval=0.1), dtype=tf.float32)

        self.flat_output = tf.matmul(self.flat_output_fw, self.out_fw_weight) + tf.matmul(self.flat_output_bw, self.out_bw_weight) + self.out_bias
        self.structured_output  = tf.reshape(self.flat_output, (tf.shape(self.input_tokens)[0], tf.shape(self.input_tokens)[1], self.out_size))


        # Gold output
        self.targets = tf.placeholder(tf.int32, shape=(None,None)) # Shape: batch_size x sequence_length
        self.flat_targets = tf.reshape(self.targets, (-1,))


        # Optimization & training
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10802, 0.95, staircase=True)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.flat_output, labels=self.flat_targets))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)



    def train_batch(self, inp_toks, inp_caps, gold_tags):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.input_tokens: inp_toks, self.input_caps: inp_caps, self.targets: gold_tags})
        return loss



    def tag_sequence(self, token_sequence):
        '''Assign POS tags to a token sequence based on trained model'''
        tokens_as_numbers = [token_to_ix[token] if token in token_to_ix else token_to_ix["UNK"] for token in normalize_sent(token_sequence)]
        caps = list(map(capitalization, token_sequence))
        predicted_tags = sess.run(self.structured_output, feed_dict={self.input_tokens: [tokens_as_numbers], self.input_caps: [caps]})
        return [ix_to_tag[np.argmax(tag_vec)] for tag_vec in predicted_tags[0]]



def train_epochs(num_epochs):
    for epoch in range(num_epochs):
        starttime = time.time()
        print("Epoch", epoch+1, "started")
        shuffle(sents)
        sum_loss = 0
        for (tokens, caps, tags) in sents:
            sum_loss += net.train_batch([tokens], [caps], [tags])
        print("Epoch", epoch+1, "finished. ({:.1f}s)".format(time.time() - starttime))
        print("Loss:", sum_loss)

        if sum_loss is np.nan:
            break
        else:
            saver.save(sess, "./model/tagging-model")
            
        if (epoch+1) % 5 == 0:
            print("Accuracy on test set:", compute_accuracy("data/devel_tokens.txt", "data/devel_tags.txt"))



def compute_accuracy(token_file, tag_file):
    num_tags = 0
    num_correct = 0
    with open(token_file) as f:
        with open(tag_file) as g:
            for line in f:
                tokens = line.strip().split("\t")
                gold_tags = next(g).strip().split("\t")
                predicted_tags = net.tag_sequence(tokens)
                assert len(tokens) == len(gold_tags) == len(predicted_tags)
                num_tags += len(gold_tags)
                for (predicted_tag, gold_tag) in zip(predicted_tags, gold_tags):
                    if predicted_tag == gold_tag:
                        num_correct += 1

    return num_correct / num_tags



# Parameters
embedding_size = 100
starter_learning_rate = 0.003

# Load data
sents, token_to_ix, ix_to_token, tag_to_ix, ix_to_tag, word_embeddings \
= load_all_data("data/train_tokens.txt", "data/train_tags.txt", "data/tagset.txt", "embeddings/glove.6B.100d.txt", embedding_size)

# Set up network
sess = tf.InteractiveSession()
net = LSTMTagger(sess, len(token_to_ix), len(tag_to_ix), embedding_size, word_embeddings, starter_learning_rate)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Optional: Load model
# saver.restore(sess, "./model/tagging-model")

# Learn
train_epochs(20)
