import tensorflow as tf 
from reader import Reader
from numpy import  newaxis
from keras.layers import LSTM, Input, Bidirectional
import numpy as np
from keras.layers import GRU
dim = 128
nh = 1
reader_to_predict = Reader('../data/glove.6B.100d.txt')
p = "Assistive technology is an umbrella term that includes assistive, adaptive, and rehabilitative devices for people with disabilities while also including the process used in selecting, locating, and using them. People who have disabilities often have difficulty performing activities of daily living (ADLs) independently, or even with assistance. ADLs are self-care activities that include toileting, mobility (ambulation), eating, bathing, dressing and grooming. Assistive technology can ameliorate the effects of disabilities that limit the ability to perform ADLs"
q = "What is assistive technology"
p_encodes,q_encodes = reader_to_predict.encode(p,q)
p_len = len(p)
q_len = len(q)
p_mask = tf.sequence_mask(len(p_encodes), tf.shape(p_encodes)[1], dtype=tf.float32, name='passage_mask')
q_mask = tf.sequence_mask(len(q_encodes), tf.shape(q_encodes)[1], dtype=tf.float32, name='question_mask')

sim_matrix = tf.matmul(p_encodes, q_encodes, transpose_b=True)

passage2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), q_encodes)

p_mask = tf.expand_dims(p_mask, -1)
passage2question_attn =tf.matmul(passage2question_attn, p_mask)
#question2passage_attn *= p_mask

attention_p_encoded = tf.concat([p_encodes,
                    p_encodes * passage2question_attn,sim_matrix], -1)

fuse_out = attention_p_encoded
start_logit = tf.layers.dense(fuse_out, 1)
end_logit = tf.layers.dense(fuse_out, 1)

max_answ_len = 50

start_prob = tf.nn.softmax(start_logit, axis=1)
end_prob = tf.nn.softmax(end_logit, axis=1)

outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                  tf.expand_dims(end_prob, axis=1))

start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for v in sess.run([start_pos,end_pos]):
            print(v)