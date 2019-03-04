import tensorflow as tf 
from openqas.reader.reader import Reader
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
p_mask = tf.sequence_mask(tf.shape(p_encodes)[0], tf.shape(p_encodes)[0], dtype=tf.float32, name='passage_mask')
q_mask = tf.sequence_mask(tf.shape(q_encodes)[0], tf.shape(q_encodes)[0], dtype=tf.float32, name='question_mask')
sim_matrix = tf.matmul(p_encodes, q_encodes, transpose_b=True)
sim_mask = tf.matmul(tf.expand_dims(p_mask, -1), tf.expand_dims(q_mask, -1), transpose_b=True)
sim_matrix = tf.subtract(sim_matrix,(1 - sim_mask) * 1e30)

passage2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), q_encodes)
p_mask = tf.expand_dims(p_mask, -1)
passage2question_attn =tf.matmul(tf.transpose(passage2question_attn), p_mask)

question2passage_attn = tf.matmul(tf.transpose(tf.nn.softmax(sim_matrix, -1)), p_encodes)
q_mask = tf.expand_dims(q_mask, -1)
question2passage_attn =tf.matmul(tf.transpose(question2passage_attn), q_mask)


attention_p_encoded = tf.concat([p_encodes,
                    tf.matmul(p_encodes , passage2question_attn ),tf.matmul(p_encodes,question2passage_attn)], -1)


start_logit = tf.layers.dense(attention_p_encoded, 1)
end_logit = tf.layers.dense(attention_p_encoded, 1)

start_logit -= (1 - p_mask) * 1e30
end_logit -= (1 - p_mask) * 1e30

max_answ_len = 50

start_prob = tf.nn.softmax(start_logit, axis=1)
end_prob = tf.nn.softmax(end_logit, axis=1)

outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                  tf.expand_dims(end_prob, axis=1))

start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(start_pos))
        print(sess.run(end_pos))


''' Training the model '''
'''
model = Model(input=[context_input, question_input], output=[answerPtrBegin_output, answerPtrEnd_output])
rms = optimizers.RMSprop(lr=0.0005)
model.compile(optimizer=rms, loss='categorical_crossentropy',
              loss_weights=[.04, 0.04], metrics=['accuracy'])
model.summary()
'''