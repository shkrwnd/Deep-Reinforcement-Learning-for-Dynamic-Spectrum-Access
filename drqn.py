import tensorflow as tf
import numpy as np
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, step_size=1 ,
                 name='QNetwork'):
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None,step_size, state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            ##########################################
         
            self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm,self.inputs_,dtype=tf.float32)
            
            self.reduced_out = self.lstm_out[:,-1,:]
            self.reduced_out = tf.reshape(self.reduced_out,shape=[-1,hidden_size])

            #########################################
            
            #self.w1 = tf.Variable(tf.random_uniform([state_size,hidden_size]))
            #self.b1 = tf.Variable(tf.constant(0.1,shape=[hidden_size]))
            #self.h1 = tf.matmul(self.inputs_,self.w1) + self.b1
            #self.h1 = tf.nn.relu(self.h1)
            #self.h1 = tf.contrib.layers.layer_norm(self.h1)
            #'''

            self.w2 = tf.Variable(tf.random_uniform([hidden_size,hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1,shape=[hidden_size]))
            self.h2 = tf.matmul(self.reduced_out,self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)

            self.w3 = tf.Variable(tf.random_uniform([hidden_size,action_size]))
            self.b3 = tf.Variable(tf.constant(0.1,shape=[action_size]))
            self.output = tf.matmul(self.h2,self.w3) + self.b3


            #self.output = tf.contrib.layers.layer_norm(self.output)
           

            '''
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

           
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,activation_fn=None)
            
            '''
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)





from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size,step_size):
        idx = np.random.choice(np.arange(len(self.buffer)-step_size), 
                               size=batch_size, replace=False)
        
        res = []                       
                             
        for i in idx:
            temp_buffer = []  
            for j in range(step_size):
                temp_buffer.append(self.buffer[i+j])
            res.append(temp_buffer)
        return res    
        

