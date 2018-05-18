from  multi_user_network_env import env_network
from drqn import QNetwork,Memory
import numpy as np
import sys
import  matplotlib.pyplot as plt 
from collections import  deque
import os
import tensorflow as tf
import time

TIME_SLOTS = 100000                            # number of time-slots to run simulation
NUM_CHANNELS = 2                               # Total number of channels
NUM_USERS = 3                                  # Total number of users
ATTEMPT_PROB = 1                               # attempt probability of ALOHA based  models 

#It creates a one hot vector of a number as num with size as len
def one_hot(num,len):
    assert num >=0 and num < len ,"error"
    vec = np.zeros([len],np.int32)
    vec[num] = 1
    return vec



#generates next-state from action and observation
def state_generator(action,obs):
    input_vector = []
    if action is None:
        print ('None')
        sys.exit()
    for user_i in range(action.size):
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        channel_alloc = obs[-1]
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)
    return input_vector

memory_size = 1000                      #size of experience replay deque
batch_size = 6                          # Num of batches to train at each time_slot
pretrain_length = batch_size            #this is done to fill the deque up to batch size before training
hidden_size = 128                       #Number of hidden neurons
learning_rate = 0.0001                  #learning rate
explore_start = .02                     #initial exploration rate
explore_stop = 0.01                     #final exploration rate
decay_rate = 0.0001                     #rate of exponential decay of exploration
gamma = 0.9                             #discount  factor
noise = 0.1
step_size=1+2+2                         #length of history sequence for each datapoint  in batch
state_size = 2 *(NUM_CHANNELS + 1)      #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)
alpha=0                                 #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo

# reseting default tensorflow computational graph
tf.reset_default_graph()

#initializing the environment
env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)

#initializing deep Q network
mainQN = QNetwork(name='main',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size)

#this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
memory = Memory(max_size=memory_size)   

#this is our input buffer which will be used for  predicting next Q-values   
history_input = deque(maxlen=step_size)

#to sample random actions for each user
action  =  env.sample()

#
obs = env.step(action)
state = state_generator(action,obs)
reward = [i[1] for i in obs[:NUM_USERS]]


##############################################
for ii in range(pretrain_length*step_size*5):
    
    action = env.sample()
    obs = env.step(action)      # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
    next_state = state_generator(action,obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    memory.add((state,action,reward,next_state))
    state = next_state
    history_input.append(state)

##############################################
    
def get_states(batch): 
    states = []
    for  i in batch:
        states_per_batch = []
        for step_i in i:
            states_per_step = []
            for user_i in step_i[0]:
                states_per_step.append(user_i)
            states_per_batch.append(states_per_step)
        states.append(states_per_batch)     
   
    return states

def get_actions(batch):
    actions = []
    for each in batch:
        actions_per_batch = []
        for step_i in each:
            actions_per_step = []
            for user_i in step_i[1]:
                actions_per_step.append(user_i)
            actions_per_batch.append(actions_per_step)
        actions.append(actions_per_batch)

    return actions

def get_rewards(batch):
    rewards = []
    for each in batch:
        rewards_per_batch = []
        for step_i in each:
            rewards_per_step = []
            for user_i in step_i[2]:
                rewards_per_step.append(user_i)
            rewards_per_batch.append(rewards_per_step)
        rewards.append(rewards_per_batch)
    return rewards

def get_next_states(batch):
    next_states = []
    for each in batch:
        next_states_per_batch = []
        for step_i in each:
            next_states_per_step = []
            for user_i in step_i[3]:
                next_states_per_step.append(user_i)
            next_states_per_batch.append(next_states_per_step)
        next_states.append(next_states_per_batch)
    return next_states        

def get_states_user(batch):
    states = []
    for user in range(NUM_USERS):
        states_per_user = []
        for each in batch:
            states_per_batch = []
            for step_i in each:
                
                try:
                    states_per_step = step_i[0][user]
                    
                except IndexError:
                    print (step_i)
                    print ("-----------")
                    
                    print ("eror")
                    
                    '''for i in batch:
                        print i
                        print "**********"'''
                    sys.exit()
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    #print len(states)
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    return np.array(actions)

def get_rewards_user(batch):
    rewards = []
    for user in range(NUM_USERS):
        rewards_per_user = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = step_i[2][user] 
                rewards_per_batch.append(rewards_per_step)
            rewards_per_user.append(rewards_per_batch)
        rewards.append(rewards_per_user)
    return np.array(rewards)




# 
def get_next_states_user(batch):
    next_states = []
    for user in range(NUM_USERS):
        next_states_per_user = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = step_i[3][user] 
                next_states_per_batch.append(next_states_per_step)
            next_states_per_user.append(next_states_per_batch)
        next_states.append(next_states_per_user)
    return np.array(next_states)



interval = 1       # debug interval

# saver object to save the checkpoints of the DQN to disk
saver = tf.train.Saver()

#initializing the session
sess = tf.Session()

#initialing all the tensorflow variables
sess.run(tf.global_variables_initializer())


#list of total rewards
total_rewards = []

# cumulative reward
cum_r = [0]

# cumulative collision
cum_collision = [0]

##########################################################################
####                      main simulation loop                    ########


for time_step in range(TIME_SLOTS):
    
    # changing beta at every 50 time-slots
    if time_step %50 == 0:
        if time_step < 5000:
            beta -=0.001

    #curent exploration probability
    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*time_step)
   

    # Exploration
    if explore_p > np.random.rand():
        #random action sampling
        action  = env.sample()
        print ("explored")
        
    # Exploitation
    else:
        #initializing action vector
        action = np.zeros([NUM_USERS],dtype=np.int32)

        #converting input history into numpy array
        state_vector = np.array(history_input)

        #print np.array(history_input)
        print ("///////////////")

        for each_user in range(NUM_USERS):
            
            #feeding the input-history-sequence of (t-1) slot for each user seperately
            feed = {mainQN.inputs_:state_vector[:,each_user].reshape(1,step_size,state_size)}

            #predicting Q-values of state respectively
            Qs = sess.run(mainQN.output,feed_dict=feed) 
            #print Qs

            #   Monte-carlo sampling from Q-values  (Boltzmann distribution)
            ##################################################################################
            prob1 = (1-alpha)*np.exp(beta*Qs)

            # Normalizing probabilities of each action  with temperature (beta) 
            prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
            #print prob 

            #   This equation is as given in the paper :
            #   Deep Multi-User Reinforcement Learning for  
            #   Distributed Dynamic Spectrum Access :
            #   @Oshri Naparstek and Kobi Cohen (equation 12)
            ########################################################################################

            #  choosing action with max probability
            action[each_user] = np.argmax(prob,axis=1)

            #action[each_user] = np.argmax(Qs,axis=1)
            if time_step % interval == 0:
                print (state_vector[:,each_user])
                print (Qs)
                print (prob, np.sum(np.exp(beta*Qs)))

    # taking action as predicted from the q values and receiving the observation from thr envionment
    obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)] 
    
    print (action)
    print (obs)

    # Generate next state from action and observation 
    next_state = state_generator(action,obs)
    print (next_state)

    # reward for all users given by environment
    reward = [i[1] for i in obs[:NUM_USERS]]
    
    # calculating sum of rewards
    sum_r =  np.sum(reward)

    #calculating cumulative reward
    cum_r.append(cum_r[-1] + sum_r)

    #If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r) 
    collision = NUM_CHANNELS - sum_r
    
    #calculating cumulative collision
    cum_collision.append(cum_collision[-1] + collision)
    
   
    #############################
    #  for co-operative policy we will give reward-sum to each user who have contributed
    #  to play co-operatively and rest 0
    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r
    #############################


    total_rewards.append(sum_r)
    print (reward)
    
    
    # add new experiences into the memory buffer as (state, action , reward , next_state) for training
    memory.add((state,action,reward,next_state))
    
    
    state = next_state
    #add new experience to generate input-history sequence for next state
    history_input.append(state)


    #  Training block starts
    ###################################################################################

    #  sampling a batch from memory buffer for training
    batch = memory.sample(batch_size,step_size)
    
    #   matrix of rank 4
    #   shape [NUM_USERS,batch_size,step_size,state_size]
    states = get_states_user(batch)      
  
    #   matrix of rank 3
    #   shape [NUM_USERS,batch_size,step_size]
    actions = get_actions_user(batch)
    
    #   matrix of rank 3
    #   shape [NUM_USERS,batch_size,step_size]
    rewards = get_rewards_user(batch)
    
    #   matrix of rank 4
    #   shape [NUM_USERS,batch_size,step_size,state_size]
    next_states = get_next_states_user(batch)
    
    #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]  
    #   first two axis are converted into first axis

    states = np.reshape(states,[-1,states.shape[2],states.shape[3]])
    actions = np.reshape(actions,[-1,actions.shape[2]])
    rewards = np.reshape(rewards,[-1,rewards.shape[2]])
    next_states = np.reshape(next_states,[-1,next_states.shape[2],next_states.shape[3]])

    #  creating target vector (possible best action)
    target_Qs = sess.run(mainQN.output,feed_dict={mainQN.inputs_:next_states})


    #  Q_target =  reward + gamma * Q_next
    targets = rewards[:,-1] + gamma * np.max(target_Qs,axis=1)
  
    #  calculating loss and train using Adam  optimizer 
    loss, _ = sess.run([mainQN.loss,mainQN.opt],
                            feed_dict={mainQN.inputs_:states,
                            mainQN.targetQs_:targets,
                            mainQN.actions_:actions[:,-1]})
    

    #   Training block ends
    ########################################################################################
    
    if  time_step %5000 == 4999:
        plt.figure(1)
        plt.subplot(211)
        #plt.plot(np.arange(1000),total_rewards,"r+")
        #plt.xlabel('Time Slots')
        #plt.ylabel('total rewards')
        #plt.title('total rewards given per time_step')
        #plt.show()
        plt.plot(np.arange(5001),cum_collision,"r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')
        #plt.show()
        plt.subplot(212)
        plt.plot(np.arange(5001),cum_r,"r-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')
        #plt.title('Cumulative reward of all users')
        plt.show()
        
        total_rewards = []
        cum_r = [0]
        cum_collision = [0]
        saver.save(sess,'checkpoints/dqn_multi-user.ckpt')
        #print time_step,loss , sum(reward) , Qs
    
    print ("*************************************************")

   






