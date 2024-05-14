import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt


class DQLAgent:
    def __init__(self, env):
        # parameter / hyperparameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    
    # initialize gym env and agent
    env = gym.make("CartPole-v0",render_mode = "human")
    agent = DQLAgent(env)
    
    batch_size = 32
    episodes = 200
    rewards_per_episode = []
    
    
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        
        #state = np.reshape(state,[1,4])
        state = np.reshape(state[0],[1,4])
        
        time = 0
        while True:
            env.render()
            # act
            action = agent.act(state) # select an action
            
            # step
            next_state, reward, done, _ ,__= env.step(action)
            #next_state = np.reshape(next_state,[1,4])
            next_state = np.reshape(state[0],[1,4])
            
            # remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay
            agent.replay(batch_size)
            
            # adjust epsilon
            #agent.adaptiveEGreedy()
            
            time += 1
            
            if done:
                #agent.targetModelUpdate()
                print("Episode: {}, time: {}".format(e,time))
                break
        
        
        rewards_per_episode.append(time)
        agent.adaptiveEGreedy()            
        print('Episode: {}, Reward: {}'.format(e,time))
        if((e%20) == 0):
            agent.model.save("C:/Users/Monster/Desktop/CartPole/deneme1.h5")
            print("V_Model kaydedildi.")       
    agent.model.save("C:/Users/Monster/Desktop/CartPole/denem1.h5")
    print("Model kaydedildi.")    

    env.close()
    # Eğitim tamamlandıktan sonra ödül grafiğini çizin
    plt.plot(rewards_per_episode)
    plt.title('200 Episode Times')
    plt.xlabel('Episode')
    plt.ylabel('Total Times')
    plt.savefig("C:/Users/Monster/Desktop/CartPole/V_2000_times_plot.png")
    plt.show()               
            
            
            
            
            
            
            
            
            
            
