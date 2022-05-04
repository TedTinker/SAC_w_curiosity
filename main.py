import gym
import time
import torch
import numpy as np
import keyboard
from collections import deque

from utils import args, list_mean, plot_losses
from model import Agent



def SAC(n_episodes=200, max_t=500, print_every=10):
    global env
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    auto_losses, trans_losses, alpha_losses, actor_losses, critic1_losses, critic2_losses = \
        [], [], [], [], [], []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state.reshape((1,state_size))
        score = 0
        auto_losses_, trans_losses_, alpha_losses_, actor_losses_, critic1_losses_, critic2_losses_ = \
            [], [], [], [], [], []
        for t in range(max_t):
            if(keyboard.is_pressed('q')): env.render()
            action = agent.act(state)
            action_v = action.numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))
            auto_loss, trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss = \
                agent.step(state, action, reward, next_state, done, t)
            auto_losses_.append(auto_loss)
            trans_losses_.append(trans_loss)
            alpha_losses_.append(alpha_loss)
            actor_losses_.append(actor_loss)
            critic1_losses_.append(critic1_loss)
            critic2_losses_.append(critic2_loss)
            state = next_state
            score += reward

            if done:
                auto_losses_ = list_mean(auto_losses_)
                trans_losses_ = list_mean(trans_losses_)
                alpha_losses_ = list_mean(alpha_losses_)
                actor_losses_ = list_mean(actor_losses_)
                critic1_losses_ = list_mean(critic1_losses_)
                critic2_losses_ = list_mean(critic2_losses_)

                auto_losses.append(auto_losses_)
                trans_losses.append(trans_losses_)
                alpha_losses.append(alpha_losses_)
                actor_losses.append(actor_losses_)
                critic1_losses.append(critic1_losses_)
                critic2_losses.append(critic2_losses_)
                plot_losses(auto_losses, trans_losses, alpha_losses, actor_losses, critic1_losses, critic2_losses)
                break 
        
        env.close()
        env = gym.make(env_name)
        scores_deque.append(score)
        average_100_scores.append(np.mean(scores_deque))
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
                
    




if __name__ == "__main__":
    env_name = args.env
    n_episodes = args.ep
    HIDDEN_SIZE = args.layer_size
    FIXED_ALPHA = args.alpha
    saved_model = args.saved_model
    
    t0 = time.time()
    env = gym.make(env_name)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size,hidden_size=HIDDEN_SIZE, action_prior="uniform") #"normal"
    
    SAC(n_episodes=args.ep, max_t=500, print_every=args.print_every)
    t1 = time.time()
    env.close()
    print("training took {} min!".format((t1-t0)/60))