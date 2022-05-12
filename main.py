#%%
import gym
import time
import numpy as np
import keyboard
from collections import deque

from utils import args, list_mean, plot_losses
from model import Agent

def SAC(n_episodes=200, max_t=500, print_every=10):
    global env
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    trans_losses, alpha_losses, actor_losses, critic1_losses, critic2_losses = \
        [], [], [], [], []

    for i_episode in range(1, n_episodes+1):
        obs = env.reset()
        obs = obs.reshape((1,obs_size))
        score = 0
        hidden = None
        trans_losses_, alpha_losses_, actor_losses_, critic1_losses_, critic2_losses_ = \
            [], [], [], [], []
        for t in range(max_t):
            #if(keyboard.is_pressed('q')): env.render()
            action, hidden = agent.act(obs, hidden)
            action_v = action.numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_obs, reward, done, info = env.step(action_v)
            next_obs = next_obs.reshape((1,obs_size))
            trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss = \
                agent.step(obs, action, reward, next_obs, done, t)
            trans_losses_.append(trans_loss)
            alpha_losses_.append(alpha_loss)
            actor_losses_.append(actor_loss)
            critic1_losses_.append(critic1_loss)
            critic2_losses_.append(critic2_loss)
            obs = next_obs
            score += reward

            if done:
                trans_losses_ = list_mean(trans_losses_)
                alpha_losses_ = list_mean(alpha_losses_)
                actor_losses_ = list_mean(actor_losses_) 
                critic1_losses_ = list_mean(critic1_losses_)
                critic2_losses_ = list_mean(critic2_losses_)

                trans_losses.append(trans_losses_)
                alpha_losses.append(alpha_losses_)
                actor_losses.append(actor_losses_)
                critic1_losses.append(critic1_losses_)
                critic2_losses.append(critic2_losses_)
                scores_deque.append(score)
                plot_losses(scores_deque, trans_losses, alpha_losses, actor_losses, critic1_losses, critic2_losses)
                break 
        
        env.close()
        env = gym.make(args.env)
        average_100_scores.append(np.mean(scores_deque))
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
                
    




if __name__ == "__main__":
    t0 = time.time()
    env = gym.make(args.env)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(
        obs_size=obs_size, 
        action_size=action_size,
        hidden_size=args.hidden_size,
        encode_size=args.encode_size,
        action_prior="uniform") #"normal"
    
    SAC(n_episodes=args.episodes, max_t=500, print_every=args.print_every)
    t1 = time.time()
    env.close()
    print("training took {} min!".format((t1-t0)/60))
# %%
