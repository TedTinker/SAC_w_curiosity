import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
from torchinfo import summary as torch_summary

import scipy.stats 
sf = scipy.stats.norm.sf

from utils import args, device
from buffer import RecurrentReplayBuffer




class Transitioner(nn.Module):
    
    def __init__(
            self, 
            state_size,
            action_size,
            hidden_size=32,
            encode_size=32):
        super(Transitioner, self).__init__()
        
        self.encode_1 = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.LeakyReLU())
        
        self.lstm = nn.LSTM(
            input_size = 32, 
            hidden_size = 32,
            batch_first=True)
        
        self.encode_2 = nn.Sequential(
            nn.Linear(32, encode_size),
            nn.LeakyReLU())
        
        self.decode = nn.Sequential(
            nn.Linear(encode_size+action_size, hidden_size),
            nn.LeakyReLU())
        
        self.mu = nn.Linear(hidden_size, state_size)
        self.log_std_linear = nn.Linear(hidden_size, state_size)
        self.to(device)
        
    def just_encode(self, x, hidden = None):
        if(len(x.shape) == 2):  sequence = False
        else:                   sequence = True
        x = self.encode_1(x)
        if(not sequence):
            x = x.view(x.shape[0], 1, x.shape[1])
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(x)
        else:               x, hidden = self.lstm(x, (hidden[0], hidden[1]))
        if(not sequence):
            x = x.view(x.shape[0], x.shape[-1])
        encoding = self.encode_2(x)
        return(encoding, hidden)
        
    def forward(self, state, action, hidden = None):
        encoding, hidden = self.just_encode(state, hidden)
        x = torch.cat((encoding, action), dim=-1)
        decoding = self.decode(x)
        mu = self.mu(decoding)
        log_std = self.log_std_linear(decoding)
        return(mu, log_std, hidden)
    
    def get_next_state(self, state, action, hidden = None):
        mu, log_std, hidden = self.forward(state, action, hidden)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        next_state = torch.tanh(mu + e * std).cpu()
        return(next_state, hidden)
        
    def DKL(self, state, next_state, action, hidden = None):
        predictions, hidden = self.get_next_state(state, action, hidden)
        divergence = nn.KLDivLoss(reduction="none")(predictions, next_state.cpu())
        divergence = divergence[:,:,0] + divergence[:,:,1] + divergence[:,:,2]
        return(divergence.unsqueeze(-1))





class Actor(nn.Module):
    
    def __init__(
            self, 
            encode_size, 
            action_size, 
            hidden_size=32, 
            log_std_min=-20, 
            log_std_max=2):
        
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(encode_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU())
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.to(device)

    def forward(self, encode):
        x = self.lin(encode)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, encode, epsilon=1e-6):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob
        
    def get_action(self, encode):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]



class Critic(nn.Module):

    def __init__(
            self, 
            encode_size, 
            action_size, 
            hidden_size=32):
        
        super(Critic, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(encode_size+action_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))
        
        self.to(device)

    def forward(self, encode, action):
        x = torch.cat((encode, action), dim=-1)
        x = self.lin(x)
        return x
    
    
    


    
    



class Agent():
    
    def __init__(
            self, 
            state_size, 
            action_size, 
            hidden_size, 
            encode_size,
            action_prior="uniform"):
        
        self.steps = 0
        
        self.encode_size = encode_size
        self.state_size = state_size 
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.state_size = state_size
        self.action_size = action_size
        
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr) 
        self._action_prior = action_prior
        
        self.transitioner = Transitioner(state_size, action_size, hidden_size, encode_size)
        self.trans_optimizer = optim.Adam(self.transitioner.parameters(), lr=args.lr)     
                   
        self.actor = Actor(encode_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)     
        
        self.critic1 = Critic(encode_size, action_size, hidden_size).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr, weight_decay=0)
        self.critic1_target = Critic(encode_size, action_size,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(encode_size, action_size, hidden_size).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr, weight_decay=0) 
        self.critic2_target = Critic(encode_size, action_size,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.memory = RecurrentReplayBuffer(state_size, action_size, max_episode_len = 10000)
        
        describe_agent(self)
        
    def step(self, state, action, reward, next_state, done, step):
        self.memory.push(state, action, reward, next_state, done, done)
        if self.memory.num_episodes > args.batch_size:
            trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss = \
                self.learn()
            return(trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss)
        return(None, None, None, None, None)
            
    def act(self, state, hidden = None):
        state = torch.from_numpy(state).float().to(device)
        encoded, hidden = self.transitioner.just_encode(state, hidden)
        action = self.actor.get_action(encoded).detach()
        return action, hidden

    def learn(self):
        
        try:
            experiences = self.memory.sample()
        except:
            return(None, None, None, None, None)
        
        self.steps += 1

        states, actions, rewards, dones, _ = experiences
        
        # Train transitioner
        pred_next_states, _ = self.transitioner.get_next_state(states[:,:-1], actions)
        trans_loss = F.mse_loss(pred_next_states.to(device), states[:,1:])
        self.trans_optimizer.zero_grad()
        trans_loss.backward()
        self.trans_optimizer.step()
        
        encoded, _ = self.transitioner.just_encode(states[:,:-1])
        encoded = encoded.detach()
        next_encoded, _ = self.transitioner.just_encode(states[:,1:])
        next_encoded = next_encoded.detach()
        
        # Update rewards with curiosity
        curiosity = args.eta * self.transitioner.DKL(states[:,:-1], states[:,1:], actions)
        rewards += curiosity.to(device)
        
        # Train critics
        next_action, log_pis_next = self.actor.evaluate(next_encoded)
        Q_target1_next = self.critic1_target(next_encoded.to(device), next_action.to(device))
        Q_target2_next = self.critic2_target(next_encoded.to(device), next_action.to(device))
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if args.alpha == None:
            Q_targets = rewards.cpu() + (args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - args.alpha * log_pis_next.squeeze(0).cpu()))
        
        Q_1 = self.critic1(encoded, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(encoded, actions).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Train actor
        if self.steps % args.d == 0:
            if args.alpha == None:
                self.alpha = torch.exp(self.log_alpha)
                actions_pred, log_pis = self.actor.evaluate(encoded)
                alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                Q = torch.min(
                    self.critic1(encoded, actions_pred), 
                    self.critic2(encoded, actions_pred))
                actor_loss = (self.alpha * log_pis.squeeze(0).cpu() - Q.cpu() - policy_prior_log_probs).mean()
            
            else:
                alpha_loss = None
                actions_pred, log_pis = self.actor.evaluate(encoded)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                Q = torch.min(
                    self.critic1(states, actions_pred.squeeze(0)), 
                    self.critic2(states, actions_pred.squeeze(0)))
                actor_loss = (args.alpha * log_pis.squeeze(0).cpu() - Q.cpu()- policy_prior_log_probs ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic1, self.critic1_target, args.tau)
            self.soft_update(self.critic2, self.critic2_target, args.tau)
            
        else:
            alpha_loss = None
            actor_loss = None
        
        if(trans_loss != None): trans_loss = trans_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()

        return(
            trans_loss, 
            alpha_loss, 
            actor_loss, 
            critic1_loss, 
            critic2_loss)
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




def describe_agent(agent):
    print("\n\n")
    print(agent.transitioner)
    print()
    print(torch_summary(agent.transitioner, ((1, agent.state_size),(1,agent.action_size))))
    
    print("\n\n")
    print(agent.actor)
    print()
    print(torch_summary(agent.actor, (1, agent.encode_size)))
    
    print("\n\n")
    print(agent.critic1)
    print()
    print(torch_summary(agent.critic1, ((1, agent.encode_size),(1,agent.action_size))))
    
if __name__ == "__main__":
    agent = Agent(
        state_size = 2,
        action_size = 1, 
        hidden_size = args.hidden_size, 
        encode_size = args.encode_size)