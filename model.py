import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
from torchinfo import summary as torch_summary

from utils import args
from buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Autoencoder(nn.Module):
    
    def __init__(
            self, 
            state_size,
            hidden_size=32,
            encode_size=32):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, encode_size),
            nn.LeakyReLU())
        
        self.decode = nn.Sequential(
            nn.Linear(encode_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, state_size))
        
    def forward(self, x):
        encoding = self.encode(x)
        decoding = self.decode(encoding)
        return(encoding, decoding)
        
        
        
        
class Transitioner(nn.Module):
    
    def __init__(
            self, 
            state_size,
            action_size, 
            hidden_size=32):
        super(Transitioner, self).__init__()
        
        self.lin = nn.Sequential(
            nn.Linear(state_size+action_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, state_size))
    
        
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        next_state = self.lin(x)
        return(next_state)




class Actor(nn.Module):
    
    def __init__(
            self, 
            state_size, 
            action_size, 
            hidden_size=32, 
            log_std_min=-20, 
            log_std_max=2):
        
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU())
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.lin(state)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob
        
    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]



class Critic(nn.Module):

    def __init__(
            self, 
            state_size, 
            action_size, 
            hidden_size=32):
        
        super(Critic, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(state_size+action_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.lin(x)
        return x
    
    
    

if __name__ == "__main__":
    
    autoencoder = Autoencoder(2)
    print("\n\n")
    print(autoencoder)
    print()
    print(torch_summary(autoencoder, (1, 2)))
    
    transitioner = Transitioner(2,1)
    print("\n\n")
    print(transitioner)
    print()
    print(torch_summary(transitioner, ((1, 2),(1,1))))

    actor = Actor(2, 1)
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, (1, 2)))
    
    critic = Critic(2, 1)
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1, 2),(1,1))))
    
    



class Agent():
    
    def __init__(
            self, 
            state_size, 
            action_size, 
            hidden_size, 
            encode_size,
            action_prior="uniform"):

        self.state_size = state_size
        self.action_size = action_size
        
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr) 
        self._action_prior = action_prior
        
        self.autoencoder = Autoencoder(state_size, hidden_size, encode_size)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=args.lr)     
           
        self.transitioner = Transitioner(state_size, action_size, hidden_size)
        self.trans_optimizer = optim.Adam(self.transitioner.parameters(), lr=args.lr)     
        
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=args.lr)     
        
        self.critic1 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr, weight_decay=0)
        self.critic1_target = Critic(state_size, action_size,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr, weight_decay=0) 
        self.critic2_target = Critic(state_size, action_size,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.memory = ReplayBuffer(action_size, int(args.memory), args.batch_size)
        
    def step(self, state, action, reward, next_state, done, step):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > args.batch_size:
            experiences = self.memory.sample()
            auto_loss, trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss = \
                self.learn(step, experiences, args.gamma)
            return(auto_loss, trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss)
        return(None, None, None, None, None, None)
            
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

    def learn(self, step, experiences, gamma, d=2):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Train autoencoder
        _, decoded = self.autoencoder(states)
        auto_loss = F.mse_loss(decoded, states)
        self.autoencoder_optimizer.zero_grad()
        auto_loss.backward()
        self.autoencoder_optimizer.step()
        
        # Train transitioner
        pred_next_states = self.transitioner(states, actions)
        trans_loss = F.mse_loss(pred_next_states, next_states)
        self.trans_optimizer.zero_grad()
        trans_loss.backward()
        self.trans_optimizer.step()
        
        # Train critics
        next_action, log_pis_next = self.actor_local.evaluate(next_states)
        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if args.alpha == None:
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - args.alpha * log_pis_next.squeeze(0).cpu()))
        
        Q_1 = self.critic1(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(states, actions).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Train actor
        if step % d == 0:
            if args.alpha == None:
                self.alpha = torch.exp(self.log_alpha)
                actions_pred, log_pis = self.actor_local.evaluate(states)
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
                    self.critic1(states, actions_pred.squeeze(0)), 
                    self.critic2(states, actions_pred.squeeze(0)))
                actor_loss = (self.alpha * log_pis.squeeze(0).cpu() - Q.cpu() - policy_prior_log_probs).mean()
            
            else:
                alpha_loss = None
                actions_pred, log_pis = self.actor_local.evaluate(states)
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
        
        if(auto_loss != None): auto_loss = auto_loss.item()
        if(trans_loss != None): trans_loss = trans_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()

        return(
            auto_loss, 
            trans_loss, 
            alpha_loss, 
            actor_loss, 
            critic1_loss, 
            critic2_loss)
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

