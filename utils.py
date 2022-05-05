import argparse

env_name = "Pendulum-v1"
#env_name = "MountainCarContinuous-v0"

parser = argparse.ArgumentParser(description="")
parser.add_argument("-env",         type=str,   default=env_name, 
                    help="Environment name")
parser.add_argument("-episodes",    type=int,   default=300, 
                    help="The amount of training episodes, default is 100")
parser.add_argument("-lr",          type=float, default=5e-4, 
                    help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-alpha",       type=float, 
                    help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-hidden_size", type=int,   default=256, 
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-encode_size", type=int,   default=4, 
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-memory",      type=int,   default=int(1e6), 
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-batch_size",  type=int,   default=4, 
                    help="Batch size, default is 256")
parser.add_argument("-tau",         type=float, default=1e-2, 
                    help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-gamma",       type=float, default=0.99, 
                    help="discount factor gamma, default is 0.99")
parser.add_argument("--print_every", type=int, default=100, 
                    help="Prints every x episodes the average reward over x episodes")
args = parser.parse_args()



import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # plt crashes without this

def list_mean(l):
    l = [l_ for l_ in l if l_ != None]
    try:
        return(sum(l)/len(l))
    except:
        return(None)
    
def get_x_y(losses):
    x = [i for i in range(len(losses)) if losses[i] != None]
    y = [l for l in losses if l != None]
    return(x, y)

    
def plot_losses(
        trans_losses, 
        alpha_losses, 
        actor_losses, 
        critic1_losses, 
        critic2_losses):
    
    trans_x, trans_y = get_x_y(trans_losses)
    alpha_x, alpha_y = get_x_y(alpha_losses)
    actor_x, actor_y = get_x_y(actor_losses)
    critic1_x, critic1_y = get_x_y(critic1_losses)
    critic2_x, critic2_y = get_x_y(critic2_losses)
    
    # First plot auto_loss and trans_loss
    plt.xlabel("Epochs")
    plt.plot(trans_x, trans_y, color = "green", label = "Trans")
    plt.ylabel("Trans losses")
    plt.legend(loc = 'upper left')
    plt.show()
    
    
    
    # Then plot losses for actor, critics, alpha
    fig, ax1 = plt.subplots()
    plt.xlabel("Epochs")

    ax1.plot(actor_x, actor_y, color='red', label = "Actor")
    ax1.set_ylabel("Actor losses")
    ax1.legend(loc = 'upper left')

    ax2 = ax1.twinx()
    ax2.plot(critic1_x, critic1_y, color='blue', linestyle = "--", label = "Critic")
    ax2.plot(critic2_x, critic2_y, color='blue', linestyle = ":", label = "Critic")
    ax2.set_ylabel("Critic losses")
    ax2.legend(loc = 'lower left')
    
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax3.plot(alpha_x, alpha_y, color = "black", label = "Alpha")
    ax3.set_ylabel("Alpha losses")
    ax3.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.show()