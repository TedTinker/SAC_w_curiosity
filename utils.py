import argparse

env_name = "Pendulum-v1"
#env_name = "MountainCarContinuous-v0"

parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default=env_name, help="Environment name")
parser.add_argument("-info", type=str, default = "Ted", help="Information or name of the run")
parser.add_argument("-ep", type=int, default=100, help="The amount of training episodes, default is 100")
parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=100, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
args = parser.parse_args()



import matplotlib.pyplot as plt

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

    
def plot_losses(alpha_losses, actor_losses, critic1_losses, critic2_losses):
    alpha_x, alpha_y = get_x_y(alpha_losses)
    actor_x, actor_y = get_x_y(actor_losses)
    critic1_x, critic1_y = get_x_y(critic1_losses)
    critic2_x, critic2_y = get_x_y(critic2_losses)
    
    plt.plot(alpha_x, alpha_y, color = "black")
    plt.xlabel("Epochs")
    plt.ylabel("Alpha losses")
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.plot(actor_x, actor_y, color='red', label = "Actor")
    ax2 = ax1.twinx()
    ax2.plot(critic1_x, critic1_y, color='blue', linestyle = "--", label = "Critic")
    ax2.plot(critic2_x, critic2_y, color='blue', linestyle = ":", label = "Critic")
    fig.tight_layout()
    plt.xlabel("Epochs")
    ax1.set_ylabel("Actor losses")
    ax2.set_ylabel("Critic losses")
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'lower left')
    plt.show()