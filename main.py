import argparse
import numpy as np
import os
import torch
import random
import function
import utils
from memory import Buf
from tqdm import tqdm
from test import test_BCQ


def interact_with_environment(buf, state_dim, action_dim, device, args, max_action):

    setting = f"{args.env}"
    buffer_name = f"{args.buffer_name}_{setting}"

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    episode_timesteps = 0


    for t in range(int(buf.size)):
        episode_timesteps += 1
        state, next_state, action, reward = buf.step()
        done_bool = buf.check()

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)


    replay_buffer.save(f"./buffers/{buffer_name}", max_action)


def train_BCQ(state_dim, action_dim, device, args):
    # For saving files
    setting = f"{args.env}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    max_action = replay_buffer.max_action

    print('creating network...')
    net = function.BCQ(replay_buffer, state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
    if args.load_model == True:
        net.load(f"./models/")
        print('parameter loaded...')

    V = []
    C = []
    A = []

    for step in range(args.max_timesteps):
        v_loss, c_loss, a_loss = net.train(iterations=int(args.eval_freq), batch_size=args.batch_size)
        V.append(v_loss)
        C.append(c_loss)
        A.append(a_loss)

        if step % 10 == 0:
            print('step-' + str(step) + ':')
            print('VAE-loss:' + str(v_loss))
            print('Q-loss:' + str(c_loss))
            print('Actor-loss:' + str(a_loss))

        if step % 100 == 0 and step != 0:
            print('model parameters being saved...')
            net.save(f"./models/")

        step += 1

    print('loss being saved...')
    with open("./results/loss-data", "a", encoding='utf-8') as f:
        for i in range(len(V)-1):
            f.write(str(V[i]) + ',')
        f.write(str(V[-1]) + '\n')

        for i in range(len(C)-1):
            f.write(str(C[i]) + ',')
        f.write(str(C[-1])+'\n')

        for i in range(len(A)-1):
            f.write(str(A[i]) + ',')
        f.write(str(A[-1]) + '\n')
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Sepsis")  # OpenAI gym environment name
    parser.add_argument("--buffer_name", default="AI")  # Prepends name to filename
    parser.add_argument("--eval_freq", default=2000, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1002, type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--hidden_size", default=30,
                        type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=64, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.7)  # Discount factor
    parser.add_argument("--tau", default=0.05)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning
    parser.add_argument("--phi", default=0.01)  # Max perturbation hyper-parameter
    parser.add_argument("--state_dim", default=39)
    parser.add_argument("--action_dim", default=5)
    parser.add_argument("--data_collection", default=True)  # If true, generate buffer
    parser.add_argument("--load_model", default=False)
    args = parser.parse_args()
    random.seed(8)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    state_dim = args.state_dim
    action_dim = args.action_dim

    device = torch.device("mps")

    if args.data_collection:
        print("-----------------------------------")
        print(f"Task:Collecting data, Env:{args.env}")
        print("-----------------------------------")
        buf = Buf()
        buf.create()
        max_action = buf.getMaxAction()
        #max value of 5 actions in buffer respectively
        interact_with_environment(buf, state_dim, action_dim, device, args, max_action)

    print("----------------------------------------------------------")
    print(f"Task:Training BCQ, Env:{args.env}, device:{torch.cuda.get_device_name(0)}")
    print("----------------------------------------------------------")
    train_BCQ(state_dim, action_dim, device, args)