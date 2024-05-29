import argparse
import numpy as np
import os
import torch
import csv
import function
import utils
from memory import Buf
from tqdm import tqdm
from evaluation import evaluate_BCQ

def test_BCQ(state_dim, action_dim, device, args, step=1900):
    # For saving files
    setting = f"{args.env}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    max_action = replay_buffer.max_action

    net = function.BCQ(replay_buffer, state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
    net.load(f"./models/")
    states, actions, new_actions, Qs, new_Qs = net.traversal()

    content = []
    raw_states = []
    die = []
    csv_file = csv.reader(open(r'./data/final_without_normalization.csv', 'r'))

    for line in csv_file:
        content.append(line)

    for i in tqdm(range(len(content[1:]))):
        item = content[i+1]
        raw_states.append(item[:40])
        die.append(item[-1])

    with open("./results/final" + "-" + str(step), "a", encoding='utf-8') as f:
        for i in tqdm(range(len(raw_states)-1)):
            for j in range(len(raw_states[0])):
                f.write(str(raw_states[i][j]) + ',')

            # Here we need to normalize the data and fill in the corresponding values
            # The detail content can be calculated by excel
            actions[i][0] = actions[i][0] * 0.2159419905977404 + 0.04903541525231953
            actions[i][1] = actions[i][1] * 0.2449570076648662 + 0.015244652243345539
            actions[i][2] = actions[i][2] * 4.156079170846117 + 0.4516164483041164
            actions[i][3] = actions[i][3] * 0.0839000963579799 + 0.004244573874546281
            actions[i][4] = actions[i][4] * 840.3034890091226 + 452.7063849007768

            new_actions[i][0] = new_actions[i][0] * 0.2159419905977404 + 0.04903541525231953
            new_actions[i][1] = new_actions[i][1] * 0.2449570076648662 + 0.015244652243345539
            new_actions[i][2] = new_actions[i][2] * 4.156079170846117 + 0.4516164483041164
            new_actions[i][3] = new_actions[i][3] * 0.0839000963579799 + 0.004244573874546281
            new_actions[i][4] = new_actions[i][4] * 840.3034890091226 + 452.7063849007768

            for j in range(len(actions[i])):
                if actions[i][j] < 0:
                    actions[i][j] = 0

            for j in range(len(new_actions[i])):
                if new_actions[i][j] < 0:
                    new_actions[i][j] = 0
                # discrete action threshold
                if new_actions[i][0] > 0.5:
                    new_actions[i][0] = 1
                if new_actions[i][0] < 0.5:
                    new_actions[i][0] = 0

            for j in range(len(actions[0])):
                f.write(str(actions[i][j].item()) + ',')
            f.write(str(Qs[i].item()) + ',')
            for j in range(len(new_actions[0])):
                f.write(str(new_actions[i][j].item()) + ',')
            f.write(str(new_Qs[i].item()) + ',')
            f.write(str(die[i]))
            f.write('\n')
    f.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Sepsis")  # OpenAI gym environment name
    parser.add_argument("--buffer_name", default="AI")  # Prepends name to filename
    parser.add_argument("--eval_freq", default=5000, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1500, type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3, type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=1, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--state_dim", default=39)
    parser.add_argument("--action_dim", default=5)
    parser.add_argument("--data_collection", default=False)  # If true, generate buffer
    args = parser.parse_args()

    state_dim = args.state_dim
    action_dim = args.action_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--------------------------------------------------------------")
    print(f"Task:Generating results, Env:{args.env}, device:{torch.cuda.get_device_name(0)}")
    print("--------------------------------------------------------------")
    test_BCQ(state_dim, action_dim, device, args)