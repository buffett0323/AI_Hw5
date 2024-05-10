import random
import os
import time
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import imageio
from tqdm import tqdm

from rl_algorithm import DQN
from custom_env import ImageEnv
import utils
# Self-added
import matplotlib.pyplot as plt


def plot_training_curves(dqn_agent):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(dqn_agent.losses)
    plt.title('Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(dqn_agent.rewards)
    plt.title('Rewards over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')

    plt.savefig('Loss_and_Rewards.png')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    # environment hyperparameters
    parser.add_argument('--env_name', type=str, default='ALE/MsPacman-v5')
    parser.add_argument('--state_dim', type=tuple, default=(4, 84, 84))
    parser.add_argument('--image_hw', type=int, default=84, help='The height and width of the image')
    parser.add_argument('--num_envs', type=int, default=4)
    # DQN hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--buffer_size', type=int, default=int(1e5))
    parser.add_argument('--target_update_interval', type=int, default=10000)
    # training hyperparameters
    parser.add_argument('--max_steps', type=int, default=int(2.5e5))
    parser.add_argument('--eval_interval', type=int, default=10000)
    # others
    parser.add_argument('--save_root', type=Path, default='./submissions')
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    # evaluation
    parser.add_argument('--eval', action="store_true", help='evaluate the model')
    parser.add_argument('--eval_model_path', type=str, default=None, help='the path of the model to evaluate')
    return parser.parse_args()


def seed_everything(seed, env):
    "Do not modify this function"
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)


def validaiton(agent, num_evals=5):
    eval_env = gym.make(args.env_name)
    eval_env = ImageEnv(eval_env)
    
    scores = 0
    for i in range(num_evals):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        state, _ = eval_env.reset()
        done = False

        while not done:
            action = agent.act(state, training=False)  # Disable training mode, e.g., no exploration
            state, reward, done, truncated, _ = eval_env.step(action)
            scores += reward
        
    return np.round(scores / num_evals, 4)

def train(agent, env):
    logging_info = {'Step': [], 'AvgScore': []}

    (state, _) = env.reset()
    
    for _ in tqdm(range(args.max_steps)):
        
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        # agent act
        action = agent.act(state)

        # Env step
        next_state, reward, done, truncated, info = env.step(action)

        # Process transition
        # Handle the case where the episode might end due to termination or truncation
        if done or truncated:
            next_state, _ = env.reset()  # Get the initial state for the new episode

        agent.process((state, action, reward, next_state, done))

        state = next_state
        
        if agent.total_steps % args.eval_interval == 0:
            avg_score = validaiton(agent)
            
            "*** YOUR CODE HERE ***"
            # utils.raiseNotDefined()
            # logging
            logging_info['Step'].append(agent.total_steps)
            logging_info['AvgScore'].append(avg_score)
            
            # save model
            print("Step: {}, AvgScore: {}".format(agent.total_steps, avg_score))
            torch.save(agent.network.state_dict(), f'model_pth/model_{agent.total_steps}.pth')



def evaluate(agent, eval_env, capture_frames=True):
    seed_everything(0, eval_env) # don't modify
    
    # load the model
    if agent is None:
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        model_path = 'model_pth/model_10000.pth'  # Update this path as needed
        agent.network.load_state_dict(torch.load(model_path))
        agent.network.eval()
    
    state, _ = eval_env.reset()
    
    # reset env
    done = False
    scores = 0
    
    # Record the evaluation video
    if capture_frames:
        writer = imageio.get_writer(save_dir / 'mspacman.mp4', fps=10)
    
    while not done:
        if capture_frames:
            writer.append_data(eval_env.render())
        else:
            eval_env.render()
        
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        action = agent.act(state, training=False)  # Ensure exploration is turned off during evaluation
        state, reward, done, _, _ = eval_env.step(action)
        scores += reward
        
    if capture_frames:
        writer.close()
    print("The score of the agent: ", scores)




def main():
    env = gym.make(args.env_name)
    env = ImageEnv(env, stack_frames=args.num_envs, image_hw=args.image_hw)

    action_dim = env.action_space.n
    state_dim = (args.num_envs, args.image_hw, args.image_hw)
    agent = DQN(state_dim=state_dim, action_dim=action_dim)
    
    # train
    train(agent, env)
    
    # evaluate
    eval_env = gym.make(args.env_name, render_mode='rgb_array')
    eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
    evaluate(agent, eval_env)
    
    # plotting losses & rewards
    plot_training_curves(agent)

    

if __name__ == "__main__":
    args = parse_args()
    
    # save_dir = args.save_root / f"{args.env_name.replace('/', '-')}__{args.exp_name}__{int(time.time())}"
    save_dir = args.save_root # do whatever you want
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    if args.eval:
        eval_env = gym.make(args.env_name, render_mode='rgb_array')
        eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
        evaluate(agent=None, eval_env=eval_env, capture_frames=False)
    else:
        main()
    