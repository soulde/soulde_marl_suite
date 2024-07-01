import cv2
import torch

from sandbox import PathPlanningSandbox, ParallelContainer
from policy import AgentParams, MADDPG, Actor, Critic
from utils import ExperimentManager, ReplayBuffer

from itertools import count
import numpy as np
import tqdm
from functools import partial


def evaluation():
    device = torch.device("cpu")
    exp = ExperimentManager('./run')

    exp.start_experiment('../configs/path_planning_eval.json')

    env = PathPlanningSandbox(exp.args['sandbox'])

    agent = MADDPG(Actor, Critic, env.state_dim, env.action_dim, device=device)

    ob, _ = env.reset()

    # print(ob.shape)
    max_step = 1000
    reward_sum = np.zeros([len(env.state_dim)])
    with tqdm.tqdm(total=max_step) as pbar:
        for c in range(max_step):
            pbar.update(1)

            ob = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(ob).squeeze(0)

            u = action.numpy()

            next_ob, rew, dones, truncation, info = env.step(u)
            reward_sum += rew

            if '_final_info' in info.keys():
                next_ob, _ = env.reset()
                pbar.set_postfix({'reward': reward_sum})
                # exp.add_scalars('reward', {'episode reward': np.mean(reward_sum[i])})
                reward_sum = np.zeros_like(reward_sum)

            ob = next_ob
            env.render()
            cv2.waitKey(1)



if __name__ == '__main__':
    evaluation()
