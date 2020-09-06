# coding: utf-8

import gym
import xsim
import numpy as np
import xtools as xt
import pandas as pd
from matplotlib import pyplot as plt
from xai.algorithms.pg import PG
from xai.utils.buffer import ReplayBuffer, BatchLoader
from xai.utils import space
from xai.utils import order


def run():
    order.please()

    config_file = "config.yaml"
    cf = xt.Config(config_file)

    env = gym.make(cf.env.name)
    xt.info("env", env)

    agent = PG(
        cf.pg.units,
        space.get_size(env.action_space),
        space.get_size(env.observation_space),
        lr=cf.pg.lr
    )
    xt.info("agent", agent)

    global_step = 0
    log = xsim.Logger()

    for epoch in range(1, cf.train.epoch+1):
        rewards = []
        rb = ReplayBuffer(max_size=cf.env.step.max)
        state = env.reset()
        for step in range(1, cf.env.step.max+1):
            action = agent.select_noisy_action(state)
            state_, reward, done, _ = env.step(action)

            rewards.append(reward)
            rb(state=state, action=action, next_state=state_, reward=reward, done=done)

            if done:
                break

            state = state_

        rb.append_column("Q", agent.compute_value(rb.get("reward"), rb.get("done")))
        loader = BatchLoader(rb, cf.train.batch_size)
        losses = []
        episode_step = 0
        while episode_step < len(rb):
            for batch in loader:
                loss = agent.train(batch)
                losses.append(loss)

                episode_step += 1
                global_step += 1

        log.store(epoch=epoch, reward=np.sum(rewards), loss=np.sum(losses)).flush()
        print("[train] epoch:{:3.0f}  step:{:6.0f}  reward:{:10.5f}  loss:{:10.5f}".format(
            epoch, global_step, np.sum(rewards), np.sum(losses)
        ))

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "epoch": result.epoch(),
        "reward": result.reward(),
        "loss": result.loss()
    })
    fig, axes = plt.subplots(nrows=2)
    result.plot(x="epoch", y="reward", ax=axes[0])
    result.plot(x="epoch", y="loss", ax=axes[1])
    plt.show()


if __name__ == '__main__':
    run()
