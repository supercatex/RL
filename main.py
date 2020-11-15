import gym
from RL import *
import matplotlib.pyplot as plt
import time


def run(env, agent, fps=60, is_render=True, is_train=False):
    state = env.reset()
    action = None
    done = False
    score = 0.0
    t1 = time.time()
    t = 0
    while not done:
        if is_render:
            env.render()

        if action is None or t % 4 == 0:
            action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        if not done:
            reward = 0.01
        else:
            reward = -1.0
        score += reward

        if is_train:
            agent.remember(state, action, next_state, reward, done)
            _agent.replay()
        state = next_state

        while time.time() - t1 < 1.0 / fps:
            time.sleep(0.001)
        # print("%.2ffps" % (1.0 / (time.time() - t1)))
        t1 = time.time()
        t += 1
    return score, t


if __name__ == "__main__":
    _env = gym.make("CartPole-v1")
    _agent = DoubleDQN(
        in_shape=(_env.observation_space.shape[0],),
        out_size=_env.action_space.n,
        memory_size=10000, min_memory_size=1000,
        epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
        learning_rate=0.00025, batch_size=32
    )
    _is_train = True

    if _is_train:
        _history = []
        _agent.compile_model()
        for _i in range(2000):
            _score, _t = run(_env, _agent, is_render=False, is_train=True)
            if _agent.epsilon < 0.5 and len(_history) > 0 and _score > max(_history):
                _agent.save_weights("best.h5")
            _agent.save_weights("model.h5")
            if _i % 20 == 0:
                _agent.update_target_network()
            _history.append(_score)
            print("Epoch: {}, score: {}, epsilon: {}".format(_i, _score, _agent.epsilon))
        _env.close()

        plt.plot(range(len(_history)), _history)
        plt.savefig("CartPole-v1-history.png")
        plt.show()
    else:
        _agent.epsilon = 0.0
        _agent.load_weights("best.h5")
        _score = run(_env, _agent, is_render=True, is_train=False)
        print("Score: {}".format(_score))

    _env.close()
    del _agent
