import gym
from tools.RL import *
import matplotlib.pyplot as plt
import time


def run(env, agent, fps=60, skip_frames=1, is_render=True, is_train=False):
    state = env.reset()
    action, done = None, False
    total_frame = 0
    score = 0.0
    t1 = time.time()
    while not done:
        if is_render:
            env.render()

        if action is None or total_frame % skip_frames == 0:
            action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        if not done:
            reward = 0.01
        else:
            reward = -1.0
        score += reward

        if done or total_frame % skip_frames == skip_frames - 1:
            if is_train:
                agent.remember(state, action, next_state, reward, done)
                agent.replay()
            state = next_state

        while time.time() - t1 < 1.0 / fps:
            time.sleep(0.001)
        t1 = time.time()

        total_frame += 1
    return score, total_frame


if __name__ == "__main__":
    _env_name = "CartPole-v1"
    _env = gym.make(_env_name)
    _agent = DeepQLearning(
        in_shape=(_env.observation_space.shape[0],),
        out_size=_env.action_space.n,
        memory_size=10000, min_memory_size=1000,
        epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1,
        learning_rate=0.00025, batch_size=32
    )
    _is_train = True

    if _is_train:
        _history = []
        _agent.compile_model()
        for _i in range(2000):
            _score, _t = run(_env, _agent, skip_frames=4, is_render=False, is_train=True)
            # if _agent.epsilon < 0.5 and len(_history) > 0 and _score > max(_history):
            #     _agent.save_weights("best.h5")
            _agent.save_weights("model.h5")
            _history.append(_score)
            print("Epoch: {}, score: {}, epsilon: {}".format(_i, _score, _agent.epsilon))
        _env.close()

        plt.plot(range(len(_history)), _history)
        plt.savefig(_env_name + "-history.png")
        plt.show()
    else:
        _agent.epsilon = 0.0
        _agent.load_weights("model.h5")
        _score = run(_env, _agent, is_render=True, is_train=False)
        print("Score: {}".format(_score))

    _env.close()
    del _agent
