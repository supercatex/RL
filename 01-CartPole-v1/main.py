import gym
from tools.RL import *
import matplotlib.pyplot as plt
import time


def run(env, agent, fps=60, is_render=True, is_train=False):
    state = env.reset()
    action, done = None, False
    total_frames = 0
    score = 0.0
    t1 = time.time()
    while not done:
        if is_render:
            env.render()

        action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        reward = 0.0
        if done:
            reward = -1.0
        score += reward

        if is_train:
            agent.remember(state, action, next_state, reward, done)
            agent.replay()
            agent.update_target_network()
        state = next_state

        while time.time() - t1 < 1.0 / fps:
            time.sleep(0.001)
        t1 = time.time()

        total_frames += 1
    return score, total_frames


if __name__ == "__main__":
    _env_name = "CartPole-v1"
    _env = gym.make(_env_name)
    _agent = DeepQLearning(
        in_shape=(_env.observation_space.shape[0],),
        out_size=_env.action_space.n,
        memory_size=10000, min_memory_size=1000,
        epsilon=1.0, epsilon_decay=0.99995, epsilon_min=0.0,
        learning_rate=0.00025, batch_size=32
    )
    _is_train = False

    if _is_train:
        _history = []
        _agent.compile_model()
        for _i in range(1000):
            _score, _frames = run(_env, _agent, is_render=False, is_train=True)
            if _i % 100 == 99:
                _agent.save_weights("model-%d.h5" % (_i + 1))
            _agent.save_weights("model.h5")

            _history.append(_frames)
            with open("log.txt", "a+") as f:
                f.write("{},{},{},{}\n".format(_i, _score, _frames, _agent.epsilon))
            print("Episode: {}, Score: {}, Frames: {}, Epsilon: {}".format(_i, _score, _frames, _agent.epsilon))
        _env.close()

        plt.plot(range(len(_history)), _history)
        plt.savefig(_env_name + "-history.png")
        plt.show()
    else:
        _agent.epsilon = 0.0
        _agent.load_weights("model-1000.h5")
        _score = run(_env, _agent, is_render=True, is_train=False)
        print("Score: {}".format(_score))

        _env.close()
    del _agent
