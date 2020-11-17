import gym
from pyglet.window import key
from tools.RL import *
import matplotlib.pyplot as plt
import time
import cv2
import pickle
import os


def key_release(k, mod):
    if k == key.ESCAPE:
        pass


def run(env, agent, fps=60, is_render=True, is_train=False):
    frames = deque(maxlen=4)
    for i in range(frames.maxlen):
        frames.append(np.zeros(agent.in_shape[:2]))
    env.reset()

    action_list = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0)
    )
    action, done = None, False

    total_frames = 0
    score = 0.0
    negative_reward_times = 0
    t1 = time.time()
    while not done:
        if is_render:
            env.render()
            if total_frames == 0:
                env.viewer.window.on_key_release = key_release
                env.reset()

        state = np.transpose(np.array(frames), (1, 2, 0))
        action = agent.get_action(state)

        next_frame, reward, done, info = env.step(action_list[action])
        negative_reward_times = 0 if reward > 0 else negative_reward_times + 1
        if negative_reward_times > 25:
            done = True
            reward = -1.0
        elif reward <= 0:
            reward = -0.01
        else:
            reward = 0.01
        score += reward

        img = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        img = img[48-20:48+20, 48-20:48+20]
        cv2.imshow("frame", img)
        cv2.waitKey(1)
        img = cv2.resize(img, (agent.in_shape[1], agent.in_shape[0]))
        img = np.array(img, dtype=np.float32) / 255
        frames.append(img)
        next_state = np.transpose(np.array(frames), (1, 2, 0))
        if is_train:
            agent.remember(state, action, next_state, reward, done)
            agent.replay()
            agent.update_target_network()

        while time.time() - t1 < 1.0 / fps:
            time.sleep(0.001)
        t1 = time.time()

        total_frames += 1
    return score, total_frames


if __name__ == "__main__":
    _env_name = "CarRacing-v0"
    _env = gym.make(_env_name)
    _agent = DeepQLearning(
        in_shape=(40, 40, 4),
        out_size=3,
        memory_size=10000, min_memory_size=1000,
        epsilon=1.0, epsilon_decay=0.99998, epsilon_min=0.0,
        learning_rate=0.00025, batch_size=32, with_cnn=True
    )
    _is_train = True

    if _is_train:
        _history = []
        if os.path.exists("memory.pickle"):
            with open("memory.pickle", "rb") as f:
                _agent.memory = pickle.load(f)
        if os.path.exists("epsilon.txt"):
            with open("epsilon.txt", "r") as f:
                _agent.epsilon = float(f.readline())
        if os.path.exists("model.h5"):
            _agent.load_weights("model.h5")
        _agent.compile_model()
        for _i in range(100):
            _score, _frames = run(_env, _agent, is_render=False, is_train=True)
            _agent.save_weights("model.h5")

            _history.append(_frames)
            with open("log.txt", "a+") as f:
                f.write("{}\t{}\t{}\t{}\n".format(_i, _score, _frames, _agent.epsilon))
            print("Epoch: {}, score: {}, frames: {}, epsilon: {}".format(_i, _score, _frames, _agent.epsilon))
        with open("memory.pickle", "wb") as f:
            pickle.dump(_agent.memory, f)
        with open("epsilon.txt", "w+") as f:
            f.write(str(_agent.epsilon))
        _env.close()

        # plt.plot(range(len(_history)), _history)
        # plt.savefig(_env_name + "-history.png")
        # plt.show()
    else:
        _agent.epsilon = 0.0
        _agent.load_weights("model.h5")
        _score = run(_env, _agent, is_render=True, is_train=False)
        print("Score: {}".format(_score))

        _env.close()
    del _agent
