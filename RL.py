import numpy as np
from typing import Tuple, List
from collections import deque
from tensorflow.keras import models, layers
from tensorflow.keras import activations
from tensorflow.keras import losses, optimizers
from tensorflow.keras import metrics
from tensorflow.keras.utils import plot_model


class DeepQLearning(object):
    def __init__(
            self,
            in_shape: Tuple,
            out_size: int,
            epsilon: float = 1.0,
            epsilon_min: float = 0.1,
            epsilon_decay: float = 0.999998,
            batch_size: int = 32,
            gamma: float = 0.99,
            memory_size: int = 10000,
            min_memory_size: int = 1000,
            learning_rate: float = 0.00025
    ):
        self.in_shape: Tuple = in_shape
        self.out_size: int = out_size
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.batch_size: int = batch_size
        self.memory_size: int = memory_size
        self.gamma: float = gamma
        self.learning_rate: float = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size
        self.q_model: models.Model = self.build_model()

    def build_model(self) -> models.Model:
        x_in = layers.Input(self.in_shape, name="State")
        x = x_in
        x = layers.Flatten()(x)
        x = layers.Dense(
            units=512,
            activation=activations.relu,
            name="Dense_1"
        )(x)
        x = layers.Dense(
            units=256,
            activation=activations.relu,
            name="Dense_2"
        )(x)
        x = layers.Dense(
            units=64,
            activation=activations.relu,
            name="Dense_3"
        )(x)
        x_out = layers.Dense(
            units=self.out_size,
            activation=activations.linear,
            name="Actions"
        )(x)
        model = models.Model(x_in, x_out)
        return model

    def compile_model(self, path=None):
        self.q_model.compile(
            loss=losses.mean_squared_error,
            optimizer=optimizers.RMSprop(
                lr=self.learning_rate,
                rho=0.95,
                epsilon=0.01
            ),
            metrics=[metrics.binary_accuracy]
        )
        self.q_model.summary()
        if path is not None:
            plot_model(self.q_model, path)

    def save_weights(self, path):
        self.q_model.save_weights(path)

    def load_weights(self, path):
        self.q_model.load_weights(path, by_name=True)

    def get_valid_actions(self) -> List[int]:
        valid_actions: List[int] = []
        for i in range(self.out_size):
            valid_actions.append(i)
        return valid_actions

    def get_action(self, state: np.ndarray) -> int:
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.get_valid_actions())
        else:
            state = np.reshape(state, (1,) + state.shape)
            values = self.q_model.predict(state)
            action = int(np.argmax(values, axis=1))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def remember(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool
    ):
        self.memory.append((state, action, reward, next_state, terminal))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        if len(self.memory) < self.min_memory_size:
            return

        mini_batch_index = np.random.choice(
            len(self.memory),
            self.batch_size,
            replace=False
        )
        mini_batch = np.array(self.memory)[mini_batch_index]
        s0, a0, s1, r1, done = zip(*mini_batch)
        s0 = np.array(s0, dtype=np.float32)
        a0 = np.array(a0, dtype=np.uint)
        s1 = np.array(s1, dtype=np.float32)
        r1 = np.array(r1, dtype=np.float32)
        done = np.array(done, dtype=np.bool)

        t0 = np.array(self.q_model.predict(s0), dtype=np.float32)
        t1 = np.array(self.q_model.predict(s1), dtype=np.float32)
        t0[range(self.batch_size), a0] = r1 + self.gamma * np.max(t1, axis=1) * np.invert(done)
        self.q_model.fit(s0, t0, epochs=1, verbose=0)


class DoubleDQN(object):
    def __init__(
            self,
            in_shape: Tuple,
            out_size: int,
            epsilon: float = 1.0,
            epsilon_min: float = 0.1,
            epsilon_decay: float = 0.999998,
            batch_size: int = 32,
            gamma: float = 0.99,
            memory_size: int = 10000,
            min_memory_size: int = 1000,
            learning_rate: float = 0.00025
    ):
        self.in_shape: Tuple = in_shape
        self.out_size: int = out_size
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.batch_size: int = batch_size
        self.memory_size: int = memory_size
        self.gamma: float = gamma
        self.learning_rate: float = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size
        self.q_model: models.Model = self.build_model()
        self.t_model: models.Model = self.build_model()

    def build_model(self) -> models.Model:
        x_in = layers.Input(self.in_shape, name="State")
        x = x_in
        x = layers.Flatten()(x)
        x = layers.Dense(
            units=512,
            activation=activations.relu,
            name="Dense_1"
        )(x)
        x = layers.Dense(
            units=256,
            activation=activations.relu,
            name="Dense_2"
        )(x)
        x = layers.Dense(
            units=64,
            activation=activations.relu,
            name="Dense_3"
        )(x)
        x_out = layers.Dense(
            units=self.out_size,
            activation=activations.linear,
            name="Actions"
        )(x)
        model = models.Model(x_in, x_out)
        return model

    def compile_model(self, path=None):
        self.q_model.compile(
            loss=losses.mean_squared_error,
            optimizer=optimizers.RMSprop(
                lr=self.learning_rate,
                rho=0.95,
                epsilon=0.01
            ),
            metrics=[metrics.binary_accuracy]
        )
        self.q_model.summary()
        if path is not None:
            plot_model(self.q_model, path)

    def save_weights(self, path):
        self.q_model.save_weights(path)

    def load_weights(self, path):
        self.q_model.load_weights(path, by_name=True)
        self.update_target_network()

    def update_target_network(self):
        self.t_model.set_weights(self.q_model.get_weights())

    def get_valid_actions(self) -> List[int]:
        valid_actions: List[int] = []
        for i in range(self.out_size):
            valid_actions.append(i)
        return valid_actions

    def get_action(self, state: np.ndarray) -> int:
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.get_valid_actions())
        else:
            state = np.reshape(state, (1,) + state.shape)
            values = self.q_model.predict(state)
            action = int(np.argmax(values, axis=1))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def remember(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool
    ):
        self.memory.append((state, action, reward, next_state, terminal))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        if len(self.memory) < self.min_memory_size:
            return

        mini_batch_index = np.random.choice(
            len(self.memory),
            self.batch_size,
            replace=False
        )
        mini_batch = np.array(self.memory)[mini_batch_index]
        s0, a0, s1, r1, done = zip(*mini_batch)
        s0 = np.array(s0, dtype=np.float32)
        a0 = np.array(a0, dtype=np.uint)
        s1 = np.array(s1, dtype=np.float32)
        r1 = np.array(r1, dtype=np.float32)
        done = np.array(done, dtype=np.bool)

        t0 = np.array(self.q_model.predict(s0), dtype=np.float32)
        t1 = np.array(self.q_model.predict(s1), dtype=np.float32)
        t2 = np.array(self.t_model.predict(s1), dtype=np.float32)
        values = t2[range(self.batch_size), np.argmax(t1, axis=1)]
        t0[range(self.batch_size), a0] = r1 + self.gamma * values * np.invert(done)
        self.q_model.fit(s0, t0, epochs=1, verbose=0)
