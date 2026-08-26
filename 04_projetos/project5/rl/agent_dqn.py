"""Tabular QL agent"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 300
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector, epsilon):
    """
    Retorna uma ação selecionada por uma política de exploração epsilon-greedy

    Args:
        state_vector (torch.FloatTensor): representação vetorial extraída
        theta (np.ndarray): matriz de pesos atual
        epsilon (float): a probabilidade de escolher um comando aleatório

    Returns:
        (int, int): os índices que descrevem a ação/objeto a ser escolhido
    """
    # TODO Your code here
    action_index, object_index = None, None
    return (action_index, object_index)


class DQN(nn.Module):
    """
    Uma implementação simples de Deep Q-Network.
    Calcula valores Q para cada tupla (ação, objeto) a partir de um vetor de estado de entrada.
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


# pragma: coderesponse template
def deep_q_learning(
    current_state_vector,
    action_index,
    object_index,
    reward,
    next_state_vector,
    terminal,
):
    """
    Atualiza os pesos da DQN para uma determinada transição

    Args:
        current_state_vector (torch.FloatTensor): representação vetorial do estado atual
        action_index (int): índice da ação atual
        object_index (int): índice do objeto atual
        reward (float): recompensa imediata que o agente recebe ao executar o comando atual
        next_state_vector (torch.FloatTensor): representação vetorial do próximo estado
        terminal (bool): True se este episódio tiver terminado

    Return:
        None
    """
    with torch.no_grad():
        q_values_action_next, q_values_object_next = model(next_state_vector)
    maxq_next = 1 / 2 * (q_values_action_next.max() + q_values_object_next.max())

    q_value_cur_state = model(current_state_vector)

    # TODO Your code here

    loss = None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# pragma: coderesponse end


def run_episode(for_training):
    """
    Executa um episódio
    Se for para treinamento, atualiza a função Q
    Se for para teste, calcula e retorna a recompensa acumulada descontada
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = None

    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state, dictionary)
        )

        # TODO Your code here

        if for_training:
            # update Q-function.
            # TODO Your code here
            pass

        if not for_training:
            # update reward
            # TODO Your code here
            pass

        # prepare next step
        # TODO Your code here

    if not for_training:
        return epi_reward


def run_epoch():
    """Executa uma época e retorna a recompensa média dos episódios de teste"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Retorna um array de recompensas de teste por época para uma execução"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test),
            )
        )
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(
        x, np.mean(epoch_rewards_test, axis=0)
    )  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(
        (
            'Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f'
            % (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)
        )
    )
    plt.show()
