"""Linear QL agent"""

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
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# pragma: coderesponse template name="linear_epsilon_greedy"
def epsilon_greedy(state_vector, theta, epsilon):
    # exploração aleatória
    if np.random.rand() < epsilon:
        action_index = np.random.randint(
            NUM_ACTIONS
        )  # Escolhe um indice de ação aleatória entre 0 e NUM_ACTIONS - 1
        object_index = np.random.randint(
            NUM_OBJECTS
        )  # Escolhe um indice de ação aleatória entre 0 e NUM_OBJECTS - 1
        return action_index, object_index  # (ação, objeto)

    # explotação
    q_values = theta.dot(
        state_vector
    )  # gera um vetor onde cada entrada é Q(s, c; \theta)
    best_c = np.argmax(
        q_values
    )  # Encontra c da ação-obheto com maior valor Q naquele estado
    action_index, object_index = index2tuple(
        best_c
    )  # Converte c de volta para o par (action_index, object_index)

    return action_index, object_index
    """
    Retorna uma ação selecionada por uma política de exploração epsilon-greedy

    Args:
        state_vector (np.ndarray): representação vetorial extraída
        theta (np.ndarray): matriz de pesos atual
        epsilon (float): a probabilidade de escolher um comando aleatório

    Return:
        (int, int): os índices que descrevem a ação/objeto a ser escolhido
    """


# pragma: coderesponse template
'''
Ideia central: Q(s, c; \theta) = \theta[c]^T \psi_R(s)
Q-learning: \theta[c] \leftarrow \theta[c]+\alpha*\delta*\psi_R(s)
onde: \delta = r + \gamma max_c' Q(s', c') - Q(s, c)
               |_________________________|   |_____|
                        target              prediction
'''


def linear_q_learning(
    theta,
    current_state_vector,
    action_index,
    object_index,
    reward,
    next_state_vector,
    terminal,
):
    '''
    índice único da ação
        - Cada ação é um par (action_index, object_index)
        - c é esse índice
        - ele indica qual linha de \theta deve ser atualizada
    '''
    c = tuple2index(action_index, object_index)

    '''
    Calcular o valor atual Q(s, c)
        - \theta[c] é o vetor de pesos da ação c
        - current_state_vector é o vetor bag-of-words do estado
        - o prodito interno dá: Q(s, c, \theta)
    '''
    q_current = theta[c].dot(current_state_vector)

    # Calcular o target
    if terminal:
        target = reward
    else:
        '''
        Calcular todos os valores Q(s', c') para o próximo estado
        Isso é exatamente: r + \gama max_c' Q(s', c')
        '''
        q_next = theta.dot(next_state_vector)
        target = reward + GAMMA * np.max(q_next)

    '''
    Calcular o erro TD
        - \delta = target - Q(s, c)
    '''
    td_error = target - q_current

    '''
    atualização de theta
        - \theta[c] \leftarrow \theta[c] + \alpha * \delta * \psi_R(s)
    '''
    theta[c] += ALPHA * td_error * current_state_vector
    """
    Atualiza theta para uma determinada transição

    Args:
        theta (np.ndarray): matriz de pesos atual
        current_state_vector (np.ndarray): representação vetorial do estado atual
        action_index (int): índice da ação atual
        object_index (int): índice do objeto atual
        reward (float): recompensa imediata que o agente recebe ao executar o comando atual
        next_state_vector (np.ndarray): representação vetorial do próximo estado
        terminal (bool): True se este episódio tiver terminado

    Returns:
        None
    """


def run_episode(for_training):
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0.0
    t = 0

    # estado inicial
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Vetor do estado atual (bag-of_words)
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(
            current_state, dictionary
        )

        # escolher ação via epsilon-greedy
        action_index, object_index = epsilon_greedy(
            current_state_vector, theta, epsilon
        )

        # executar ação
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )

        # vetor do próximo estado
        next_state = next_room_desc + next_quest_desc
        next_state_vector = utils.extract_bow_feature_vector(next_state, dictionary)

        # treinamento
        if for_training:
            linear_q_learning(
                theta,
                current_state_vector,
                action_index,
                object_index,
                reward,
                next_state_vector,
                terminal,
            )

        # teste → acumular recompensa descontada
        if not for_training:
            epi_reward += (GAMMA**t) * reward
            t += 1

        # avançar estado
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward
    """
    Executa um episódio
    Se for para treinamento, atualiza a função Q
    Se for para teste, calcula e retorna a recompensa acumulada descontada

    Args:
        for_training (bool): True se for para treinamento

    Returns:
        None
    """


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global theta
    theta = np.zeros([action_dim, state_dim])

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
    action_dim = NUM_ACTIONS * NUM_OBJECTS

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
