"""Tabular QL agent"""

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
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
# politica ε‑greedy | \pi (s)=\arg \max _{c\in C}Q(s,c)
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    # Explotação (exploit)
    if (
        np.random.rand() < epsilon
    ):  # jugar uma moeda de probabilidade ε | escolher ação aleatória
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
        return action_index, object_index

    # Exploração (random)
    q_values = q_func[
        state_1, state_2
    ]  # pegar todos os Q_valores do estado | isso é uma matriz -> shape: (NUM_ACTIONS, NUM_OBJECTS)
    best_action, best_object = np.unravel_index(
        np.argmax(q_values), q_values.shape
    )  # encontrar o índice de maior valor.

    return best_action, best_object

    """
    Retorna uma ação selecionada por uma política de exploração epsilon-Greedy

    Args:
        state_1, state_2 (int, int): dois índices que descrevem o estado atual
        q_func (np.ndarray): função Q atual
        epsilon (float): a probabilidade de escolher um comando aleatório

    Returns:
        (int, int): os índices que descrevem a ação/objeto a ser escolhido
    """


# pragma: coderesponse end


# pragma: coderesponse template
# Unica atualização Q-learning: Q(s,c)\leftarrow (1-\alpha )Q(s,c)+\alpha [R(s,c)+\gamma \max _{c'\in C}Q(s',c')]
# Se o episódio terminou: Q(s, c)\leftarrow (1-\alpha)!(s, c) + \alpha R(s, c)
def tabular_q_learning(
    q_func,
    current_state_1,  # índice da sala (0-3)
    current_state_2,  # índice da missão (0-3)
    # logo, o estado oculto é: h = (r, q)
    action_index,  # índice da ação (go, eat, watch, etc.)
    object_index,  # índice do objeto (apple, TV, bed, etc.)
    # Isso forma o comando? c = (a, b)
    reward,  # valor imediato R(h, c)
    next_state_1,  # sala seguinte
    next_state_2,  # missão (que nunca muda)
    terminal,  # = True | se o episódio acabou
):
    # Valor atual !(s, c)
    current_q = q_func[current_state_1, current_state_2, action_index, object_index]
    if terminal:
        # Episódio terminou: não ha valor futuro
        target = reward
    else:
        # Valor futuro ótimo max_{c'} Q{s', c'}
        next_q_max = np.max(q_func[next_state_1, next_state_2])
        target = reward + GAMMA * next_q_max

    # Atualização Q-Learning
    updated_q = (1 - ALPHA) * current_q + ALPHA * target
    # Gravar o novo valor
    q_func[current_state_1, current_state_2, action_index, object_index] = updated_q

    return None  # This function shouldn't return anything

    """
    Atualiza a q_func para uma determinada transição
    Args:
        q_func (np.ndarray): função Q atual
        current_state_1, current_state_2 (int, int): dois índices que descrevem o estado atual
        action_index (int): índice da ação atual
        object_index (int): índice do objeto atual
        reward (float): a recompensa imediata que o agente recebe ao executar o comando atual
        next_state_1, next_state_2 (int, int): dois índices que descrevem o próximo estado
        terminal (bool): True se este episódio tiver terminado
    Returns:
        None
    """


# pragma: coderesponse template
'''
    for_training: diz se esse episódio é usado para aprender (True) ou para avaliar (False).
    A função tem dois comportamentos:
        treino → atualiza Q, não retorna recompensa.
        teste → não atualiza Q, retorna a soma descontada das recompensas.
'''


def run_episode(for_training):
    epsilon = TRAINING_EP if for_training else TESTING_EP
    '''
    Se for treino: usa TRAINING_EP (tipicamente maior, mais exploração).
    Se for teste: usa TESTING_EP (tipicamente pequeno, quase sempre exploit).
    '''
    # Recompensa acumulada descontada (somente para teste)
    epi_reward = 0.0  # vai acumular \sum \gamma^t r_t
    t = 0  # é o índice do passo, usado no expoente de \gamma^t

    # Estado inicial | começando um novo jogo
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    '''
    Chama o ambiente para iniciar um episódio.
    Retorna:
        descrição textual da sala inicial,
        descrição textual da missão,
        terminal = False (episódio recém-iniciado).
    '''

    # Mapear texto -> índice de estado
    state_1 = dict_room_desc[current_room_desc]
    state_2 = dict_quest_desc[current_quest_desc]
    '''
    Usa os dicionários fornecidos:
        dict_room_desc: texto da sala → índice inteiro.
        dict_quest_desc: texto da missão → índice inteiro.
    Isso transforma o estado textual s = (S_r, s_q) em estado tabular (state_1, state_2)
    '''
    # Loop principal do epísodio
    '''
    Enquanto o episódio não terminou, seguimos:
        escolhendo ações,
        avançando o ambiente,
        atualizando Q (se treino),
        acumulando recompensa (se teste).
    '''
    while not terminal:
        # 1. Escolher ação via epsilon-greedy
        action_index, object_index = epsilon_greedy(state_1, state_2, q_func, epsilon)
        '''
        Usa a função que você implementou:
            com probabilidade ε → ação aleatória.
            com probabilidade 1-ε → ação com maior Q naquele estado.
        Retorna índices da ação e do objeto, que formam o comando  c = (a, b)
        '''
        # 2. Executar ação no ambiente
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )
        '''
        Aplica o comando no estado atual.
        Retorna:
            descrição da próxima sala,
            descrição da missão (não muda),
            recompensa imediata,
            se o episódio terminou (terminal=True ou False)
        '''
        # 3. Mapear próximo estado para índice
        next_state_1 = dict_room_desc[next_room_desc]
        next_state_2 = dict_quest_desc[next_quest_desc]
        '''
        De novo, texto -> índice
        Agora temos o par (next_state_1, next_state_2) para usar na atualização de Q
        '''
        # 4. Atualizar Q (somente treinamento)
        if for_training:
            tabular_q_learning(
                q_func,
                state_1,
                state_2,
                action_index,
                object_index,
                reward,
                next_state_1,
                next_state_2,
                terminal,
            )
        '''
        Chama sua função de Q-learning tabular.
        Atualiza apenas um valor Q(s, c) com base na transição atual.
        Se terminal=True, o target não inclui valor futuro.
        '''
        # 5. Acumular recompensa descontada (somente teste)
        if not for_training:
            epi_reward += (GAMMA**t) * reward
            t += 1
        '''
        Soma \gamma^t r_t ao total do episódio
        Incrementa t para o proximo passo
        Isso implenta extamente \sum_(t=0)^\inf \gamma^t r_t
        '''
        # 6. Avançar para o próximo estado
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc
        state_1 = next_state_1
        state_2 = next_state_2
        '''
        Atualiza:
            descrições textuais,
            índices tabulares.
            Prepara o próximo passo do while.
        '''

    # Retornar recompensa apenas na fase de teste
    if not for_training:
        return epi_reward
    '''
    Em treino: não retorna nada (a função é usada só para atualizar Q).
    Em teste: retorna a recompensa acumulada descontada daquele episódio.
    '''
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
    """Executa uma época e retorna a recompensa média dos episódios de teste"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Retorna um array de recompensas de teste por época para uma execução"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

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
    # Carregamento de dados e construção dos dicionários que utilizam um índice único para cada estado
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    # Experimento com vários valores de alpha
    alphas = [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]
    results = {}

    for alpha in alphas:
        ALPHA = alpha
        print(f"\n=== Rodando experimento com alpha = {alpha} ===")
        epoch_rewards_test = []

        for _ in range(NUM_RUNS):
            epoch_rewards_test.append(run())

        results[alpha] = np.mean(epoch_rewards_test, axis=0)

    # Plot das curvas de convergência
    fig, axis = plt.subplots()

    for alpha in alphas:
        axis.plot(results[alpha], label=f"alpha={alpha}")

    axis.set_xlabel("Epochs")
    axis.set_ylabel("Reward")
    axis.set_title("Convergência para diferentes valores de alpha")
    axis.legend()
    plt.show()

    # epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    # for _ in range(NUM_RUNS):
    #     epoch_rewards_test.append(run())

    # epoch_rewards_test = np.array(epoch_rewards_test)

    # x = np.arange(NUM_EPOCHS)
    # fig, axis = plt.subplots()
    # axis.plot(
    #     x, np.mean(epoch_rewards_test, axis=0)
    # )  # plot reward per epoch averaged per run
    # axis.set_xlabel('Epochs')
    # axis.set_ylabel('reward')
    # axis.set_title(
    #     (
    #         'Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f'
    #         % (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)
    #     )
    # )
    # plt.show()
