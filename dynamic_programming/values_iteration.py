import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for _ in range(max_iter):
        values_next = np.full_like(values, float("-inf"))
        for state in range(mdp.observation_space.n):
            for action in range(mdp.action_space.n):
                mdp.reset_state(state)
                next_state, reward, done, _ = mdp.step(action)
                if done:
                    value = reward
                else:
                    value = reward + gamma * values[next_state]
                values_next[state] = max(values_next[state], value)
        if np.allclose(values, values_next):
            break
        values = values_next
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))

    # BEGIN SOLUTION
    def positions(height, width):
        for row in range(height):
            for col in range(width):
                yield (row, col)

    for _ in range(max_iter):
        values_next = values.copy()
        for state in positions(env.height, env.width):
            if np.all(env.moving_prob[state] == 0):
                continue
            values_next[state] = float("-inf")
            for action in range(env.action_space.n):
                env.set_state(*state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    value = reward
                else:
                    value = reward + gamma * values[next_state]
                values_next[state] = max(values_next[state], value)
        if np.allclose(values, values_next, atol=theta):
            break
        values = values_next
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))

    # BEGIN SOLUTION
    def positions(height, width):
        for row in range(height):
            for col in range(width):
                yield (row, col)

    for _ in range(max_iter):
        values_next = values.copy()
        delta = np.zeros_like(values)
        for row, col in positions(env.height, env.width):
            env.set_state(row, col)
            delta[row, col] = value_iteration_per_state(
                env, values_next, gamma, values, theta
            )
        if np.all(delta < theta):
            break
        values = values_next
    return values
    # END SOLUTION
