import warnings
warnings.filterwarnings(action='ignore')

import os
import random
import pandas as pd
import numpy as np
import plotly.io as pio
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from matplotlib.colors import ListedColormap

pio.renderers.default = 'iframe_connected'
my_path = os.path.abspath('')


def discover_NBG(clean, vis=False):

    # trace variants and their frequencies
    traces = clean.groupby('case:concept:name', observed=True)['concept:name'] \
                  .transform(lambda x: '>>'.join(x))

    variant_freq = traces.value_counts().rename_axis('trace').reset_index(name='counts')
    variant_freq['var_id'] = ['var_' + str(i) for i in range(len(variant_freq))]
    variant_freq['element'] = variant_freq['trace'].apply(lambda x: list(x.split('>>')))

    # prepare the data for graph construction
    clean2 = clean[['case:concept:name', 'concept:name', 'order']] \
        .rename(columns={"concept:name": "from", "order": "level"}, copy=False)
    clean2['FROM'] = clean2['from'].astype(str) + '_' + clean2['level'].astype(str)

    NBGs = []
    actset = clean['concept:name'].unique()

    for i in range(len(actset)):
        cat_1 = variant_freq[variant_freq['element'].apply(lambda x: actset[i] in x)].copy()
        cat_1['duplicate'] = cat_1['element'].apply(
            lambda x: sum(pd.Series(x).isin([actset[i]]))
        )

        cat_2 = pd.DataFrame({
            'var_id': np.repeat(cat_1['var_id'], cat_1['element'].apply(len)),
            'concept:name': sum(cat_1['element'], []),
            'counts': np.repeat(cat_1['counts'], cat_1['element'].apply(len))
        })

        cat_2['level'] = 0
        cat_2.loc[cat_2['concept:name'] == actset[i], 'level'] = 1
        cat_2a = cat_2.groupby('var_id')['level']
        cat_2['level'] = cat_2a.cumcount() - cat_2a.transform(lambda x: (x.values).argmax())

        # handle activities that occur several times in one trace
        if sum(cat_1['duplicate'] > 1) > 0:
            for _, row in cat_1[cat_1['duplicate'] > 1].iterrows():
                temp1 = cat_2[cat_2['var_id'] == str(row['var_id'])]
                cat_3 = pd.concat([temp1] * (row['duplicate'] - 1))
                cat_3['var_id'] = np.repeat(
                    ["{}_dup_{}".format(a, b) for a, b in zip(list(cat_3['var_id']),
                                                             range(row['duplicate'] - 1))],
                    len(temp1)
                )

                locs = np.where(temp1['concept:name'] == actset[i])
                temp2 = np.repeat(locs[0][1:], len(temp1))
                cat_3['level'] = list(range(len(temp1))) * (row['duplicate'] - 1) - temp2
                cat_2 = pd.concat([cat_2, cat_3])

        cat_3 = cat_2.rename(columns={"concept:name": "from"})
        cat_3['FROM'] = cat_3['from'] + '_' + cat_3['level'].astype(str)
        cat_3.reset_index(drop=True, inplace=True)

        cat_4 = cat_3.drop(index=cat_3.index[len(cat_3) - 1]).copy()
        cat_4['TO'] = list(cat_3.drop(index=cat_3.index[0])['FROM'])

        cat_4 = cat_4.drop(cat_4[cat_4['FROM'].str.contains("End")].index)
        cat_4['next_level'] = cat_4['level'] + 1
        cat_4.reset_index(drop=True, inplace=True)

        cat_5 = cat_4.groupby(['FROM', 'TO', 'level'])['counts'] \
                     .sum().reset_index(name='counts') \
                     .sort_values(by=['level'], axis=0)
        cat_5.reset_index(drop=True, inplace=True)

        cat_5['likelihood'] = 0.0

        total_counts_post = cat_5.groupby('FROM')['counts'].transform('sum')
        total_counts_prior = cat_5.groupby('TO')['counts'].transform('sum')

        cat_5.loc[cat_5.level >= 0, 'likelihood'] = \
            (cat_5['counts'] / total_counts_post).loc[cat_5.level >= 0]
        cat_5.loc[cat_5.level < 0, 'likelihood'] = \
            (cat_5['counts'] / total_counts_prior).loc[cat_5.level < 0]

        G = nx.from_pandas_edgelist(
            cat_5, 'FROM', 'TO',
            edge_attr='likelihood',
            create_using=nx.DiGraph()
        )

        node_dic = cat_3[['from', 'FROM', 'level']].drop_duplicates().reset_index(drop=True)
        nx.set_node_attributes(
            G,
            name='level',
            values=pd.Series(node_dic['level'].values, index=node_dic['FROM']).to_dict()
        )
        nx.set_node_attributes(
            G,
            name='activity',
            values=pd.Series(node_dic['from'].values, index=node_dic['FROM']).to_dict()
        )

        color = ['yellow' if data['activity'] == actset[i] else '#FF1744'
                 for _, data in G.nodes(data=True)]

        pos = nx.multipartite_layout(G, subset_key='level')
        plt.figure(1, figsize=(12, 9))
        plt.title("NBG for " + actset[i])
        nx.draw(G, pos, with_labels=True, node_color=color)

        edge_labels = nx.get_edge_attributes(G, 'likelihood')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, font_color='blue')

        NBGs.append(G)

        if vis:
            os.makedirs(os.path.join(my_path, 'image'), exist_ok=True)
            plt.savefig(os.path.join(my_path, 'image', actset[i] + '.png'), format="PNG")

        plt.close()

    return NBGs



def discover_pattern(episode_history):
    df_explain = pd.DataFrame({
        'action': [hist[1] for hist in episode_history],
        'loc': [hist[2] for hist in episode_history],
        'label': [hist[3] for hist in episode_history],
        'rework': [hist[4] for hist in episode_history]
    })

    if len(df_explain) == 0:
        return ''

    df_explain['action_label'] = df_explain[['action', 'label']] \
        .apply(lambda x: str(x[0]) + '_' + str(x[1]), axis=1)

    df_explain['predict_patterns'] = ''
    df_explain['predict_patterns_simple'] = ''

    # interpret as a remove -> insert pattern
    loc_insert = (df_explain['action'] == 'remove')
    if sum(loc_insert) > 0:
        df_explain.loc[loc_insert, 'predict_patterns'] = \
            df_explain.loc[loc_insert, ['loc', 'label']] \
                      .apply(lambda x: 'insert_' + str(x[0]) + '_' + str(x[1]), axis=1)

    # interpret as an inject -> skip pattern
    loc_skip = (df_explain['action'] == 'inject')
    if sum(loc_skip) > 0:
        df_explain.loc[loc_skip, 'predict_patterns'] = \
            df_explain.loc[loc_skip, ['loc', 'label']] \
                      .apply(lambda x: 'skip_' + str(x[0]) + '_' + str(x[1]), axis=1)

    # rework
    loc_rework = (df_explain['rework'] == True)
    if sum(loc_rework) > 0:
        df_explain.loc[loc_rework, 'predict_patterns'] = \
            df_explain.loc[loc_rework, ['loc', 'label']] \
                      .apply(lambda x: 'rework_' + str(x[0]) + '_' + str(x[1]), axis=1)

    # move
    loc_moved = (~loc_rework) & \
                (df_explain['label'].duplicated(keep=False)) & \
                (~df_explain['action_label'].duplicated(keep=False))

    if sum(loc_moved) > 0:
        df_explain.loc[loc_moved, 'predict_patterns'] = \
            df_explain.loc[loc_moved, ['action', 'loc', 'label']] \
                      .apply(lambda x: 'move_' + str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]),
                             axis=1)
        df_explain.loc[loc_moved, 'predict_patterns_simple'] = \
            df_explain.loc[loc_moved, ['label']] \
                      .apply(lambda x: 'move_' + str(x[0]), axis=1)

    # replace
    def check_different_actions_activities(group):
        if len(group) < 2:
            return False
        first_row = group.iloc[0]
        for _, row in group.iloc[1:].iterrows():
            if row['action'] != first_row['action'] and row['label'] != first_row['label']:
                return True
        return False

    matching_locs = df_explain.groupby('loc').filter(check_different_actions_activities)['loc'].unique()
    loc_replace = (~loc_rework) & (df_explain['loc'].isin(matching_locs))

    if sum(loc_replace) > 0:
        df_explain.loc[loc_replace, 'predict_patterns'] = \
            df_explain.loc[loc_replace, ['action', 'label']] \
                      .apply(lambda x: 'replace_' + str(x[0]) + '_' + str(x[1]), axis=1)

    return str(df_explain['predict_patterns'].tolist())


# ────────────────────────────────────────────────────────────
# process_case_id_wrapper: RL training (or inference) for a single case.
# The repair memory is handled by the caller (sim._process_case_with_memory).
# ────────────────────────────────────────────────────────────
def process_case_id_wrapper(
    case_id_for_processing,
    p_anomaly_nolabel,
    p_filtered_actset,
    p_act_freq,
    p_actset,
    p_NBGs,
    p_num_epi,
    alpha,
    agent,                  # shared agent created by the caller
    train_mode=True,        # True = training ('validation') phase, False = test phase
    case_type='val',        # value of the 'type' column in the result CSV ('val' or 'test')
    max_step=8              # maximum number of steps per episode
):

    obs_case = p_anomaly_nolabel.loc[p_anomaly_nolabel['case:concept:name'] == case_id_for_processing].reset_index(drop=True)
    env = P_BEAR_RL(obs_case, p_actset, p_filtered_actset, p_NBGs, alpha=alpha)

    # training: num_epi episodes; test: at most 100 episodes, stopping at the first success (see below)
    episodes = p_num_epi if train_mode else 100 
    
    all_episode_histories = []

    for _ in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        game_over = False
        episode_history = []

        while not done and not game_over:
            # action selection: epsilon-greedy in both phases (greedy_only=0)
            action = agent.select_action(state, greedy_only=0)

            next_state, reward, done, game_over, info, loc, label_act, rework = env.step(action,
                max_step=max_step,
                deterministic=0
            )

            # Q-table update only in the training phase (train_mode=True)
            if train_mode:
                agent.update_q_value(state, action, reward, next_state)
                agent.decay_exploration_rate()

            steps += 1

            if label_act is not None:
                episode_history.append((state.copy(), env.action_labels[action].split('_')[0], loc, label_act, rework, reward, next_state.copy()))
            state = next_state

        if done and not game_over:
            adm_last = env.ADM(next_state, 0, p_NBGs)

            if len(adm_last) > 0 and len(adm_last[0]) > 0:
                score = np.sum(adm_last) / (len(adm_last) * len(adm_last[0]))
            else:
                score = 0

            all_episode_histories.append((steps - 1, score, episode_history))

            if not train_mode:
                break


    if len(all_episode_histories) > 0:
        check_clean = [h[0] for h in all_episode_histories]

        # among episodes with steps-1 >= 0, select those with the fewest steps
        positive_histories = [hist for hist in all_episode_histories if hist[0] >= 0]

        if len(positive_histories) > 0:
            min_step = min(h[0] for h in positive_histories)
            shortest_episodes = [hist for hist in positive_histories if hist[0] == min_step]
            best_episode = min(shortest_episodes, key=lambda x: x[1])

            if len(best_episode[2]) > 0:
                env.state = best_episode[2][-1][-1]
                env.state['predict_patterns'] = discover_pattern(best_episode[2])
            else:
                env.state = env.init_state.copy()
                env.state['predict_patterns'] = ''

            case_result_df = env.state.copy()
            rl_steps = best_episode[0]
            rl_success = 1
        else:
            state = env.init_state.copy()
            state['predict_patterns'] = ''
            case_result_df = state
            rl_steps = 0
            rl_success = 0
    else:
        state = env.init_state.copy()
        state['predict_patterns'] = ''
        case_result_df = state
        rl_steps = 0
        rl_success = 0

    # record metadata in the result (type, success flag, number of steps)
    case_result_df['type'] = case_type
    case_result_df['rl_steps'] = rl_steps
    case_result_df['rl_success'] = rl_success

    return case_result_df



class P_BEAR_RL:
    def __init__(self, case, actset, filtered_actset, NBGs, alpha):
        self.init_state = case.copy()
        self.state = case.copy()
        self.filtered_actset = filtered_actset
        self.actset = actset
        self.alpha = alpha
        self.action_space = list(range(len(filtered_actset) + 2))  # 0-1: remove, 2-: inject
        self.action_labels = ["remove_A", "remove_B"] + [f'inject_{item}' for item in filtered_actset]
        self.current_step = 0
        self.NBGs = NBGs

        adm_case = self.ADM(case, threshold=alpha, NBGs=NBGs)
        if len(adm_case) > 0:
            self.initial_anomalies = np.sum(adm_case, axis=0)
        else:
            self.initial_anomalies = np.array([])

    def ADM(self, obs_case, threshold, NBGs):
        actlist = obs_case['concept:name'].tolist()
        invalid_act = [act for act in actlist if act not in self.actset]

        v_matrix = []
        for order, act in enumerate(actlist):
            if act not in invalid_act:
                edge_inTrace = [
                    tuple([actlist[temp] + '_' + str(temp - order),
                           actlist[temp + 1] + '_' + str(temp + 1 - order)])
                    for temp in range(len(actlist) - 1)
                ]

                voting_NBG = NBGs[np.where(self.actset == act)[0][0]]
                edge_labels = nx.get_edge_attributes(voting_NBG, 'likelihood')

                valid_edge = []
                for edge, likelihood in edge_labels.items():
                    if likelihood > threshold:
                        valid_edge.append(edge)

                vote = [0 if temp in valid_edge else 1 for temp in edge_inTrace]
                past_vote = vote[0:order]
                post_vote = vote[order:len(vote)]

                past_vote = (np.cumsum(np.cumsum(past_vote[::-1])) == 1).astype(int)[::-1]
                post_vote = (np.cumsum(np.cumsum(post_vote)) == 1).astype(int)

                vote = past_vote.tolist() + post_vote.tolist()
                v_matrix.append(vote)

        return v_matrix

    def loc_anomalies(self, mat):
        adm_mat = self.ADM(obs_case=mat, threshold=self.alpha, NBGs=self.NBGs)
        if len(adm_mat) == 0:
            return []   
        anomalous_edge = np.sum(adm_mat, axis=0)
        indices_greater_than_one = np.where(anomalous_edge > 1)[0].tolist()
        return indices_greater_than_one

    def reset(self):
        self.state = self.init_state.copy()
        self.current_step = 0
        return self.state

    def _select_loc(self, indices_greater_than_one, mode='left'):
        # deterministic location rule (used only when deterministic=True)
        if len(indices_greater_than_one) == 0:
            return None

        if mode == 'left':
            return min(indices_greater_than_one)    # first anomaly location
        elif mode == 'right':
            return max(indices_greater_than_one)    # last anomaly location
        else:
            return min(indices_greater_than_one)

    def step(self, action, max_step=30, deterministic=False):   
        self.current_step += 1
        reward = -1
        done = False
        game_over = False
        rework = False
        idx = 0
        act = None

        prev_state = self.state.copy()
        indices_greater_than_one = self.loc_anomalies(prev_state)

        if len(indices_greater_than_one) == 0:
            done = True
            return self.state, reward, done, game_over, {}, idx, act, rework


        if action == 0:  # remove left-like
            if deterministic:
                loc = self._select_loc(indices_greater_than_one, mode='left')
            else:
                loc = random.sample(indices_greater_than_one, 1)[0]

            if loc is not None and loc > 0:
                act = prev_state['concept:name'][loc]
                idx = prev_state['order'][loc]
                repaired_case = prev_state.drop(index=loc).reset_index(drop=True)
                self.state = repaired_case

                if loc < len(prev_state) - 1 and act == prev_state['concept:name'][loc + 1]:
                    rework = True

        elif action == 1:  # remove right-like
            if deterministic:
                base_loc = self._select_loc(indices_greater_than_one, mode='left')
                loc = base_loc + 1 if base_loc is not None else None
            else:
                loc = random.sample(indices_greater_than_one, 1)[0] + 1

            if loc is not None and loc < len(prev_state) - 1:
                act = prev_state['concept:name'][loc]
                idx = prev_state['order'][loc]
                repaired_case = prev_state.drop(index=loc).reset_index(drop=True)
                self.state = repaired_case

                if loc > 0 and act == prev_state['concept:name'][loc - 1]:
                    rework = True

        elif action > 1:  # inject
            if deterministic:
                base_loc = self._select_loc(indices_greater_than_one, mode='left')
                loc = base_loc + 1 if base_loc is not None else None
            else:
                loc = random.sample(indices_greater_than_one, 1)[0] + 1

            if loc is not None and loc > 0:
                idx = prev_state['order'][loc - 1] + 1
                cut1 = prev_state.loc[0:(loc - 1)]
                cut2 = prev_state.loc[loc:len(prev_state)]
                insert = prev_state.loc[0:0].copy()
                insert['concept:name'] = self.filtered_actset[action - 2]
                act = self.filtered_actset[action - 2]

                repaired_case = pd.concat([cut1, insert, cut2]).reset_index(drop=True)
                self.state = repaired_case
                loc = loc - 1

        indices_greater_than_one_new = self.loc_anomalies(self.state)

        if len(indices_greater_than_one_new) < len(indices_greater_than_one):
            reward += 5 - self.current_step
        elif len(indices_greater_than_one_new) == len(indices_greater_than_one):
            reward -= 1
        else:
            reward -= 5

        if self.current_step > max_step:
            game_over = True
            reward -= 10

        if len(self.state) < 4:
            game_over = True

        done = self._is_done()
        return self.state, reward, done, game_over, {}, idx, act, rework

    def _is_done(self):
        indices_greater_than_one = self.loc_anomalies(self.state)
        return len(indices_greater_than_one) == 0

    def render(self):
        matrix_np = np.array(self.ADM(self.state, threshold=self.alpha, NBGs=self.NBGs))

        colors = ['white', 'red']
        cmap = ListedColormap(colors)

        plt.imshow(matrix_np, cmap=cmap, vmin=0, vmax=1)
        plt.title('Anomaly Detection Matrix')
        plt.colorbar(label='(0: normal, 1: anomaly)')
        plt.xticks([])
        plt.yticks([])
        plt.show()



class QLearningAgent:
    def __init__(
        self,
        filtered_actset,
        act_freq,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.5,
        exploration_decay_rate=0.0001
    ):
        self.filtered_actset = filtered_actset
        self.action_space = list(range(len(filtered_actset) + 2))  # 0-1: remove, 2-: inject
        self.act_freq = act_freq
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = {}

    def _get_state_key(self, state):
        return tuple(state.flatten())

    def get_q_value(self, state, action):
        state_key = self._get_state_key(np.array(state['concept:name']))
        return self.q_table.get((state_key, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        state_key = self._get_state_key(np.array(state['concept:name']))
        next_state_key = self._get_state_key(np.array(next_state['concept:name']))
        
        best_next_q = 0.0
        if any((next_state_key, a) in self.q_table for a in self.action_space):
            best_next_q = np.max([
                self.q_table.get((next_state_key, a), 0.0)
                for a in self.action_space
            ])
        
        old_q = self.q_table.get((state_key, action), 0.0)
        new_q = old_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - old_q
        )

        self.q_table[(state_key, action)] = new_q

    def select_action(self, state, greedy_only=False):

        if greedy_only:
            # greedy only: action with the highest Q-value (no exploration)
            q_values = [self.get_q_value(state, a) for a in self.action_space]
            return int(np.argmax(q_values))

        # default: epsilon-greedy
        if random.random() < self.exploration_rate:
            if random.random() < 0.5:
                return random.choice([0, 1])
            else:
                # sample the inject action by activity frequency
                return random.choices(
                    range(2, len(self.filtered_actset) + 2),
                    weights=self.act_freq.prob,
                    k=1
                )[0]
        else:
            q_values = [self.get_q_value(state, a) for a in self.action_space]
            return int(np.argmax(q_values))

    def decay_exploration_rate(self):
        self.exploration_rate = max(
            0.01,
            self.exploration_rate * (1 - self.exploration_decay_rate)
        )