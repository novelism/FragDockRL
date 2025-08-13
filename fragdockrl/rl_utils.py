#!/usr/bin/env python

#import os
#import sys
import numpy as np
import torch
#import torch.nn as nn
#import torch.optim as optim
import copy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def cal_z_bb(net, device, bb_fp, batch_size_search):
    """
    Convert building block fingerprints (X space) into latent representations (Z space).

    Performs the transformation X -> Z using a given PyTorch model.

    Parameters
    ----------
    net : torch.nn.Module
        PyTorch model used for encoding fingerprints into latent space.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    bb_fp : torch.Tensor
        Torch tensor containing building block fingerprints (X space).
    batch_size_search : int
        Number of samples per batch during processing.

    Returns
    -------
    torch.Tensor
        Torch tensor containing latent representations (Z space) of the building blocks.
    """
    z_bb_list = list()
    num_mol_bb = bb_fp.shape[0]
    num_batch = int(np.ceil(num_mol_bb/batch_size_search))
    for idx_b in range(num_batch):
        ini = idx_b*batch_size_search
        fin = (idx_b+1)*batch_size_search
        x_batch = bb_fp[ini:fin].to(device)
        z_bb_batch = net.fo_i(x_batch).data
        z_bb_list.append(z_bb_batch)
    z_bb = torch.concatenate(z_bb_list)
    return z_bb


def cal_q(net, device, z_state, z_bb, batch_size_search):
    """
    Compute Q(state, action) values from latent representations Z_state and Z_action.

    Parameters
    ----------
    net : torch.nn.Module
        PyTorch model used to compute Q-values from latent representations.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    Z_state : torch.Tensor
        Torch tensor containing latent state representations.
    Z_bb : torch.Tensor
        Torch tensor containing latent action (building block) representations.
    batch_size : int
        Number of samples per batch during processing.

    Returns
    -------
    torch.Tensor
        Torch tensor containing Q(state, action) values.
    """
    num_mol_bb = z_bb.shape[0]
    num_batch = int(np.ceil(num_mol_bb/batch_size_search))
    y_list = list()
    z_state_batch = z_state.repeat(batch_size_search, 1)

    for idx_b in range(num_batch):
        ini = idx_b*batch_size_search
        fin = (idx_b+1)*batch_size_search
        z_bb_batch = z_bb[ini:fin].to(device)
        num_d = z_bb_batch.shape[0]
        z_state0 = z_state_batch[0:num_d]
        y = net.fo_f(z_state0, z_bb_batch)
        y_list.append(y.data)  # .cpu())
    y = torch.concatenate(y_list)
    return y


def get_action(m, net, device, z_state, z_bb, batch_size_search=256,
               temperature=1.0, eps=0.1, p_stop=0.1):
    """
    Select an action from the Q(S, A) function output.

    The action selection is performed using three strategies:
        - idx == 0: Terminate synthesis (probability = p_stop)
        - idx == 1: Select the action with the maximum Q-value
        - idx == 2: Select an action probabilistically according to Q-values

    Parameters
    ----------
    m : object
        Placeholder for molecule or related data (unused in current implementation).
    net : torch.nn.Module
        PyTorch model used to compute Q-values.
    device : torch.device
        Device on which to perform computation ('cpu' or 'cuda').
    z_state : torch.Tensor
        Latent state representation tensor.
    z_bb : torch.Tensor
        Latent action (building block) representation tensor.
    batch_size_search : int, optional
        Number of samples per batch during Q-value computation (default is 256).
    temperature : float, optional
        Temperature parameter for softmax action selection when using probabilistic strategy (default is 1.0).
    eps : float, optional
        Probability of selecting the probabilistic strategy (idx == 2) (default is 0.1).
    p_stop : float, optional
        Probability of terminating synthesis (idx == 0) (default is 0.1).

    Returns
    -------
    int
        Selected action index.
    """

    y = cal_q(net, device, z_state, z_bb, batch_size_search)

    p0 = torch.tensor([p_stop, eps, 1.0-p_stop-eps], dtype=torch.float32)
    ss = p0.multinomial(num_samples=1, replacement=True)
    if ss == 0:
        idx = 0
    elif ss == 1:
        idx = torch.argmax(y, axis=0)
    else:
        p = torch.softmax(y/temperature, dim=0)[:, 0]
        idx = p.multinomial(num_samples=1, replacement=True)

    return idx


def possible_reaction(m, reactant_id_dict, df_reaction):
    """
    Search for possible synthetic reactions for the given molecule.

    This function checks the given molecule against a dictionary of possible reactants
    and identifies reactions where the molecule matches one of the required reactants.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        RDKit Mol object representing the molecule to test.
    reactant_id_dict : dict
        Dictionary mapping reactant IDs to tuples:
        (reactant_smarts, RDKit Mol object, reaction_name).
    df_reaction : pandas.DataFrame
        DataFrame containing reaction information.

    Returns
    -------
    list of tuple
        List of tuples containing:
        (reactant_id, reaction_id, reactant_num, reactant_id_cc, reactant_num_cc, reaction_name)
        for reactions where the given molecule can participate.
    """

    reaction_key = reactant_id_dict.keys()
    c_reaction_list = list()
    product_list = list()

    for reactant_id in reaction_key:
        reactant_id2 = reactant_id.split('_')
        smarts_reactant_ref, m_reactant_ref, reaction_name = reactant_id_dict[reactant_id]
        match_atoms = m.GetSubstructMatch(m_reactant_ref)
        if len(match_atoms) == 0:
            continue

        reaction_id = int(reactant_id2[0])
        reactant_num = int(reactant_id2[1])

        if reactant_num == 1:
            reactant_num_cc = 2
        elif reactant_num == 2:
            reactant_num_cc = 1
        reactant_id_cc = '%d_%d' % (reaction_id, reactant_num_cc)
        if reactant_id_cc not in reaction_key:
            continue
        c_reaction_list.append((reactant_id, reaction_id, reactant_num,
                               reactant_id_cc, reactant_num_cc, reaction_name))
    return c_reaction_list


def run_step(m, action_id, c_reaction_list, df_bb, df_reaction, mol_bb_dict,
             penalty_score=-2.0):
    """
    Perform a single reinforcement learning reaction step to generate a new molecule.

    The function applies the selected reaction (action_id) to the current molecule.
    If the reaction fails or produces multiple distinct products, a penalty is applied.
    If `action_id` is 0, the process terminates without modifying the molecule.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        Current molecule to which the reaction will be applied.
    action_id : int
        Selected reaction or building block ID.
    c_reaction_list : list of tuple
        List of candidate reaction tuples:
        (reactant_id, reaction_id, reactant_num, reactant_id_cc, reactant_num_cc, reaction_name).
    df_bb : pandas.DataFrame
        DataFrame containing building block information, indexed by action_id.
    df_reaction : pandas.DataFrame
        DataFrame containing reaction templates (including 'smirks' column).
    mol_bb_dict : dict
        Dictionary mapping building block IDs to RDKit Mol objects.
    penalty_score : float, optional
        Reward penalty for failed reactions (default is -2.0).

    Returns
    -------
    tuple
        (m_new, reward, done, status_code) where:
            m_new : rdkit.Chem.Mol
                Resulting molecule after the reaction step.
            reward : float
                Reward value for the step (0 for success, penalty for failure).
            done : bool
                True if synthesis is terminated, otherwise False.
            status_code : int
                Indicator of outcome:
                    -1 : Synthesis terminated (stop)
                     0 : No product generated
                     1 : Single product generated
                     2 : Multiple products generated (failure)
    """

    done = False
    if action_id == 0:
        m_new = copy.copy(m)
        reward = 0
        done = True
        return (m_new, reward, done, -1)  # 'stop'
    else:
        m_cc = mol_bb_dict[action_id]
        m_new_list = list()
        product_list = list()
        tmp_list = list()
        for c_reaction in c_reaction_list:
            (reactant_id, reaction_id, reactant_num, reactant_id_cc,
             reactant_num_cc, reaction_name) = c_reaction

            if df_bb.at[action_id, reactant_id_cc]:
                smirks = df_reaction.at[reaction_id, 'smirks']
                rxn = AllChem.ReactionFromSmarts(smirks)
                if reactant_num == 1:
                    m_new_list = rxn.RunReactants((m, m_cc))
                else:
                    m_new_list = rxn.RunReactants((m_cc, m))

            for m_t1 in m_new_list:
                for m_new_m in m_t1:
                    smi_tmp = Chem.MolToSmiles(m_new_m)
                    if smi_tmp not in tmp_list:
                        tmp_list.append(smi_tmp)
                        product_list.append(m_new_m)
        if len(product_list) == 0:
            reward = penalty_score
            m_new = m
            return (m_new, reward, done, 0)  # 'none product'

        if len(product_list) == 1:
            try:
                m_new = Chem.RemoveAllHs(product_list[0])
                reward = 0
            except Exception as e:
                reward = penalty_score
                m_new = m
            done = False
            return (m_new, reward, done, 1)  # 'single product'
        else:
            reward = penalty_score
            m_new = m
            return (m_new, reward, done, 2)  # 'multi product'


def run_ep(m, net, device, bb_fp, batch_size_search, df_reaction, df_bb,
           reactant_id_dict, mol_bb_dict, idx_bb_dict,
           temperature=1.0, eps=0.1, p_stop=0.2,
           max_step=5, penalty_score=-2.0):
    """
    Run a single reinforcement learning episode of molecular synthesis.

    This function iteratively applies synthesis steps to a starting molecule `m`
    using a reinforcement learning policy defined by `net`. At each step,
    possible reactions are identified, candidate building blocks are selected,
    and the chosen action is applied via `run_step`. The episode ends when a stop
    action is selected, no further reactions are possible, or `max_step` is reached.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        Initial molecule for the episode.
    net : torch.nn.Module
        Neural network model used to compute latent embeddings and Q-values.
    device : torch.device
        Torch device ('cpu' or 'cuda') for model inference.
    bb_fp : torch.Tensor
        Building block fingerprints (X_bb) as a torch tensor.
    batch_size_search : int
        Batch size for model inference.
    df_reaction : pandas.DataFrame
        DataFrame containing reaction information (including SMIRKS).
    df_bb : pandas.DataFrame
        DataFrame containing building block presence/compatibility flags.
    reactant_id_dict : dict
        Dictionary mapping reactant IDs to tuples:
        (reactant_smarts, rdkit Mol object, reaction_name).
    mol_bb_dict : dict
        Mapping of building block IDs to RDKit Mol objects.
    idx_bb_dict : dict
        Mapping between building block index and ID.
    temperature : float, optional
        Temperature parameter for Boltzmann-based probabilistic action selection (default=1.0).
    eps : float, optional
        Probability of selecting the maximum Q-value action (default=0.1).
    p_stop : float, optional
        Probability of selecting the stop action (default=0.2).
    max_step : int, optional
        Maximum number of synthesis steps, including termination (default=5).
    penalty_score : float, optional
        Reward penalty for failed synthesis steps (default=-2.0).

    Returns
    -------
    list
        A list of episode steps, where each step is:
        [m, action_id, m_new, reward, possible_reaction_bb_bool, done, product_info, count_update]
        - m : RDKit Mol object before the step
        - action_id : Selected building block ID
        - m_new : RDKit Mol object after the step
        - reward : Step reward
        - possible_reaction_bb_bool : Boolean array of possible building blocks at the step
        - done : Boolean flag indicating episode termination
        - product_info : int
            0 = no product
            1 = single product
            2 = multiple products
            -1 = stop
        - count_update : Number of successful updates so far
    """

    ep_list = list()
    ep_step = 0
    done = False

    z_bb = cal_z_bb(net, device, bb_fp, batch_size_search)
    count_update = 0

    while True:
        ep_step += 1
        c_reaction_list = possible_reaction(m, reactant_id_dict, df_reaction)
        c_possible_reaction = [x[3] for x in c_reaction_list]
        df_bb_tmp = df_bb[c_possible_reaction].any(axis=1)
        possible_reaction_bb_bool = df_bb_tmp.values
        if count_update > 0:
            possible_reaction_bb_bool[0] = True
        z_bb_poss = z_bb[possible_reaction_bb_bool]
        poss_bb_idx_id = df_bb_tmp[df_bb_tmp].index

        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
        torch_bool = torch.tensor(fp, dtype=torch.float32)[None, :]
        z_state = net.fc_i(torch_bool.to(device))

        if ep_step >= max_step:
            action = 0
            action_id = 0
            done = True

        elif z_bb_poss.shape[0] == 0:
            action = 0
            action_id = 0
            done = True

        else:
            if count_update > 1:
                action = get_action(m, net, device, z_state, z_bb_poss,
                                    batch_size_search, temperature=temperature,
                                    eps=eps, p_stop=p_stop)
            else:
                action = get_action(m, net, device, z_state, z_bb_poss,
                                    batch_size_search, temperature=temperature,
                                    eps=0.0, p_stop=0.0)
            # action_id = idx_bb_dict[int(action)]
            action_id = poss_bb_idx_id[int(action)]

        results = run_step(m, action_id, c_reaction_list, df_bb,
                           df_reaction, mol_bb_dict, penalty_score=penalty_score)
        m_new, reward, done, product_info = results
        if product_info == 1:
            count_update += 1

        ep = [m, action_id, m_new, reward, possible_reaction_bb_bool,
              done, product_info, count_update]

        ep_list.append(ep)
        m = m_new

        if done:
            break

    return ep_list


def search_ep_batch(m_start, net, device, num_ep_batch, bb_fp,
                    batch_size_search, df_reaction, df_bb, reactant_id_dict,
                    mol_bb_dict, idx_bb_dict, temperature=1.0, eps=0.1,
                    p_stop=0.2, max_step=5, penalty_score=-2.0):
    """
    Generate a batch of new molecules by running multiple synthesis episodes.

    Parameters
    ----------
    m_start : rdkit.Chem.Mol
        Starting molecule for each episode.
    net : torch.nn.Module
        Neural network model used for synthesis policy.
    device : torch.device
        Device ('cpu' or 'cuda') for model inference.
    num_ep_batch : int
        Number of episodes to generate in the batch.
    bb_fp : torch.Tensor
        Building block fingerprints tensor.
    batch_size_search : int
        Batch size for model inference.
    df_reaction : pandas.DataFrame
        DataFrame containing reaction templates.
    df_bb : pandas.DataFrame
        DataFrame containing building block information.
    reactant_id_dict : dict
        Dictionary mapping reactant IDs to reaction information.
    mol_bb_dict : dict
        Dictionary mapping building block IDs to RDKit Mol objects.
    idx_bb_dict : dict
        Mapping between building block index and ID.
    temperature : float, optional
        Temperature for probabilistic action selection (default=1.0).
    eps : float, optional
        Probability of selecting max Q-value action (default=0.1).
    p_stop : float, optional
        Probability of stop action selection (default=0.2).
    max_step : int, optional
        Maximum number of synthesis steps per episode (default=5).
    penalty_score : float, optional
        Reward penalty for failed synthesis (default=-2.0).

    Returns
    -------
    list
        List of episode results generated in the batch.
    """

    ep_list_batch = list()

    for i in range(num_ep_batch):
        m = copy.copy(m_start)
        ep = run_ep(m, net, device, bb_fp, batch_size_search, df_reaction,
                    df_bb, reactant_id_dict, mol_bb_dict, idx_bb_dict,
                    temperature=temperature, eps=eps, p_stop=p_stop,
                    max_step=max_step, penalty_score=penalty_score)
        ep_list_batch.append(ep)
    return ep_list_batch


def train(shot_list, net, net_2, device, bb_fp, bb_idx_dict, batch_size_train,
          batch_size_search, optimizer, loss_function, gamma, max_iter=4):
    """
    Train the Q-function using a list of training snapshots.

    Parameters
    ----------
    shot_list : list
        List of training snapshots, each containing state, action, reward, and related info.
    net : torch.nn.Module
        Neural network model to be trained (Q-function approximator).
    net_2 : torch.nn.Module
        Target network used for stable Q-value estimation; updated from `net` after each iteration.
    device : torch.device
        Device ('cpu' or 'cuda') for computation.
    bb_fp : torch.Tensor
        Fingerprints of building blocks.
    bb_idx_dict : dict
        Mapping from building block IDs to their indices in `bb_fp`.
    batch_size_train : int
        Batch size for training iterations.
    batch_size_search : int
        Batch size used for Q-value computation inside the network.
    optimizer : torch.optim.Optimizer
        Optimizer for neural network parameter updates.
    loss_function : callable
        Loss function to minimize (e.g., MSELoss).
    gamma : float
        Discount factor for future rewards.
    max_iter : int, optional
        Maximum number of training iterations over the dataset (default is 4).

    Returns
    -------
    list
        List of tuples with training iteration number and corresponding loss value.
    """

    num_snap_shot = len(shot_list)
    idx_shot = np.arange(num_snap_shot)
    np.random.shuffle(idx_shot)
    loss_list = list()

    max_iter0 = min(max_iter, int(np.ceil(num_snap_shot / batch_size_train)))
    for iteration in range(0, max_iter0):

        ini = iteration*batch_size_train
        fin = (iteration+1)*batch_size_train
        idx_sample = idx_shot[ini:fin]
        q_new_max_list = list()
        x_state_list = list()
        x_action_list = list()
        reward_list = list()

        z_bb = cal_z_bb(net_2, device, bb_fp, batch_size_search)

        for idx_s in idx_sample:
            shot0 = shot_list[idx_s]
            m = shot0[0]
            bb_id = shot0[1]
            bb_idx = bb_idx_dict[bb_id]
            bb_fp0 = bb_fp[bb_idx]
            m_new = shot0[2]
            reward = shot0[3]
            possible_reaction_bb_bool_next = shot0[4]
            done = shot0[5]
#            info = shot0[6]
#            count_update = shot0[7]
#            ep_idx = shot0[8]
#            i_step = shot0[9]
#            count_train = shot0[10]
#            r_step = shot0[11]

            z_bb_poss = z_bb[possible_reaction_bb_bool_next]
            if done:
                q_new_max = 0
            elif z_bb_poss.shape[0] == 0:
                q_new_max = 0
            else:

                fp_new = AllChem.GetMorganFingerprintAsBitVect(
                    m, radius=2, nBits=1024)
                x_state_new = torch.tensor(
                    fp_new, dtype=torch.float32)[None, :]
                z_state_new = net_2.fc_i(x_state_new.to(device))

                q_new = cal_q(net_2, device, z_state_new,
                              z_bb_poss, batch_size_search)

                q_new_max = torch.max(q_new)
            q_new_max_list.append([q_new_max])

            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
            x_state_list.append(fp)
            x_action_list.append(bb_fp0[None, :])

            reward_list.append([reward])

        reward = torch.tensor(reward_list, dtype=torch.float32).to(device)
        q_new_max = torch.tensor(
            q_new_max_list, dtype=torch.float32).to(device)
        x_state = torch.tensor(x_state_list, dtype=torch.float32).to(device)
        x_action = torch.concat(x_action_list, axis=0).to(device)
        qsa = net.forward(x_state, x_action)

        optimizer.zero_grad()
        loss = loss_function(qsa, reward + gamma*q_new_max)
        loss.backward(retain_graph=True)
        optimizer.step()

        net_2.load_state_dict(net.state_dict())

        loss_list.append([iteration, loss.data.cpu()])
#        print(iteration, loss.data.cpu())
    return loss_list


def test(shot_list, net, device, bb_fp, bb_idx_dict, batch_size_train,
         batch_size_search, loss_function, gamma, max_iter=4):
    """
    Test the accuracy of the trained Q-function using a list of snapshots.

    Parameters
    ----------
    shot_list : list
        List of testing snapshots containing states, actions, rewards, and related info.
    net : torch.nn.Module
        Trained neural network model (Q-function approximator).
    device : torch.device
        Device ('cpu' or 'cuda') for computation.
    bb_fp : torch.Tensor
        Fingerprints of building blocks.
    bb_idx_dict : dict
        Mapping from building block IDs to their indices in `bb_fp`.
    batch_size_train : int
        Batch size for evaluation.
    batch_size_search : int
        Batch size used for Q-value computation inside the network.
    loss_function : callable
        Loss function (not used inside this function but kept for interface consistency).
    gamma : float
        Discount factor for future rewards.
    max_iter : int, optional
        Number of iterations for evaluation (default=4).

    Returns
    -------
    tuple of torch.Tensor
        - qsa: Q-values predicted by the network.
        - reward: Actual rewards from the snapshots.
        - q_new_max: Maximum predicted Q-values for the next states.
    """

    num_snap_shot = len(shot_list)
    idx_shot = np.arange(num_snap_shot)
    np.random.shuffle(idx_shot)
    loss_list = list()

    max_iter0 = min(max_iter, int(np.ceil(num_snap_shot / batch_size_train)))
    for iteration in range(0, max_iter0):

        ini = iteration*batch_size_train
        fin = (iteration+1)*batch_size_train
        idx_sample = idx_shot[ini:fin]
        q_new_max_list = list()
        x_state_list = list()
        x_action_list = list()
        reward_list = list()

        z_bb = cal_z_bb(net, device, bb_fp, batch_size_search)

        for idx_s in idx_sample:
            shot0 = shot_list[idx_s]
            m = shot0[0]
            bb_id = shot0[1]
            bb_idx = bb_idx_dict[bb_id]
            bb_fp0 = bb_fp[bb_idx]
            m_new = shot0[2]
            reward = shot0[3]
            possible_reaction_bb_bool_next = shot0[4]
            done = shot0[5]
#            info = shot0[6]
#            count_update = shot0[7]
#            ep_idx = shot0[8]
#            i_step = shot0[9]
#            count_train = shot0[10]
#            r_step = shot0[11]

            z_bb_poss = z_bb[possible_reaction_bb_bool_next]
            if done:
                q_new_max = 0
            elif z_bb_poss.shape[0] == 0:
                q_new_max = 0
            else:

                fp_new = AllChem.GetMorganFingerprintAsBitVect(
                    m, radius=2, nBits=1024)
                x_state_new = torch.tensor(
                    fp_new, dtype=torch.float32)[None, :]
                z_state_new = net.fc_i(x_state_new.to(device))

                q_new = cal_q(net, device, z_state_new, z_bb_poss,
                              batch_size_search)

                q_new_max = torch.max(q_new)
            q_new_max_list.append([q_new_max])

            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
            x_state_list.append(fp)
            x_action_list.append(bb_fp0[None, :])

            reward_list.append([reward])

        reward = torch.tensor(reward_list, dtype=torch.float32)
        q_new_max = torch.tensor(
            q_new_max_list, dtype=torch.float32)
        x_state = torch.tensor(x_state_list, dtype=torch.float32).to(device)
        x_action = torch.concat(x_action_list, axis=0).to(device)
        qsa = net.forward(x_state, x_action).data.cpu()

    return qsa, reward, q_new_max
