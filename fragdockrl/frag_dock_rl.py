#!/usr/bin/env python

import os
# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import copy
import pandas as pd
from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors
import time
import pickle

from . import final_reward, model, rl_utils, utils
# import tether_dock


def save_ep(ep_dir, p_list, ep_simple_list, shot_list, ep_tree_dict):

    score_list_file = ep_dir + '/score_list.npy'
    ep_file = ep_dir + '/ep_simple.pkl'
    snap_shot_file = ep_dir + '/snap_shot.pkl'
    ep_tree_file = ep_dir + '/ep_tree.pkl'

    np.save(score_list_file, p_list)

    with open(ep_file, 'wb') as f:
        pickle.dump(ep_simple_list, f)

    with open(snap_shot_file, 'wb') as f:
        pickle.dump(shot_list, f)

    with open(ep_tree_file, 'wb') as f:
        pickle.dump(ep_tree_dict, f)


def load_ep(ep_dir):
    score_list_file = ep_dir + '/score_list.npy'
    ep_file = ep_dir + '/ep_simple.pkl'
    snap_shot_file = ep_dir + '/snap_shot.pkl'
    ep_tree_file = ep_dir + '/ep_tree.pkl'

    p_list = np.load(score_list_file)

    with open(ep_file, 'rb') as f:
        ep_simple_list = pickle.load(f)

    with open(snap_shot_file, 'rb') as f:
        shot_list = pickle.load(f)

    with open(ep_tree_file, 'rb') as f:
        ep_tree_dict = pickle.load(f)

    return p_list, ep_simple_list, shot_list, ep_tree_dict


def cal_frag_dock_rl(params_dict):
    ligand_pdb_file = params_dict['ligand_pdb_file']
    start_smi = params_dict['start_smi']
    smi_ref_com = params_dict['smi_ref_com']
    ref_atom_idx_ref = params_dict['ref_atom_idx_ref']
    num_sub_proc = params_dict['num_sub_proc']
    ep_dir = params_dict['ep_dir']
    save_dir = params_dict['save_dir']
    tmp_dir = params_dict['tmp_dir']
    penalty_score = params_dict['penalty_score']
    max_step = params_dict['max_step']
    num_ep_batch = params_dict['num_ep_batch']
    batch_size_train = params_dict['batch_size_train']
    batch_size_search = params_dict['batch_size_search']
    gamma = params_dict['gamma']
    max_epoch = params_dict['max_epoch']
    temperature0 = params_dict['temperature0']
    temp_reduce = params_dict['temp_reduce']
    temperature_min = params_dict['temperature_min']
    cut_para_dict = params_dict['cut_para_dict']
    penelty_para_dict = params_dict['penelty_para_dict']
    ligand_pdb_file = params_dict['ligand_pdb_file']
    building_block_file = params_dict['building_block_file']
    reaction_file = params_dict['reaction_file']
    m_bb_file = params_dict['m_bb_file']
    bb_fp_pkl = params_dict['bb_fp_pkl']

    smina_para_dict = params_dict['smina_para_dict']
    device = params_dict['device']
    net = params_dict['net']
    net_2 = params_dict['net_2']
    optimizer = params_dict['optimizer']
    loss_function = params_dict['loss_function']

    if not os.path.exists(ep_dir):
        os.mkdir(ep_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    dd = utils.load_reaction_data(
        building_block_file, reaction_file, m_bb_file)
    df_bb, df_reaction, reactant_id_dict, mol_bb_dict = dd
#    reaction_key = reactant_id_dict.keys()
    df_bb_fp = pd.read_pickle(bb_fp_pkl)

    # Building block that does not react with any reaction template
    b_gg = df_bb[df_bb.columns[2:]].any(axis=1)
    b_gg.loc[0] = True   # except stop code
    df_bb_fp = df_bb_fp[b_gg]
    df_bb = df_bb[b_gg]

    bb_idx_dict = {x: i for i, x in enumerate(df_bb_fp.index)}
    idx_bb_dict = {i: x for i, x in enumerate(df_bb_fp.index)}
    bb_fp = torch.tensor(df_bb_fp.values, dtype=torch.float32)

    batch_size_search = min(batch_size_search, df_bb.shape[0])

    m_ref = Chem.MolFromPDBFile(ligand_pdb_file, removeHs=True)

    m_start_h = utils.prep_ref_mol_simple(start_smi, m_ref)
    m_ref_com_h = utils.prep_ref_mol_simple(smi_ref_com, m_ref)

    start_mol_file = 'mol_start.pdb'
    ref_mol_coor_file = 'mol_ref_coor.pdb'

    Chem.MolToPDBFile(m_start_h, start_mol_file)
    Chem.MolToPDBFile(m_ref_com_h, ref_mol_coor_file)

    m_start = Chem.RemoveHs(m_start_h)
    m_ref_com = Chem.RemoveHs(m_ref_com_h)

    # ep_list = list()
    ep_tree_dict = dict()
    shot_list = list()
    ep_simple_list = list()
    p_list = np.array([]).reshape(0, 13)
    # [idx, cumulative_reward, final_reward, dock_score, mol_wt, num_rb, logp, num_hd, num_ha, num_ring, tpsa, num_heavy_atoms]

    # temperature = temperature0 + temperature_min
    # print(temperature)

    fp_log = open('run_log.txt', 'a')
    for i_gen in range(0, max_epoch):

        temperature = temperature0 * \
            np.power(temp_reduce, i_gen) + temperature_min
        print("generation:", i_gen, temperature)
        line_out = 'generation: %d %.6f\n' % (i_gen, temperature)
        fp_log.write(line_out)

        st = time.time()
        start_idx = i_gen*num_ep_batch
        ep_list_batch0 = rl_utils.search_ep_batch(m_start, net, device, num_ep_batch,
                                                  bb_fp, batch_size_search, df_reaction,
                                                  df_bb, reactant_id_dict,
                                                  mol_bb_dict, idx_bb_dict,
                                                  temperature=temperature, eps=0.00, p_stop=0.0,
                                                  max_step=max_step,
                                                  penalty_score=penalty_score)
        et1 = time.time()
        print('search time:', et1-st)
        line_out = 'search time: %.3f\n' % (et1-st)
        fp_log.write(line_out)
        # Separate search code and property calculation code for parallelization.
        results = final_reward.dock_ep_list(ep_list_batch0, m_ref_com, tmp_dir,
                                            ref_atom_idx_ref, cut_para_dict,
                                            smina_para_dict=smina_para_dict,
                                            penelty_para_dict=penelty_para_dict,
                                            num_sub_proc=num_sub_proc)
        ep_list_batch2, dock_property_batch = results
        dock_score_batch = [x['dock_score'] for x in dock_property_batch]
        mean_dock_reward = -np.mean(dock_score_batch)
        print("dock_mean_reward:", i_gen, mean_dock_reward)
        line_out = 'dock_mean_reward: %d %.3f\n' % (i_gen, mean_dock_reward)
        fp_log.write(line_out)

        ep_list_batch = [[i+start_idx, ep_list_batch2[i],
                          dock_property_batch[i]] for i in range(len(ep_list_batch2))]

        p_list_batch, ep_simple_list_batch = utils.extract_ep_simple(
            ep_list_batch)

        p_list = np.concatenate([p_list, p_list_batch])
        ep_simple_list += ep_simple_list_batch
        shot_list_batch = utils.shot_from_ep(ep_list_batch, ep_tree_dict)
        shot_list += shot_list_batch

        print(len(shot_list))
        shot_list = shot_list[-40000:]

        et2 = time.time()
        print('docking time', et2-et1)
        line_out = 'docking time: %.3f\n' % (et2-et1)
        fp_log.write(line_out)

        # update_new_shot
        for r_step in range(max_step):
            if r_step == 0:
                num_repeat = 3
            elif r_step == 1:
                num_repeat = 2
            else:
                num_repeat = 1
            for i_rr in range(0, num_repeat):
                shot_list_r_step = [
                    x for x in shot_list_batch if x[11] <= r_step]
                loss_list = rl_utils.train(shot_list_r_step, net, net_2, device, bb_fp, bb_idx_dict,
                                           batch_size_train, batch_size_search, optimizer,
                                           loss_function, gamma, max_iter=4)
                print(r_step, i_rr, loss_list)
                line_out = 'loss_new_shot: %d %d ' % (r_step, i_rr)
                tmp_ll = list()
                for loss_g in loss_list:
                    tmp_ll.append('%.4f' % loss_g[1])
                line_out += ','.join(tmp_ll)
                line_out += '\n'
                fp_log.write(line_out)

        # update_shot
        if len(shot_list) < 2000:
            max_iter0 = 0
        elif len(shot_list) < 5000:
            max_iter0 = 1
        elif len(shot_list) < 10000:
            max_iter0 = 2
        elif len(shot_list) < 20000:
            max_iter0 = 3
        else:
            max_iter0 = 4
        if max_iter0 > 0:
            loss_list = rl_utils.train(shot_list, net, net_2, device, bb_fp, bb_idx_dict,
                                       batch_size_train, batch_size_search, optimizer,
                                       loss_function, gamma, max_iter=max_iter0)

            for loss_g in loss_list:
                print(loss_g[0], loss_g[1])
                line_out = 'loss_old_shot: %d %.4f\n' % (loss_g[0], loss_g[1])
                fp_log.write(line_out)

        torch.save(net, save_dir + '/net_%d.torch' % i_gen)
        save_ep(ep_dir, p_list, ep_simple_list, shot_list, ep_tree_dict)

        et3 = time.time()
        print('training time:', et3-et2)
        line_out = 'training time: %.3f\n' % (et3-et2)
        fp_log.write(line_out)

    fp_log.close()

    torch.save(net, save_dir + '/net.torch')


def main():

    protein_dir = '../fix'

# config.txt
#    protein_pdb_file = protein_dir + '/2P2IA_receptor.pdb'
#    protein_pdbqt_file = protein_dir + '/2P2IA_receptor.pdbqt'
    ligand_pdb_file = protein_dir + '/2P2IA_608.pdb'

    cut_para_dict = {
        'num_rb': 12,
        'num_heavy_atoms': 60,
        'mol_wt': 650,
        'timeout_docking': 120,
        'cut_score': 3,  # clip docking score
        'cut_rmsd': 2.5,
    }

    # penelty_para_dict = {'w_logp': 1.0, 'w_mw': 1.0, 'w_ha': 1.0, 'w_hd': 1.0,}
    penelty_para_dict = {'w_logp': 0.0, 'w_mw': 0.0, 'w_ha': 0.0, 'w_hd': 0.0}

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_num = torch.cuda.current_device()
        device = torch.device("cuda:%d" % device_num)
        torch.set_num_threads(2)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(14)

    print(device)

    net = model.Net(input_dim=1024, hidden_dim1=1024, latent_dim=512,
                    hidden_dim2=1024, output_dim=1, num_layer=2).to(device)
    net_2 = model.Net(input_dim=1024, hidden_dim1=1024, latent_dim=512,
                      hidden_dim2=1024, output_dim=1, num_layer=2).to(device)

    # net = torch.load('save/net.torch', weights_only=False).to(device)
    net_2.load_state_dict(net.state_dict())

    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    data_dir = '../data'
    building_block_file = data_dir + '/bb_reaction.pkl'
    reaction_file = data_dir + '/smirks_reactant.pkl'
    m_bb_file = data_dir + '/m_bb.pkl'
    bb_fp_pkl = data_dir + '/bb_fp_r2_1024.pkl'

    smina_para_dict = {
        'smina_run': 'smina',
        'prepare_ligand_run': 'prepare_ligand4',
        'smina_config_file': 'config.txt'
    }

    params_dict = {
        'start_smi': 'NCc1ccncc1',
        'smi_ref_com': 'Cc1ccncc1',
        'ref_atom_idx_ref': 0,
        'num_sub_proc': 16,
        'ep_dir': 'ep',
        'save_dir': 'save',
        'tmp_dir': 'tmp',
        'penalty_score': -2.0,  # step reward
        'max_step': 4,
        'num_ep_batch': 200,
        'batch_size_train': 256,
        'batch_size_search': 5096,
        'gamma': 0.9,
        'max_epoch': 400,
        'temperature0': 0.45,
        'temp_reduce': 0.99,
        'temperature_min': 0.05,
        'cut_para_dict': cut_para_dict,
        'penelty_para_dict': penelty_para_dict,
        'ligand_pdb_file': ligand_pdb_file,
        'building_block_file': building_block_file,
        'reaction_file': reaction_file,
        'm_bb_file': m_bb_file,
        'bb_fp_pkl': bb_fp_pkl,
        'smina_para_dict': smina_para_dict,

        'device': device,
        'net': net,
        'net_2': net_2,
        'optimizer': optimizer,
        'loss_function': loss_function,
    }
    cal_frag_dock_rl(params_dict)


if __name__ == '__main__':
    main()
