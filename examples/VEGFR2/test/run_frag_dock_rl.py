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

from fragdockrl import cal_frag_dock_rl, model

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
        'max_epoch': 400,  # 2 for test
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
