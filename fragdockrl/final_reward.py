import os
import numpy as np
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import Manager
from rdkit import Chem
from rdkit.Chem import Descriptors
from .tether_dock import run_tethered_docking


def creator(q, data, num_sub_proc):
    """
    Create jobs for parallel processing.

    Puts the final molecule from each episode into queue `q` with its index,
    then adds 'DONE' signals for each subprocess to indicate completion.

    Parameters
    ----------
    q : multiprocessing.Queue
        Queue for inter-process communication.
    data : list
        List of episodes, each containing steps with the final molecule at index 2 of the last step.
    num_sub_proc : int
        Number of subprocesses; equals number of 'DONE' signals added.
    """
    for ep_idx, ep in enumerate(data):
        end_ep = ep[-1]
        m_end = end_ep[2]
        q.put((ep_idx, m_end))

    for i in range(0, num_sub_proc):
        q.put('DONE')


def worker(q, m_ref_com, tmp_dir, root_atom_idx_ref, cut_para_dict, smina_para_dict, return_dict):
    """
    Perform parallel docking and property calculations for molecules from a queue.

    Continuously processes molecules from queue `q`, calculates molecular properties,
    applies filtering based on cutoff parameters, runs tethered docking if criteria are met,
    and stores results in `return_dict`.

    Parameters
    ----------
    q : multiprocessing.Queue
        Queue from which jobs (episode index and molecule) are received.
    m_ref_com : rdkit.Chem.Mol
        Reference molecule with 3D conformation for docking.
    tmp_dir : str
        Temporary directory path for docking files.
    root_atom_idx_ref : int
        Root atom index in the reference molecule.
    cut_para_dict : dict
        Dictionary of cutoff parameters including 'num_rb', 'num_heavy_atoms', 'mol_wt',
        'timeout_docking', 'cut_score', and 'cut_rmsd'.
    smina_para_dict : dict
        Dictionary of smina parameters including 'smina_run',
        'prepare_ligand_run', and 'smina_config_file' .
    return_dict : multiprocessing.Manager().dict
        Shared dictionary for storing results keyed by episode index.

    Notes
    -----
    - Processing stops when the string 'DONE' is received from the queue.
    - Applies property-based filtering before docking.
    - Docking failures or excessive RMSD values are penalized with cutoff scores.
    """

    smina_run = smina_para_dict['smina_run']
    prepare_ligand_run = smina_para_dict['prepare_ligand_run']
    smina_config_file = smina_para_dict['smina_config_file']

    cut_num_rb = cut_para_dict['num_rb']
    cut_num_heavy_atoms = cut_para_dict['num_heavy_atoms']
    cut_mol_wt = cut_para_dict['mol_wt']
    timeout_docking = cut_para_dict['timeout_docking']
    cut_score = cut_para_dict['cut_score']
    cut_rmsd = cut_para_dict['cut_rmsd']

    pid = os.getpid()
    tmp_id = pid
    while True:
        qqq = q.get()
        if qqq == 'DONE':
            # print('proc =', os.getpid())
            break

        ep_idx = qqq[0]
        m_new = qqq[1]

        mol_wt = Descriptors.MolWt(m_new)
        logp = Descriptors.MolLogP(m_new)
        num_hd = Descriptors.NumHDonors(m_new)
        num_ha = Descriptors.NumHAcceptors(m_new)
        tpsa = Descriptors.TPSA(m_new)

        num_rb = Chem.rdMolDescriptors.CalcNumRotatableBonds(m_new)
        num_ring = Chem.rdMolDescriptors.CalcNumRings(m_new)
        num_heavy_atoms = m_new.GetNumHeavyAtoms()

        p_dict = {'mol_wt': mol_wt,
                  'num_rb': num_rb,
                  'logp': logp,
                  'num_hd': num_hd,
                  'num_ha': num_ha,
                  'num_ring': num_ring,
                  'tpsa': tpsa,
                  'num_heavy_atoms': num_heavy_atoms,
                  }

        if (num_rb > cut_num_rb) or (num_heavy_atoms > cut_num_heavy_atoms) or (mol_wt > cut_mol_wt):
            dock_score = cut_score
            p_dict['dock_score'] = dock_score
            p_dict['dock_RMSD'] = 9.9
            return_dict[ep_idx] = p_dict
            continue
        try:
            dock_score0, rmsd1 = run_tethered_docking(m_new, m_ref_com,
                                                      tmp_dir, tmp_id,
                                                      smina_run=smina_run,
                                                      smina_config_file=smina_config_file,
                                                      root_atom_idx_ref=root_atom_idx_ref,
                                                      prepare_ligand_run=prepare_ligand_run,
                                                      timeout_docking=timeout_docking)

            if rmsd1 > cut_rmsd:
                dock_score0 = cut_score
            dock_score = min(dock_score0, cut_score)

        except Exception as e:
            dock_score = cut_score
            rmsd1 = 9.9
        p_dict['dock_score'] = dock_score
        p_dict['dock_RMSD'] = rmsd1

        return_dict[ep_idx] = p_dict


def dock_ep_list(ep_list_batch, m_ref_com, tmp_dir, root_atom_idx_ref,
                 cut_para_dict, smina_para_dict, penelty_para_dict=None,
                 num_sub_proc=16):
    """
    Perform docking and property calculations on a batch of generated molecules with parallel processing.

    Parameters
    ----------
    ep_list_batch : list
        List of episodes, each containing synthesis steps and molecule data.
    m_ref_com : rdkit.Chem.Mol
        Reference molecule with 3D conformation for docking.
    tmp_dir : str
        Temporary directory path for docking files.
    root_atom_idx_ref : int
        Root atom index in the reference molecule.
    cut_para_dict : dict
        Dictionary containing cutoff parameters for docking and filtering.
    smina_para_dict : dict
        Dictionary of smina parameters including 'smina_run',
        'prepare_ligand_run', and 'smina_config_file' .
    penelty_para_dict : dict, optional
        Dictionary of penalty weights for properties including 'w_logp', 'w_mw', 'w_ha', and 'w_hd'.
        Defaults to zero weights if None.
    num_sub_proc : int, optional
       Number of subprocesses to use for parallel docking (default is 16).

    returns
    ------
    tuple
       - Updated ep_list_batch with modified rewards incorporating docking scores and penalties.
       - List of dictionaries with docking properties for each episode.
    """

    if penelty_para_dict is None:
        penelty_para_dict = {'w_logp': 0.0,
                             'w_mw': 0.0, 'w_ha': 0.0, 'w_hd': 0.0, }
    w_logp = penelty_para_dict['w_logp']
    w_mw = penelty_para_dict['w_mw']
    w_ha = penelty_para_dict['w_ha']
    w_hd = penelty_para_dict['w_hd']

    q1 = Queue()
    manager = Manager()
    return_dict = manager.dict()
    proc_master = Process(target=creator,
                          args=(q1, ep_list_batch, num_sub_proc))
    proc_master.start()

    procs = list()
    for sub_id in range(0, num_sub_proc):
        proc = Process(target=worker, args=(
            q1, m_ref_com, tmp_dir, root_atom_idx_ref, cut_para_dict,
            smina_para_dict, return_dict))
        procs.append(proc)
        proc.start()

    q1.close()
    q1.join_thread()

    proc_master.join()
    for proc in procs:
        proc.join()

    keys = sorted(return_dict.keys())
    dock_property_list = list()
    for ep_idx in range(0, len(ep_list_batch)):
        ep = ep_list_batch[ep_idx]
        reward_old = ep[-1][3]
        p_dict = return_dict[ep_idx]
        mol_wt = p_dict['mol_wt']
        logp = p_dict['logp']
        num_hd = p_dict['num_hd']
        num_ha = p_dict['num_ha']
        dock_score = p_dict['dock_score']
        dock_rmsd = p_dict['dock_RMSD']

        reward = reward_old - dock_score
        if w_logp > 0:
            reward += -w_logp*np.power((logp-2.0)/3.0, 2)
        if w_mw > 0:
            reward += -w_mw*np.power((mol_wt-350.0)/150.0, 2)
        if w_ha > 0:
            if num_ha > 8:
                p_ha = - (num_ha - 8.0)/3.0
            else:
                p_ha = 0
            reward += w_ha*p_ha
        if w_hd > 0:
            if num_ha > 4:
                p_ha = - (num_ha - 4.0)/2.0
            else:
                p_ha = 0
            reward += w_ha*p_ha

        ep[-1][3] = reward
        dock_property_list.append(p_dict)

    return ep_list_batch, dock_property_list
