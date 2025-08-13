import numpy as np
import pandas as pd
import pickle
import copy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
# from rdkit.Chem import Descriptors


def prep_ref_mol_simple(smi, m_ref):
    """
    Extract core structure from a molecular object including 3D conformation
    by performing Maximum Common Substructure (MCS) search.

    Parameters
    ----------
    core_smiles : str
        SMILES string of the core structure to be extracted.
    ref_mol : rdkit.Chem.Mol
        RDKit Mol object of the reference molecule, must include conformer information.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit Mol object representing the extracted core structure.
    """
    m = Chem.MolFromSmiles(smi)
    mcs = rdFMCS.FindMCS(
        [m, m_ref], ringMatchesRingOnly=True, completeRingsOnly=True)
    smarts = mcs.smartsString
    mcs_query = Chem.MolFromSmarts(smarts)
    match_atoms_ref = m_ref.GetSubstructMatch(mcs_query)
    match_atoms_list = m.GetSubstructMatches(mcs_query, uniquify=False)
    num_models = len(match_atoms_list)

    m_h = Chem.AddHs(m)
    conf_ref = m_ref.GetConformer()
    positions_ref = conf_ref.GetPositions()
    num_atoms_mcs = len(match_atoms_ref)

    for i_model in range(len(match_atoms_list)):
        coordmap = dict()
        for i in range(num_atoms_mcs):
            idx_m = match_atoms_list[i_model][i]
            idx_ref = match_atoms_ref[i]
            coordmap[idx_m] = conf_ref.GetAtomPosition(idx_ref)
        cids = AllChem.EmbedMultipleConfs(m_h, numConfs=1, numThreads=1,
                                          clearConfs=False,
                                          useRandomCoords=True,
                                          coordMap=coordmap)

    return m_h


def prep_ref_mol(smi, m_ref, m_smarts):
    """
    Align 3D coordinates of a reference molecule to a SMARTS core structure and apply the coordinates to the molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to be aligned.
    ref_mol : rdkit.Chem.Mol
        RDKit Mol object of the reference molecule, must include conformer information.
    m_smarts : rdkit.Chem.Mol
        RDKit Mol object converted from SMARTS representing the core structure.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit Mol object with aligned coordinates applied.
    """

    m_new = Chem.MolFromSmiles(smi)
    m_h = Chem.AddHs(m_new)
    m_h_new = copy.copy(m_h)
    AllChem.EmbedMolecule(m_h)

    match_atoms_list = m_h.GetSubstructMatches(m_smarts, uniquify=False)
    match_atoms_list_ref = m_ref.GetSubstructMatches(m_smarts, uniquify=False)

#    atom_map_list = list()
    match_list = list()
    for match_atoms in match_atoms_list:
        for match_atoms_ref in match_atoms_list_ref:
            atom_map = list()
            for idx in range(len(match_atoms)):
                atom_map.append((match_atoms[idx], match_atoms_ref[idx]))

            m_h0 = copy.copy(m_h)
            rmsd = Chem.rdMolAlign.AlignMol(m_h0, m_ref, atomMap=atom_map)
            match_list.append((m_h0, rmsd, atom_map))

    return match_list


def get_reactant_id_dict(df_reaction):
    """
    Generate a reaction ID dictionary from a DataFrame containing reaction SMIRKS.

    The dictionary format is:
        {reaction_id: (reactant_smarts, RDKit Mol object, reaction_name)}

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing reaction information in SMIRKS format.

    Returns
    -------
    dict
        Dictionary mapping reaction IDs to tuples of (reactant_smarts, RDKit Mol object, reaction_name).
    """

    reactant_id_dict = dict()
    for i in df_reaction.index:
        df0 = df_reaction.loc[i]
        smirks = df0['smirks']
        reaction_name = df0['name']
        reactant_s, product = smirks.split('>>')
        reactant_list = reactant_s.split('.')
        reactant1 = reactant_list[0]

        reactant_id = '%d_1' % i
        m1 = Chem.MolFromSmarts(reactant1)
        reactant_id_dict[reactant_id] = (reactant1, m1, reaction_name)

        if len(reactant_list) >= 2:
            reactant2 = reactant_list[1]
            reactant_id = '%d_2' % i
            reactant_id_dict[reactant_id] = reactant2
            m2 = Chem.MolFromSmarts(reactant2)
            reactant_id_dict[reactant_id] = (reactant2, m2, reaction_name)
    return reactant_id_dict


def load_reaction_data(building_block_file, reaction_file, m_bb_file):
    """
    Load preprocessed building block information from files and convert reaction DataFrame to a dictionary.
    The function also generates `reactant_id_dict` using `get_reactant_id_dict(df_reaction)`.

    Parameters
    ----------
    building_block_file : str
        Path to the building block file.
    reaction_file : str
        Path to the reaction file.
    m_bb_file : str
        Path to the molecular building block file.

    Returns
    -------
    tuple
        A tuple containing:
        - df_bb (pandas.DataFrame): DataFrame of building blocks.
        - df_reaction (pandas.DataFrame): DataFrame of reactions.
        - reactant_id_dict (dict): Dictionary mapping reactant IDs.
        - mol_bb_dict (dict): Dictionary mapping building block IDs to RDKit Mol objects.
    """

    df_bb = pd.read_pickle(building_block_file)
    df_reaction = pd.read_pickle(reaction_file)

    reactant_id_dict = get_reactant_id_dict(df_reaction)

    with open(m_bb_file, 'rb') as f:
        mol_bb_dict = pickle.load(f)

    return df_bb, df_reaction, reactant_id_dict, mol_bb_dict


def shot_from_ep(ep_list_batch, ep_tree_dict):
    """
    Extract training snapshots from episodes.

    Parameters
    ----------
    ep_list_batch : list
        List of episodes, where each episode is a tuple (episode_index, episode_steps).
    ep_tree_dict : dict
        Dictionary used to track the tree structure of building blocks during extraction.

    Returns
    -------
    list
        List of snapshots needed for training, each containing:
        (m, bb_id, m_new, reward, possible_reaction_bb_bool_next, done, product_info,
         count_update, ep_idx, i_step, count_train, r_step)
    """

    shot_list = list()
    count_train = 0
    for ep0 in ep_list_batch:
        ep_idx = ep0[0]
        ep_s = ep0[1]
        tmp_dict = ep_tree_dict
        num_step = len(ep_s)
        shot_end = ep_s[-1]
        count_end = shot_end[7]
        for i_step in range(num_step):
            shot = ep_s[i_step]
            m, bb_id, m_new, reward, possible_reaction_bb_bool, done, product_info, count_update = shot
            if done:
                r_step = count_end - count_update
            else:
                r_step = count_end - count_update + 1

            if i_step < len(ep_s)-1:
                ep_s_next = ep_s[i_step+1]
                possible_reaction_bb_bool_next = ep_s_next[4]
            else:
                possible_reaction_bb_bool_next = possible_reaction_bb_bool
            if bb_id not in tmp_dict:
                tmp_dict[bb_id] = dict()
            shot_list.append((m, bb_id, m_new, reward, possible_reaction_bb_bool_next,
                              done, product_info, count_update, ep_idx, i_step, count_train, r_step))

            if product_info == 1:
                tmp_dict = tmp_dict[bb_id]
    return shot_list


def extract_ep_simple(ep_list):
    """
    Extract simplified episode summaries and property lists from episode data.

    Parameters
    ----------
    ep_list : list
        List of episodes, where each episode contains:
        - episode index
        - list of episode steps
        - property dictionary with molecular descriptors and docking scores.

    Returns
    -------
    tuple
        - property_list : numpy.ndarray
            Array of episode properties including rewards and molecular descriptors.
        - simple_list : list of dict
            List of simplified episode dictionaries containing:
            index, step info, final reward, cumulative reward, and final SMILES string.
    """

    p_list = list()
    simple_list = list()
    for ep in ep_list:
        idx = ep[0]
        ep_s = ep[1]
        p_dict = ep[2]

        dock_score = p_dict['dock_score']
        dock_rmsd = p_dict['dock_RMSD']
        mol_wt = p_dict['mol_wt']
        num_rb = p_dict['num_rb']
        logp = p_dict['logp']
        num_hd = p_dict['num_hd']
        num_ha = p_dict['num_ha']
        num_ring = p_dict['num_ring']
        tpsa = p_dict['tpsa']
        num_heavy_atoms = p_dict['num_heavy_atoms']

        num_step = len(ep_s)

        simple_ep_list = list()

        cumulative_reward = 0
        for i_step in range(num_step):
            shot = ep_s[i_step]
            m, bb_id, m_new, reward, possible_reaction_bb_bool, done, product_info, check_update = shot
            cumulative_reward += reward
            check_update = False
            if product_info == 1:
                check_update = True
            simple_ep_list.append([bb_id, check_update])
        m_gg = ep[1][-1][0]
        smi_final = Chem.MolToSmiles(m_gg)
        final_reward = ep[1][-1][3]
        p_list.append([idx, cumulative_reward, final_reward, dock_score, dock_rmsd,
                      mol_wt, num_rb, logp, num_hd, num_ha, num_ring, tpsa, num_heavy_atoms])

        simple_dict = p_dict
        simple_dict['idx'] = idx
        simple_dict['ep'] = simple_ep_list
        simple_dict['final_reward'] = final_reward
        simple_dict['cumulative_reward'] = cumulative_reward
        simple_dict['SMILES'] = smi_final

        simple_list.append(simple_dict)
    property_list = np.array(p_list)
    return property_list, simple_list
