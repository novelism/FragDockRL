import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import subprocess


def gen_conf_ref(m_new_m):
    m_new = Chem.RemoveAllHs(m_new_m)
    m_new_h = Chem.AddHs(m_new)
    num_atoms_mcs = m_new_h.GetNumAtoms()
    coordmap = dict()
    conf0 = m_new_h.GetConformer()

    for i in range(num_atoms_mcs):
        pos = conf0.GetAtomPosition(i)
        if np.linalg.norm((pos.x, pos.y, pos.z)) == 0:
            continue
        coordmap[i] = pos
    cids = AllChem.EmbedMultipleConfs(m_new_h, numConfs=1, numThreads=1,
                                      clearConfs=True, useRandomCoords=True,
                                      coordMap=coordmap)
    return m_new_h


def gen_conf_ref_mol(m, m_ref):
    m_new = Chem.RemoveAllHs(m)
    m_h = Chem.AddHs(m_new)
    check_error = AllChem.EmbedMolecule(m_h)
    if check_error:
        return 'error'

    match = m_h.GetSubstructMatch(m_ref)
    atommap = [[x, i] for i, x in enumerate(match)]
    rmsd = Chem.rdMolAlign.AlignMol(m_h, m_ref, atomMap=atommap)

    return m_h, rmsd


def find_root_atom_idx(m_new, m_ref_com, match_idx=0, atom_idx=0):
    match_atoms_list = m_new.GetSubstructMatches(m_ref_com, uniquify=False)
    root_atom_idx = match_atoms_list[match_idx][atom_idx]
    return root_atom_idx


def pdbqt_to_flex(pdbqt0, flex_pdbqt):
    fp = open(pdbqt0)
    lines = fp.readlines()
    fp.close()

    fp_out = open(flex_pdbqt, 'w')
    residue_name = 'UNL'
    chain_id = ' '
    res_idx = '   1'
    line_begin = 'BEGIN_RES %s %s %s\n' % (residue_name, chain_id, res_idx)
    fp_out.write(line_begin)
    for line in lines:
        if line[0:7] == 'TORSDOF':
            continue
        fp_out.write(line)
    line_end = 'END_RES %s %s %s\n' % (residue_name, chain_id, res_idx)
    fp_out.write(line_end)
    fp_out.close()


def split_pdbqt_dict(pdbqt):
    fp = open(pdbqt)
    lines = fp.readlines()
    fp.close()

    model_dict = dict()

    for line in lines:
        if line[0:5] == 'MODEL':
            model_id = int(line[6:].strip())
            model_dict[model_id] = list()
            continue
        elif line[0:6] == 'ENDMDL':
            continue
        model_dict[model_id].append(line)
    return model_dict


def run_tethered_docking(m_new_m, m_ref_com, out_dir, mol_id='molid',
                         smina_run='smina', smina_config_file='config.txt',
                         root_atom_idx_ref=0, prepare_ligand_run='prepare_ligand4.py',
                         timeout_docking=120, use_ref_align=False):
    """
    Perform tethered docking using a docking molecule and a reference molecule.

    Parameters
    ----------
    m_new_m : rdkit.Chem.Mol
        RDKit Mol object of the molecule to be docked.
    m_ref_con : rdkit.Chem.Mol
        RDKit Mol object of the reference molecule.
    out_dir : str
        Output directory path.
    mol_id : str, optional
        Molecule ID (default is "molid").
    smina_run : str, optional
        Smina executable file name (default is "smina").
    smina_config_file : str, optional
        Smina configuration file name (default is "config.txt").
    root_atom_idx_ref : int, optional
        Root atom index in the reference molecule (default is 0).
    prepare_ligand_run : str, optional
        Prepare_ligand4 script name (default is "prepare_ligand4.py"; sometimes "prepare_ligand4" depending on system).
    timeout_docking : int, optional
        Maximum docking time in seconds (default is 120).
    use_ref_align : bool, optional
        Method to use reference coordinates. If True, re-search with MCS alignment;
        if False, maintain coordinates of non-zero atoms. It is recommended to keep False in the current version.

    Returns
    -------
    tuple
        Tuple containing:
        - Docking_Affinity (float): The docking affinity score.
        - RMSD (float): Root-mean-square deviation value.
    """

    if use_ref_align:
        m_new_h, rmsd = gen_conf_ref_mol(m_new_m, m_ref_com)
    else:
        m_new_h = gen_conf_ref(m_new_m)

    root_atom_idx = find_root_atom_idx(
        m_new_h, m_ref_com, match_idx=0, atom_idx=root_atom_idx_ref)

    test_pdb_file = out_dir + '/test_mol_%s.pdb' % mol_id
    test_pdbqt_file = out_dir + '/test_mol_%s.pdbqt' % mol_id
    Chem.MolToPDBFile(m_new_h, test_pdb_file, flavor=4)
#    prepare_ligand_run = 'prepare_ligand4.py'
    run_line_prepare_ligand = '%s -l %s -o %s -R %d' % (
        prepare_ligand_run, test_pdb_file, test_pdbqt_file, root_atom_idx)
    results = subprocess.check_output(run_line_prepare_ligand.split(),
                                      stderr=subprocess.STDOUT,
                                      timeout=10,
                                      universal_newlines=True)

    test_pdbqt_flex_file = out_dir + '/test_mol_flex_%s.pdbqt' % mol_id
    pdbqt_to_flex(test_pdbqt_file, test_pdbqt_flex_file)

    test_pdbqt_flex_out_file = out_dir + '/test_mol_flex_%s_out.pdbqt' % mol_id

    run_line_smina_flex = '%s --config %s --flex %s --out_flex %s --no_lig --exhaustiveness 1 --cpu 1' % (
        smina_run, smina_config_file,
        test_pdbqt_flex_file, test_pdbqt_flex_out_file)

    results_smina_flex = subprocess.check_output(run_line_smina_flex.split(),
                                                 stderr=subprocess.STDOUT,
                                                 timeout=timeout_docking,
                                                 universal_newlines=True)

    test_pdbqt_flex_out_01_file = out_dir + \
        '/test_mol_flex_%s_out_01.pdbqt' % mol_id
    flex_dock_model_dict = split_pdbqt_dict(test_pdbqt_flex_out_file)
    dock_model_id_list = sorted(flex_dock_model_dict.keys())
    dock_model1_id = dock_model_id_list[0]
    dock_model1 = flex_dock_model_dict[dock_model1_id]
    with open(test_pdbqt_flex_out_01_file, 'w')as fp_out:
        fp_out.write(''.join(dock_model1))

    test_pdbqt_flex_out_min_01_file = out_dir + \
        '/test_mol_flex_%s_out_min_01.pdbqt' % mol_id
    run_line_smina_minimize = '%s --config %s --ligand %s --out %s --minimize' % (
        smina_run, smina_config_file,
        test_pdbqt_flex_out_01_file,
        test_pdbqt_flex_out_min_01_file)

    results_smina = subprocess.check_output(run_line_smina_minimize.split(),
                                            stderr=subprocess.STDOUT,
                                            timeout=20,
                                            universal_newlines=True)

    dock_score = None
    rmsd1 = None
    lines = results_smina.split('\n')
    for line in lines:
        if line.startswith('Affinity'):
            dock_score = float(line.strip().split()[1])
        elif line.startswith('RMSD'):
            rmsd1 = float(line.strip().split()[1])

    return (dock_score, rmsd1)
