import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure


def read_spectra(file_path):
    '''
    A function that reads "xmu.dat" in faff folder and returns a list of following arrays:
        e_list: energy grids
        mu_list: mu
        e_fermi: fermi energy
        mu_norm_factor:  
        mu0_list: mu0 
    Parameters:
    -----------
    file_path: str
        The path of the "xmu.dat" file
    
    Return:
    -------
        [
            e_list, # energy grids
            mu_list, # mu
            e_fermi, # fermi energy
            mu_norm_factor,
            mu0_list, # mu0 
        ]
    '''
    with open(os.path.join(file_path, "xmu.dat")) as f: 
        content = f.readlines()
    content = [x.strip() for x in content] 
    for i, line in enumerate(content):
        if 'used to normalize' in line:
            mu_norm_factor = float(line.split()[-1])
        if '#  omega' in line[:10]:
            #print('start_from_line: ', i+1)
            res = content[i+1:]
            break
    e_list = []
    mu_list = []
    mu0_list = []
    for line in res:
        e_list.append(float(line.split()[0]))
        mu_list.append(float(line.split()[3]))
        mu0_list.append(float(line.split()[4]))
        
        if abs(float(line.split()[2])) < 0.001:
            e_fermi = float(line.split()[0])

    return [e_list, mu_list, e_fermi, mu_norm_factor, mu0_list]


def feff_folder_to_dict(path, structure=False, feff=False, charge_transfer=False):
    '''
    Read feff data from database folder. Data include: pymatgen structure, feff spectra, charge transfer.

    Parameters:
    -----------
    path: str
        the absolute path of the database folder.
    structure: Bool
        If true, read pymatgen structure
    feff: Bool
        If true, read feff spectra including xafs and xanes
    charge_transfer: Bool
        If true, read charge transfer for the target site

    Return:
    ------
    data_feff: dict
        a dictionary of the following form as an example:
        {"mp-780188":{
            'py_structure': {pymatgen.core.Structure},
                  'feff_7': {
                      'charge_transfer': int, 
                      'feff_res_xanes': [e_list, mu_list, e_fermi, mu_norm_factor, mu0_list]), 
                      'feff_res_exafs': [same as above...])}, 
                  'feff_9': {...}, 
                  'feff_10':{...}
            },
         "mp-780183": {...},
         ...
        }
    
    '''
    data_feff = {}
    for mp_path in glob.glob(os.path.join(path, 'm*')):
        mp_id = mp_path.split('/')[-1]
        data_feff[mp_id] = {}
        if structure:
            py_str = Structure.from_file(os.path.join(mp_path, 'POSCAR'))
            data_feff[mp_id]['py_structure'] = py_str
        for feff_path in glob.glob(os.path.join(mp_path, 'feff*')):
            site_idx = int(feff_path.split('/')[-1].split('_')[1])
            data_feff[mp_id][f'feff_{site_idx}'] = {}
            if charge_transfer:
                log1_file = os.path.join(feff_path, 'XANES', 'log1.dat')
                with open(log1_file, 'tr') as f:
                    all_lines = f.readlines()
                    for i, l in enumerate(all_lines):
                        if l.startswith("Charge transfer:"):
                            ct = -1 * float(all_lines[i+1].split()[1]) # the opposite of # electron transfer
                data_feff[mp_id][f'feff_{site_idx}']['charge_transfer'] = ct
            if feff:
                # [e_list, mu_list, e_fermi, mu_norm_factor, mu0_list]
                try:
                    data_feff[mp_id][f'feff_{site_idx}']['feff_res_xanes'] = read_spectra(os.path.join(feff_path, 'XANES'))  
                except:
                    pass
                try:
                    data_feff[mp_id][f'feff_{site_idx}']['feff_res_exafs'] = read_spectra(os.path.join(feff_path, 'EXAFS'))
                except:
                    pass
    return data_feff


def feff_dict_to_list(feff_dict):
    mpid_list = []
    structure_list = []
    xanes_list = []
    raw_energy_grids = []
    ct_list = []

    for mpid, property_dict in feff_dict.items():
        
        if not isinstance(property_dict,dict): continue
        
        try:
            structure = property_dict['py_structure']
        except:
            continue

        for property_str, pdata in property_dict.items():
            if property_str.startswith('feff_'):
                site_index = int(property_str.split('_')[1])
                try:
                    charge_transfer = pdata['charge_transfer']
                except KeyError:
                    charge_tranfer = None
                try:
                    energy_grid, xanes = pdata['feff_res_xanes'][:2]
                except KeyError:
                    energy_grid, xanes = None, None
                if energy_grid is None: continue
                
                
                mpid_list.append((mpid, site_index))
                structure_list.append((structure))
                ct_list.append(charge_transfer)
                raw_energy_grids.append(energy_grid)
                xanes_list.append(xanes)

    xanes_list = np.stack(xanes_list)
    raw_energy_grids = np.stack(raw_energy_grids)
    ct_list = np.stack(ct_list)

    return mpid_list, structure_list, ct_list, raw_energy_grids, xanes_list


def concatenate_df(df_list):
    '''
    Concatenate different descriptor dataframes together and drop NA rows. So only those materials(sites)
    with all desriptors available will be kept.
    Parameters:
    -----------
    df_list: List[pandas.core.DataFrame]
        A list of datagrames that must have ['MPID','SITE'] as two-level index. This could be obtained by
        reading a csv dataframe file and set index: "pd.read_csv(fname).set_index(['MPID','SITE']"
    '''
    original_sizes = [len(df) for df in df_list]
    df_together = pd.concat(df_list,axis=1)
    df_together.replace(to_replace="None", value=np.NaN, inplace=True)
    df_together.dropna(inplace=True)
    df_together.reset_index(level=['MPID','SITE'],inplace=True)
    concatenated_size = len(df_together)
    return df_together, original_sizes, concatenated_size


def filter_boxplot_outliers(x, hi=True, lo=True):
    '''
    Remove boxplot outliers (those points beyond "caps") and make boxplot before and after removal.
    '''
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    value_dict = axs[0].boxplot(x)
    lower_cap, higher_cap = [w.get_ydata()[0] for w in value_dict['caps']]
    select_outlier = np.zeros_like(x, dtype=bool)
    if hi:
        select_outlier = select_outlier | (x > higher_cap)
    if lo:
        select_outlier = select_outlier | (x < lower_cap)
    print(select_outlier.sum(), len(x))
    _ = axs[1].boxplot(x[~select_outlier])
    return select_outlier