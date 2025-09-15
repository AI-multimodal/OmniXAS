from cmath import pi
import os, glob
import numpy as np
import itertools
import math
from scipy.special import sph_harm
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matminer.featurizers.structure import GlobalInstabilityIndex
from matminer.featurizers.site.bonding import AverageBondAngle
from matminer.featurizers.utils.stats import PropertyStats
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.chemenv.coordination_environments\
        .coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments\
    .chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments\
    .structure_environments import LightStructureEnvironments as LSE
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



bva = BVAnalyzer()
gii = GlobalInstabilityIndex(r_cut=4.0, disordered_pymatgen=False)
strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()


class BondAngle(AverageBondAngle):
    
    """Rewrite this part so it returns either minimum, average angle or standard
    deviation of angle.
    """
    def __init__(self, method):
        super().__init__(method)
    
    def featurize(self, strc, idx, stats="mean"):
        """ Three options are: "mean", "minimum" and "deviation". By default, 
        it calculates average bond angle. Original code see: 
        https://github.com/hackingmaterials/matminer/blob/main/matminer/featurizers/site/bonding.py
        """
        
        # Compute nearest neighbors of the indexed site
        nns = self.method.get_nn_info(strc, idx)
        if len(nns) == 0:
            raise IndexError("Input structure has no bonds.")
        center = strc[idx].coords

        sites = [i["site"].coords for i in nns]

        # Calculate bond angles for each neighbor
        bond_angles = np.empty((len(sites), len(sites)))
        bond_angles.fill(np.nan)
        for i, i_site in enumerate(sites):
            for j, j_site in enumerate(sites):
                if j == i:
                    continue
                dot = np.dot(i_site - center, j_site - center) / (
                    np.linalg.norm(i_site - center) * np.linalg.norm(j_site - center)
                )
                if np.isnan(np.arccos(dot)):
                    bond_angles[i, j] = bond_angles[j, i] = np.arccos(round(dot, 5))
                else:
                    bond_angles[i, j] = bond_angles[j, i] = np.arccos(dot)
        # Take the minimum bond angle of each neighbor
        if stats == "max":
            max_bond_angles = np.nanmax(bond_angles, axis=1)
            return [PropertyStats.maximum(max_bond_angles)]
        else:
            min_bond_angles = np.nanmin(bond_angles, axis=1)
            if stats == "std":
                return [PropertyStats.std_dev(min_bond_angles)]
            elif stats == "min":
                return [PropertyStats.minimum(min_bond_angles)]
            elif stats == "mean":
                return [PropertyStats.mean(min_bond_angles)]
            else:
                raise Exception(f"Invalid statistic type: {stats}")


def cal_q_l(angular_qn, angles, bv_list=[]):
    """Computes an element of the q-vector order parameter.

    Parameters
    ----------
    angular_qn : int
        The angular momentum quantum number. It is a positive integer standing
        for the degree of the spherical harmonic function.
    angles : list[list[float]]
        A list of the [phi, theta] values. E.g. [[2.3, math.pi], [0.83, 0.0],
        [0.83, math.pi]]. Often also called [polar angle, azimuthal angle].
    bv_list: the list of bond valence of corresponding anions: BVS=np.sum(bv_list). 
        If none, then no weight is applied.

    Returns
    -------
    float
        The q-value for the specified angular quantum number.
    """
    if len(bv_list)==0:
        bv_list = np.ones(len(angles))
    else:
        bv_list = np.array(bv_list)
    assert len(bv_list) == len(angles)

    q_value_list = []
    for m in range(-angular_qn, angular_qn+1):
        sph_harm_list = []
        for i, angle in enumerate(angles):
            sph_harm_list.append(bv_list[i]*sph_harm(m, angular_qn, angle[1], angle[0]))
        sph_harm_sum_weighted = abs(np.sum(sph_harm_list)/np.sum(bv_list))**2.0
        q_value_list.append(sph_harm_sum_weighted)

    q_value = np.sqrt(np.sum(q_value_list)*4.0*np.pi/(2.0*angular_qn+1.0))
    return q_value

def calculate_bond_lengths(structure, site_index, neighbors):
    '''
    Calculate the bond lengths between the center atom and its neighbors.

    Parameters:
    -----------
    structure: pymatgen.core.structure.Structure
        The structure to work on
    site_index: int
        The index of the target site
    neighbors: Sequence[pymatgen.core.structure.PeriodicNeighbor]
        A list of neighbor sites around structure[site_index] found through either
        `structure.get_neighbors()` or `structure.get_neighbor_list()`.
    
    Returns:
    --------
    bond_lengths: Sequence[float]
        The bond lengths for all the neighboring sites
    '''
    
    center_coords = structure[site_index].coords
    
    bond_lengths = []
    for neighbor in neighbors:
        vector = neighbor.coords-center_coords
        distance = np.sqrt(np.sum(vector**2))
        bond_lengths.append(distance)
    assert len(bond_lengths) == len(neighbors)
    
    return bond_lengths

def calculate_bond_angles(structure, site_index, neighbors):
    '''
    Calculate bond angles of neighbors.
    
    Parameters:
    -----------
    structure: pymatgen.core.structure.Structure
        The structure to work on
    site_index: int
        The index of the target site
    neighbors: Sequence[pymatgen.core.structure.PeriodicNeighbor]
        A list of neighbor sites around structure[site_index] found through either
        `structure.get_neighbors()` or `structure.get_neighbor_list()`.
    
    Returns:
    --------
    bond_angles: Sequence[float]
        The bond angles for all the neighboring sites
    '''
    center_coords = structure[site_index].coords
    bond_angles = []
    for neighbor in neighbors:
        vector = neighbor.coords - center_coords        
        xy = np.linalg.norm(vector[:2])
        theta = np.arctan2(xy, vector[2])
        phi = np.arctan2(vector[1], vector[0])
        bond_angles.append([theta, phi])
    assert len(bond_angles) == len(neighbors)
    
    return bond_angles


def calculate_bvs_matminer(structure, site_index, neighbors):
    '''
    Calculate BVS from matminer function: calc_bv_sum()
    '''
    # make sure the structure has oxication state
    try:
        structure = bva.get_oxi_state_decorated_structure(structure)
    except:
        pass
    site_atom = structure[site_index] 

    neighbor_indices = [nb.index for nb in neighbors]
    error = 0 # error code: 0 -> success
    
    # get valences
    try:
        valences = bva.get_valences(structure)
    except ValueError:
        return None, 1 # error code: 1-> valence can't be assigned;
    valence_list = list(zip([e.symbol for e in structure.species], valences))
    if ('O',-1) in [valence_list[i] for i in neighbor_indices]:
        error=3 # error code: 3-> Oxygen has -1 oxidation state
        return None, error
    
    # calculate bvs
    try:
        bvs = gii.calc_bv_sum(valences[site_index],  # valence for site_index
                              str(site_atom.species.elements[0])[:2], # element for site_index
                              neighbors) # neighbors for site_index
    except ValueError as ex:
        error = 2 # error code:  2-> value not found in the table
        ex.args
        return None, error
    
    return bvs, error


def calculate_bvs_sc(structure, site_index, neighbor_list, guess=3,
                     R_map = lambda ion, value: 1, B_map=lambda ion, value: 0.37,
                     alpha=0.2, criteria=1e-6, current_iter=0, max_iter=1e5):
    '''
    Wei's self consistent way of calculating bvs
    '''
    neighbor_indices = [nb.index for nb in neighbor_list]
    ions = [str(structure.species[i]) for i in neighbor_indices]
    bond_lengths = calculate_bond_lengths(structure, site_index, neighbor_list)
    
    return  update_bvs_sc(ions, bond_lengths, R_map=R_map, B_map=B_map,
                          alpha=alpha,criteria=criteria,guess=guess, 
                          max_iter=max_iter)

    
def update_bvs_sc(anion_list, bond_lengths, guess=3, 
                    R_map = lambda ion, value: 1, B_map=lambda ion, value: 0.37,
                    alpha=0.2, criteria=0.000001,current_iter=0, max_iter=10000):
    """
    Calculate bond valence sum from bond lengths using iterative method untilr result converges.

    Examples of input parameters:
    anion_list = ['O', 'O', 'Cl', 'Br']
    bond_lengths = [2.0, 2.3, 2.6, 2.9], in angstrom
    guess = 3, the initial guess, is not very critical
    criteria = 0.000001, convergence criteria
    """

    # take the previous iteration and add 1 to be current iteration
    current_iter += 1 
    if current_iter > max_iter:
        BVS_new = None
        bv_list = None
        return None, [], []
    
    BVS = 0.0
    bv_list = []
    BVS_evolution = []
    for i, anion in enumerate(anion_list):
        try:
            bv = np.exp((R_map(anion, guess) - bond_lengths[i])/B_map(anion,guess)) 
        except:
            return None, [], []
        bv_list.append(bv)
    BVS = np.sum(bv_list)

    if abs(BVS - guess) > criteria: # iterate until converge
        guess = (BVS - guess)*alpha + BVS
        BVS_new, bv_list, BVS_evolution = update_bvs_sc(anion_list, bond_lengths, 
                                                        guess=guess, 
                                                        R_map = R_map, B_map=B_map,
                                                        criteria=criteria, 
                                                        current_iter=current_iter,
                                                        max_iter=max_iter)
    else:
        BVS_new = BVS
    BVS_evolution.append(BVS)


    return BVS_new, bv_list, BVS_evolution

def get_neighbors(
    structure, 
    site_index, 
    *,
    method='standard', 
    maximum_distance_factor=2.0,
    r_max=3.0,
    dr=3.0,
    has_oxi_state=False,
    allowed_elements=["O"]):
    '''
    Return a list of neighbors for a given site index in the structure.
    
    Parameters:
    -----------
    structure: pymatgen.core.Structure
        The structure to work with.
    site_index:
        The index of the site to find neighbors for.
    method: str
        The method used to count neighbor, see "get_neighbors()".
        Options are: 
            simple: simple cut-off method
            standard: local geometry finder (lgf) method with centering_type="standard"
            centroid: local geometry finder (lgf) method with centering_type="centroid"
            central_site: local geometry finder (lgf) method with centering_type="central_site"
    maximum_distance_factor: float, default 2.0.
        A factor used in local geometry finder method.
    has_oxi_state: Bool
        If True, add oxidation state to the structure. Default is False, it might deteriorate the 
        CN counting.
    allowed_elements: List[str] or "all"
        A list allowed elements for the neighbor to be calculated. If "all" then all neighbors are 
        allowed.
    r_max: float. Default 3.0.
        The cut-off radius for neighbor finding method. If `method` is "shell", `r_max` defines the 
        outer radius of the shell.
    dr: float. Default 3.0.
        Valid only if `method` is "shell". `dr` defines the thickness of the shell in which 
        neighbors are to be found. E.g. `r_max=3` and `dr=3` will define a shell between `r=0` and 
        `r=3`; `r_max=6` and `dr=4` defines a shell between `r=2` and `r=6`.
    
    Returns:
    --------
    neighbor_list: List[pymatgen.core.structure.PeriodicNeighbor]
        The list of neighboring sites.
    '''

    if has_oxi_state:
        try:
            structure_ = bva.get_oxi_state_decorated_structure(structure)
        except:
            structure_ = structure.copy()
            has_oxi_state = False
    else:
        structure.remove_oxidation_states()
        structure_ = structure.copy()
    target_site = structure_[site_index]
    
    if allowed_elements == "all":
        elements_list = ["O", "Cl", "F", "Br", "P", "S", "N", "B", "C", "Si"]
    else:
        elements_list = allowed_elements

    neighbors = []
    if method in ["simple", "shell"]:
        if method == "simple":
            neighbors = structure_.get_neighbors(r=r_max, site=target_site)
        elif method == "shell":
            neighbors = structure_.get_neighbors_in_shell(target_site.coords, r_max-dr/2, dr/2)
    
    elif method in ['centroid','central_site','standard']:
        lgf = LocalGeometryFinder() # needs to initialize 
        lgf.setup_structure(structure_)
        lgf.setup_parameters(centering_type=method, include_central_site_in_centroid=False)
        mdf = maximum_distance_factor
        
        # "se" is a structure_Environments object
        se = lgf.compute_structure_environments(
            maximum_distance_factor=mdf, only_cations=False, only_indices=[site_index]
        )
        # a lighter version of "se" obtained by applying a “strategy” on "se"
        lse = LSE.from_structure_environments(
            strategy=strategy, structure_environments=se
        )
        try:
            neighbors = lse.neighbors_sets[site_index][0].neighj_sites
        except IndexError:
            return [], 1  # error code: 1 -> no neighbors found with this strategy
        if max([nb.index for nb in neighbors])>len(structure_):
            return [], 2  # error code: 2 -> neighbor index out of structure range

    elif method == 'CrystalNN':
        # "https://pymatgen.org/pymatgen.analysis.local_env.html#
        # pymatgen.analysis.local_env.CrystalNN.NNData"
        cnn = CrystalNN()
        neighbors = [nbr['site'] for nbr in cnn.get_nn_info(structure_,site_index)]
    
    # Discard if neighbors list contains element other than allowed elements.
    neighbors_list_to_return = []
    for neighbor in neighbors:
        if neighbor == target_site: continue
        element_w_oxi_state = str(structure_.species[neighbor.index])
        element = ''.join(filter(lambda s:s.isalpha(),element_w_oxi_state))
        if (allowed_elements != "all") and (element not in elements_list):
            return [], 3  # contains element other than allowed element list
        neighbors_list_to_return.append(neighbor)

    return neighbors_list_to_return, 0 # error code: 0 -> normal exit
    


def calculate_oxygen_cn(structure, neighbor_list, method='simple', r_max=3.0, has_oxi_state=False):

    '''
    Return the averaged coordination number, standard deviation of the average oxygen-target distance, 
    as well as the minimum of oxygen-target distances of all oxygens that are neighbors of the absorber site.
    
    Parameters:
    -----------
    structure: pymatgen.core.Structure
        The structure to work with
    site_index: int
        The index of the obsorber site whose neighbors are given by "neighbor_list".
    neighbor_list: List[pymatgen.core.structure.PeriodicNeighbor]
        The list of neighboring sites.
    method: str
        The method used to count neighbor, see "get_neighbors()".
    anion_list: List[str], default ["O"]
        A list allowed elements for the neighbor to be calculated. If "all" then all anions are allowed.
    r_max: float. Default 3.0.
        The cut-off radius for neighbor finding method.
    has_oxi_state: Bool
        If True, add oxidation state to the structure. Default is False, it might deteriorate the CN counting.

    '''
    structure_ = structure.copy()
    
    try:
        oxygen_cn_list = []
        for nbr in neighbor_list:
            secondary_neighbors, _ = get_neighbors(
                structure_, 
                nbr.index, 
                method=method, 
                r_max=r_max,
                allowed_elements="all", 
                has_oxi_state=has_oxi_state
            )
            oxygen_cn_list.append(len(secondary_neighbors))
        oxygen_cn_avg = np.mean(oxygen_cn_list).tolist()
    except: 
        oxygen_cn_avg = None

    return oxygen_cn_avg


def calculate_oodist(coord_list, minimum=True):
    """Given a set of oxygen coordinates, calculate their minimum/mean distance.
    """
    
    oodist_list = []
    for pair in itertools.combinations(coord_list, 2):
        oodist_list.append(np.linalg.norm(pair[0]-pair[1]))
    if minimum:
        return np.min(oodist_list).tolist()
    else:
        return np.mean(oodist_list).tolist()
    

def calculate_angle(structure, site, stats="mean"):
    """Calculate angle pairwise for all the coordinates and take mimimum/average.
    """
    
    aba = BondAngle(CrystalNN())
    angle = aba.featurize(structure, site, stats=stats)[0]
    return angle * 180 / pi
    

def calculate_rstd(coord_list, return_bondlength=False):
    """ Coordinates of the neighbors centered at target site.
    """
    
    dist_list = []
    for coord in coord_list:
        dist_list.append(np.linalg.norm(coord))
    if return_bondlength:
        return np.mean(dist_list).tolist()
    else:
        return np.std(dist_list).tolist()


def calculate_spec(xanes, raw_grid, std_grid):
    '''
    xanes: xanes data defined on the original energy grid.
    raw_grid: the original grid.
    std_grid: the new grid xanes data is mapped to.
    
    return:
    -------
    std_spec: the new xanes data mapped on "std_grid"
    error: error code, 0-> success; 1-> error
    '''
    error = 0
    try:
        spec_fn = interp1d(raw_grid, xanes, kind='quadratic', 
                           bounds_error=False, 
                           fill_value=(xanes[0],xanes[-1]))
        std_spec = spec_fn(std_grid)
    except:
        error = 1
    return std_spec, error


def calculate_sphericity(coord_list):
    """
    Calculate the sphericity of a given set of coordinates.
    For the definition of convex hull, refer to https://en.wikipedia.org/wiki/Convex_hull
    For the definition of sphericity, refer to https://www.wikiwand.com/en/Sphericity
    """
    # breakpoint()
    try:
        ch = ConvexHull(coord_list)
        area, volume = ch.area, ch.volume
        sphericity = ((math.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3))) / area
    except Exception as ex:
        if "coplanar" in ex.args[0]:
            sphericity = 0.0
        else:
            sphericity = None
    return sphericity

def calculate_symmetry(structure):
    """
    See: https://pymatgen.org/pymatgen.symmetry.analyzer.html
    also: sga.get_space_group_number(), sga.get_crystal_system()
    """
    sga = SpacegroupAnalyzer(structure)
    so = sga.get_symmetry_operations()
    return so

if __name__ == "__main__":
    for a in np.linspace(0,3,31):
        coords = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, a]
        ])
        sphericity = calculate_sphericity(coords)
        print(f"{a:.1f}, {sphericity:.4f}")