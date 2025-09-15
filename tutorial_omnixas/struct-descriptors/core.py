import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from descriptors import utils
from pymatgen.analysis.bond_valence import BVAnalyzer

class DescriptorEnabledStructure(Structure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize descriptors
        self.clear()
    
    def from_structure(s):
            """A simple helper method for instantiating this class from another
            structure (as opposed to using the inherited from_file method, for
            example."""

            r = DescriptorEnabledStructure(s.lattice, s.species, s.frac_coords)
            assert r.lattice == s.lattice
            assert r.species == s.species
            assert (r.frac_coords == s.frac_coords).all()
            return r

    def clear(self):
        self.cn = None
        self.ocn = None
        self.bvs = None
        self.oxi_state = None
        self.bv_list = []
        self.Rstd = None
        self.min_oo_dist = None
        self.mean_oo_dist = None
        self.min_oo_angle = None
        self.max_oo_angle = None
        self.mean_oo_angle = None
        self.std_oo_angle = None
        self.q_list = []
        self.target_site = None
        self.target_site_index = None
        self.neighbor_list = []
        self.R_map = None
        self.B_map = None
        self.sphericity = None
        self.noise = None
        self.symmetry_operations = None

    def show_descriptors(self,verbose=False):
        self.descriptor_dict = {
            "OS": self.oxi_state,
            "CN": self.cn,
            "OCN": self.ocn,
            "BVS": self.bvs,
            "Rstd": self.Rstd,
            "MOOD": self.min_oo_dist,
            "NOOD": self.mean_oo_dist,
            "MOOA": self.min_oo_angle,
            "NNMA": self.max_oo_angle,
            "NOOA": self.mean_oo_angle,
            "Astd": self.std_oo_angle,
            "QVEC": self.q_list,
            "Sphericity": self.sphericity,
            "symtry_operation": self.symmetry_operations,
        }
        if verbose:
            print(f"Target site index: {self.target_site_index}")
            try:
                print(f"Neighbor indices: {[nbr.index for nbr in self.neighbor_list]}")
            except:
                print(f"Neighbor indices: {self.neighbor_list}")
            print(f"OS: {self.oxi_state}")
            print(f"CN: {self.cn}")
            print(f"OCN: {self.ocn}")
            print(f"BVS: {self.bvs}")
            print(f"Rstd: {self.Rstd}")
            print(f"MOOD: {self.min_oo_dist}")
            print(f"NOOD: {self.mean_oo_dist}")
            print(f"MOOA: {self.min_oo_angle}")
            print(f"NNMA: {self.max_oo_angle}")
            print(f"NOOA: {self.mean_oo_angle}")
            print(f"Astd: {self.std_oo_angle}")
            print(f"QVEC: {self.q_list}")
            print(f"Sphericity: {self.sphericity}")
        

    def update_R_map(self, R_value_dict):
        '''
        Example of input of its type is "dict":
            R_dict = {
                'anion': ['O','Cl','F', 'Br'], 
                'Ti2+': [None, 2.31, 2.15, 2.49], 
                'Ti3+': [1.791, 2.22, 1.723, None], 
                'Ti4+': [1.815, 2.19, 1.76, 2.36],
                'Ti9+': [1.79, 2.184, None, 2.32]
            }
        Example of input if its type is "pandas.core.frame.DataFrame":
                    anion	Ti2+	Ti3+	Ti4+	Ti9+
                0   	O	NaN	    1.791	1.815	1.790
                1	    Cl	2.31	2.220	2.190	2.184
                2	    F	2.15	1.723	1.760	NaN
                3	    Br	2.49	NaN	    2.360	2.320           
        '''
        if isinstance(R_value_dict, dict):
            R_df = pd.DataFrame(data=R_value_dict)
        elif isinstance(R_value_dict, pd.core.frame.DataFrame):
            R_df = R_value_dict
        else:
            raise "R_map failed. Please check the input format!"
        
        R_func_dict = {}
        for i, elem in enumerate(R_df['anion'].to_list()):
            oxi_states = []
            R_values = []
            for j, col in enumerate(R_df.columns.to_list()):    
                if col.startswith('anion') or np.isnan(R_df.iloc[i,j]):
                    continue
                oxi_states.append(float(''.join(filter(lambda e:e.isnumeric(),col))))
                R_values.append(R_df.iloc[i,j])
            if len(oxi_states)<=2:
                R_func_dict[elem] = np.poly1d(np.polyfit(oxi_states, R_values, 1))
            else:    
                R_func_dict[elem] = np.poly1d(np.polyfit(oxi_states, R_values, 2))

        self.R_map = lambda ion, guess: R_func_dict[ion](guess)

    def update_B_map(self,B_value_dict):
        self.B_map = lambda site, dist: 0.37

    
    def update_neighbor_list(self,site_index,method='simple', r_max=3.0, dr=3.0,
                         allowed_elements=["O", "F", "Cl", "Br", "S"], has_oxi_state=True):
        '''
        Update the target site and its neighbors, as well as reset descriptors (except for CN, which is the # of neighbors)

        Parameters:
        -----------
        method: string
            Method used to find neighbors for a given sites. Available options are:
                'simple': siple counting for a given cutoff radious. 
                'standard':
                'centroid':
                'CrystalNN': Use CrystalNN to find neighbors.
        has_oxi_state: bool
            If True, create oxidation state decorated structure by "bva.get_oxi_state_decorated_structure(structure)". 
            Default is True.
        allowed_elements: List[str]
            A list containing the symbols of elements to be considered as neighbors.
        '''
        self.target_site_index = site_index
        self.target_site = self.sites[self.target_site_index]
        self.neighbor_list, _ = utils.get_neighbors(
            self,
            site_index,
            method=method, 
            r_max=r_max, dr=dr,
            allowed_elements=allowed_elements,
            has_oxi_state=has_oxi_state
        )
        self.cn = len(self.neighbor_list)
        self.bvs = None
        self.bv_list = []
        self.q_list = []


    def update_descriptor_q(self, angular_qn_min=1, angular_qn_max=12, bvs_weight=False, method='Wei'):
        """
        Calculate q descriptor baesd on the current set of neighbors. Note always run "update_neighbor_list" 
        before this method if you are not sure. 
        
        Parameters:
        -----------
        bvs_weight: Bool
            If True, calculate q_vector weighted by bond valence sum of the target site. Default is False.
        method: string
            Method used to calculate bvs for the target site only if bvs_weight is True. Default value is 
            "Wei", which is the self-consistent way of calculating the bond valences. 
        """
        
        if len(self.neighbor_list)==0 or self.target_site_index is None:
            self.q_list = []
            print("Error: No neighbors found! q_vector is reset!")
            return
        site_index = self.target_site_index

        # whether q is weighted by bv_list
        if bvs_weight:
            if len(self.bv_list)==0: # Otherwise no need to calculate bvs again
                self.update_descriptor_bvs(method=method)
            bv_list = self.bv_list
        else:
            bv_list = []
        
        self.q_list = []
        neighbor_list = self.neighbor_list
        bond_angles = utils.calculate_bond_angles(self,site_index,neighbor_list)
        for order in range(angular_qn_min, angular_qn_max+1):
            q_value = utils.cal_q_l(order, bond_angles, bv_list=bv_list)
            self.q_list.append(q_value)
        self.q_list = np.array(self.q_list)


    def update_descriptor_bvs(self, method='Wei', alpha = 0.2, criteria=0.000001, guess=3.0, max_iter=10000):
        """
        Self-consistent calculation of the bond valence sum.
        Examples of input parameters:
            ions = ['O', 'O', 'Cl', 'Br']
            bond_lengths = [2.0, 2.3, 2.6, 2.9], in angstrom
            BVS_trial = 3, the initial BVS_trial, is not very critical
            criteria = 0.000001, convergence criteria
            components = False. If true, return BVS and bv_list, the list of 
                BV's where BVS = np.sum(bv_list)
        
        """
        if len(self.neighbor_list)==0 or self.target_site_index is None:
            self.bvs = None
            self.bv_list = []
            print("Error: No neighbors found! BVS is reset!")
        
        site_index = self.target_site_index
        neighbor_list = self.neighbor_list

        if method == 'Wei':
            assert not (self.R_map is None or self.B_map is None)
            self.bvs, self.bv_list, _ = utils.calculate_bvs_sc(self, site_index, neighbor_list,
                                                         R_map = self.R_map, B_map = self.B_map)
        elif method == 'Matminer':
            self.bvs, _ = utils.calculate_bvs_matminer(self, site_index, neighbor_list)
        else:
            self.bvs = None
            self.bv_list = []

    def update_cn(self):
        '''
        Update coordination number or the oxygen coordination number.
        '''
        if len(self.neighbor_list)==0 or self.neighbor_list is None:
            self.cn = None
            self.ocn = None
            print("Error: No neighbors found! ocn is reset!")
            return
        else:
            self.cn == len(self.neighbor_list)

    
    def update_ocn(
        self, 
        method='simple', 
        r_max=3.0, 
        has_oxi_state=False
    ):
        '''
        Update coordination number or the oxygen coordination number.
        '''
        if len(self.neighbor_list)==0 or self.neighbor_list is None:
            self.ocn = None
            print("Error: No neighbors found! q_vector is reset!")
            return

        self.ocn = utils.calculate_oxygen_cn(
            self, 
            self.neighbor_list, 
            method=method, 
            r_max=r_max,
            has_oxi_state=has_oxi_state
        )


    def update_rstd(self, return_bondlength=False):
        target_coord = self[self.target_site_index].coords
        coord_list = [nbr.coords-target_coord for nbr in self.neighbor_list]
        self.Rstd = utils.calculate_rstd(coord_list, return_bondlength=return_bondlength)


    def update_oo_dist(self, minimum=True):
        if len(self.neighbor_list) <= 1:
            return
        else:
            coord_list = [nbr.coords for nbr in self.neighbor_list]
            oo_dist = utils.calculate_oodist(coord_list, minimum=minimum)
            if minimum:
                self.min_oo_dist = round(oo_dist, 2)
            else:
                self.mean_oo_dist = round(oo_dist, 2)


    def update_angle(self, stats="mean"):
        """
        Parameters
        ----------
        reduce : str
            One of the following:
            `max`: calculate the maximum angle among all the 1st shell pairs
            `min`: calculate the minimum angle among all the 1st shell pairs
            `std`: calculate the standard deviation of minimum angle among all 
                   the 1st shell pairs
            `mean`: calculate the average of the minimum angles among all the 
                   1st shell pairs
        """
        if len(self.neighbor_list) <= 1:
            return
        else:
            idx = self.target_site_index
            oo_angle = utils.calculate_angle(self, idx, stats=stats)
            if stats == "min":
                self.min_oo_angle = round(oo_angle, 2)
            elif stats == "mean":
                self.mean_oo_angle = round(oo_angle, 2)
            elif stats == "std":
                self.std_oo_angle = round(oo_angle, 4)
            elif stats == "max":
                self.max_oo_angle = round(oo_angle, 2)

    def update_sphericity(self):
        coord_list = [nbr.coords for nbr in self.neighbor_list]
        self.sphericity = utils.calculate_sphericity(coord_list)

    def update_symmetry(self):
        self.symmetry_operations = len(utils.calculate_symmetry(self))
    
    def update_oxidation_state(self, bva=None):
        if bva is None:
            bva = BVAnalyzer()
        try:
            self.oxi_state = bva.get_valences(self)[self.target_site_index]
        except ValueError:
            self.oxi_state = None
