# %%
from functools import cache

import numpy as np
import torch
import yaml
from loguru import logger
from matgl import load_model
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.utils.cutoff import polynomial_cutoff
from omegaconf import DictConfig

from omnixas.data.xas import MaterialStructure
from omnixas.utils.constants import Element, SpectrumType


class M3GNetFeaturizer:
    def __init__(self, model=None, n_blocks=None):
        self.model = model or M3GNetFeaturizer._load_default_featurizer()
        self.model.eval()
        self.n_blocks = n_blocks or self.model.n_blocks

    def featurize(self, structure):
        graph_converter = Structure2Graph(self.model.element_types, self.model.cutoff)
        g, state_attr = graph_converter.get_graph(structure)

        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)

        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)

        with torch.no_grad():
            expanded_dists = self.model.bond_expansion(g.edata["bond_dist"])

            l_g = create_line_graph(g, self.model.threebody_cutoff)

            l_g.apply_edges(compute_theta_and_phi)
            g.edata["rbf"] = expanded_dists
            three_body_basis = self.model.basis_expansion(l_g)
            three_body_cutoff = polynomial_cutoff(
                g.edata["bond_dist"], self.model.threebody_cutoff
            )
            node_feat, edge_feat, state_feat = self.model.embedding(
                node_types, g.edata["rbf"], state_attr
            )

            for i in range(self.n_blocks):
                edge_feat = self.model.three_body_interactions[i](
                    g,
                    l_g,
                    three_body_basis,
                    three_body_cutoff,
                    node_feat,
                    edge_feat,
                )
                edge_feat, node_feat, state_feat = self.model.graph_layers[i](
                    g, edge_feat, node_feat, state_feat
                )
        return np.array(node_feat.detach().numpy())

    @cache
    @staticmethod
    def _load_default_featurizer():
        with open("config/paths.yaml") as f:
            paths = yaml.safe_load(f)
            path = DictConfig(paths).models.m3gnet
        logger.info(f"Loading m3gnet model from {path}")
        model = load_model(path).model
        model.eval()
        return model


class M3GNetSiteFeaturizer(M3GNetFeaturizer):
    def featurize(self, structure: MaterialStructure, site_index: int):
        return super().featurize(structure.root)[site_index]


# %%

if __name__ == "__main__":
    from tqdm import tqdm

    from omnixas.data import (
        ElementSpectrum,
        MLData,
    )
    from omnixas.utils import DEFAULTFILEHANDLER

    # elements, spectrum_type = ElementsVASP, SpectrumType.VASP
    # elements, spectrum_type = ElementsFEFF, SpectrumType.FEFF

    # for element in elements:

    spectrum_type = SpectrumType.VASP
    for element in [Element.Ti]:
        spectra = DEFAULTFILEHANDLER().fetch_serialized_objects(
            ElementSpectrum,
            element=element,
            type=spectrum_type,
        )

        def save_ml_data(
            spectrum,
            file_handler=DEFAULTFILEHANDLER(),
            featurizer=M3GNetSiteFeaturizer(),
        ):
            index = 0 if spectrum.type == SpectrumType.VASP else spectrum.index
            features = featurizer.featurize(spectrum.material.structure, index)
            ml_data = MLData(X=features, y=np.array(spectrum.intensities))
            file_handler.serialize_json(
                ml_data,
                supplemental_info={
                    **spectrum.dict(),
                    "index_string": spectrum.index_string,
                },
            )

        for spectrum in tqdm(list(spectra), desc=f"Featurizing {element}"):
            save_ml_data(spectrum)

# %%
