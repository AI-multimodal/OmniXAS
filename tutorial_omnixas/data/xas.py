# %%
from enum import StrEnum
from pathlib import Path
from typing import List, Optional, Self

import numpy as np
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    RootModel,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pymatgen.core import Structure as PymatgenStructure
from pymatgen.core.sites import PeriodicSite

from omnixas.core.periodic_table import Element, SpectrumType
from omnixas.utils.readable_enums import ReadableEnums


class MaterialIDPrefix(StrEnum):
    mp = "mp"
    mvc = "mvc"  # depricated in materials project


class MaterialID(RootModel, validate_assignment=True):
    """A class representing a unique identifier for materials.

    The MaterialID follows the format "prefix-number" where prefix must be one of
    the valid MaterialIDPrefix values and number should be an integer.

    Args:
        root (str): The material ID string in the format "prefix-number"

    Raises:
        ValueError: If the ID string is empty or doesn't follow the required format

    Examples:
        >>> from omnixas.data import MaterialID
        >>> material_id = MaterialID(root="MP-1234")
        >>> str(material_id)
        'MP-1234'
    """

    root: str = Field(..., description="Material ID")

    @field_validator("root")
    @classmethod
    def _check_id(cls, v):
        if not v:
            msg = "Material ID cannot be empty"
            logger.error(msg)
            raise ValueError(msg)
        prefix, id_str = v.split("-")
        if prefix not in [m.value for m in MaterialIDPrefix]:
            msg = f"Unexpected prefix {prefix} in material ID {v}"
            msg += f" Must be one of {[m.value for m in MaterialIDPrefix]}"
            logger.warning(msg)
        if not id_str.isdigit():
            msg = f"Expected integer after prefix in material ID {v}"
            logger.warning(msg)
        return v

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.root

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, other):
        return self.root == other.root


@ReadableEnums()
class EnergyGrid(RootModel, validate_assignment=True):
    """A class representing an energy grid with monotonically increasing values.

    The energy grid must contain at least one non-negative float value and all values
    must be strictly increasing.

    Args:
        root (List[float]): List of non-negative energy values

    Raises:
        ValueError: If the values are not monotonically increasing or contain
        negative values

    Examples:
        >>> grid = EnergyGrid(root=[0.0, 1.0, 2.0, 3.0])
        >>> len(grid)
        4
        >>> grid[0]
        0.0
    """

    root: List[NonNegativeFloat] = Field(None, min_length=1)

    @field_validator("root")
    @classmethod
    def _check_monotonic(cls, v):
        if not all(np.diff(v) > 0):
            msg = "Energies must be monotonically increasing"
            logger.error(msg)
            raise ValueError(msg)
        return v

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)


@ReadableEnums()
class IntensityValues(RootModel, validate_assignment=True):
    """A class representing intensity values for spectral data.

    The intensity values must be non-negative and contain at least one value.

    Args:
        root (List[float]): List of non-negative intensity values

    Raises:
        ValueError: If any intensity value is negative

    Examples:
        >>> intensities = IntensityValues(root=[0.5, 1.2, 0.8])
        >>> len(intensities)
        3
        >>> intensities[0]
        0.5
    """

    root: List[NonNegativeFloat] = Field(None, min_length=1)

    @field_validator("root")
    @classmethod
    def _check_non_negative(cls, v):
        if not all(i >= 0 for i in v):
            msg = "All intensities must be non-negative"
            logger.error(msg)
            raise ValueError(msg)
        return v

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)


@ReadableEnums()
class Spectrum(BaseModel, validate_assignment=True):
    """A class representing spectral data with corresponding energies and intensities.

    The spectrum must have matching lengths for energies and intensities if
    both are provided.

    Args:
        intensities (IntensityValues, optional): The intensity values of the spectrum
        energies (EnergyGrid, optional): The energy grid of the spectrum

    Raises:
        ValueError: If the lengths of energies and intensities don't match

    Examples:
        >>> from omnixas.data import EnergyGrid, IntensityValues
        >>> spectrum = Spectrum(
        ...     energies=EnergyGrid(root=[0.0, 1.0, 2.0]),
        ...     intensities=IntensityValues(root=[0.5, 1.2, 0.8])
        ... )
    """

    intensities: Optional[IntensityValues] = Field(None)
    energies: Optional[EnergyGrid] = Field(None)

    @model_validator(mode="after")
    def _check_length(cls, values):
        energies = values.energies
        intensities = values.intensities
        if energies is not None and len(energies) != len(intensities):
            msg = "Energies and intensities must have the same length"
            logger.error(msg)
            raise ValueError(msg)
        return values


@ReadableEnums()
class MaterialStructure(RootModel, validate_assignment=True):
    """A class representing the structure of a material using Pymatgen.

    Wraps a Pymatgen Structure object with additional functionality.

    Args:
        root (PymatgenStructure): The underlying Pymatgen structure

    Examples:
        >>> # Load structure from a CIF file
        >>> structure = MaterialStructure.from_file(Path("material.cif"))
        >>> # Access atomic sites
        >>> first_site = structure.sites[0]
    """

    root: PymatgenStructure = Field(None, description="Pymatgen structure")

    @classmethod
    def from_file(cls, path: Path):
        """Create a MaterialStructure from a structure file.

        Args:
            path (Path): Path to the structure file (supported formats include
            CIF, POSCAR, etc.)

        Returns:
            MaterialStructure: A new MaterialStructure instance
        """
        return cls(root=PymatgenStructure.from_file(path))

    @field_serializer("root")
    def serialize_structure(self, pymatgen_structure):
        return pymatgen_structure.as_dict()

    @property
    def sites(self):
        return self.root.sites

    def __eq__(self, other):
        return self.root.as_dict() == other.root.as_dict()


@ReadableEnums()
class Material(BaseModel):
    """A class representing a material with its identifier and optional structure.

    Args:
        id (MaterialID): The unique identifier for the material
        structure (MaterialStructure, optional): The crystal structure of the material

    Examples:
        >>> material = Material(
        ...     id=MaterialID(root="MP-1234"),
        ...     structure=MaterialStructure.from_file(Path("structure.cif"))
        ... )
    """

    id: MaterialID
    structure: Optional[MaterialStructure] = Field(None)

    def __name__(self):
        return self.id.root


@ReadableEnums()
class PymatgenSite(RootModel, validate_assignment=True):
    """A class wrapping a Pymatgen PeriodicSite object.

    Args:
        root (PeriodicSite): The Pymatgen periodic site object

    Examples:
        >>> # Create from structure index
        >>> site = PymatgenSite.from_site_index(structure, 0)
    """

    root: PeriodicSite = Field(None, description="Pymatgen site")

    @model_serializer()
    def serialize_model(self):
        return self.as_dict()

    @classmethod
    def from_site_index(cls, structure: MaterialStructure, index: int):
        """Create a PymatgenSite from a structure and site index.

        Args:
            structure (MaterialStructure): The material structure containing the site
            index (int): The index of the site in the structure

        Returns:
            PymatgenSite: A new PymatgenSite instance
        """
        return cls(structure.sites[index])


@ReadableEnums()
class SiteSpectrum(Spectrum):
    """A class representing spectral data for a specific site in a material.

    Args:
        type (SpectrumType): The type of spectrum
        index (NonNegativeInt): The index of the site in the material structure
        material (Material, optional): The material containing the site
        intensities (IntensityValues, optional): The intensity values of the spectrum
        energies (EnergyGrid, optional): The energy grid of the spectrum

    Examples:
        >>> from omnixas.data import Material, MaterialStructure
        >>> from omnixas.data import EnergyGrid, IntensityValues
        >>> site_spectrum = SiteSpectrum(
        ...     type=SpectrumType.FEFF,
        ...     index=0,
        ...     material=material,
        ...     energies=EnergyGrid(root=[0.0, 1.0, 2.0]),
        ...     intensities=IntensityValues(root=[0.5, 1.2, 0.8])
        ... )
    """

    type: SpectrumType
    index: NonNegativeInt
    material: Optional[Material]

    @property
    def site(self) -> PymatgenSite:
        return self.material.structure.sites[self.index]

    @property
    def index_string(self) -> str:  # helper for file i/o
        return f"{self.index:03d}"


@ReadableEnums()
class ElementSpectrum(SiteSpectrum, validate_assignment=True):
    """A class representing spectral data for a specific element at a site.

    This class extends SiteSpectrum to include element-specific validation and
    information.

    Args:
        element (Element): The element for which the spectrum is calculated
        type (SpectrumType): The type of spectrum
        index (NonNegativeInt): The index of the site in the material structure
        material (Material, optional): The material containing the site
        intensities (IntensityValues, optional): The intensity values of the spectrum
        energies (EnergyGrid, optional): The energy grid of the spectrum

    Raises:
        ValueError: If the specified element doesn't match the element at the site

    Examples:
        >>> from omnixas.data import Material, MaterialStructure
        >>> from omnixas.data import EnergyGrid, IntensityValues
        >>> element_spectrum = ElementSpectrum(
        ...     element=Element.Fe,
        ...     type=SpectrumType.FEFF,
        ...     index=0,
        ...     material=material,
        ...     energies=EnergyGrid(root=[0.0, 1.0, 2.0]),
        ...     intensities=IntensityValues(root=[0.5, 1.2, 0.8])
        ... )
    """

    element: Element

    @model_validator(mode="after")
    def _validate_element(self) -> Self:
        asked_element = self.element
        if (
            self.type != SpectrumType.VASP
        ):  # TODO: remove in deployment # coz vasp sims were done this way
            try:
                site_element = Element(ElementSpectrum.extract_element(self.site))
                site_element = Element(site_element)
                if asked_element != site_element:
                    msg = f"Element {asked_element} does not match"
                    msg += f" site element {site_element} for {self.material.id}"
                    logger.warning(msg)

            except Exception:
                msg = f"Could not validate if element {asked_element}"
                msg += f" is same as site element for {self.material.id}"
                logger.warning(msg)
        return self

    @staticmethod
    def extract_element(site: PymatgenSite) -> str:
        species = site.species
        if not species.is_element:
            msg = "Site must be an element"
            logger.error(msg)
            raise ValueError(msg)
        return str(list(dict(species).keys())[0])
