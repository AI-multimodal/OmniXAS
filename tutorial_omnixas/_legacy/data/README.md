---
  pdf_document: null
  geometry: margin=2cm
  output: pdf_document
---

# Methods and parameters for processing RAW data

```yaml
transformations:
  quarter_eV_resolution: 0.25 # used for interpolation
  e_range_diff: 35 # energy range used

  e_start:
    Co: 7709.282
    Cr: 5989.168
    Cu: 8983.173
    Fe: 7111.23
    Mn: 6537.886
    Ni: 8332.181
    Ti: 4964.504
    V: 5464.097
    ALL: 0 # dummy for universal e_start

  emperical_offset:
    VASP:
      Ti: 5114.08973
      Cu: 9499.797

  gamma:
    Ti: 0.89 # put same value in trucation offset
    Cu: 3.458 # 1.729 # or 3.458 #Based on email conv with J.K.

  e_core:
    Ti: -4864.0371
    Cu: -8850.2467
```

# Parsing and Pre-processing Algorithm for RAW FEFF and VASP Data

## VASP Data (`RAWDataVASP`)

### VASP: Search Patterns and Calculations

- **ID Search Pattern**: Regular expression `"(mp-\d+|mvc-\d+)"` used to
  identify IDs.
- **Ground State Energy (E_GS)**: Extracted from the `OSZICAR` file located in
  `base_dir/id/SimulationType/SCF/`.
- **Chemical Potential (mu)**:
  - If `xmu_avg.dat` is missing, it is generated from `mu2.txt` using average
    calculation: `mu2/3`
  - Located in `base_dir/id/SimulationType/site/`.
- **Conduction Band Minimum (e_cbm)**: Extracted from `efermi.txt` located in
  `base_dir/id/SimulationType/site/`.
- **Charging Energy (E_ch)**: Extracted from the `OSZICAR` file located in
  `base_dir/id/SimulationType/site/`.
- **Volume Calculation**:
  - Lattice vectors read from the `POSCAR` file located in
    `base_dir/id/SimulationType/site/`.
  - Volume Vector Parsing: Lattice vectors vx, vy, vz, extracted from
    horizontal lines 2, 3, and 4 in the POSCAR file, are used to compute the
    unit cell volume using the scalar triple product formula, with the volume
    then converted to Bohr^3 units.
- **Core Electron Energy (e_core)**:
  - Values are dynamically loaded from a YAML configuration file
    (`cfg/transformations.yaml`).

## FEFF Data (`RAWDataFEFF`)

### FEFF: Search Patterns and Calculations

- **ID Search Pattern**: Same regular expression as VASP (`"(mp-\d+|mvc-\d+)"`) for consistency.
- **Mu Parameter (mu)**:
  - Extracted from the `xmu.dat` file located in `base_dir/id/FEFF-XANES/site/`.
  - Normalization factor identified from a line starting with "`# xsedge+`" in the file.
  - Absorption coefficient multiplied by the normalization factor.

Both VASP and FEFF classes rely on a consistent directory structure and specific file naming conventions. The algorithms detail where each file is located, how certain values are calculated, and the patterns used to search and organize the data effectively.

## Data Transformation Algorithms with Explicit Formulas

### VASP Data Processing

1. **Truncation**:

   - Minimum Energy Calculation: `min_energy = (e_cbm - e_core) - start_offset`
   - Default `start_offset = 0 eV` (need to determine this again for alignment with FEFF)
   - Truncate away energy < `min_energy`.
   - Truncate away spectra <= 0

2. **Scaling**:

   - Omega Calculation: `omega = spectra * energy`
   - Scaling Formula: `spectra = (omega * volume) / (Rydberg constant * 2 * inverse fine-structure constant)`

3. **Broadening**:

   - Lorentzian Broadening: `broadened_amplitude = lorentz_broaden(energy, energy, spectra, gamma)`
   - `lorentz_broaden` applies the Lorentzian function across the energy spectrum.
   - `gamma` is read from config file

4. **Alignment**:

   - Theoretical Offset: `theoretical_offset = (e_core - e_cbm) + (E_ch - E_GS)`
   - Apply theoretical and empirical offsets to energy: `energy += theoretical_offset + empirical_offset`
   - `emperical_offset` is computed by comparing the `VASP` spectral with experimental spectra.

### FEFF Data Processing

1. **Truncation**:

   - Truncate the spectral data to exclude negative values: `spectra = spectra[spectra >= 0]`

2. **Scaling**:

   - Scaling Formula: `spectra = spectra / (Bohr radius^2)`
   - Converts spectra from atomic units to standard units.

3. **Alignment with VASP**:

   - FEFF also has optional `align` process that is used to align FEFF spectra
     with VASP spectra using method `compare_between_spectra` from `LighShow`. This is not used in transfer learning.

### Data Transformation Common to both VASP and FEFF

1. **Emperical Truncation**:

   - Truncation based on `e_start` and `e_range_diff` in configuration file
   - `e_start` is the starting energy based on emperical observations
   - `e_range_diff` is the range of energy to keep based on emperical observations

2. **Resampling**:

   - Resample the spectral data to a uniform energy grid between `e_start` and
     `e_end`:
     - `e_start` is the starting energy based on emperical observations
     - `e_end` is determined by `e_start` and `e_range_diff` (loaded from
       config file)
     - Number of points in the range is determined by `quarter_eV_resolution`,
       which is also loaded from config file
