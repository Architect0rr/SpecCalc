# -*- coding: utf-8 -*-

"""This is a settings file for Absorption Spectrum Calculator

"""

from typing import Dict, Tuple, Union, List

file: str = 'ds'  # file, that contains experimental spectrum

settings: Dict[str, Union[int, float]] = {
    'pressure': 1,  # pressure in atm
    'vnmin': 7188.7,  # Min wavenumber
    'vnmax': 7190,  # Max wavenumber
    'tmin': 250,  # Min temperature
    'tmax': 1500,  # Max temperature
    'dt': 100,  # Temperature resolution (less - slower)
}


substance: Dict[str, Union[str, float]] = {
    'TableName': 'H20',  # for different substances use different tables (table names)
    'molNumber': 1,  # HITRAN Mulecule ID. For regular H2O it is 1. For more see: https://hitran.org/docs/molec-meta/
    'isID': 1,  # HITRAN Isotopologue ID. For regular H2O it is 1. For more see: https://hitran.org/docs/iso-meta/
}


advanced: Dict[str, Union[List[Tuple[int, float]], Tuple[int, float, float]]] = {
    'FixedCoeffTemp': [],
    'IteratedTemp': (296, 0.9, 0.1),
}
