import numpy as np
from aiida import orm, plugins

StructureData = plugins.DataFactory("core.structure")


def split_structure(structure, fixed_atoms, magnetization_per_site, fragments):
    ase_geo = structure.get_ase()

    allfixed = [0 for i in ase_geo]
    mps = []
    fixed = ""
    if "all" not in fragments:
        yield {
            "label": "all",
            "structure": structure,
            "fixed_atoms": fixed_atoms,
            "magnetization_per_site": magnetization_per_site,
        }

    for f in fixed_atoms:
        allfixed[f] = 1

    for fragment_label, fragment in fragments.items():
        fragment = sorted(fragment)

        if magnetization_per_site or fixed_atoms:
            tuples = [
                (e, *np.round(p, 2))
                for e, p in zip(
                    ase_geo[fragment].get_chemical_symbols(),
                    ase_geo[fragment].positions,
                )
            ]
            if magnetization_per_site:
                mps = [
                    m
                    for at, m in zip(ase_geo, list(magnetization_per_site))
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                if all(m == 0 for m in mps):
                    mps = []
            if fixed_atoms:
                fixed = [
                    f
                    for at, f in zip(ase_geo, allfixed)
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]

        yield {
            "label": fragment_label,
            "structure": StructureData(ase=ase_geo[fragment]),
            "fixed_atoms": orm.List(list=np.nonzero(fixed)[0].tolist()),
            "magnetization_per_site": orm.List(list=mps),
        }
