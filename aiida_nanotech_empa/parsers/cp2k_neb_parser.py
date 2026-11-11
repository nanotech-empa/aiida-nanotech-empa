import re

import ase
import numpy as np
from aiida import common, engine, orm, parsers, plugins

from ..utils import common_utils
from ..utils.cp2k_results import calculation_boundary, parse_neb_status

StructureData = plugins.DataFactory("structure")
ArrayData = plugins.DataFactory("array")


class Cp2kNebParser(parsers.Parser):
    """Parser for the output of CP2K NEB calculations."""

    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""

        try:
            _ = self.retrieved
        except common.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        exit_code = self._parse_stdout()
        if exit_code is not None:
            return exit_code

        try:
            trajectory_exit = self._parse_trajectory()
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            self.logger.warning("Could not parse the saved NEB structures: %s", exc)
            trajectory_exit = self.exit_codes.ERROR_STRUCTURE_PARSE

        status = self._neb_status
        if status["aborted"]:
            return self.exit_codes.ERROR_OUTPUT_CONTAINS_ABORT
        if status["exceeded_walltime"]:
            return self.exit_codes.ERROR_OUT_OF_WALLTIME
        if status["electronic_converged"] is False:
            return self.exit_codes.ERROR_SCF_NOT_CONVERGED
        if status["band_converged"] is False:
            return self.exit_codes.ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED
        if status["band_converged"] is None or status["electronic_converged"] is None:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        return trajectory_exit or engine.ExitCode(0)

    def _parse_stdout(self):
        """Basic CP2K output file parser."""

        exit_code, output_string = common_utils.read_retrieved(
            self, self.node.process_class._DEFAULT_OUTPUT_FILE
        )
        if exit_code is not None:
            return exit_code

        outputs = {self.node.process_class._DEFAULT_OUTPUT_FILE: output_string}
        for name in self.retrieved.base.repository.list_object_names():
            if re.fullmatch(r"aiida-BAND\d+\.out", name):
                outputs[name] = self.retrieved.base.repository.get_object_content(name)
        expected = (
            self.node.inputs.parameters.get_dict()
            .get("MOTION", {})
            .get("BAND", {})
            .get("NUMBER_OF_REPLICA")
        )
        result_dict = parse_neb_status(outputs, expected)
        self._neb_status = result_dict

        energy_str_list = re.findall(
            r"ENERGIES \[au\] = (.*?)BAND", output_string, re.DOTALL
        )
        energies_list = [
            np.array(e_str.split(), dtype=float) for e_str in energy_str_list
        ]

        energies_arr_data = ArrayData()
        energies_arr_data.set_array("energies", np.array(energies_list))

        dist_str_list = re.findall(
            r"DISTANCES REP = (.*?)ENERGIES", output_string, re.DOTALL
        )
        distances_list = [
            np.array(["0.0"] + ds.split(), dtype=float) for ds in dist_str_list
        ]

        distances_arr_data = ArrayData()
        distances_arr_data.set_array("distances", np.array(distances_list))

        self.out("output_parameters", orm.Dict(dict=result_dict))
        self.out("replica_energies", energies_arr_data)
        self.out("replica_distances", distances_arr_data)

        return None

    def _parse_trajectory(self):
        """CP2K trajectory parser."""

        exit_code, output_string = common_utils.read_retrieved(
            self, self.node.process_class._DEFAULT_RESTART_FILE_NAME
        )
        if exit_code is not None:
            return (
                self.exit_codes.ERROR_STRUCTURE_PARSE
                if exit_code == self.exit_codes.ERROR_OUTPUT_MISSING
                else exit_code
            )

        m = re.search(r"\n\s*&CELL\n(.*?)\n\s*&END CELL\n", output_string, re.DOTALL)
        cell_lines = [line.strip().split() for line in m.group(1).split("\n")]
        cell_str = [
            line[1:] for line in cell_lines if line and line[0] in ("A", "B", "C")
        ]
        cell = np.array(cell_str, np.float64)

        matches = re.findall(
            r"\n\s*&COORD\n(.*?)\n\s*&END COORD\n", output_string, re.DOTALL
        )

        coord_line_sets = [
            [line.strip().split() for line in m.split("\n")] for m in matches
        ]
        coord_set_with_elements = coord_line_sets[-1]
        replica_coord_line_sets = coord_line_sets[:-1]

        # Remove integers from element names and create tags.
        element_list = [
            re.sub(r"[0-9]+", "", line[0]) for line in coord_set_with_elements
        ]

        # Extract tags from the element labels
        tags = []
        for line in coord_set_with_elements:
            try:
                to_append = re.findall(r"\d+", line[0])[0]
            except IndexError:
                to_append = 0

            tags.append(int(to_append))

        boundary = calculation_boundary(self.node.inputs.parameters.get_dict())
        for i_rep, rep_coord_lines in enumerate(replica_coord_line_sets):
            positions = np.array(rep_coord_lines, np.float64)
            ase_atoms = ase.Atoms(
                symbols=element_list,
                positions=positions,
                cell=cell,
                pbc=boundary["pbc"],
            )
            ase_atoms.set_tags(tags)
            replica_label = f"opt_replica_{str(i_rep).zfill(3)}"
            self.out(replica_label, StructureData(ase=ase_atoms, label=replica_label))

        return None
