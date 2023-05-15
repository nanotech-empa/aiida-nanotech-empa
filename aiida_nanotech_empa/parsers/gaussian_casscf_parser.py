from aiida import orm
from aiida_gaussian.parsers.gaussian import GaussianBaseParser

from ..helpers import HART_2_EV


class GaussianCasscfParser(GaussianBaseParser):
    """AiiDA parser for Gaussian CASSCF calculations."""

    def _parse_log(self, log_file_string, _inputs):
        """Overwrite the basic log parser."""

        # Parse with cclib.
        property_dict = self._parse_log_cclib(log_file_string)

        if property_dict is None:
            return self.exit_codes.ERROR_OUTPUT_PARSING

        property_dict.update(self._parse_electron_numbers(log_file_string))

        property_dict.update(self._parse_casscf(log_file_string))

        self.out("output_parameters", orm.Dict(dict=property_dict))

        if "casscf_energy_ev" in property_dict:
            self.out("casscf_energy_ev", orm.Float(property_dict["casscf_energy_ev"]))
        if "casmp2_energy_ev" in property_dict:
            self.out("casmp2_energy_ev", orm.Float(property_dict["casmp2_energy_ev"]))

        exit_code = self._final_checks_on_log(log_file_string, property_dict)
        if exit_code is not None:
            return exit_code

        return None

    def _parse_casscf(self, log_file_string):
        parsed_data = {}

        for line in log_file_string.splitlines():
            if "     eigenvalue " in line.lower():
                parsed_data["casscf_energy_ev"] = (
                    float(line.split()[-1].replace("D", "E")) * HART_2_EV
                )
            if "EUMP2 =" in line:
                parsed_data["casmp2_energy_ev"] = (
                    float(line.split()[-1].replace("D", "E")) * HART_2_EV
                )

        return parsed_data
