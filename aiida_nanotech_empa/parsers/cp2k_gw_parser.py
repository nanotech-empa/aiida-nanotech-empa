# -*- coding: utf-8 -*-
"""AiiDA-CP2K output parser for GW calculations."""

from aiida.common import OutputParsingError
from aiida.orm import Dict

from aiida_cp2k.utils import parse_cp2k_output_advanced
from aiida_cp2k.parsers import Cp2kBaseParser

HART_2_EV = 27.21138602


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Cp2kGWParser(Cp2kBaseParser):
    """AiiDA parser class for the output of CP2K GW calculations."""
    def _parse_stdout(self):

        fname = self.node.process_class._DEFAULT_OUTPUT_FILE  # pylint: disable=protected-access
        if fname not in self.retrieved.list_object_names():
            raise OutputParsingError("Cp2k output file not retrieved")

        try:
            output_string = self.retrieved.get_object_content(fname)
        except IOError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        # CP2K advanced parsing provided by aiida-cp2k
        result_dict = parse_cp2k_output_advanced(output_string)

        # nwarnings is the last thing to be printed in th eCP2K output file:
        # if it is not there, CP2K didn't finish properly
        if 'nwarnings' not in result_dict:
            raise OutputParsingError("CP2K did not finish properly.")

        if "aborted" in result_dict:
            return self.exit_codes.ERROR_OUTPUT_CONTAINS_ABORT

        # Standard output parameters
        self.out("std_output_parameters", Dict(dict=result_dict))

        # Custom GW parsing
        gw_output_parameters = self._parse_cp2k_gw_output(output_string)
        self.out("gw_output_parameters", Dict(dict=gw_output_parameters))

        return None

    def _parse_cp2k_gw_output(self, output_string):  # noqa
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        lines = output_string.splitlines()

        results = {}

        def add_res_list(name):
            if name not in results:
                results[name] = []

        i_line = 0
        while i_line < len(lines):
            line = lines[i_line]

            # ----------------------------------------------------------------
            # num electrons
            if "Number of electrons" in line:
                add_res_list('num_el')
                n = int(line.split()[-1])
                spin_line = lines[i_line - 2]
                if "Spin " in spin_line:
                    results['nspin'] = 2
                    spin = int(spin_line.split()[-1]) - 1
                else:
                    spin = 0
                    results['nspin'] = 1

                if len(results['num_el']) == spin:
                    results['num_el'].append(n)
                else:
                    print("Warning: something is wrong with num. el. parsing.")
            # ----------------------------------------------------------------
            # Energy (overwrite so that we only have the last (converged) one)
            if "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" in line:
                results['energy'] = float(line.split()[-1]) * HART_2_EV
            # ----------------------------------------------------------------
            # Occupied eigenvalues (normal SCF)
            if "eigenvalues of the occupied" in line.lower():
                add_res_list('evals')
                spin = int(line.split()[-1]) - 1
                results['evals'].append([])
                i_line += 2
                while True:
                    vals = lines[i_line].split()
                    if len(vals) == 0 or not is_number(vals[0]):
                        break
                    results['evals'][spin] += [
                        float(v) * HART_2_EV for v in vals
                    ]
                    i_line += 1
            # ----------------------------------------------------------------
            # Unoccupied eigenvalues (normal SCF)
            if "lowest eigenvalues of the unoccupied" in line.lower():
                spin = int(line.split()[-1]) - 1
                i_line += 2
                while True:
                    if 'reached convergence in' in lines[i_line].lower():
                        i_line += 1
                        continue
                    vals = lines[i_line].split()
                    if len(vals) == 0 or not is_number(vals[0]):
                        break
                    results['evals'][spin] += [
                        float(v) * HART_2_EV for v in vals
                    ]
                    i_line += 1
            # ----------------------------------------------------------------
            # GW output
            if "Sigx-vxc (eV)" in line and "E_GW (eV)" in line:
                add_res_list('mo')
                add_res_list('occ')
                add_res_list('gw_eval')
                add_res_list('g0w0_eval')
                add_res_list('g0w0_e_scf')

                i_line += 1

                gw_mo = []
                gw_occ = []
                gw_e_scf = []
                gw_eval = []

                while True:
                    line_loc = lines[i_line]
                    if "GW HOMO-LUMO gap" in line_loc:

                        spin = 1 if "Beta" in line_loc else 0

                        if len(results['mo']) > spin:
                            # we already have a set, overwrite with later iteration
                            results['mo'][spin] = gw_mo
                            results['occ'][spin] = gw_occ
                            results['gw_eval'][spin] = gw_eval
                        else:
                            results['mo'].append(gw_mo)
                            results['occ'].append(gw_occ)
                            results['gw_eval'].append(gw_eval)
                            results['g0w0_eval'].append(gw_eval)
                            results['g0w0_e_scf'].append(gw_e_scf)

                        break

                    vals = line_loc.split()
                    # header & example line:
                    #     Molecular orbital   E_SCF (eV)       Sigc (eV)   Sigx-vxc (eV)       E_GW (eV)
                    #        1 ( occ )           -26.079           6.728         -10.116         -26.068
                    if len(vals) == 8 and is_number(vals[0]):
                        gw_mo.append(int(vals[0]) -
                                     1)  # start orb count from 0
                        gw_occ.append(1 if vals[2] == 'occ' else 0)
                        gw_e_scf.append(float(vals[4]))
                        gw_eval.append(float(vals[7]))
                    i_line += 1
            # ----------------------------------------------------------------
            # IC output
            if "E_n before ic corr" in line and "Delta E_ic" in line:
                add_res_list('mo')
                add_res_list('occ')
                add_res_list('ic_en')
                add_res_list('ic_delta')

                i_line += 1

                ic_mo = []
                ic_occ = []
                ic_en = []
                ic_delta = []

                while True:
                    line_loc = lines[i_line]
                    if "IC HOMO-LUMO gap" in line_loc:

                        spin = 1 if "Beta" in line_loc else 0

                        if len(results['mo']) > spin:
                            # we already have a set, overwrite with later iteration
                            results['mo'][spin] = ic_mo
                            results['occ'][spin] = ic_occ
                            results['ic_en'][spin] = ic_en
                            results['ic_delta'][spin] = ic_delta
                        else:
                            results['mo'].append(ic_mo)
                            results['occ'].append(ic_occ)
                            results['ic_en'].append(ic_en)
                            results['ic_delta'].append(ic_delta)

                        break

                    vals = line_loc.split()
                    # header & example line:
                    #           MO     E_n before ic corr           Delta E_ic    E_n after ic corr
                    #   70 ( occ )                -11.735                1.031              -10.705
                    if len(vals) == 7 and is_number(vals[0]):
                        ic_mo.append(int(vals[0]) -
                                     1)  # start orb count from 0
                        ic_occ.append(1 if vals[2] == 'occ' else 0)
                        ic_en.append(float(vals[4]))
                        ic_delta.append(float(vals[5]))
                    i_line += 1

            # ----------------------------------------------------------------
            i_line += 1

        # ----------------------------------------------------------------
        # Determine HOMO indexes w.r.t. outputted eigenvalues
        results['homo'] = []

        if 'occ' in results:
            # In case of GW and IC, the MO count doesn't start from 0
            # so use the occupations
            for i_spin in range(results['nspin']):
                results['homo'].append(results['occ'][i_spin].index(0) - 1)
        else:
            # In case of normal SCF, use the electron numbers
            for i_spin in range(results['nspin']):
                if results['nspin'] == 1:
                    results['homo'].append(
                        int(results['num_el'][i_spin] / 2) - 1)
                else:
                    results['homo'].append(results['num_el'][i_spin] - 1)
                # Also create 'mo' and 'occ' arrays
                add_res_list('occ')
                add_res_list('mo')
                occ = [1 for i in range(len(results['evals'][i_spin]))]
                occ[results['homo'][i_spin] + 1:] = 0
                mo = list(range(len(results['evals'][i_spin])))
                results['occ'].append(occ)
                results['mo'].append(mo)

        return results