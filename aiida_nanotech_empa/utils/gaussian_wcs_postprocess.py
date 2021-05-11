"""Module that post-processes the output of the gaussian.spin workchain

By default, a compact report is generated together with images of spin
densities for different spin solutions 
"""

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


def select_frontier_orbital_energies(out_params, n_orb=4):

    nspin = len(out_params['homos'])
    mo_e = np.array(out_params['moenergies'])

    indexes = []
    energies = [[]]
    occs = [[]]

    if nspin == 1:

        homo_s0 = out_params['homos'][0]

        i_start = homo_s0 - n_orb
        i_end = homo_s0 + n_orb

        for i_orb in range(i_end, i_start, -1):
            indexes.append(i_orb)
            energies[0].append(mo_e[0][i_orb])
            occs[0].append(2 if i_orb <= homo_s0 else 0)

    elif nspin == 2:

        energies.append([])
        occs.append([])

        homo_s0 = out_params['homos'][0]
        homo_s1 = out_params['homos'][1]

        central_i = int(np.round((homo_s0 + homo_s1) / 2))

        i_start = central_i - n_orb
        i_end = central_i + n_orb

        for i_orb in range(i_end, i_start, -1):
            indexes.append(i_orb)
            energies[0].append(mo_e[0][i_orb])
            energies[1].append(mo_e[1][i_orb])
            occs[0].append(1 if i_orb <= homo_s0 else 0)
            occs[1].append(1 if i_orb <= homo_s1 else 0)

    return {"indexes": indexes, "energies": energies, "occs": occs}


def _get_orb_energies_str(out_params, n_orb=4):

    orb_dict = select_frontier_orbital_energies(out_params, n_orb)

    inds = orb_dict['indexes']
    ens = orb_dict['energies']
    occs = orb_dict['occs']

    nspin = len(orb_dict['energies'])

    if nspin == 1:
        header = "{:>10} {:>16} {:>6}".format("i_orb", "E (eV)", "occ")
        lines = [header]

        for z in zip(inds, ens[0], occs[0]):
            lines.append("{:>10} {:>16.4f} {:>6}".format(*z))

    elif nspin == 2:

        header = "{:>10} {:>16} {:>9} {:>16} {:>9}".format(
            "i_orb", "E(up) (eV)", "occ(up)", "E(down) (eV)", "occ(down)")
        lines = [header]

        for z in zip(inds, ens[0], occs[0], ens[1], occs[1]):
            lines.append("{:>10} {:>16.4f} {:>9} {:>16.4f} {:>9}".format(*z))
    return "\n".join(lines)


def get_spin_exp_values(out_params):
    spin_dict = out_params["spin_expectation_values"][-1]
    s_ideal = (out_params['mult'] - 1) / 2
    s2_ideal = s_ideal * (s_ideal + 1)
    return {
        'S**2': spin_dict['S**2'],
        'S': spin_dict['S'],
        'S**2_ideal': s2_ideal,
        'S_ideal': s_ideal
    }


def _get_spin_exp_values_str(out_params):
    s = ""
    if ("spin_expectation_values" in out_params
            and len(out_params["spin_expectation_values"]) > 0):
        s_e = get_spin_exp_values(out_params)
        s += "S**2: {0:.3f}, ideal: {1:.2f}\n".format(s_e['S**2'],
                                                      s_e['S**2_ideal'])
        s += "S:    {0:.3f}, ideal: {1:.2f}".format(s_e['S'], s_e['S_ideal'])
    return s


def _get_out_params_str(out_params):
    s = _get_spin_exp_values_str(out_params)
    s += "\n\n"
    s += "MO energies:\n"
    s += _get_orb_energies_str(out_params, 4)
    s += "\n"
    s += "GAP:       {:.4f} eV\n".format(out_params['gap'])
    if 'gap_a' in dict(out_params):
        s += "GAP alpha: {:.4f} eV\n".format(out_params['gap_a'])
        s += "GAP beta:  {:.4f} eV\n".format(out_params['gap_b'])
    return s


def _get_natorb_analysis_str(natorb_params, out_params, n_orb=4):

    no_occs = natorb_params['no_occs']
    no_occs_sp = natorb_params['no_occs_sp']

    i_start = out_params['homos'][0] - n_orb
    i_end = out_params['homos'][0] + n_orb

    lines = ["{:>20} {:>16} {:>16}".format("i_no", "occ", "sp. proj. occ")]

    for i_no in range(i_end, i_start, -1):
        lines.append("{:>20} {:16.4f} {:16.4f}".format(i_no, no_occs[i_no],
                                                       no_occs_sp[i_no]))
    lines.append("---------------------------------------------------------")
    lines.append("{:>20} {:16.4f} {:16.4f}".format(
        "Standard num odd:", natorb_params['std_num_odd'],
        natorb_params['std_num_odd_sp']))
    lines.append("{:>20} {:16.4f} {:16.4f}".format(
        "HG num odd:", natorb_params['hg_num_odd'],
        natorb_params['hg_num_odd_sp']))
    return "\n".join(lines)


def get_pil_image(cube_image_folder, name):
    with cube_image_folder.open(name, 'rb') as f:
        pil_image = Image.open(f)
        pil_image.load()
    return pil_image


def plot_cube_images(cube_image_folder,
                     name_contains=None,
                     show=True,
                     save_image_loc=None,
                     save_prefix=""):

    if name_contains is None:
        name_contains = ['z+']

    rows = {}
    image_names = cube_image_folder.list_object_names()
    for imag_name in image_names:

        if not all([e in imag_name for e in name_contains]):
            continue

        label = imag_name.split("_")[0]
        if label not in rows:
            rows[label] = []
        pil_image = get_pil_image(cube_image_folder, imag_name)
        rows[label].append((imag_name, pil_image))

        if save_image_loc is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(pil_image)
            plt.axis('off')
            image_file_name = save_image_loc + "/{}{}".format(
                save_prefix, imag_name)
            plt.savefig(image_file_name, dpi=400, bbox_inches='tight')
            plt.close()
            print(f"saved {image_file_name}")

    if show:
        for label in rows:
            n_imag_row = len(rows[label])
            plt.figure(figsize=(5 * n_imag_row, 5))
            for i, pi in enumerate(rows[label]):
                plt.subplot(1, n_imag_row, i + 1)
                plt.imshow(pi[1])
                plt.title(pi[0])
                plt.axis('off')
            plt.show()


def _show_spin_density(cube_image_folder,
                       nb,
                       save_image_loc=None,
                       save_prefix=""):
    if nb:
        name_contains = ['spin', 'z+', 'iv0.010']
        show = True
    else:
        name_contains = ['spin', 'z+']
        show = False
    plot_cube_images(cube_image_folder,
                     name_contains=name_contains,
                     show=show,
                     save_image_loc=save_image_loc,
                     save_prefix=save_prefix)


def make_report(wc_node, nb=False, save_image_loc=None):
    """ Function that generates a report for a gaussian spin workchain run"""
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-statements
    print("Functional:", wc_node.inputs.functional.value)
    print("Basis set OPT:", wc_node.inputs.basis_set_opt.value)
    print("Basis set SCF:", wc_node.inputs.basis_set_scf.value)
    print("Multplicity list:", list(wc_node.inputs.multiplicity_list))

    gs_energy = wc_node.outputs["gs_energy"].value
    gs_multiplicity = wc_node.outputs["gs_multiplicity"].value
    gs_out_params = dict(wc_node.outputs["gs_out_params"])

    gs_ip = wc_node.outputs["gs_ionization_potential"].value
    gs_ea = wc_node.outputs["gs_electron_affinity"].value

    print()
    print("##############################################################")
    print("#### GROUND STATE: MULTIPLICITY {}".format(gs_multiplicity))
    print("##############################################################")

    _show_spin_density(wc_node.outputs.gs_cube_images,
                       nb,
                       save_image_loc=save_image_loc,
                       save_prefix='gs_')

    print(f"Energy (eV): {gs_energy:10.4f}")
    print()
    print(f"IP    (eV): {gs_ip:8.4f}")
    print(f"EA    (eV): {gs_ea:8.4f} (accurate only with a diffuse basis)")
    print(f"IP-EA (eV): {gs_ip-gs_ea:8.4f}")
    print()
    print(_get_out_params_str(gs_out_params))

    if 'gs_natorb_params' in wc_node.outputs:
        print()
        print("Natural orbital occupation analysis:")
        print(
            _get_natorb_analysis_str(dict(wc_node.outputs.gs_natorb_params),
                                     gs_out_params))
    print()

    if 'gs_hf_out_params' in wc_node.outputs:

        gs_hf_out_params = dict(wc_node.outputs.gs_hf_out_params)

        print("##############################################################")
        print("#### GROUND STATE HF")
        print("##############################################################")

        if "gs_hf_cube_images" in wc_node.outputs:
            _show_spin_density(wc_node.outputs.gs_hf_cube_images,
                               nb,
                               save_image_loc=save_image_loc,
                               save_prefix='gs_hf_')

        print()
        print(_get_out_params_str(gs_hf_out_params))

        if 'gs_hf_natorb_params' in wc_node.outputs:
            print()
            print("Natural orbital occupation analysis:")
            print(
                _get_natorb_analysis_str(
                    dict(wc_node.outputs.gs_hf_natorb_params),
                    gs_hf_out_params))
        print()

    for mult in list(wc_node.inputs.multiplicity_list):

        if mult == gs_multiplicity:
            continue

        print("##############################################################")
        print("#### MULTIPLICITY {}".format(mult))
        print("##############################################################")

        cube_label = f"m{mult}_vert_cube_images"
        _show_spin_density(wc_node.outputs[cube_label],
                           nb,
                           save_image_loc=save_image_loc,
                           save_prefix=f'm{mult}_')

        adia_label = f"m{mult}_opt_energy"
        vert_label = f"m{mult}_vert_energy"

        vert_out_params = dict(wc_node.outputs[f'm{mult}_vert_out_params'])

        opt_en = wc_node.outputs[adia_label].value
        adia_ex = opt_en - gs_energy
        vert_en = wc_node.outputs[vert_label].value
        vert_ex = vert_en - gs_energy

        print()
        print(f"adia. exc (meV): {adia_ex*1000:8.1f}")
        print(f"vert. exc (meV): {vert_ex*1000:8.1f}")
        print()
        print(_get_out_params_str(vert_out_params))
        print()
