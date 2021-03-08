"""Module that post-processes the output of the gaussian.spin workchain

By default, a compact report is generated together with images of spin
densities for different spin solutions 
"""

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


def print_orb_energies(prop_dict, n_orb=4):

    nspin = len(prop_dict['homos'])
    mo_e = np.array(prop_dict['moenergies'])

    if nspin == 1:

        homo_s0 = prop_dict['homos'][0]

        i_start = homo_s0 - n_orb
        i_end = homo_s0 + n_orb

        print("{:>10} {:>16} {:>6}".format("i_orb", "E (eV)", "occ"))

        for i_orb in range(i_end, i_start, -1):
            occ = 2 if i_orb <= homo_s0 else 0
            print("{:>10} {:16.4f} {:6}".format(i_orb, mo_e[0][i_orb], occ))

        print("GAP: {:.4f} eV".format(mo_e[0][homo_s0 + 1] - mo_e[0][homo_s0]))

    elif nspin == 2:

        homo_s0 = prop_dict['homos'][0]
        homo_s1 = prop_dict['homos'][1]

        i_start = (homo_s0 + homo_s1) // 2 - n_orb
        i_end = (homo_s0 + homo_s1) // 2 + n_orb

        print("{:>10} {:>16} {:>6} {:>16} {:>6}".format(
            "i_orb", "E(up) (eV)", "occ(up)", "E(down) (eV)", "occ(down)"))

        max_homo = np.max([mo_e[0, homo_s0], mo_e[1, homo_s1]])
        min_lumo = np.min([mo_e[0, homo_s0 + 1], mo_e[1, homo_s1 + 1]])

        for i_orb in range(i_end, i_start, -1):
            occ0 = 1 if i_orb <= homo_s0 else 0
            occ1 = 1 if i_orb <= homo_s1 else 0
            print("{:10d} {:16.4f} {:6d} {:16.4f} {:6d}".format(
                i_orb, mo_e[0, i_orb], occ0, mo_e[1, i_orb], occ1))

        print("GAP alpha: {:.4f} eV".format(mo_e[0, homo_s0 + 1] -
                                            mo_e[0, homo_s0]))
        print("GAP beta:  {:.4f} eV".format(mo_e[1, homo_s1 + 1] -
                                            mo_e[1, homo_s1]))
        print("GAP eff:   {:.4f} eV".format(min_lumo - max_homo))


def print_out_params(out_params):

    if out_params["spin_expectation_values"]:
        spin_dict = out_params["spin_expectation_values"][-1]
        s_ideal = (out_params['mult'] - 1) / 2
        print("S**2: {0:.3f}, ideal: {1:.2f}".format(spin_dict['S**2'],
                                                     s_ideal * (s_ideal + 1)))
        print("S:    {0:.3f}, ideal: {1:.2f}".format(spin_dict['S'], s_ideal))
    print()

    if "moenergies" in out_params:
        print("MO energies:")
        print_orb_energies(out_params)


def print_no_analysis(natorb_dict, prop_dict, n_orb=4):

    no_occs = natorb_dict['no_occs']
    no_occs_sp = natorb_dict['no_occs_sp']

    i_start = prop_dict['homos'][0] - n_orb
    i_end = prop_dict['homos'][0] + n_orb

    print("{:>20} {:>16} {:>16}".format("i_no", "occ", "sp. proj. occ"))

    for i_no in range(i_end, i_start, -1):
        print("{:>20} {:16.4f} {:16.4f}".format(i_no, no_occs[i_no],
                                                no_occs_sp[i_no]))
    print("---------------------------------------------------------")
    print("{:>20} {:16.4f} {:16.4f}".format("Standard num odd:",
                                            natorb_dict['std_num_odd'],
                                            natorb_dict['std_num_odd_sp']))
    print("{:>20} {:16.4f} {:16.4f}".format("HG num odd:",
                                            natorb_dict['hg_num_odd'],
                                            natorb_dict['hg_num_odd_sp']))


def plot_spin_wc(cube_image_folder):
    #pylint: disable=protected-access
    folder_path = cube_image_folder._repository._get_base_folder().abspath
    image_names = cube_image_folder.list_object_names()
    for imag_name in image_names:
        if "spin" in imag_name and "z+" in imag_name and "iv0.010" in imag_name:
            pil_image = Image.open(folder_path + "/" + imag_name)
            plt.imshow(pil_image)
            plt.title(imag_name)
            plt.axis('off')
            plt.show()


def make_report(wc_node):
    #pylint: disable=too-many-locals
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
    print("############################################################")
    print("#### GROUND STATE: MULTIPLICITY {}".format(gs_multiplicity))
    print("############################################################")

    plot_spin_wc(wc_node.outputs.gs_cube_images)

    print(f"Energy (eV): {gs_energy:10.4f}")
    print()
    print(f"IP (eV): {gs_ip:8.4f}")
    print(f"EA (eV): {gs_ea:8.4f} (accurate only with a diffuse basis)")
    print()
    print_out_params(gs_out_params)

    if 'gs_natorb_params' in wc_node.outputs:
        print()
        print("Natural orbital occupation analysis:")
        print_no_analysis(dict(wc_node.outputs.gs_natorb_params),
                          gs_out_params,
                          n_orb=4)
    print()

    if 'gs_hf_out_params' in wc_node.outputs:

        gs_hf_out_params = dict(wc_node.outputs.gs_hf_out_params)

        print("############################################################")
        print("#### GROUND STATE HF")
        print("############################################################")

        if "gs_hf_cube_images" in wc_node.outputs:
            plot_spin_wc(wc_node.outputs.gs_hf_cube_images)

        print()
        print_out_params(gs_hf_out_params)

        if 'gs_hf_natorb_params' in wc_node.outputs:
            print()
            print("Natural orbital occupation analysis:")
            print_no_analysis(dict(wc_node.outputs.gs_hf_natorb_params),
                              gs_hf_out_params,
                              n_orb=4)
        print()

    for mult in list(wc_node.inputs.multiplicity_list):

        if mult == gs_multiplicity:
            continue

        print("############################################################")
        print("#### MULTIPLICITY {}".format(mult))
        print("############################################################")

        cube_label = f"m{mult}_opt_cube_images"
        plot_spin_wc(wc_node.outputs[cube_label])

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
        print_out_params(vert_out_params)
        print()
