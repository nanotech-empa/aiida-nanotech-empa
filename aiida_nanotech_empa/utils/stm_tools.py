import matplotlib.pyplot as plt
import numpy as np

from ..helpers import ANG_TO_BOHR, HART_2_EV
from . import igor


def process_cube_planes_array(cpa):
    x_arr = cpa.get_array("x_arr")
    y_arr = cpa.get_array("y_arr")
    h_arr = cpa.get_array("h_arr")

    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # center x and y arrays
    x_arr -= np.mean(x_arr)
    y_arr -= np.mean(y_arr)

    extent = [x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]]

    planes_dict = {}
    for aname in cpa.get_arraynames():
        if aname.startswith("cube_"):
            n_split = aname.split("_")
            i_mo = n_split[1]
            if not i_mo.isnumeric():
                continue
            # number in filename corresponds to cubegen convention,
            # where counting starts from 1
            i_mo = int(i_mo) - 1

            i_spin = 0
            if len(n_split) > 2:
                spin_let = n_split[2]
                if spin_let == "b":
                    i_spin = 1

            if i_mo not in planes_dict:
                planes_dict[i_mo] = [None]
            if i_spin == 1 and len(planes_dict[i_mo]) == 1:
                planes_dict[i_mo].append(None)

            planes_dict[i_mo][i_spin] = cpa.get_array(aname)

    cpa_dict = {
        "mo_planes": planes_dict,
        "dx": dx,
        "dy": dy,
        "extent": extent,
        "heights": h_arr,
    }
    return cpa_dict


def extrapolate_morb(orb_plane, dx, dy, energy_wrt_vacuum, delta_h):
    """
    dx, dy, delta_h - in ang
    energy_wrt_vacuum - in eV
    """

    # Convert everything to a.u.
    dx = dx * ANG_TO_BOHR
    dy = dy * ANG_TO_BOHR
    delta_h = delta_h * ANG_TO_BOHR

    energy_wrt_vacuum = energy_wrt_vacuum / HART_2_EV

    if energy_wrt_vacuum >= 0.0:
        print("Warning: unbound state, can't extrapolate! Constant extrapolation.")
        energy_wrt_vacuum = 0.0

    fourier = np.fft.rfft2(orb_plane)
    # NB: rfft2 takes REAL fourier transform over last (y) axis and COMPLEX over other (x) axes
    # dv in BOHR, so k is in 1/bohr
    kx_arr = 2 * np.pi * np.fft.fftfreq(orb_plane.shape[0], dx)
    ky_arr = 2 * np.pi * np.fft.rfftfreq(orb_plane.shape[1], dy)

    kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr, indexing="ij")

    prefactors = np.exp(
        -np.sqrt(kx_grid**2 + ky_grid**2 - 2 * energy_wrt_vacuum) * delta_h
    )

    return np.fft.irfft2(fourier * prefactors, orb_plane.shape)


def gaussian(x, fwhm):
    sigma = fwhm / 2.3548
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def get_orb_mapping(i_mo, i_spin, h, extrap_h, cpa_dict, extrap_energy):
    if h > extrap_h:
        plane_h = extrap_h
    else:
        plane_h = h

    try:
        plane_h_ind = np.where(np.isclose(cpa_dict["heights"], plane_h, atol=0.05))[0][
            0
        ]
    except IndexError:
        print("Error: specified height was not calculated!")
        return None

    plane = cpa_dict["mo_planes"][i_mo][i_spin][:, :, plane_h_ind]

    if h > extrap_h:
        orb_plane = extrapolate_morb(
            plane, cpa_dict["dx"], cpa_dict["dy"], extrap_energy, h - extrap_h
        )
    else:
        orb_plane = plane

    # potentially apply p-tip

    return orb_plane


def get_sts_mapping(energy, fwhm, h, extrap_h, cpa_dict, sop):
    energies = sop["moenergies"]
    nspin = len(sop["moenergies"])

    final_map = None

    for i_spin in range(nspin):
        for i_mo, e in enumerate(energies[i_spin]):
            if np.abs(e - energy) <= 1.5 * fwhm:
                broad_coef = gaussian(e - energy, fwhm)

                if i_mo not in cpa_dict["mo_planes"]:
                    print(
                        f"Missing MO{i_mo}, that potentially contributes to sts at E={energy}"
                    )
                    continue

                orb_map = get_orb_mapping(i_mo, i_spin, h, extrap_h, cpa_dict, e)

                if final_map is None:
                    final_map = broad_coef * orb_map**2
                else:
                    final_map += broad_coef * orb_map**2
    return final_map


def save_figure_and_igor(data_2d, filename, title, **imshow_args):
    plt.figure(figsize=(5, 5))
    plt.imshow(data_2d.T, **imshow_args)
    plt.title(title)
    plt.savefig(filename + ".png", dpi=250, bbox_inches="tight")
    plt.close()

    extent = imshow_args["extent"]

    igorwave = igor.Wave2d(
        data=data_2d,
        xmin=extent[0],
        xmax=extent[1],
        xlabel="x [Angstroms]",
        ymin=extent[2],
        ymax=extent[3],
        ylabel="y [Angstroms]",
    )
    igorwave.write(filename + ".itx")


def get_rel_homo_label(i_mo, i_homo):
    i_mo_wrt_homo = i_mo - i_homo
    if i_mo_wrt_homo < 0:
        label = "homo%+d" % i_mo_wrt_homo
    elif i_mo_wrt_homo == 0:
        label = "homo"
    elif i_mo_wrt_homo == 1:
        label = "lumo"
    elif i_mo_wrt_homo > 1:
        label = "lumo%+d" % (i_mo_wrt_homo - 1)
    return label


def plot_mapping(
    sop,
    cpa,
    i_mo,
    i_spin,
    h=3.0,
    extrap_h=3.0,
    fwhm=0.05,
    save_dir=None,
    ax=None,
    kind="orb",
):
    """
    kind: ['orb', 'orb2', 'sts']
    """

    cpa_dict = process_cube_planes_array(cpa)
    en = sop["moenergies"][i_spin][i_mo]
    i_homo = sop["homos"][i_spin]
    rel_homo_label = get_rel_homo_label(i_mo, i_homo)

    i_mo_f1 = i_mo + 1  # index gaussian convention (count starts from 1)

    if kind == "orb":
        data = get_orb_mapping(i_mo, i_spin, h, extrap_h, cpa_dict, en)
        label = f"MO{i_mo_f1} s{i_spin} {rel_homo_label}\nh{h:.1f} E={en:.2f}"
        fname = f"orb{i_mo_f1}_s{i_spin}_{rel_homo_label}_h{h:.1f}_eh{extrap_h:.1f}_e{en:.2f}"
        amax = np.max(np.abs(data))
        imshow_args = {
            "cmap": "seismic",
            "extent": cpa_dict["extent"],
            "origin": "lower",
            "vmin": -amax,
            "vmax": amax,
        }

    elif kind == "orb2":
        data = get_orb_mapping(i_mo, i_spin, h, extrap_h, cpa_dict, en) ** 2
        label = f"MO{i_mo_f1}^2 s{i_spin} {rel_homo_label}\nh{h:.1f} E={en:.2f}"
        fname = f"dens{i_mo_f1}_s{i_spin}_{rel_homo_label}_h{h:.1f}_eh{extrap_h:.1f}_e{en:.2f}"
        imshow_args = {
            "cmap": "seismic",
            "extent": cpa_dict["extent"],
            "origin": "lower",
        }

    elif kind == "sts":
        data = get_sts_mapping(en, fwhm, h, extrap_h, cpa_dict, sop)
        label = f"STS h{h:.1f} at E={en:.2f}"
        fname = f"sts_fwhm{fwhm:.2f}_mo{i_mo_f1}_s{i_spin}_{rel_homo_label}_h{h:.1f}_eh{extrap_h:.1f}_e{en:.2f}"
        imshow_args = {
            "cmap": "seismic",
            "extent": cpa_dict["extent"],
            "origin": "lower",
        }

    if save_dir is not None:
        save_figure_and_igor(data, f"{save_dir}/{fname}", label, **imshow_args)

    show_plot = False
    if ax is None:
        ax = plt.gca()
        show_plot = True

    ax.imshow(data.T, **imshow_args)
    ax.set_title(label, loc="left")
    ax.axis("off")

    if show_plot:
        plt.show()


def plot_no_mapping(
    nop, cpa, i_no, h=3.0, extrap_h=3.0, save_dir=None, ax=None, kind="orb"
):
    """
    kind: ['orb', 'orb2']
    """

    cpa_dict = process_cube_planes_array(cpa)

    no_occ = nop["nooccnos"][i_no]

    extrap_energy = -2.0  # eV

    i_no_f1 = i_no + 1  # index gaussian convention (count starts from 1)

    if kind == "orb":
        data = get_orb_mapping(i_no, 0, h, extrap_h, cpa_dict, extrap_energy)
        label = f"NO{i_no_f1} h{h:.1f} occ={no_occ:.4f}"
        fname = f"no{i_no_f1}_h{h:.1f}_eh{extrap_h:.1f}_occ{no_occ:.4f}"
        amax = np.max(np.abs(data))
        imshow_args = {
            "cmap": "seismic",
            "extent": cpa_dict["extent"],
            "origin": "lower",
            "vmin": -amax,
            "vmax": amax,
        }

    elif kind == "orb2":
        data = get_orb_mapping(i_no, 0, h, extrap_h, cpa_dict, extrap_energy) ** 2
        label = f"NO{i_no_f1}^2 h{h:.1f} occ={no_occ:.4f}"
        fname = f"NO_dens{i_no_f1}_h{h:.1f}_eh{extrap_h:.1f}_occ{no_occ:.4f}"
        imshow_args = {
            "cmap": "seismic",
            "extent": cpa_dict["extent"],
            "origin": "lower",
        }

    if save_dir is not None:
        save_figure_and_igor(data, f"{save_dir}/{fname}", label, **imshow_args)

    show_plot = False
    if ax is None:
        ax = plt.gca()
        show_plot = True

    ax.imshow(data.T, **imshow_args)
    ax.set_title(label, loc="left")
    ax.axis("off")

    if show_plot:
        plt.show()
