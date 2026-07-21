import copy
import json

import numpy as np
from aiida import engine, orm, plugins
from aiida.common import exceptions
from aiida.common.links import LinkType

PwCalculation = plugins.CalculationFactory("quantumespresso.pw")
QeBanduppyCalculation = plugins.CalculationFactory("nanotech_empa.qe_banduppy")


class QeBanduppyUnfoldingWorkChain(engine.WorkChain):
    """Unfold QE supercell bands onto a primitive-cell path with BandUPpy."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("pw_code", valid_type=orm.AbstractCode)
        spec.input("banduppy_code", valid_type=orm.AbstractCode)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("parent_folder", valid_type=orm.RemoteData, required=False)
        spec.input("template_remote_folder", valid_type=orm.RemoteData)
        spec.input("template_metadata", valid_type=orm.Dict, required=False)
        spec.input("unfolding_parameters", valid_type=orm.Dict)
        spec.input(
            "run_reference_bands",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
        )
        spec.input_namespace("pseudos", valid_type=orm.Data, dynamic=True)
        spec.input("settings", valid_type=orm.Dict, required=False)
        spec.input("parallelization", valid_type=orm.Dict, required=False)
        spec.input("pw_metadata_options", valid_type=orm.Dict, required=False)
        spec.input("reference_pw_metadata_options", valid_type=orm.Dict, required=False)
        spec.input("banduppy_metadata_options", valid_type=orm.Dict, required=False)
        spec.outline(
            cls.setup,
            cls.inspect_template_remote,
            cls.prepare_folded_kpoints,
            engine.if_(cls.should_run_reference)(
                cls.prepare_reference_inputs,
                cls.run_reference_scf,
                cls.inspect_reference_scf,
                cls.run_reference_bands,
                cls.inspect_reference_bands,
            ),
            cls.run_folded_qe,
            cls.inspect_folded_qe,
            cls.run_banduppy,
            cls.inspect_banduppy,
            cls.results,
        )
        spec.output(
            "folded_qe_remote_folder", valid_type=orm.RemoteData, required=False
        )
        spec.output("folded_kpoints", valid_type=orm.KpointsData, required=False)
        spec.output("reference_structure", valid_type=orm.StructureData, required=False)
        spec.output("reference_scf_kpoints", valid_type=orm.KpointsData, required=False)
        spec.output("reference_kpoints", valid_type=orm.KpointsData, required=False)
        spec.output(
            "reference_scf_remote_folder", valid_type=orm.RemoteData, required=False
        )
        spec.output(
            "reference_bands_remote_folder", valid_type=orm.RemoteData, required=False
        )
        spec.output("reference_bands", valid_type=orm.BandsData, required=False)
        spec.output("reference_bands_parameters", valid_type=orm.Dict, required=False)
        spec.output("mapping_arrays", valid_type=orm.ArrayData, required=False)
        spec.output("mapping_data", valid_type=orm.Dict, required=False)
        spec.output("banduppy_retrieved", valid_type=orm.FolderData, required=False)
        spec.outputs.dynamic = True
        spec.exit_code(
            300,
            "ERROR_TEMPLATE_REMOTE_EMPTY",
            message="The template QE remote folder is missing or empty.",
        )
        spec.exit_code(
            310,
            "ERROR_FOLDED_QE_FAILED",
            message="The folded-kpoints QE calculation failed.",
        )
        spec.exit_code(
            315,
            "ERROR_REFERENCE_SCF_FAILED",
            message="The primitive reference SCF calculation failed.",
        )
        spec.exit_code(
            316,
            "ERROR_REFERENCE_BANDS_FAILED",
            message="The primitive reference bands calculation failed.",
        )
        spec.exit_code(
            320,
            "ERROR_FOLDED_QE_REMOTE_EMPTY",
            message="The folded QE remote folder is missing or empty.",
        )
        spec.exit_code(
            330, "ERROR_BANDUPPY_FAILED", message="The BandUPpy calculation failed."
        )
        spec.exit_code(
            340,
            "ERROR_BANDUPPY_OUTPUT_MISSING",
            message="The BandUPpy calculation did not retrieve unfolding_bands.npz.",
        )

    def setup(self):
        self.ctx.unfolding_parameters = self.inputs.unfolding_parameters.get_dict()
        self.ctx.template_metadata = (
            self.inputs.template_metadata.get_dict()
            if "template_metadata" in self.inputs
            else {}
        )
        self.report(
            "Starting QE BandUPpy unfolding"
            + (
                f" from template UUID {self.ctx.template_metadata.get('template_uuid')}"
                if self.ctx.template_metadata.get("template_uuid")
                else ""
            )
        )

    def inspect_template_remote(self):
        if not _remote_folder_has_entries(self.inputs.template_remote_folder):
            return self.exit_codes.ERROR_TEMPLATE_REMOTE_EMPTY

    def should_run_reference(self):
        return bool(self.inputs.run_reference_bands.value)

    def prepare_folded_kpoints(self):
        import banduppy

        params = self.ctx.unfolding_parameters
        matrix = np.asarray(params["supercell_matrix"], dtype=int)
        path = params["path"]
        labels = params["labels"]
        npoints = params.get("npoints_per_segment", 20)
        if isinstance(npoints, (list, tuple)):
            npoints = tuple(int(value) for value in npoints)
        else:
            npoints = int(npoints)

        band_unfold = banduppy.Unfolding(supercell=matrix, print_log=None)
        (
            kpoints_pbz_full,
            _kpoints_pbz_unique,
            kpoints_sbz,
            mapping,
            special_labels,
        ) = band_unfold.generate_SC_Kpts_from_pc_k_path(
            pathPBZ=path,
            nk=npoints,
            labels=labels,
            kpts_weights=1.0,
            save_all_kpts=False,
            save_sc_kpts=False,
            file_format="qe",
        )

        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.inputs.structure)
        kpoints.set_kpoints(
            kpoints_sbz[:, :3], cartesian=False, weights=np.ones(len(kpoints_sbz))
        )
        kpoints.label = "BandUPpy folded supercell k-points"
        kpoints.description = json.dumps(
            {
                "source": "BandUPpy",
                "supercell_matrix": matrix.tolist(),
                "primitive_path": path,
                "labels": labels,
                "npoints_per_segment": (
                    list(npoints) if isinstance(npoints, tuple) else npoints
                ),
                "kpoint_spacing": params.get("kpoint_spacing"),
            }
        )
        kpoints.store()

        arrays = orm.ArrayData()
        arrays.set_array("supercell_matrix", matrix)
        arrays.set_array("kpoints_pbz_full", np.asarray(kpoints_pbz_full))
        arrays.set_array("kpoints_sbz", np.asarray(kpoints_sbz))
        arrays.set_array(
            "primitive_kline",
            _primitive_kline(
                self.inputs.structure, matrix, np.asarray(kpoints_pbz_full)[:, :3]
            ),
        )
        arrays.label = "BandUPpy QE unfolding k-point arrays"
        arrays.store()

        self.ctx.folded_kpoints = kpoints
        self.ctx.mapping_arrays = arrays
        self.ctx.mapping_data = orm.Dict(dict=_jsonable_mapping(mapping))
        self.ctx.special_labels = orm.Dict(
            dict=_jsonable_special_labels(special_labels)
        )
        self.out("folded_kpoints", kpoints)
        self.out("mapping_arrays", arrays)
        self.out("mapping_data", self.ctx.mapping_data)

    def prepare_reference_inputs(self):
        params = self.ctx.unfolding_parameters
        matrix = np.asarray(params["supercell_matrix"], dtype=int)
        tolerance = float(params.get("primitive_atom_tolerance", 0.08))
        reference_structure = _build_reference_primitive_structure(
            self.inputs.structure, matrix, tolerance=tolerance
        )
        reference_structure.label = "BandUPpy pristine primitive reference structure"
        reference_structure.store()

        reference_scf_kpoints = _reference_scf_kpoints(
            self.inputs.structure,
            reference_structure,
            matrix,
            self.inputs.parent_folder if "parent_folder" in self.inputs else None,
        )
        reference_scf_kpoints.store()

        reference_kpoints = orm.KpointsData()
        reference_kpoints.set_cell_from_structure(reference_structure)
        reference_kpoints.set_kpoints(
            self.ctx.mapping_arrays.get_array("kpoints_pbz_full")[:, :3],
            cartesian=False,
            weights=np.ones(len(self.ctx.mapping_arrays.get_array("kpoints_pbz_full"))),
        )
        try:
            reference_kpoints.labels = _deduplicated_kpoint_labels(
                self.ctx.special_labels.get_dict(),
                self.ctx.mapping_arrays.get_array("kpoints_pbz_full")[:, :3],
            )
        except Exception:
            pass
        reference_kpoints.label = "BandUPpy primitive reference k-points"
        reference_kpoints.store()

        self.ctx.reference_structure = reference_structure
        self.ctx.reference_scf_kpoints = reference_scf_kpoints
        self.ctx.reference_kpoints = reference_kpoints
        self.out("reference_structure", reference_structure)
        self.out("reference_scf_kpoints", reference_scf_kpoints)
        self.out("reference_kpoints", reference_kpoints)

    def run_reference_scf(self):
        matrix = np.asarray(
            self.ctx.unfolding_parameters["supercell_matrix"], dtype=int
        )
        builder = PwCalculation.get_builder()
        builder.code = self.inputs.pw_code
        builder.structure = self.ctx.reference_structure
        builder.pseudos = _filter_pseudos_for_structure(
            dict(self.inputs.pseudos), self.ctx.reference_structure
        )
        builder.kpoints = self.ctx.reference_scf_kpoints
        builder.parameters = orm.Dict(
            dict=_reference_qe_parameters(
                self.inputs.parameters.get_dict(), matrix, calculation="scf"
            )
        )
        if "settings" in self.inputs:
            builder.settings = self.inputs.settings
        if "parallelization" in self.inputs:
            builder.parallelization = self.inputs.parallelization
        if "reference_pw_metadata_options" in self.inputs:
            builder.metadata.options = (
                self.inputs.reference_pw_metadata_options.get_dict()
            )
        elif "pw_metadata_options" in self.inputs:
            builder.metadata.options = self.inputs.pw_metadata_options.get_dict()
        builder.metadata.label = "BandUPpy pristine primitive reference SCF"
        builder.metadata.description = (
            "SCF calculation for the ideal primitive-cell reference bandstructure."
        )
        self.report("Submitting primitive reference SCF calculation")
        return engine.ToContext(reference_scf=self.submit(builder))

    def inspect_reference_scf(self):
        calc = self.ctx.reference_scf
        if not calc.is_finished_ok:
            return self.exit_codes.ERROR_REFERENCE_SCF_FAILED
        self.out("reference_scf_remote_folder", calc.outputs.remote_folder)

    def run_reference_bands(self):
        matrix = np.asarray(
            self.ctx.unfolding_parameters["supercell_matrix"], dtype=int
        )
        builder = PwCalculation.get_builder()
        builder.code = self.inputs.pw_code
        builder.structure = self.ctx.reference_structure
        builder.pseudos = _filter_pseudos_for_structure(
            dict(self.inputs.pseudos), self.ctx.reference_structure
        )
        builder.kpoints = self.ctx.reference_kpoints
        builder.parameters = orm.Dict(
            dict=_reference_qe_parameters(
                self.inputs.parameters.get_dict(), matrix, calculation="bands"
            )
        )
        builder.parent_folder = self.ctx.reference_scf.outputs.remote_folder
        if "settings" in self.inputs:
            builder.settings = self.inputs.settings
        if "parallelization" in self.inputs:
            builder.parallelization = self.inputs.parallelization
        if "reference_pw_metadata_options" in self.inputs:
            builder.metadata.options = (
                self.inputs.reference_pw_metadata_options.get_dict()
            )
        elif "pw_metadata_options" in self.inputs:
            builder.metadata.options = self.inputs.pw_metadata_options.get_dict()
        builder.metadata.label = "BandUPpy pristine primitive reference bands"
        builder.metadata.description = (
            "QE bands calculation for the ideal primitive-cell reference path."
        )
        self.report("Submitting primitive reference bands calculation")
        return engine.ToContext(reference_bands_calc=self.submit(builder))

    def inspect_reference_bands(self):
        calc = self.ctx.reference_bands_calc
        if not calc.is_finished_ok:
            return self.exit_codes.ERROR_REFERENCE_BANDS_FAILED
        self.out("reference_bands_remote_folder", calc.outputs.remote_folder)
        self.out("reference_bands", calc.outputs.output_band)
        self.out("reference_bands_parameters", calc.outputs.output_parameters)

    def run_folded_qe(self):
        builder = PwCalculation.get_builder()
        builder.code = self.inputs.pw_code
        builder.structure = self.inputs.structure
        builder.pseudos = dict(self.inputs.pseudos)
        builder.kpoints = self.ctx.folded_kpoints
        builder.parameters = orm.Dict(
            dict=_folded_qe_parameters(
                self.inputs.parameters.get_dict(), self.ctx.unfolding_parameters
            )
        )
        if "parent_folder" in self.inputs:
            builder.parent_folder = self.inputs.parent_folder
        if "settings" in self.inputs:
            builder.settings = self.inputs.settings
        if "parallelization" in self.inputs:
            builder.parallelization = self.inputs.parallelization
        if "pw_metadata_options" in self.inputs:
            builder.metadata.options = self.inputs.pw_metadata_options.get_dict()
        builder.metadata.label = "BandUPpy folded-kpoints QE bands"
        builder.metadata.description = (
            "QE bands calculation on folded supercell k-points for BandUPpy unfolding."
        )
        self.report("Submitting folded-kpoints QE calculation")
        return engine.ToContext(folded_qe=self.submit(builder))

    def inspect_folded_qe(self):
        calc = self.ctx.folded_qe
        if not calc.is_finished_ok:
            return self.exit_codes.ERROR_FOLDED_QE_FAILED
        self.out("folded_qe_remote_folder", calc.outputs.remote_folder)
        if not _remote_folder_has_entries(calc.outputs.remote_folder):
            return self.exit_codes.ERROR_FOLDED_QE_REMOTE_EMPTY

    def run_banduppy(self):
        calc = self.ctx.folded_qe
        control = calc.inputs.parameters.get_dict().get("CONTROL", {})
        outputs = _created_outputs(calc)
        output_parameters = outputs.get("output_parameters")
        fermi_energy = None
        if output_parameters is not None:
            fermi_energy = output_parameters.get_dict().get("fermi_energy")
        params = copy.deepcopy(self.ctx.unfolding_parameters)
        params.setdefault("kpoint_batch_size", 1)
        params.update(
            {
                "prefix": control.get("prefix", "aiida"),
                "outdir": control.get("outdir", "./out/"),
                "fermi_energy": fermi_energy,
            }
        )

        builder = QeBanduppyCalculation.get_builder()
        builder.code = self.inputs.banduppy_code
        builder.folded_qe_remote_folder = calc.outputs.remote_folder
        builder.parameters = orm.Dict(dict=params)
        builder.mapping_arrays = self.ctx.mapping_arrays
        builder.mapping_data = self.ctx.mapping_data
        builder.special_labels = self.ctx.special_labels
        if "banduppy_metadata_options" in self.inputs:
            builder.metadata.options = self.inputs.banduppy_metadata_options.get_dict()
        builder.metadata.label = "QE BandUPpy unfolding"
        builder.metadata.description = (
            "BandUPpy unfolding of a QE folded-kpoints bands calculation."
        )
        self.report("Submitting BandUPpy unfolding calculation")
        return engine.ToContext(banduppy=self.submit(builder))

    def inspect_banduppy(self):
        if not self.ctx.banduppy.is_finished_ok:
            return self.exit_codes.ERROR_BANDUPPY_FAILED
        retrieved = self.ctx.banduppy.outputs.retrieved
        if "unfolding_bands.npz" not in retrieved.base.repository.list_object_names():
            self.report("BandUPpy output unfolding_bands.npz was not retrieved")
            return self.exit_codes.ERROR_BANDUPPY_OUTPUT_MISSING

    def results(self):
        self.out("banduppy_retrieved", self.ctx.banduppy.outputs.retrieved)


def _folded_qe_parameters(parameters, unfolding_parameters=None):
    result = copy.deepcopy(parameters)
    result.setdefault("CONTROL", {})
    result["CONTROL"]["calculation"] = "bands"
    diagonalization = (unfolding_parameters or {}).get("folded_diagonalization")
    if diagonalization and diagonalization != "inherit":
        result.setdefault("ELECTRONS", {})["diagonalization"] = str(diagonalization)
    return result


def _reference_qe_parameters(parameters, supercell_matrix, *, calculation):
    result = copy.deepcopy(parameters)
    result.setdefault("CONTROL", {})
    result.setdefault("SYSTEM", {})
    result.setdefault("ELECTRONS", {})
    result["CONTROL"]["calculation"] = calculation
    result["CONTROL"]["restart_mode"] = "from_scratch"

    system = result["SYSTEM"]
    original_nbnd = system.get("nbnd")
    for key in (
        "tot_charge",
        "nspin",
        "starting_magnetization",
        "tot_magnetization",
        "constrained_magnetization",
    ):
        system.pop(key, None)
    for key in list(system):
        if str(key).startswith("starting_ns_eigenvalue"):
            system.pop(key, None)

    det = max(1, int(round(abs(float(np.linalg.det(supercell_matrix))))))
    if original_nbnd is not None and calculation == "bands":
        system["nbnd"] = max(1, int(np.ceil(float(original_nbnd) / det)))
    else:
        system.pop("nbnd", None)

    if calculation == "scf":
        for key in ("startingpot", "startingwfc", "diago_full_acc"):
            result["ELECTRONS"].pop(key, None)
    return result


def _filter_pseudos_for_structure(pseudos, structure):
    missing = [kind.name for kind in structure.kinds if kind.name not in pseudos]
    if missing:
        raise ValueError(
            f"No pseudo was provided for primitive reference kinds: {missing}"
        )
    return {kind.name: pseudos[kind.name] for kind in structure.kinds}


def _build_reference_primitive_structure(structure, supercell_matrix, *, tolerance):
    supercell_matrix = np.asarray(supercell_matrix, dtype=float)
    supercell_cell = np.asarray(structure.cell, dtype=float)
    primitive_cell = np.linalg.solve(supercell_matrix, supercell_cell)
    kind_by_name = {kind.name: kind for kind in structure.kinds}
    clusters = []
    for site in structure.sites:
        frac_sc = np.asarray(site.position, dtype=float) @ np.linalg.inv(supercell_cell)
        frac_primitive = np.mod(frac_sc @ supercell_matrix, 1.0)
        assigned = False
        for cluster in clusters:
            if (
                _periodic_fractional_distance(frac_primitive, cluster["center"])
                <= tolerance
            ):
                cluster["items"].append((frac_primitive, site.kind_name))
                cluster["center"] = _periodic_mean(
                    [item[0] for item in cluster["items"]]
                )
                assigned = True
                break
        if not assigned:
            clusters.append(
                {"center": frac_primitive, "items": [(frac_primitive, site.kind_name)]}
            )

    reference = orm.StructureData(cell=primitive_cell, pbc=structure.pbc)
    metadata = []
    for cluster in sorted(
        clusters, key=lambda item: tuple(np.round(item["center"], 10))
    ):
        kind_counts = {}
        for _, kind_name in cluster["items"]:
            kind_counts[kind_name] = kind_counts.get(kind_name, 0) + 1
        majority_kind = sorted(
            kind_counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        majority_positions = [
            frac for frac, kind_name in cluster["items"] if kind_name == majority_kind
        ]
        frac = _periodic_mean(majority_positions)
        kind = kind_by_name[majority_kind]
        symbols = kind.symbols[0] if len(kind.symbols) == 1 else kind.symbols
        reference.append_atom(
            position=np.asarray(frac, dtype=float) @ primitive_cell,
            symbols=symbols,
            name=kind.name,
        )
        metadata.append(
            {
                "kind": majority_kind,
                "counts": kind_counts,
                "fractional_position": [float(value) for value in frac],
            }
        )
    reference.description = json.dumps(
        {
            "source": "clustered from defective supercell",
            "supercell_matrix": np.asarray(supercell_matrix, dtype=int).tolist(),
            "primitive_atom_tolerance": float(tolerance),
            "clusters": metadata,
        }
    )
    return reference


def _periodic_fractional_distance(left, right):
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta -= np.rint(delta)
    return float(np.linalg.norm(delta))


def _periodic_mean(points):
    points = np.asarray(points, dtype=float)
    result = []
    for axis in range(points.shape[1]):
        angles = 2.0 * np.pi * points[:, axis]
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        result.append((mean_angle / (2.0 * np.pi)) % 1.0)
    return np.asarray(result, dtype=float)


def _reference_scf_kpoints(
    supercell_structure, reference_structure, supercell_matrix, parent_folder
):
    mesh = np.ones(3, dtype=int)
    offset = [0.0, 0.0, 0.0]
    if parent_folder is not None:
        creator = _created_by(parent_folder)
        if creator is not None and "kpoints" in creator.inputs:
            try:
                mesh, offset = creator.inputs.kpoints.get_kpoints_mesh()
                mesh = np.asarray(mesh, dtype=int)
            except Exception:
                mesh = np.ones(3, dtype=int)
    factors = np.maximum(
        1,
        np.rint(
            np.linalg.norm(np.asarray(supercell_matrix, dtype=float), axis=0)
        ).astype(int),
    )
    primitive_mesh = np.maximum(1, mesh * factors)
    primitive_mesh = [
        int(value) if pbc else 1
        for value, pbc in zip(primitive_mesh, reference_structure.pbc)
    ]
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(reference_structure)
    kpoints.set_kpoints_mesh(primitive_mesh, offset=offset)
    kpoints.label = "BandUPpy primitive reference SCF k-points"
    kpoints.description = json.dumps(
        {
            "source_supercell_mesh": [int(value) for value in mesh],
            "supercell_matrix_factors": [int(value) for value in factors],
        }
    )
    return kpoints


def _created_by(data_node):
    incoming = data_node.base.links.get_incoming(link_type=LinkType.CREATE).all()
    return incoming[0].node if incoming else None


def _deduplicated_kpoint_labels(special_labels, kpoints):
    labels = []
    previous_label = None
    previous_point = None
    for index_text, label in sorted(
        special_labels.items(), key=lambda item: int(item[0])
    ):
        index = int(index_text)
        point = np.asarray(kpoints[index], dtype=float)
        if (
            previous_label == label
            and previous_point is not None
            and np.allclose(point, previous_point)
        ):
            continue
        labels.append((index, str(label)))
        previous_label = label
        previous_point = point
    return labels


def _remote_folder_has_entries(remote_folder):
    try:
        with remote_folder.computer.get_transport() as transport:
            path = remote_folder.get_remote_path()
            return transport.isdir(path) and bool(transport.listdir(path))
    except exceptions.TransportTaskException:
        return False


def _jsonable_mapping(mapping):
    return {
        str(int(k_index)): {
            str(int(unique_index)): [int(item) for item in k_indices]
            for unique_index, k_indices in unique_map.items()
        }
        for k_index, unique_map in mapping.items()
    }


def _jsonable_special_labels(special_labels):
    return {str(key): str(value) for key, value in dict(special_labels).items()}


def _primitive_kline(structure, supercell_matrix, fractional_kpoints):
    supercell_lattice = np.asarray(structure.cell, dtype=float)
    primitive_lattice = np.linalg.solve(
        np.asarray(supercell_matrix, dtype=float), supercell_lattice
    )
    reciprocal_lattice = 2.0 * np.pi * np.linalg.inv(primitive_lattice).T
    cartesian_kpoints = np.asarray(fractional_kpoints, dtype=float) @ reciprocal_lattice
    if len(cartesian_kpoints) == 0:
        return np.array([], dtype=float)
    distances = np.linalg.norm(np.diff(cartesian_kpoints, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(distances)))


def _created_outputs(calc):
    return {
        triple.link_label: triple.node
        for triple in calc.base.links.get_outgoing().all()
    }
