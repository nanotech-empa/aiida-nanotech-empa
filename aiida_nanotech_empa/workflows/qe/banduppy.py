import copy
import json

import numpy as np
from aiida import engine, orm, plugins
from aiida.common import exceptions


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
        spec.input_namespace("pseudos", valid_type=orm.Data, dynamic=True)
        spec.input("settings", valid_type=orm.Dict, required=False)
        spec.input("parallelization", valid_type=orm.Dict, required=False)
        spec.input("pw_metadata_options", valid_type=orm.Dict, required=False)
        spec.input("banduppy_metadata_options", valid_type=orm.Dict, required=False)
        spec.outline(
            cls.setup,
            cls.inspect_template_remote,
            cls.prepare_folded_kpoints,
            cls.run_folded_qe,
            cls.inspect_folded_qe,
            cls.run_banduppy,
            cls.inspect_banduppy,
            cls.results,
        )
        spec.output("folded_qe_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("folded_kpoints", valid_type=orm.KpointsData, required=False)
        spec.output("mapping_arrays", valid_type=orm.ArrayData, required=False)
        spec.output("mapping_data", valid_type=orm.Dict, required=False)
        spec.output("banduppy_retrieved", valid_type=orm.FolderData, required=False)
        spec.outputs.dynamic = True
        spec.exit_code(300, "ERROR_TEMPLATE_REMOTE_EMPTY", message="The template QE remote folder is missing or empty.")
        spec.exit_code(310, "ERROR_FOLDED_QE_FAILED", message="The folded-kpoints QE calculation failed.")
        spec.exit_code(320, "ERROR_FOLDED_QE_REMOTE_EMPTY", message="The folded QE remote folder is missing or empty.")
        spec.exit_code(330, "ERROR_BANDUPPY_FAILED", message="The BandUPpy calculation failed.")

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
        kpoints.set_kpoints(kpoints_sbz[:, :3], cartesian=False, weights=np.ones(len(kpoints_sbz)))
        kpoints.label = "BandUPpy folded supercell k-points"
        kpoints.description = json.dumps(
            {
                "source": "BandUPpy",
                "supercell_matrix": matrix.tolist(),
                "primitive_path": path,
                "labels": labels,
                "npoints_per_segment": list(npoints) if isinstance(npoints, tuple) else npoints,
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
            _primitive_kline(self.inputs.structure, matrix, np.asarray(kpoints_pbz_full)[:, :3]),
        )
        arrays.label = "BandUPpy QE unfolding k-point arrays"
        arrays.store()

        self.ctx.folded_kpoints = kpoints
        self.ctx.mapping_arrays = arrays
        self.ctx.mapping_data = orm.Dict(dict=_jsonable_mapping(mapping))
        self.ctx.special_labels = orm.Dict(dict=_jsonable_special_labels(special_labels))
        self.out("folded_kpoints", kpoints)
        self.out("mapping_arrays", arrays)
        self.out("mapping_data", self.ctx.mapping_data)

    def run_folded_qe(self):
        builder = PwCalculation.get_builder()
        builder.code = self.inputs.pw_code
        builder.structure = self.inputs.structure
        builder.pseudos = dict(self.inputs.pseudos)
        builder.kpoints = self.ctx.folded_kpoints
        builder.parameters = orm.Dict(dict=_folded_qe_parameters(self.inputs.parameters.get_dict()))
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

    def results(self):
        self.out("banduppy_retrieved", self.ctx.banduppy.outputs.retrieved)


def _folded_qe_parameters(parameters):
    result = copy.deepcopy(parameters)
    result.setdefault("CONTROL", {})
    result["CONTROL"]["calculation"] = "bands"
    return result


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
    primitive_lattice = np.linalg.solve(np.asarray(supercell_matrix, dtype=float), supercell_lattice)
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
