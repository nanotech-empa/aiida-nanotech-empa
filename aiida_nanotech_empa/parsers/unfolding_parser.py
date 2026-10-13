from aiida import parsers


class Cp2kUnfoldingParser(parsers.Parser):
    """Check that the unfolded band file (and, if requested, the PDOS
    projection file) was produced and retrieved."""

    def parse(self, **kwargs):
        retrieved_names = self.retrieved.base.repository.list_object_names()
        if self.node.inputs.output_filename.value not in retrieved_names:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING
        if (
            self.node.inputs.parse_pdos_projections.value
            and self.node.inputs.pdos_projection_filename.value not in retrieved_names
        ):
            return self.exit_codes.ERROR_PDOS_PROJECTION_FILE_MISSING
