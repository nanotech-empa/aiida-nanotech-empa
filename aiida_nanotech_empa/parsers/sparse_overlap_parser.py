from aiida import parsers


class SparseOverlapParser(parsers.Parser):
    """Check that the sparse overlap file was produced and retrieved."""

    def parse(self, **kwargs):
        output_filename = self.node.inputs.output_filename.value
        if output_filename not in self.retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING
