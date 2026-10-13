from aiida import parsers

from ..plugins.bader import BADER_OUTPUT_FILES


class BaderParser(parsers.Parser):
    """Check that the Bader output files were produced and retrieved."""

    def parse(self, **kwargs):
        retrieved = set(self.retrieved.base.repository.list_object_names())
        if not set(BADER_OUTPUT_FILES) <= retrieved:
            return self.exit_codes.ERROR_OUTPUT_FILES_MISSING
