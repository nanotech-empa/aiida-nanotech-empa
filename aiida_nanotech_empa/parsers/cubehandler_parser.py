"""Check requested rendered outputs before accepting a CubeHandler calculation."""

from aiida.common.exceptions import NotExistent
from aiida.parsers import Parser

from ..utils.cubehandler_results import rendered_images


class CubeHandlerParser(Parser):
    def parse(self, **kwargs):
        parameters = self.node.inputs.parameters.get_dict()
        if not any(
            step.get("command") == "render" for step in parameters.get("steps", [])
        ):
            return None
        try:
            rendered_images(parameters, self.retrieved.base.repository.open)
        except (OSError, ValueError, SyntaxError, NotExistent) as exc:
            self.logger.error(
                "Requested rendered output is unavailable or invalid: %s", exc
            )
            return self.exit_codes.ERROR_RENDER_OUTPUT
        return None
