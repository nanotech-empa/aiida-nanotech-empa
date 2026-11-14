"""Validate explicitly requested rendered images in CubeHandler retrievals."""

from pathlib import PurePosixPath

from PIL import Image


def rendered_images(parameters, open_file):
    """Return image metadata, raising on absent, corrupt or mismatched renders.

    Non-render steps retain their existing output contract. ``open_file`` is the
    retrieved repository's binary-open function; no remote access is performed.
    """
    results = []
    for step in parameters.get("steps", []):
        if step.get("command") != "render":
            continue
        options = step.get("options", {})
        image_format = options.get("format", "png").lower()
        filename = options.get("output")
        if filename is None and step.get("args"):
            filename = f"{PurePosixPath(step['args'][0]).stem}_render.{image_format}"
        if not isinstance(filename, str) or not filename:
            raise ValueError("A render step must record its output filename")
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Render output must be a relative retrieved path")
        if not path.suffix:
            path = path.with_suffix(f".{image_format}")
        with open_file(str(path), mode="rb") as handle, Image.open(handle) as image:
            size, actual_format = image.size, image.format
            image.verify()
        expected = image_format.upper()
        expected = {"JPG": "JPEG", "TIF": "TIFF", "PNM": "PPM", "PS": "EPS"}.get(
            expected, expected
        )
        if actual_format != expected:
            raise ValueError(f"Render format mismatch for {filename}")
        for index, key in enumerate(("width", "height")):
            if key in options and size[index] != options[key]:
                raise ValueError(f"Render {key} mismatch for {filename}")
        results.append({"filename": str(path), "size": size, "format": actual_format})
    return results
