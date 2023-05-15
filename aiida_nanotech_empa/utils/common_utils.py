import base64
import tempfile


def check_if_calc_ok(self_, prev_calc):
    """Checks if a calculation finished well.

    Args:
        self_: The workchain instance, used for reporting.
        prev_calc (CalcNode): a calculation step

    Returns:
        Bool: True if workchain can continue, False otherwise
    """
    if not prev_calc.is_finished_ok:
        if prev_calc.is_excepted:
            self_.report("ERROR: previous step excepted.")
            return False
        if prev_calc.exit_status is not None and prev_calc.exit_status >= 500:
            self_.report("Warning: previous step: " + prev_calc.exit_message)
        else:
            self_.report("ERROR: previous step: " + str(prev_calc.exit_message))
            return False

    return True


def thumbnail(ase_struc, file_format=None):
    """Prepare binary information."""

    file_format = file_format if file_format else "png"
    with tempfile.NamedTemporaryFile() as tmp:
        ase_struc.write(tmp.name, format=file_format)
        with open(tmp.name, "rb") as raw:
            return base64.b64encode(raw.read()).decode()


def add_extras(node, app, uuid):
    if app not in node.base.extras.all:
        node.base.extras.set(app, [uuid])
    else:
        workflows_list = node.base.extras.all[app]
        workflows_list.append(uuid)
        node.base.extras.set(app, workflows_list)
