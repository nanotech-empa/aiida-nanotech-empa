import base64
from tempfile import NamedTemporaryFile


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
    with NamedTemporaryFile() as tmp:
        ase_struc.write(tmp.name, format=file_format)
        with open(tmp.name, "rb") as raw:
            return base64.b64encode(raw.read()).decode()
        
def add_extras(node,app,extras_label,uuid):
    if app not in node.extras:
        node.set_extra(app,{label:[uuid]})
    else:
        app_extras = node.extras[app]
        if extras_label not in app_extras:
            extras_list = []
        else:
            extras_list = app_extras[extras_label]
        extras_list.append(uuid)
        app_extras[extras_label] = extras_list
        node.set_extra(app, app_extras)        
