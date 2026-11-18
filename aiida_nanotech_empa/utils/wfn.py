"""Access optional CP2K restart files using the destination account.

Imported computers describe provenance; they need not have local AuthInfo.
Connection failures on the selected destination remain errors. Only unavailable
optional files (including permission-denied files) mean "start without a WFN".
"""

import errno
import logging
import posixpath
import shlex

from aiida import common, orm

LOGGER = logging.getLogger(__name__)


def available_wfn_folder(
    source_folder, filename, selected_computer=None, copy_to=None, reporter=None
):
    """Return a usable restart folder, or None for an unavailable optional WFN.

    The returned folder is unstored when an imported computer must be replaced
    by the selected access identity. The original RemoteData is never modified.
    Explicit restart inputs do not call this automatic-discovery helper.
    """
    report = reporter or LOGGER.warning
    source_computer = source_folder.computer
    computer = selected_computer or source_computer
    if computer.hostname != source_computer.hostname:
        report("Automatic WFN reuse skipped: source and destination hosts differ.")
        return None

    user = orm.User.collection.get_default()
    if not computer.is_user_configured(user):
        if selected_computer is not None:
            raise common.InvalidOperation(
                f"Selected CP2K computer '{computer.label}' is not configured "
                "for the current AiiDA user."
            )
        report("Automatic WFN reuse skipped: source computer is not configured.")
        return None

    if not computer.is_user_enabled(user):
        raise common.InvalidOperation(
            f"Selected CP2K computer '{computer.label}' is disabled for this user."
        )

    directory = source_folder.get_remote_path()
    path = posixpath.join(directory, filename)
    # Opening the selected transport is deliberately outside file-error handling:
    # a broken destination connection must not masquerade as a missing WFN.
    with computer.get_transport() as transport:
        try:
            quoted = shlex.quote(path)
            status, _, _ = transport.exec_command_wait(
                f"test -f {quoted} && test -r {quoted}"
            )
            if status == 1:
                report(
                    "Automatic WFN reuse skipped: restart file is absent or "
                    f"unreadable with the account selected on '{computer.label}'."
                )
                return None
            if status != 0:
                raise RuntimeError(
                    f"WFN access check failed on '{computer.label}' "
                    f"with command exit status {status}."
                )
            if copy_to is not None and copy_to != filename:
                transport.copyfile(path, posixpath.join(directory, copy_to))
        except OSError as exc:
            if not isinstance(exc, (FileNotFoundError, PermissionError)) and (
                exc.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM)
            ):
                raise
            report(
                "Automatic WFN reuse skipped: restart file cannot be accessed "
                f"or staged with the account selected on '{computer.label}'."
            )
            return None

    if computer.uuid == source_computer.uuid:
        return source_folder

    folder = orm.RemoteData(computer=computer, remote_path=directory)
    folder.base.extras.set("wfn_source_remote_folder_uuid", source_folder.uuid)
    report(
        "Automatic WFN reuse verified through the selected computer "
        f"'{computer.label}'; imported source computer was left unchanged."
    )
    return folder
