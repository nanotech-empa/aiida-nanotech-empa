"""Optional WFN lookup on imported provenance, with real isolated AiiDA data."""

import errno

import pytest
from aiida import common, orm

from aiida_nanotech_empa.utils.wfn import available_wfn_folder


@pytest.fixture(autouse=True)
def isolated_profile(aiida_profile_tmp):
    assert aiida_profile_tmp.is_test_profile


@pytest.fixture
def imported_folder(aiida_localhost, tmp_path):
    computer = orm.Computer(
        label="unconfigured-imported-computer",
        hostname=aiida_localhost.hostname,
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir=str(tmp_path),
    ).store()
    directory = tmp_path / "old user's folder"
    directory.mkdir()
    folder = orm.RemoteData(computer=computer, remote_path=str(directory)).store()
    assert not computer.is_user_configured(orm.User.collection.get_default())
    return folder, directory


@pytest.mark.parametrize("available", [False, True])
def test_imported_computer_uses_destination_account(
    aiida_localhost, imported_folder, monkeypatch, available
):
    folder, directory = imported_folder
    if available:
        (directory / "aiida-RESTART.wfn").write_bytes(b"original WFN")
    before = folder.base.attributes.all
    extras_before = folder.base.extras.all
    auth_count = orm.QueryBuilder().append(orm.AuthInfo).count()
    node_count = orm.QueryBuilder().append(orm.Node).count()
    real_transport = orm.Computer.get_transport

    def destination_only(computer, *args, **kwargs):
        assert computer.uuid == aiida_localhost.uuid
        return real_transport(computer, *args, **kwargs)

    monkeypatch.setattr(orm.Computer, "get_transport", destination_only)
    messages = []
    result = available_wfn_folder(
        folder, "aiida-RESTART.wfn", aiida_localhost, reporter=messages.append
    )
    if available:
        assert result.computer.uuid == aiida_localhost.uuid
        assert result.get_remote_path() == str(directory)
        assert not result.is_stored
        assert result.base.extras.get("wfn_source_remote_folder_uuid") == folder.uuid
        assert (directory / "aiida-RESTART.wfn").read_bytes() == b"original WFN"
    else:
        assert result is None
        assert "absent or unreadable" in messages[-1]
    assert messages
    assert folder.base.attributes.all == before
    assert folder.base.extras.all == extras_before
    assert orm.QueryBuilder().append(orm.AuthInfo).count() == auth_count
    assert orm.QueryBuilder().append(orm.Node).count() == node_count


def test_unreadable_optional_file_is_not_reused(aiida_localhost, imported_folder):
    folder, directory = imported_folder
    path = directory / "aiida-RESTART.wfn"
    path.write_bytes(b"not readable by this account")
    path.chmod(0)
    try:
        assert available_wfn_folder(folder, path.name, aiida_localhost) is None
    finally:
        path.chmod(0o600)


def test_missing_legacy_source_auth_is_optional(imported_folder):
    folder, _ = imported_folder
    assert available_wfn_folder(folder, "aiida-RESTART.wfn") is None


def test_unconfigured_destination_is_an_error(imported_folder):
    folder, _ = imported_folder
    with pytest.raises(common.InvalidOperation, match="not configured"):
        available_wfn_folder(folder, "aiida-RESTART.wfn", folder.computer)


def test_destination_connection_failure_is_not_hidden(
    aiida_localhost, imported_folder, monkeypatch
):
    folder, _ = imported_folder

    def broken_destination(*args, **kwargs):
        raise ConnectionError("destination unavailable")

    monkeypatch.setattr(orm.Computer, "get_transport", broken_destination)
    with pytest.raises(ConnectionError, match="destination unavailable"):
        available_wfn_folder(folder, "aiida-RESTART.wfn", aiida_localhost)


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), FileNotFoundError("removed")]
)
def test_optional_file_disappearing_or_denied_during_staging(
    aiida_localhost, imported_folder, monkeypatch, error
):
    folder, directory = imported_folder
    source = directory / "aiida-BAND2-RESTART.wfn"
    source.write_bytes(b"immutable image WFN")
    transport_type = type(aiida_localhost.get_transport())

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(transport_type, "copyfile", unavailable)
    assert (
        available_wfn_folder(
            folder,
            source.name,
            aiida_localhost,
            copy_to="aiida-RESTART.wfn",
        )
        is None
    )
    assert source.read_bytes() == b"immutable image WFN"


def test_neb_alias_uses_destination_without_changing_original(
    aiida_localhost, imported_folder
):
    folder, directory = imported_folder
    source = directory / "aiida-BAND2-RESTART.wfn"
    source.write_bytes(b"selected image WFN")
    result = available_wfn_folder(
        folder, source.name, aiida_localhost, copy_to="aiida-RESTART.wfn"
    )
    assert result.computer.uuid == aiida_localhost.uuid
    assert (directory / "aiida-RESTART.wfn").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("status", [2, 127])
def test_failed_destination_probe_command_is_not_missing_file(
    aiida_localhost, imported_folder, monkeypatch, status
):
    folder, _ = imported_folder
    transport_type = type(aiida_localhost.get_transport())
    monkeypatch.setattr(
        transport_type, "exec_command_wait", lambda *args, **kwargs: (status, "", "")
    )
    with pytest.raises(RuntimeError, match=f"exit status {status}"):
        available_wfn_folder(folder, "aiida-RESTART.wfn", aiida_localhost)


def test_unexpected_filesystem_error_is_not_hidden(
    aiida_localhost, imported_folder, monkeypatch
):
    folder, _ = imported_folder
    transport_type = type(aiida_localhost.get_transport())

    def io_error(*args, **kwargs):
        raise OSError(errno.EIO, "I/O failure")

    monkeypatch.setattr(transport_type, "exec_command_wait", io_error)
    with pytest.raises(OSError, match="I/O failure"):
        available_wfn_folder(folder, "aiida-RESTART.wfn", aiida_localhost)


def test_same_computer_returns_existing_provenance(aiida_localhost, tmp_path):
    (tmp_path / "aiida-RESTART.wfn").write_bytes(b"existing WFN")
    folder = orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path)).store()
    assert available_wfn_folder(folder, "aiida-RESTART.wfn", aiida_localhost) == folder
