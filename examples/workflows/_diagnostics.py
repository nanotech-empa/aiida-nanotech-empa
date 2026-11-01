"""Failure diagnostics for real-code examples; independent of calculation inputs."""

from aiida import orm


def assert_finished_ok(identifier):
    """Show child scheduler output when a real-code example fails."""
    process = orm.load_node(identifier)
    if process.is_finished_ok:
        return
    identifiers = [str(process.uuid)] + [
        str(node.uuid) for node in process.called_descendants
    ]
    del process
    lines = []
    for uuid in identifiers:
        node = orm.load_node(uuid)
        lines.append(
            f"{node.process_label or node.node_type} {uuid}: {node.process_state}, exit_status={node.exit_status}, exit_message={node.exit_message}"
        )
        if isinstance(node, orm.CalcJobNode):
            # A parser-less calculation can finish successfully despite a failed
            # executable and missing NPZ output. Include its scheduler streams too.
            for label, read in (
                ("scheduler stderr", node.get_scheduler_stderr),
                ("scheduler stdout", node.get_scheduler_stdout),
            ):
                try:
                    content = read()
                except (OSError, KeyError) as exc:
                    content = f"Cannot read {label}: {exc}"
                if content:
                    lines.append(
                        f"--- {label} for {uuid} (last 16000 characters) ---\n{content[-16000:]}"
                    )
        del node
    raise AssertionError(
        "Real-code workflow did not finish successfully:\n" + "\n".join(lines)
    )
