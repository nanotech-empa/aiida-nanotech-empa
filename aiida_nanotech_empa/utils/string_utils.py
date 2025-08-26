def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def list_to_string_range(lst, shift=1):
    """Converts a list like [0, 2, 3, 4] into a string like '1 3..5'.
    Shift used when e.g. for a user interface numbering starts from 1 not from 0"""
    return " ".join(
        [
            (
                f"{t[0] + shift}..{t[1] + shift}"
                if isinstance(t, tuple)
                else str(t + shift)
            )
            for t in find_ranges(sorted(lst))
        ]
    )
