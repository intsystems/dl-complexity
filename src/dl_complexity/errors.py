"""Public errors: missing information is never replaced by a zero score."""


class InputError(ValueError):
    """Invalid, inconsistent or insufficient mathematical inputs."""


class PreparationRequired(InputError):
    """The caller must explicitly prepare the method's artifacts."""
