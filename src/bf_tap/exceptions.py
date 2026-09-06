class BFTapError(Exception):
    """Base class for expected, user-facing pipeline failures."""


class ContractError(BFTapError):
    """Input data or configuration violates a frozen contract."""


class ProtectedLabelError(BFTapError):
    """Protected labels were requested outside the scoring gate."""
