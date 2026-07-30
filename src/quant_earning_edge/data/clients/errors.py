"""Shared external-provider failure types."""


class ProviderRequestError(RuntimeError):
    """A provider could not return a successful response."""


class ProviderResponseError(RuntimeError):
    """A provider returned data that violated its expected contract."""
