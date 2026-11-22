"""Domain errors mapped to HTTP status codes in the API layer."""


class LocalizationError(Exception):
    """Base class for localization pipeline errors."""


class BadRequestError(LocalizationError):
    """Invalid input that should become HTTP 400."""


class LocalizationNotConverged(LocalizationError):
    """Matching failed / confidence too low → HTTP 422."""


class MapFetchError(LocalizationError):
    """Map tiles unavailable or invalid → typically HTTP 500."""
