"""Version identifiers. Bump EXTRACTOR_VERSION whenever extraction output can change."""

from importlib.metadata import PackageNotFoundError, version

import essentia

EXTRACTOR_VERSION = "1.0.0"
# The exact installed build (e.g. 2.1b6.dev1389); essentia.__version__ only says "2.1-beta6-dev".
try:
    ESSENTIA_VERSION = version("essentia")
except PackageNotFoundError:  # built from source without package metadata
    ESSENTIA_VERSION = essentia.__version__


def pipeline_version(model_version: str) -> str:
    """Identifies everything that determines an analysis result. Cached results are only
    reused for the same pipeline version; a change in any part creates new rows."""
    return f"extractor-{EXTRACTOR_VERSION}+essentia-{ESSENTIA_VERSION}+model-{model_version}"
