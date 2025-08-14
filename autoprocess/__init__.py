try:
    from ._version import version as __version__
except ImportError:
    # Fallback for editable installs or when _version.py doesn't exist
    try:
        import setuptools_scm

        __version__ = setuptools_scm.get_version()
    except ImportError:
        __version__ = "unknown"
