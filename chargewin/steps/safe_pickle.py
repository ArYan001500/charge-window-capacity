"""Restricted unpickler for the HUST pickles: only numpy, pandas and builtin containers are allowed."""
import io
import pickle

ALLOWED_PREFIXES = ("numpy", "pandas")
ALLOWED_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "slice", "range", "complex", "bytearray", "object"}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(ALLOWED_PREFIXES):
            return super().find_class(module, name)
        if module == "builtins" and name in ALLOWED_BUILTINS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def safe_loads(data: bytes):
    return SafeUnpickler(io.BytesIO(data)).load()
