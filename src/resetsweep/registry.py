"""Name-to-class registries for pluggable components (methods, scores, ...)."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable


class Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._classes: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            existing = self._classes.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"{self._kind} {name!r} is already registered to {existing.__qualname__}"
                )
            self._classes[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {name!r}; valid: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._classes)

    def construct(self, name: str, /, *args: object, **settings: object) -> object:
        cls = self.get(name)
        signature = inspect.signature(cls)
        try:
            signature.bind(*args, **settings)
        except TypeError as exc:
            valid = [
                p.name
                for p in signature.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            raise ValueError(
                f"cannot construct {self._kind} {name!r}: {exc}; valid: {', '.join(valid)}"
            ) from exc
        return cls(*args, **settings)

    def load(self, package: str) -> None:
        """Import every non-underscore module of `package` so its classes register."""
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.name.startswith("_"):
                importlib.import_module(f"{package}.{info.name}")


def make_registry(kind: str) -> Registry:
    return Registry(kind)
