import sys
import textwrap
from pathlib import Path

import pytest

from resetsweep.registry import make_registry


def test_register_and_get_returns_class() -> None:
    registry = make_registry("method")

    @registry.register("a")
    class A:
        pass

    assert registry.get("a") is A


def test_register_returns_the_class_unchanged() -> None:
    registry = make_registry("method")

    class A:
        pass

    assert registry.register("a")(A) is A


def test_duplicate_name_different_class_raises_value_error() -> None:
    registry = make_registry("method")

    @registry.register("a")
    class First:
        pass

    class Second:
        pass

    with pytest.raises(ValueError, match="'a' is already registered"):
        registry.register("a")(Second)


def test_registering_same_class_twice_is_allowed() -> None:
    registry = make_registry("method")

    class A:
        pass

    registry.register("a")(A)
    registry.register("a")(A)

    assert registry.names() == ["a"]


def test_get_unknown_lists_valid_names() -> None:
    registry = make_registry("method")
    registry.register("b")(type("B", (), {}))
    registry.register("a")(type("A", (), {}))

    with pytest.raises(KeyError, match="valid: a, b"):
        registry.get("zzz")


def test_names_are_sorted() -> None:
    registry = make_registry("method")
    for name in ("c", "a", "b"):
        registry.register(name)(type(name.upper(), (), {}))

    assert registry.names() == ["a", "b", "c"]


class _Configurable:
    def __init__(self, x: int, *, y: int = 1) -> None:
        self.x = x
        self.y = y


def test_construct_passes_settings() -> None:
    registry = make_registry("method")
    registry.register("c")(_Configurable)

    obj = registry.construct("c", 1, y=5)

    assert isinstance(obj, _Configurable)
    assert (obj.x, obj.y) == (1, 5)


def test_construct_rejects_unknown_setting_listing_valid() -> None:
    registry = make_registry("method")
    registry.register("c")(_Configurable)

    with pytest.raises(ValueError, match=r"valid: x, y"):
        registry.construct("c", 1, z=2)


def test_construct_unknown_name_raises_key_error() -> None:
    registry = make_registry("method")

    with pytest.raises(KeyError, match="unknown method 'nope'"):
        registry.construct("nope")


def test_load_imports_public_modules_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "plugpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_reg.py").write_text(
        "from resetsweep.registry import make_registry\nregistry = make_registry('plug')\n"
    )
    for module, name in (("alpha", "alpha"), ("_hidden", "hidden")):
        (pkg / f"{module}.py").write_text(
            textwrap.dedent(
                f"""
                from plugpkg._reg import registry

                @registry.register("{name}")
                class Plug:
                    pass
                """
            )
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in [k for k in sys.modules if k.startswith("plugpkg")]:
        monkeypatch.delitem(sys.modules, key)

    from plugpkg._reg import registry  # type: ignore[import-not-found]

    registry.load("plugpkg")
    registry.load("plugpkg")  # second call is a no-op, not a duplicate-name error

    assert registry.names() == ["alpha"]
    for key in [k for k in sys.modules if k.startswith("plugpkg")]:
        monkeypatch.delitem(sys.modules, key)
