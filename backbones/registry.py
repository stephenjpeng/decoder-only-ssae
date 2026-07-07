from __future__ import annotations

from typing import Callable, Type

from backbones.base import Backbone

_REGISTRY: dict[str, Type[Backbone]] = {}


def register_backbone(name: str) -> Callable[[Type[Backbone]], Type[Backbone]]:
    def decorator(cls: Type[Backbone]) -> Type[Backbone]:
        if name in _REGISTRY:
            raise ValueError(f"backbone {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_backbone(name: str, **kwargs) -> Backbone:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_backbones() -> list[str]:
    return sorted(_REGISTRY)
