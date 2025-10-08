"""Utility helpers for dynamic component initialization within the trainer."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast, overload

OptionalDefaults = Dict[str, Callable[[], Any] | Any]
T = TypeVar("T")


def load_class(path: str) -> type[Any]:
    """Load a class from its fully qualified import path."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_component(
    cls: type[Any],
    provided_kwargs: Optional[Dict[str, Any]] = None,
    optional_defaults: Optional[OptionalDefaults] = None,
) -> Any:
    """Instantiate `cls`, filling optional kwargs when the constructor accepts them."""
    kwargs = dict(provided_kwargs or {})
    if optional_defaults:
        signature = inspect.signature(cls.__init__)
        for name, value in optional_defaults.items():
            if name in kwargs or name not in signature.parameters:
                continue
            kwargs[name] = value() if callable(value) else value
    return cls(**kwargs)


def instantiate_from_spec(
    spec: Union[str, Dict[str, Any]],
    *,
    spec_name: str,
    optional_defaults: Optional[OptionalDefaults] = None,
    dict_requires_type: bool = True,
    dict_default_cls: type[Any] | None = None,
) -> Any:
    """Instantiate a component from a string or dict spec."""
    if isinstance(spec, str):
        cls = load_class(spec)
        return instantiate_component(cls, optional_defaults=optional_defaults)

    if isinstance(spec, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        spec_conf = dict(spec)
        type_path = spec_conf.pop("type", None)
        if type_path is None:
            if dict_requires_type:
                raise ValueError(f"{spec_name} dict must have a 'type' key with the class full name")
            if dict_default_cls is None:
                raise ValueError(f"{spec_name} dict missing 'type' and no default class provided")
            cls = dict_default_cls
        else:
            cls = load_class(type_path)
        return instantiate_component(cls, spec_conf, optional_defaults)

    raise TypeError(f"{spec_name} spec must be a string or dict (got {type(spec)}).")


def _ensure_expected_type(
    instance: Any,
    expected_type: type[T],
    spec_name: str,
    type_error_fmt: str | None,
) -> T:
    if not isinstance(instance, expected_type):
        type_name = str(type(instance))  # type: ignore
        if type_error_fmt:
            raise TypeError(type_error_fmt.format(type_name=type_name, expected_type=expected_type.__name__))
        raise TypeError(f"{spec_name} factory returned {type_name}, which is not a {expected_type.__name__} subclass.")
    return instance


@overload
def build_component(
    spec: Union[T, str, Dict[str, Any], type[T], Callable[[], T], None],
    *,
    expected_type: type[T],
    spec_name: str,
    default_factory: Callable[[], T],
    allow_none: bool = ...,
    optional_defaults: Optional[OptionalDefaults] = ...,
    dict_requires_type: bool = ...,
    dict_default_cls: type[T] | None = ...,
    allow_class: bool = ...,
    allow_factory: bool = ...,
    allow_str: bool = ...,
    type_error_fmt: str | None = ...,
    invalid_type_error_fmt: str | None = ...,
) -> T: ...


@overload
def build_component(
    spec: Union[T, str, Dict[str, Any], type[T], Callable[[], T], None],
    *,
    expected_type: type[T],
    spec_name: str,
    default_factory: None = ...,
    allow_none: bool,
    optional_defaults: Optional[OptionalDefaults] = ...,
    dict_requires_type: bool = ...,
    dict_default_cls: type[T] | None = ...,
    allow_class: bool = ...,
    allow_factory: bool = ...,
    allow_str: bool = ...,
    type_error_fmt: str | None = ...,
    invalid_type_error_fmt: str | None = ...,
) -> T | None: ...


@overload
def build_component(
    spec: Union[T, str, Dict[str, Any], type[T], Callable[[], T], None],
    *,
    expected_type: type[T],
    spec_name: str,
    default_factory: None = ...,
    allow_none: bool = ...,
    optional_defaults: Optional[OptionalDefaults] = ...,
    dict_requires_type: bool = ...,
    dict_default_cls: type[T] | None = ...,
    allow_class: bool = ...,
    allow_factory: bool = ...,
    allow_str: bool = ...,
    type_error_fmt: str | None = ...,
    invalid_type_error_fmt: str | None = ...,
) -> T | None: ...


def build_component(
    spec: Union[T, str, Dict[str, Any], type[T], Callable[[], T], None],
    *,
    expected_type: type[T],
    spec_name: str,
    default_factory: Callable[[], T] | None = None,
    allow_none: bool = False,
    optional_defaults: Optional[OptionalDefaults] = None,
    dict_requires_type: bool = True,
    dict_default_cls: type[T] | None = None,
    allow_class: bool = False,
    allow_factory: bool = False,
    allow_str: bool = True,
    type_error_fmt: str | None = None,
    invalid_type_error_fmt: str | None = None,
) -> T | None:
    """Return a component instance created from ``spec``."""
    if isinstance(spec, expected_type):
        return cast(T, spec)

    if spec is None:
        if default_factory is not None:
            instance = default_factory()
            return _ensure_expected_type(instance, expected_type, spec_name, type_error_fmt)
        if allow_none:
            return None
        raise ValueError(
            invalid_type_error_fmt.format(actual_type=type(spec), expected_type=expected_type.__name__)
            if invalid_type_error_fmt
            else f"{spec_name} cannot be None."
        )

    if allow_class and isinstance(spec, type) and issubclass(spec, expected_type):
        instance = instantiate_component(spec, optional_defaults=optional_defaults)
        return _ensure_expected_type(instance, expected_type, spec_name, type_error_fmt)

    if allow_factory and callable(spec) and not isinstance(spec, type):  # type: ignore
        instance = spec()
        return _ensure_expected_type(instance, expected_type, spec_name, type_error_fmt)

    if allow_str and isinstance(spec, str):
        instance = instantiate_from_spec(
            spec,
            spec_name=spec_name,
            optional_defaults=optional_defaults,
            dict_requires_type=dict_requires_type,
            dict_default_cls=dict_default_cls,
        )
        return _ensure_expected_type(instance, expected_type, spec_name, type_error_fmt)

    if isinstance(spec, dict):
        instance = instantiate_from_spec(
            spec,  # type: ignore
            spec_name=spec_name,
            optional_defaults=optional_defaults,
            dict_requires_type=dict_requires_type,
            dict_default_cls=dict_default_cls,
        )
        return _ensure_expected_type(instance, expected_type, spec_name, type_error_fmt)

    if invalid_type_error_fmt:
        raise ValueError(invalid_type_error_fmt.format(actual_type=type(spec), expected_type=expected_type.__name__))  # type: ignore[arg-type]

    type_name = str(type(spec))  # type: ignore
    raise ValueError(f"Invalid {spec_name} type: {type_name}. Expected {expected_type.__name__}, str, dict, or None.")


__all__ = ["OptionalDefaults", "build_component", "instantiate_component", "instantiate_from_spec", "load_class"]
