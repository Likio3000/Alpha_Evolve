from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["BaseModel", "Field", "ConfigDict", "ValidationError"]


_UNSET = object()
_RESERVED_FIELD_NAMES = {"model_config", "__fields__"}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


class ValidationError(Exception):
    """Light-weight replacement for :class:`pydantic.ValidationError`."""

    def __init__(self, errors: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__("Validation failed")
        self._errors = errors or []

    def errors(self) -> List[Dict[str, Any]]:
        return list(self._errors)


@dataclasses.dataclass
class FieldInfo:
    default: Any = _UNSET
    required: bool = False
    ge: Optional[float] = None
    le: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ConfigDict(dict):
    """Simple dictionary wrapper mirroring :func:`pydantic.ConfigDict`."""

    def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - trivial
        super().__init__(**kwargs)


def Field(default: Any = _UNSET, **kwargs: Any) -> FieldInfo:
    """Capture metadata for a model field."""

    required = False
    if default is Ellipsis:
        required = True
        default = _UNSET
    info = FieldInfo(default=default, required=required)
    for key in ("ge", "le", "gt", "lt", "min_length", "max_length"):
        if key in kwargs:
            setattr(info, key, kwargs[key])
    return info


class _BaseModelMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: Dict[str, Any]):
        annotations: Dict[str, Any] = {}
        for base in reversed(bases):
            annotations.update(getattr(base, "__annotations__", {}))
        annotations.update(namespace.get("__annotations__", {}))

        fields: Dict[str, FieldInfo] = {}
        for attr, _ in annotations.items():
            if attr in _RESERVED_FIELD_NAMES:
                continue
            raw_default = namespace.get(attr, getattr(bases[-1], attr, _UNSET)) if bases else namespace.get(attr, _UNSET)
            if isinstance(raw_default, FieldInfo):
                info = raw_default
            elif raw_default is _UNSET:
                info = FieldInfo(default=_UNSET, required=True)
            else:
                info = FieldInfo(default=raw_default, required=False)
            fields[attr] = info
            if attr in namespace:
                namespace.pop(attr)
        namespace.setdefault("model_config", ConfigDict())
        namespace["__fields__"] = fields
        return super().__new__(mcls, name, bases, namespace)


class BaseModel(metaclass=_BaseModelMeta):
    model_config: ConfigDict
    __fields__: Dict[str, FieldInfo]

    def __init__(self, **data: Any) -> None:
        errors: List[Dict[str, Any]] = []
        values: Dict[str, Any] = {}
        provided = dict(data)
        if isinstance(self.model_config, Mapping):
            config_extra = self.model_config.get("extra", "ignore")
        else:
            config_extra = getattr(self.model_config, "extra", "ignore")

        for name, info in self.__fields__.items():
            if name in provided:
                values[name] = provided.pop(name)
            elif not info.required:
                if info.default is _UNSET:
                    values[name] = None
                else:
                    default_value = info.default
                    if isinstance(default_value, (dict, list, set)):
                        default_value = default_value.copy()  # pragma: no cover - defensive
                    values[name] = default_value
            else:
                errors.append({"loc": [name], "msg": "Field required"})

        if provided:
            if config_extra == "forbid":
                for extra_name in provided:
                    errors.append({"loc": [extra_name], "msg": "Extra fields not permitted"})
            elif config_extra == "allow":
                values.update(provided)

        for name, value in list(values.items()):
            if name not in self.__fields__:
                continue
            err = self._validate_value(name, value)
            if err is not None:
                errors.append(err)

        if errors:
            raise ValidationError(errors)

        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data: Any) -> "BaseModel":
        if not _is_mapping(data):
            raise ValidationError([{"loc": ["__root__"], "msg": "Input should be a mapping"}])
        return cls(**dict(data))

    def model_dump(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__fields__
            if hasattr(self, name)
        }

    def _validate_value(self, name: str, value: Any) -> Optional[Dict[str, Any]]:
        info = self.__fields__[name]
        if value is None:
            return None
        for attr, op in (("ge", lambda a, b: a >= b), ("le", lambda a, b: a <= b),
                         ("gt", lambda a, b: a > b), ("lt", lambda a, b: a < b)):
            bound = getattr(info, attr)
            if bound is not None:
                try:
                    if not op(value, bound):
                        return {"loc": [name], "msg": f"Value should be {attr} {bound}"}
                except TypeError:
                    return {"loc": [name], "msg": "Invalid comparison"}
        length = None
        if isinstance(value, (str, list, tuple, set, dict)):
            length = len(value)
        if info.min_length is not None and length is not None and length < info.min_length:
            return {"loc": [name], "msg": f"Length should be >= {info.min_length}"}
        if info.max_length is not None and length is not None and length > info.max_length:
            return {"loc": [name], "msg": f"Length should be <= {info.max_length}"}
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        fields = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in self.__fields__)
        return f"{self.__class__.__name__}({fields})"
