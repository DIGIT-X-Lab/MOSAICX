"""Convert tabular schema/catalog files into MOSAICX template specs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mosaicx.pipelines.schema_gen import FieldSpec, SchemaSpec, normalize_schema_spec

_TRUE_VALUES = {"1", "true", "yes", "y", "ja", "required", "mandatory", "pflicht"}
_FALSE_VALUES = {"0", "false", "no", "n", "nein", "optional", "none", ""}

_CATALOG_COLUMNS = {
    "field_name",
    "field_type",
    "field_description",
    "field_is_mandatory",
}
_STANDARD_NAME_COLUMNS = {"field_name", "name"}
_STANDARD_TYPE_COLUMNS = {"type", "field_type"}
_STANDARD_DESCRIPTION_COLUMNS = {"description", "field_description", "note", "field_note"}
_STANDARD_REQUIRED_COLUMNS = {"required", "field_is_mandatory", "mandatory"}
_STANDARD_VALUES_COLUMNS = {"values", "enum_values", "field_values", "allowed_values"}
_CATALOG_VALUE_COLUMNS = {"field_value", "value_code", "code"}
_VALUE_LABEL_COLUMNS = {
    "field_value_shortdesc",
    "field_value_description",
    "value_description",
    "value_label",
    "label",
}
_CATALOG_ID_COLUMNS = {"catalog_id", "catalog_name"}
_CATALOG_VERSION_COLUMNS = {"catalog_version", "catalog_version_description"}


@dataclass(frozen=True)
class TableTemplateColumns:
    """Column mapping for table-driven template creation."""

    name: str | None = None
    type: str | None = None
    description: str | None = None
    required: str | None = None
    values: str | None = None
    value_label: str | None = None
    split_by: str | None = None
    catalog_id: str | None = None
    catalog_version: str | None = None


def table_to_schema_spec(
    path: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    columns: TableTemplateColumns | None = None,
) -> SchemaSpec:
    """Infer a :class:`SchemaSpec` from a CSV or Excel file.

    Two common table shapes are supported:

    - Field catalogs with columns such as ``field_name``, ``field_type``,
      ``field_description``, ``field_is_mandatory``, and optional
      ``field_value`` rows for enum values.
    - Ordinary data tables, where each source column becomes a template field.
    """
    specs = table_to_schema_specs(
        path,
        name=name,
        description=description,
        columns=columns,
    )
    if len(specs) != 1:
        raise ValueError("table_to_schema_spec expected one schema; use table_to_schema_specs for split output.")
    return specs[0]


def table_to_schema_specs(
    path: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    columns: TableTemplateColumns | None = None,
) -> list[SchemaSpec]:
    """Infer one or more :class:`SchemaSpec` objects from a CSV or Excel file.

    When ``columns.split_by`` is set, rows are grouped by that column and one
    schema is produced per group. This is useful for data dictionaries that
    contain many logical forms in one file.
    """
    table_path = Path(path)
    df = _read_table(table_path)
    df = _clean_dataframe(df)
    if df.empty or not list(df.columns):
        raise ValueError(f"Table has no usable columns: {table_path}")

    mapping = _resolve_column_mapping(df, columns or TableTemplateColumns())
    split_by = mapping.split_by

    specs: list[SchemaSpec] = []
    if split_by is None:
        specs.append(
            _dataframe_to_best_schema_spec(
                df,
                table_path=table_path,
                name=name,
                mapping=mapping,
            )
        )
    else:
        for raw_group_value, group in df.groupby(split_by, sort=False, dropna=False):
            group_value = str(raw_group_value).strip()
            if not group_value:
                continue
            group_name = _split_group_class_name(group_value, prefix=name)
            spec = _dataframe_to_best_schema_spec(
                group,
                table_path=table_path,
                name=group_name,
                mapping=mapping,
                group_label=f"{split_by}={group_value}",
            )
            specs.append(spec)

    if description:
        extra = " ".join(str(description).split())
        for spec in specs:
            if spec.description:
                spec.description = f"{spec.description} {extra}"
            else:
                spec.description = extra

    return [
        normalize_schema_spec(spec, default_class_name=_default_class_name(table_path))
        for spec in specs
    ]


def _read_table(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a core dependency
        raise RuntimeError("pandas is required for --from-table.") from exc

    if suffix in {".csv", ".tsv"}:
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str)
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(
        f"Unsupported table format {suffix!r}. Use CSV, TSV, XLSX, or XLS."
    )


def _clean_dataframe(df: Any) -> Any:
    df = df.copy()
    df.columns = [_clean_column_name(str(col)) for col in df.columns]
    df = df.loc[:, [bool(str(col).strip()) for col in df.columns]]
    return df.fillna("")


def _clean_column_name(name: str) -> str:
    return " ".join(name.lstrip("\ufeff").strip().split())


def _resolve_column_mapping(
    df: Any,
    columns: TableTemplateColumns,
) -> TableTemplateColumns:
    return TableTemplateColumns(
        name=_resolve_column(df, columns.name, _STANDARD_NAME_COLUMNS, required=False),
        type=_resolve_column(df, columns.type, _STANDARD_TYPE_COLUMNS, required=False),
        description=_resolve_column(
            df,
            columns.description,
            _STANDARD_DESCRIPTION_COLUMNS,
            required=False,
        ),
        required=_resolve_column(
            df,
            columns.required,
            _STANDARD_REQUIRED_COLUMNS,
            required=False,
        ),
        values=_resolve_column(
            df,
            columns.values,
            _STANDARD_VALUES_COLUMNS | _CATALOG_VALUE_COLUMNS,
            required=False,
        ),
        value_label=_resolve_column(
            df,
            columns.value_label,
            _VALUE_LABEL_COLUMNS,
            required=False,
        ),
        split_by=_resolve_column(df, columns.split_by, set(), required=False),
        catalog_id=_resolve_column(
            df,
            columns.catalog_id,
            _CATALOG_ID_COLUMNS,
            required=False,
        ),
        catalog_version=_resolve_column(
            df,
            columns.catalog_version,
            _CATALOG_VERSION_COLUMNS,
            required=False,
        ),
    )


def _resolve_column(
    df: Any,
    explicit: str | None,
    candidates: set[str],
    *,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Column {explicit!r} not found. Available columns: {', '.join(map(str, df.columns))}"
            )
        return explicit
    found = _first_matching_column(df, candidates)
    if required and found is None:
        raise ValueError(
            f"Missing required column. Expected one of: {', '.join(sorted(candidates))}"
        )
    return found


def _dataframe_to_best_schema_spec(
    df: Any,
    *,
    table_path: Path,
    name: str | None,
    mapping: TableTemplateColumns,
    group_label: str | None = None,
) -> SchemaSpec:
    if mapping.name is not None and mapping.type is not None:
        if _looks_like_field_catalog(df, mapping):
            return _catalog_to_schema_spec(
                df,
                table_path=table_path,
                name=name,
                mapping=mapping,
                group_label=group_label,
            )
        return _standard_table_to_schema_spec(
            df,
            table_path=table_path,
            name=name,
            mapping=mapping,
            group_label=group_label,
        )
    return _dataframe_to_schema_spec(df, table_path=table_path, name=name)


def _looks_like_field_catalog(df: Any, mapping: TableTemplateColumns) -> bool:
    if mapping.name is None or mapping.type is None or mapping.values is None:
        return False
    if mapping.values in _STANDARD_VALUES_COLUMNS and "field_value" not in df.columns:
        return False
    nonempty_fields = [
        str(value).strip()
        for value in df[mapping.name].tolist()
        if str(value).strip()
    ]
    return len(nonempty_fields) != len(set(nonempty_fields))


def _looks_like_standard_template_table(df: Any) -> bool:
    columns = set(df.columns)
    return bool(columns & _STANDARD_NAME_COLUMNS) and bool(columns & _STANDARD_TYPE_COLUMNS)


def _standard_table_to_schema_spec(
    df: Any,
    *,
    table_path: Path,
    name: str | None,
    mapping: TableTemplateColumns | None = None,
    group_label: str | None = None,
) -> SchemaSpec:
    mapping = mapping or _resolve_column_mapping(df, TableTemplateColumns())
    name_col = mapping.name
    type_col = mapping.type
    desc_col = mapping.description
    required_col = mapping.required
    values_col = mapping.values
    if name_col is None or type_col is None:
        raise ValueError("Standard template tables require field_name/name and type columns.")

    fields: list[FieldSpec] = []
    for _, row in df.iterrows():
        field_name = str(row.get(name_col, "")).strip()
        if not field_name:
            continue
        field_type_raw = str(row.get(type_col, "")).strip()
        enum_values = _parse_values(str(row.get(values_col, "") if values_col else ""))
        field_type = _catalog_field_type(field_type_raw, has_values=bool(enum_values))
        value_labels = _parse_value_labels(str(row.get(values_col, "") if values_col else ""))
        fields.append(
            FieldSpec(
                name=field_name,
                type=field_type,
                description=str(row.get(desc_col, "") if desc_col else "").strip(),
                required=_standard_required(
                    str(row.get(required_col, "") if required_col else "")
                ),
                enum_values=enum_values or None,
                value_labels=value_labels or None,
            )
        )

    return SchemaSpec(
        class_name=name or _default_class_name(table_path),
        description=(
            f"Template inferred from MOSAICX template table {table_path.name} "
            f"({len(fields)} fields)."
            + (f" Source group: {group_label}." if group_label else "")
        ),
        fields=fields,
    )


def _catalog_to_schema_spec(
    df: Any,
    *,
    table_path: Path,
    name: str | None,
    mapping: TableTemplateColumns | None = None,
    group_label: str | None = None,
) -> SchemaSpec:
    mapping = mapping or _resolve_column_mapping(df, TableTemplateColumns())
    if mapping.name is None or mapping.type is None:
        raise ValueError("Field catalogs require field name and type columns.")
    fields: list[FieldSpec] = []
    grouped = df.groupby(mapping.name, sort=False, dropna=False)

    for raw_field_name, group in grouped:
        field_name = _first_nonempty([raw_field_name])
        if not field_name:
            continue

        field_type = _first_nonempty(group.get(mapping.type, []))
        field_values = _unique_nonempty(group.get(mapping.values, [])) if mapping.values else []
        labels = _value_labels(group, mapping)
        value_labels = dict(labels)

        description_parts = [
            _first_nonempty(group.get(mapping.description, [])) if mapping.description else "",
        ]
        metadata = _catalog_metadata(group, mapping)
        field_note = _first_nonempty(group.get("field_note", [])) if "field_note" in group else ""
        if field_note:
            metadata["field_note"] = field_note

        fields.append(
            FieldSpec(
                name=field_name,
                type=_catalog_field_type(field_type, has_values=bool(field_values)),
                description=" ".join(
                    part.strip() for part in description_parts if part and part.strip()
                ),
                required=_catalog_required(group.get("field_is_mandatory", [])),
                enum_values=field_values or None,
                value_labels=value_labels or None,
                metadata=metadata or None,
            )
        )

    return SchemaSpec(
        class_name=name or _default_class_name(table_path),
        description=(
            f"Template inferred from field catalog {table_path.name} "
            f"({len(df)} rows, {len(fields)} fields)."
            + (f" Source group: {group_label}." if group_label else "")
        ),
        fields=fields,
    )


def _dataframe_to_schema_spec(
    df: Any,
    *,
    table_path: Path,
    name: str | None,
) -> SchemaSpec:
    fields: list[FieldSpec] = []
    for col in df.columns:
        values = [str(value).strip() for value in df[col].tolist()]
        nonempty = [value for value in values if value]
        field_type, enum_values = _infer_column_type(nonempty)
        fields.append(
            FieldSpec(
                name=str(col),
                type=field_type,
                description=f"Column {col!r} inferred from {table_path.name}.",
                required=len(nonempty) == len(values) and bool(values),
                enum_values=enum_values,
            )
        )

    return SchemaSpec(
        class_name=name or _default_class_name(table_path),
        description=(
            f"Template inferred from table {table_path.name} "
            f"({len(df)} rows, {len(fields)} columns)."
        ),
        fields=fields,
    )


def _default_class_name(path: Path) -> str:
    words = re.findall(r"[A-Za-z0-9]+", path.stem)
    if not words:
        return "TableTemplate"
    return "".join(word[:1].upper() + word[1:] for word in words) + "Template"


def _split_group_class_name(value: str, *, prefix: str | None = None) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    base = "".join(word[:1].upper() + word[1:] for word in words) or "Template"
    if prefix:
        return f"{prefix}{base}"
    return base


def _first_matching_column(df: Any, candidates: set[str]) -> str | None:
    for column in df.columns:
        if str(column) in candidates:
            return str(column)
    return None


def _first_nonempty(values: Any) -> str:
    for value in list(values):
        text = str(value).strip()
        if text:
            return text
    return ""


def _unique_nonempty(values: Any, *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in list(values):
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _value_labels(group: Any, mapping: TableTemplateColumns) -> list[tuple[str, str]]:
    if mapping.values is None or mapping.values not in group:
        return []
    label_col = mapping.value_label
    if label_col is None:
        return []

    labels: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, row in group.iterrows():
        value = str(row.get(mapping.values, "")).strip()
        label = str(row.get(label_col, "")).strip()
        if not value or not label or value in seen:
            continue
        seen.add(value)
        labels.append((value, label))
    return labels


def _catalog_metadata(group: Any, mapping: TableTemplateColumns) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if mapping.catalog_id:
        values = _unique_nonempty(group.get(mapping.catalog_id, []), limit=5)
        if values:
            metadata[mapping.catalog_id] = values[0] if len(values) == 1 else values
    if mapping.catalog_version:
        values = _unique_nonempty(group.get(mapping.catalog_version, []), limit=8)
        if values:
            metadata[mapping.catalog_version] = values[0] if len(values) == 1 else values
    return metadata


def _catalog_field_type(field_type: str, *, has_values: bool) -> str:
    normalized = field_type.strip().lower()
    if has_values or normalized in {
        "combobox",
        "combo",
        "select",
        "dropdown",
        "radio",
        "catalog",
        "lookup",
    }:
        return "enum"
    if normalized in {"checkbox", "check", "bool", "boolean"}:
        return "bool"
    if normalized in {"int", "integer", "long"}:
        return "int"
    if normalized in {"float", "double", "decimal", "number", "numeric"}:
        return "float"
    return "str"


def _catalog_required(values: Any) -> bool:
    value = _first_nonempty(values).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return False


def _standard_required(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return True


def _parse_values(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if "|" in text:
        parts = text.split("|")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = text.split(",")
    return _unique_nonempty(part.split("=", 1)[0] for part in parts)


def _parse_value_labels(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    if "|" in text:
        parts = text.split("|")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = text.split(",")
    labels: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        value, label = part.split("=", 1)
        value = value.strip()
        label = label.strip()
        if value and label:
            labels[value] = label
    return labels


def _infer_column_type(values: list[str]) -> tuple[str, list[str] | None]:
    if not values:
        return "str", None

    lowered = {value.lower() for value in values}
    if lowered <= _TRUE_VALUES.union(_FALSE_VALUES) and not lowered <= {"0", "1"}:
        return "bool", None

    if all(_is_int(value) for value in values):
        return "int", None
    if all(_is_float(value) for value in values):
        return "float", None

    unique = _unique_nonempty(values)
    if _should_be_enum(unique, len(values)):
        return "enum", unique

    return "str", None


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value.replace(",", "."))
    except ValueError:
        return False
    return True


def _should_be_enum(unique_values: list[str], total_values: int) -> bool:
    if not unique_values or len(unique_values) > 20:
        return False
    if len(unique_values) == total_values and total_values > 5:
        return False
    if any(len(value) > 80 for value in unique_values):
        return False
    return len(unique_values) <= 8 or (len(unique_values) / max(total_values, 1)) <= 0.5
