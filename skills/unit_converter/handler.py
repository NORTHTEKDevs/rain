"""unit_converter: common unit conversions.

Handles patterns like:
    "100 km to miles"
    "5 lbs in kg"
    "convert 32 F to C"
    "200 g in oz"

Supports length, mass, temperature, volume. Returns the converted value
with sensible rounding.
"""

from __future__ import annotations

import re


# Conversion factors to a canonical SI unit per category.
_LENGTH_TO_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "nm": 1852.0, "nautical_mile": 1852.0, "nmi": 1852.0,
}
_MASS_TO_KG = {
    "kg": 1.0, "kilo": 1.0, "kilos": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
    "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    "ton": 1000.0, "tons": 1000.0, "tonne": 1000.0, "tonnes": 1000.0,
}
_VOLUME_TO_L = {
    "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
    "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    "cup": 0.24, "cups": 0.24,
    "tbsp": 0.0147868, "tablespoon": 0.0147868, "tablespoons": 0.0147868,
    "tsp": 0.00492892, "teaspoon": 0.00492892, "teaspoons": 0.00492892,
    "fl_oz": 0.0295735, "floz": 0.0295735, "fluid_ounce": 0.0295735,
}

_TEMP_UNITS = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}


def _convert_temp(value: float, from_u: str, to_u: str) -> float | None:
    f, t = from_u.lower(), to_u.lower()
    # Normalise to a single letter.
    norm = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}
    f = norm.get(f, f)
    t = norm.get(t, t)
    # to Kelvin
    if f == "c":
        k = value + 273.15
    elif f == "f":
        k = (value - 32.0) * 5.0 / 9.0 + 273.15
    elif f == "k":
        k = value
    else:
        return None
    # from Kelvin
    if t == "c":
        return k - 273.15
    if t == "f":
        return (k - 273.15) * 9.0 / 5.0 + 32.0
    if t == "k":
        return k
    return None


_RX = re.compile(
    r"([\d.]+)\s*([a-zA-Z]+)\s+(?:to|in|into|as)\s+([a-zA-Z]+)", re.I
)


def _format(n: float) -> str:
    if abs(n) >= 1000 or (abs(n) < 0.01 and n != 0):
        return f"{n:.6g}"
    return f"{n:.4g}"


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "unit_converter: empty query"
    m = _RX.search(query_text)
    if not m:
        return f"unit_converter: could not parse {query_text!r}; expected '<n> <unit> to <unit>'"
    try:
        value = float(m.group(1))
    except ValueError:
        return "unit_converter: could not parse number"
    from_u = m.group(2).lower()
    to_u = m.group(3).lower()

    # Temperature?
    if from_u in _TEMP_UNITS or to_u in _TEMP_UNITS:
        if from_u not in _TEMP_UNITS or to_u not in _TEMP_UNITS:
            return f"unit_converter: temp units must both be C/F/K; got {from_u} -> {to_u}"
        result = _convert_temp(value, from_u, to_u)
        if result is None:
            return f"unit_converter: bad temp units {from_u} {to_u}"
        return f"{_format(value)} {from_u.upper()} = {_format(result)} {to_u.upper()}"

    # Try each category.
    for table, label in [
        (_LENGTH_TO_M, "length"),
        (_MASS_TO_KG, "mass"),
        (_VOLUME_TO_L, "volume"),
    ]:
        if from_u in table and to_u in table:
            si = value * table[from_u]
            result = si / table[to_u]
            return f"{_format(value)} {from_u} = {_format(result)} {to_u}"

    return f"unit_converter: unknown or mismatched units ({from_u} -> {to_u})"
