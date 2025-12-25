"""TensorFlow data type constants and fill values for missing data handling."""

from tensorflow import (
    float32,
    float64,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bool,
    string,
)


"""Data types for TensorFlow"""
DTYPE = {
    "float32": float32,
    "float64": float64,
    "int32": int32,
    "int64": int64,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
    "bool": bool,
    "string": string,
}

"""FillNA layer fill values for different data types"""
FILLNA_VALUES = {
    float32: 0.0,
    float64: 0.0,
    int32: -1,
    int64: -1,
    uint8: 0,
    uint16: 0,
    uint32: 0,
    uint64: 0,
    bool: False,
    string: "N/A",
}
