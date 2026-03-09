"""
File ingestion and parsing (CSV and Excel).

Handles reading and parsing uploaded CSV/Excel files, extracting column headers
and sample rows for preview and profiling.
"""
import csv
from io import StringIO, BytesIO
from typing import List, Dict, Any

from fastapi import UploadFile


EXCEL_EXTENSIONS = (".xlsx", ".xls")


def _read_excel_preview(file: UploadFile, sample_rows: int = 50) -> Dict[str, Any]:
    """
    Read uploaded Excel file and extract headers plus sample rows.

    Uses openpyxl (xlsx) or xlrd (xls) via pandas for broad compatibility.
    """
    import pandas as pd

    raw = file.file.read()
    file.file.seek(0)

    fname = (file.filename or "").lower()
    engine = "openpyxl" if fname.endswith(".xlsx") else "xlrd"

    df = pd.read_excel(BytesIO(raw), engine=engine, nrows=sample_rows)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    headers = list(df.columns)
    rows: List[Dict[str, Any]] = df.astype(str).to_dict(orient="records")

    return {"columns": headers, "sample_rows": rows}


def read_file_preview(file: UploadFile, sample_rows: int = 50) -> Dict[str, Any]:
    """
    Unified reader: dispatches to CSV or Excel reader based on file extension.
    """
    fname = (file.filename or "").lower()
    if fname.endswith(EXCEL_EXTENSIONS):
        return _read_excel_preview(file, sample_rows)
    return read_csv_preview(file, sample_rows)


def read_csv_preview(file: UploadFile, sample_rows: int = 50) -> Dict[str, Any]:
    """
    Read uploaded CSV file and extract headers plus sample rows.
    
    This function:
    - Decodes the file content (handles encoding issues gracefully)
    - Parses CSV format using DictReader (column name → value mapping)
    - Extracts only a sample of rows to avoid loading large files into memory
    - Resets file pointer so the file can be re-read if needed
    
    Args:
        file: FastAPI UploadFile object containing the CSV file
        sample_rows: Number of rows to sample (default: 50)
    
    Returns:
        Dictionary with:
        - "columns": List of column names (headers)
        - "sample_rows": List of dictionaries, each representing a row
    """
    # Read file content and decode as UTF-8
    # errors="replace" handles encoding issues by replacing invalid characters
    # This makes the function tolerant of "dirty" CSV files
    content = file.file.read().decode("utf-8", errors="replace")
    
    # Parse CSV using DictReader (creates dict per row: column_name → value)
    reader = csv.DictReader(StringIO(content))
    headers = reader.fieldnames or []
    headers = [header.lower().replace(' ', '_') for header in headers]
    
    # Extract sample rows (limit to avoid memory issues with large files)
    rows: List[Dict[str, Any]] = []
    for _, row in zip(range(sample_rows), reader):
        rows.append(row)
    
    # Reset file pointer to beginning so file can be re-read if needed
    # (e.g., for profiling after preview)
    file.file.seek(0)
    
    return {"columns": headers, "sample_rows": rows}