import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from io import BytesIO

import pandas as pd

logger = logging.getLogger(__name__)


YEAR_TARGETS = ["2022", "2023", "2024"]

REVENUE_ALIASES = [
    "net revenue",
    "revenue",
    "total revenue",
    "net sales",
    "sales",
]

EBITDA_ALIASES = [
    "reported ebitda",
    "ebitda",
    "e b i t d a",
]

# Labels for Net Working Capital table (case-insensitive substring match on first column)
NWC_YEARS = ["2023", "2024"]
NWC_LABELS = {
    "cash": ["cash", "cash & cash equivalents", "cash and cash equivalents"],
    "accounts_receivable": ["accounts receivable", "accounts recievable", "a/r", "ar"],
    "other_current_assets": ["other current assets"],
    "accounts_payable": ["accounts payable", "a/p", "ap"],
    "credit_cards_payable": ["credit card", "credit cards payable", "credit card payable"],
    "other_current_liabilities": ["other current liabilities"],
    "nwc_adjusted": ["nwc, adjusted"],
}


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    # Normalize common unicode spaces and punctuation
    s = re.sub(r"\s+", " ", s)
    # Remove trailing footnote markers like (a), [1], * etc. for matching
    s = re.sub(r"\(.*?\)$", "", s).strip()
    return s


def _is_year_header(text: str, year: str) -> bool:
    if not text:
        return False
    # Match variants like FY 2023, 2023A, 2023 (A), CY2023, 2023E
    pattern = rf"(^|[^0-9]){year}(?:\s*[A-Za-z()])?($|[^0-9])|fy\s*{year}|cy\s*{year}"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _extract_year_from_value(value, prefer_datetime=False) -> Optional[str]:
    """Extract year from various value types including datetime objects and strings."""
    if value is None:
        return None
    
    # Handle datetime objects (monthly/daily data)
    if hasattr(value, 'year'):
        year_str = str(value.year)
        if year_str in YEAR_TARGETS:
            return year_str
    
    # If we prefer datetime objects, skip non-datetime values
    if prefer_datetime:
        return None
    
    # Handle strings
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
        
    for y in YEAR_TARGETS:
        if _is_year_header(text, y):
            return y
    # Handle two-digit year formats like Dec-23, '23, 12/23
    # Map 23->2023, 24->2024
    two_digit_map = {"23": "2023", "24": "2024"}
    m = re.search(r"(^|[^0-9])'?([0-9]{2})([^0-9]|$)", text)
    if m:
        yy = m.group(2)
        if yy in two_digit_map and any(k.endswith(two_digit_map[yy][-2:]) for k in two_digit_map):
            # Verify that context likely refers to a date by checking month names or separators nearby
            if re.search(r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|/|-", text, flags=re.IGNORECASE):
                return two_digit_map[yy]
    return None


def _looks_like_year_cell(text: str) -> Optional[str]:
    if not text:
        return None
    for y in YEAR_TARGETS:
        if _is_year_header(text, y):
            return y
    return None


def _match_label(text: str, aliases: List[str]) -> bool:
    if not text:
        return False
    for alias in aliases:
        if alias in text:
            return True
    return False


def _parse_number(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "n/a"}:
        return None
    # Remove currency symbols and commas
    s = s.replace("$", "").replace(",", "")
    # Handle parentheses for negatives
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # Remove percent signs and any trailing notes
    s = re.sub(r"[%\s]+$", "", s)
    # Remove footnote superscripts like ^1 or *
    s = re.sub(r"\^?\*?\d*$", "", s)
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


def _extract_from_sheet(df: pd.DataFrame, sheet_name: str = "Unknown") -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
    """Try multiple orientations to locate rows/columns for metrics and year values.
    Returns: (revenue_values, ebitda_values, source_info)
    """
    
    # Strategy 0: Look for annual summary columns first (more reliable than summing monthly)
    try:
        # Check first few rows for datetime headers to find the layout
        for header_row_idx in range(min(6, len(df))):
            header_row = df.iloc[header_row_idx]
            
            # Look for year headers (both datetime and plain year integers)
            year_summary_cols = {y: None for y in YEAR_TARGETS}
            for col_idx, cell_value in enumerate(header_row):
                # Check for plain year integers (summary columns)
                if pd.notna(cell_value) and isinstance(cell_value, (int, float)):
                    try:
                        year = str(int(cell_value))
                        if year in YEAR_TARGETS and year_summary_cols[year] is None:
                            year_summary_cols[year] = col_idx
                    except (ValueError, OverflowError):
                        pass
            
            # If we found annual summary columns, use those
            if any(col is not None for col in year_summary_cols.values()):
                first_col_series = df.iloc[:, 0].astype(str).map(_normalize_string)
                revenue_row_indices = first_col_series[first_col_series.apply(lambda t: _match_label(t, REVENUE_ALIASES))].index
                ebitda_row_indices = first_col_series[first_col_series.apply(lambda t: _match_label(t, EBITDA_ALIASES))].index
                
                revenue_values = {y: None for y in YEAR_TARGETS}
                ebitda_values = {y: None for y in YEAR_TARGETS}
                
                # Get revenue values from summary columns
                if len(revenue_row_indices) > 0:
                    revenue_row = df.iloc[revenue_row_indices[0]]
                    for year, col_idx in year_summary_cols.items():
                        if col_idx is not None and col_idx < len(revenue_row):
                            val = _parse_number(revenue_row.iloc[col_idx])
                            if val is not None:
                                revenue_values[year] = val
                
                # Get EBITDA values from summary columns
                if len(ebitda_row_indices) > 0:
                    ebitda_row = df.iloc[ebitda_row_indices[0]]
                    for year, col_idx in year_summary_cols.items():
                        if col_idx is not None and col_idx < len(ebitda_row):
                            val = _parse_number(ebitda_row.iloc[col_idx])
                            if val is not None:
                                ebitda_values[year] = val
                
                # Return if we found any values
                if any(v is not None for v in revenue_values.values()) or any(v is not None for v in ebitda_values.values()):
                    # Build source info for found values
                    source_info = {}
                    for year in YEAR_TARGETS:
                        if revenue_values.get(year) is not None:
                            source_info[f"revenue_{year}"] = f"{sheet_name} (annual summary)"
                        if ebitda_values.get(year) is not None:
                            source_info[f"ebitda_{year}"] = f"{sheet_name} (annual summary)"
                    return revenue_values, ebitda_values, source_info
            
            # Fallback: Sum monthly values if no annual summaries found
            year_to_cols = {y: [] for y in YEAR_TARGETS}
            for col_idx, cell_value in enumerate(header_row):
                year = _extract_year_from_value(cell_value, prefer_datetime=True)
                if year:
                    year_to_cols[year].append(col_idx)
            
            # Only proceed if we found year columns
            if any(cols for cols in year_to_cols.values()):
                first_col_series = df.iloc[:, 0].astype(str).map(_normalize_string)
                revenue_row_indices = first_col_series[first_col_series.apply(lambda t: _match_label(t, REVENUE_ALIASES))].index
                ebitda_row_indices = first_col_series[first_col_series.apply(lambda t: _match_label(t, EBITDA_ALIASES))].index
                
                revenue_values = {y: None for y in YEAR_TARGETS}
                ebitda_values = {y: None for y in YEAR_TARGETS}
                
                # Sum revenue values by year
                if len(revenue_row_indices) > 0:
                    revenue_row = df.iloc[revenue_row_indices[0]]
                    for year, cols in year_to_cols.items():
                        if cols:
                            year_total = 0
                            valid_count = 0
                            for col_idx in cols:
                                if col_idx < len(revenue_row):
                                    val = _parse_number(revenue_row.iloc[col_idx])
                                    if val is not None:
                                        year_total += val
                                        valid_count += 1
                            if valid_count > 0:
                                revenue_values[year] = year_total
                
                # Sum EBITDA values by year
                if len(ebitda_row_indices) > 0:
                    ebitda_row = df.iloc[ebitda_row_indices[0]]
                    for year, cols in year_to_cols.items():
                        if cols:
                            year_total = 0
                            valid_count = 0
                            for col_idx in cols:
                                if col_idx < len(ebitda_row):
                                    val = _parse_number(ebitda_row.iloc[col_idx])
                                    if val is not None:
                                        year_total += val
                                        valid_count += 1
                            if valid_count > 0:
                                ebitda_values[year] = year_total
                
                # Return if we found any values
                if any(v is not None for v in revenue_values.values()) or any(v is not None for v in ebitda_values.values()):
                    # Build source info for found values (monthly summation)
                    source_info = {}
                    for year in YEAR_TARGETS:
                        if revenue_values.get(year) is not None:
                            source_info[f"revenue_{year}"] = f"{sheet_name} (monthly totals)"
                        if ebitda_values.get(year) is not None:
                            source_info[f"ebitda_{year}"] = f"{sheet_name} (monthly totals)"
                    return revenue_values, ebitda_values, source_info
    except Exception:
        pass
    
    # Strategy 1: Assume first row is headers (years across columns), first column has labels
    try:
        labeled_df = df.copy()
        labeled_df.columns = [_normalize_string(c) for c in labeled_df.columns]
        # Map header columns to target years
        year_to_col = {}
        for col in labeled_df.columns:
            y = _looks_like_year_cell(col)
            if y and y not in year_to_col:
                year_to_col[y] = col
        # Find label column: prefer the first column
        label_col = labeled_df.columns[0]
        labels_norm = labeled_df[label_col].astype(str).map(_normalize_string)
        # Identify rows for metrics
        revenue_row_idx = labels_norm[labels_norm.apply(lambda t: _match_label(t, REVENUE_ALIASES))].index
        ebitda_row_idx = labels_norm[labels_norm.apply(lambda t: _match_label(t, EBITDA_ALIASES))].index
        revenue_values: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
        ebitda_values: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
        if len(revenue_row_idx) > 0:
            row = labeled_df.loc[revenue_row_idx[0]]
            for y, col in year_to_col.items():
                if y in YEAR_TARGETS:
                    revenue_values[y] = _parse_number(row.get(col))
        if len(ebitda_row_idx) > 0:
            row = labeled_df.loc[ebitda_row_idx[0]]
            for y, col in year_to_col.items():
                if y in YEAR_TARGETS:
                    ebitda_values[y] = _parse_number(row.get(col))
        # If any values found, return
        if any(v is not None for v in revenue_values.values()) or any(v is not None for v in ebitda_values.values()):
            # Build source info for strategy 1
            source_info = {}
            for year in YEAR_TARGETS:
                if revenue_values.get(year) is not None:
                    source_info[f"revenue_{year}"] = f"{sheet_name} (headers strategy)"
                if ebitda_values.get(year) is not None:
                    source_info[f"ebitda_{year}"] = f"{sheet_name} (headers strategy)"
            return revenue_values, ebitda_values, source_info
    except Exception:
        pass

    # Strategy 2: Years in first column (rows = years), metrics across columns
    try:
        labeled_df = df.copy()
        labeled_df.columns = [_normalize_string(c) for c in labeled_df.columns]
        # Detect if first column looks like years
        first_col_series = labeled_df.iloc[:, 0].astype(str).map(_normalize_string)
        row_to_year: Dict[int, str] = {}
        for idx, cell in first_col_series.items():
            y = _looks_like_year_cell(cell)
            if y:
                row_to_year[idx] = y
        # Find metric columns by header name
        col_for_revenue = None
        col_for_ebitda = None
        for col in labeled_df.columns:
            col_norm = _normalize_string(col)
            if col_for_revenue is None and _match_label(col_norm, REVENUE_ALIASES):
                col_for_revenue = col
            if col_for_ebitda is None and _match_label(col_norm, EBITDA_ALIASES):
                col_for_ebitda = col
        revenue_values = {y: None for y in YEAR_TARGETS}
        ebitda_values = {y: None for y in YEAR_TARGETS}
        if row_to_year and (col_for_revenue or col_for_ebitda):
            for row_idx, y in row_to_year.items():
                if y in YEAR_TARGETS:
                    if col_for_revenue:
                        revenue_values[y] = _parse_number(labeled_df.at[row_idx, col_for_revenue])
                    if col_for_ebitda:
                        ebitda_values[y] = _parse_number(labeled_df.at[row_idx, col_for_ebitda])
            if any(v is not None for v in revenue_values.values()) or any(v is not None for v in ebitda_values.values()):
                # Build source info for strategy 2
                source_info = {}
                for year in YEAR_TARGETS:
                    if revenue_values.get(year) is not None:
                        source_info[f"revenue_{year}"] = f"{sheet_name} (years in rows)"
                    if ebitda_values.get(year) is not None:
                        source_info[f"ebitda_{year}"] = f"{sheet_name} (years in rows)"
                return revenue_values, ebitda_values, source_info
    except Exception:
        pass

    # Strategy 3: Fallback scan - find any cell matching labels, then scan rightwards or below for year/value pairs
    try:
        values_found_rev: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
        values_found_ebitda: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
        arr = df.fillna("").astype(str).values
        nrows, ncols = arr.shape
        for r in range(nrows):
            for c in range(ncols):
                cell = _normalize_string(arr[r][c])
                if _match_label(cell, REVENUE_ALIASES) or _match_label(cell, EBITDA_ALIASES):
                    is_rev = _match_label(cell, REVENUE_ALIASES)
                    # scan right for year headers on same row or one row above
                    for cc in range(c + 1, min(c + 8, ncols)):
                        header_here = _normalize_string(arr[r][cc])
                        header_above = _normalize_string(arr[r - 1][cc]) if r - 1 >= 0 else ""
                        for hdr in [header_here, header_above]:
                            y = _looks_like_year_cell(hdr)
                            if y:
                                val = _parse_number(arr[r][cc])
                                if is_rev:
                                    values_found_rev[y] = values_found_rev[y] or val
                                else:
                                    values_found_ebitda[y] = values_found_ebitda[y] or val
                    # scan below for year labels (vertical orientation)
                    for rr in range(r + 1, min(r + 10, nrows)):
                        y = _looks_like_year_cell(_normalize_string(arr[rr][c]))
                        if y:
                            val = _parse_number(arr[rr][c + 1]) if c + 1 < ncols else None
                            if is_rev:
                                values_found_rev[y] = values_found_rev[y] or val
                            else:
                                values_found_ebitda[y] = values_found_ebitda[y] or val
        if any(v is not None for v in values_found_rev.values()) or any(v is not None for v in values_found_ebitda.values()):
            # Build source info for strategy 3 (fallback scan)
            source_info = {}
            for year in YEAR_TARGETS:
                if values_found_rev.get(year) is not None:
                    source_info[f"revenue_{year}"] = f"{sheet_name} (pattern scan)"
                if values_found_ebitda.get(year) is not None:
                    source_info[f"ebitda_{year}"] = f"{sheet_name} (pattern scan)"
            return values_found_rev, values_found_ebitda, source_info
    except Exception:
        pass

    return {y: None for y in YEAR_TARGETS}, {y: None for y in YEAR_TARGETS}, {}


def extract_three_year_pnl_from_excel(file_path: str) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
    """Extract Net Revenue and Reported EBITDA values for 2022-2024 from all sheets of an Excel file.
    Prefers data from sheets that explicitly reference "recast". If no values are found on
    any recast sheets, falls back to non-recast sheets.
    Returns: (revenue_dict, ebitda_dict, source_info_dict)
    """
    try:
        excel = pd.ExcelFile(file_path)
        # Process sheets in preferred order: recast sheets first, then others
        recast_sheets = [s for s in excel.sheet_names if "recast" in s.lower()]
        regular_sheets = [s for s in excel.sheet_names if "recast" not in s.lower()]

        def process_sheets(sheets: List[str]) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
            rev_agg: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
            ebitda_agg: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
            source_agg: Dict[str, str] = {}
            for sheet in sheets:
                try:
                    df = pd.read_excel(excel, sheet_name=sheet)
                    # Try with headers first
                    rev, ebitda, sources = _extract_from_sheet(df, sheet)
                    # If nothing, try header=None to treat all cells raw
                    if not any(v is not None for v in rev.values()) and not any(v is not None for v in ebitda.values()):
                        df_no_header = pd.read_excel(excel, sheet_name=sheet, header=None)
                        rev, ebitda, sources = _extract_from_sheet(df_no_header, sheet)
                    for y in YEAR_TARGETS:
                        if rev[y] is not None and rev_agg[y] is None:
                            rev_agg[y] = rev[y]
                            if f"revenue_{y}" in sources:
                                source_agg[f"revenue_{y}"] = sources[f"revenue_{y}"]
                        if ebitda[y] is not None and ebitda_agg[y] is None:
                            ebitda_agg[y] = ebitda[y]
                            if f"ebitda_{y}" in sources:
                                source_agg[f"ebitda_{y}"] = sources[f"ebitda_{y}"]
                except Exception as sheet_err:
                    logger.debug(f"Sheet parse failed for '{sheet}' in {file_path}: {sheet_err}")
            return rev_agg, ebitda_agg, source_agg

        # First attempt: use recast sheets
        rec_rev, rec_ebitda, rec_sources = process_sheets(recast_sheets)
        if any(v is not None for v in rec_rev.values()) or any(v is not None for v in rec_ebitda.values()):
            logger.info("Using P&L values from recast sheets where available")
            return rec_rev, rec_ebitda, rec_sources

        # Fallback: use regular sheets
        reg_rev, reg_ebitda, reg_sources = process_sheets(regular_sheets)
        return reg_rev, reg_ebitda, reg_sources
    except Exception as e:
        logger.warning(f"Failed to parse Excel for financial extraction: {file_path} :: {e}")
        return {y: None for y in YEAR_TARGETS}, {y: None for y in YEAR_TARGETS}, {}


def extract_three_year_pnl_from_excel_bytes(file_bytes: bytes, file_label: str = "<bytes>") -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
    """Same as extract_three_year_pnl_from_excel but reads from in-memory bytes (e.g., GCS download).
    Prefers data from recast sheets and falls back to non-recast sheets.
    Returns: (revenue_dict, ebitda_dict, source_info_dict)
    """
    try:
        bio = BytesIO(file_bytes)
        excel = pd.ExcelFile(bio)

        recast_sheets = [s for s in excel.sheet_names if "recast" in s.lower()]
        regular_sheets = [s for s in excel.sheet_names if "recast" not in s.lower()]

        def process_sheets(sheets: List[str]) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
            rev_agg: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
            ebitda_agg: Dict[str, Optional[float]] = {y: None for y in YEAR_TARGETS}
            source_agg: Dict[str, str] = {}
            for sheet in sheets:
                try:
                    df = pd.read_excel(excel, sheet_name=sheet)
                    rev, ebitda, sources = _extract_from_sheet(df, sheet)
                    if not any(v is not None for v in rev.values()) and not any(v is not None for v in ebitda.values()):
                        df_no_header = pd.read_excel(excel, sheet_name=sheet, header=None)
                        rev, ebitda, sources = _extract_from_sheet(df_no_header, sheet)
                    for y in YEAR_TARGETS:
                        if rev[y] is not None and rev_agg[y] is None:
                            rev_agg[y] = rev[y]
                            if f"revenue_{y}" in sources:
                                source_agg[f"revenue_{y}"] = sources[f"revenue_{y}"]
                        if ebitda[y] is not None and ebitda_agg[y] is None:
                            ebitda_agg[y] = ebitda[y]
                            if f"ebitda_{y}" in sources:
                                source_agg[f"ebitda_{y}"] = sources[f"ebitda_{y}"]
                except Exception as sheet_err:
                    logger.debug(f"Sheet parse failed for '{sheet}' in {file_label}: {sheet_err}")
            return rev_agg, ebitda_agg, source_agg

        rec_rev, rec_ebitda, rec_sources = process_sheets(recast_sheets)
        if any(v is not None for v in rec_rev.values()) or any(v is not None for v in rec_ebitda.values()):
            logger.info("Using P&L values from recast sheets where available (bytes)")
            return rec_rev, rec_ebitda, rec_sources

        reg_rev, reg_ebitda, reg_sources = process_sheets(regular_sheets)
        return reg_rev, reg_ebitda, reg_sources
    except Exception as e:
        logger.warning(f"Failed to parse Excel bytes for financial extraction [{file_label}]: {e}")
        return {y: None for y in YEAR_TARGETS}, {y: None for y in YEAR_TARGETS}, {}


def build_three_year_pnl_table_html(
    revenue: Dict[str, Optional[float]],
    ebitda: Dict[str, Optional[float]],
    sources: Dict[str, str] = None
) -> str:
    def fmt(v: Optional[float]) -> str:
        if v is None:
            return "Not available"
        # Format with commas, no trailing .0 if integer-like
        if abs(v - int(v)) < 1e-6:
            return f"{int(v):,}"
        return f"{v:,.2f}"

    years = YEAR_TARGETS
    rows = [
        ("Net Revenue", [fmt(revenue.get(y)) for y in years]),
        ("Gross Profit", ["-" for _ in years]),
        ("SG&A", ["-" for _ in years]),
        ("Reported EBITDA", [fmt(ebitda.get(y)) for y in years]),
    ]

    # Return raw HTML table; caller should ensure it is inserted safely
    parts = []
    parts.append('<table class="financial-table">')
    parts.append('<thead>')
    parts.append('<tr><th></th>' + ''.join(f'<th>{y}</th>' for y in years) + '</tr>')
    parts.append('</thead>')
    parts.append('<tbody>')
    for label, vals in rows:
        parts.append('<tr>' + f'<td>{label}</td>' + ''.join(f'<td>{v}</td>' for v in vals) + '</tr>')
    parts.append('</tbody>')
    parts.append('</table>')
    
    # Add source information if available
    if sources:
        parts.append('<div style="margin-top: 8px; font-size: 9pt; color: #666;">')
        parts.append('<strong>Sources:</strong><br/>')
        
        # Group sources by sheet
        sheet_sources = {}
        for key, source in sources.items():
            if key.startswith('revenue_'):
                year = key.replace('revenue_', '')
                metric = f"Net Revenue {year}"
            elif key.startswith('ebitda_'):
                year = key.replace('ebitda_', '')
                metric = f"Reported EBITDA {year}"
            else:
                continue
                
            if source not in sheet_sources:
                sheet_sources[source] = []
            sheet_sources[source].append(metric)
        
        # Display grouped sources
        for source, metrics in sheet_sources.items():
            parts.append(f"• {', '.join(metrics)}: {source}<br/>")
        
        parts.append('</div>')
    
    return "".join(parts)


def _find_latest_year_value(row_values: List[object], header_values: List[object], target_year: str) -> Optional[float]:
    """Given a row of values and a parallel header row with months/dates, return the latest
    available value within the specified year. Works when headers are datetime-like or strings
    that contain the year. Assumes row_values indexes align with header_values.
    """
    latest_idx = None
    latest_month = -1
    for idx, hdr in enumerate(header_values):
        year = _extract_year_from_value(hdr, prefer_datetime=False)
        if year == target_year:
            # Determine month ordering if possible
            month_num = None
            if hasattr(hdr, 'month'):
                month_num = int(hdr.month)
            else:
                # Try to parse things like 'Jan 2024', '2024-09', etc.
                text = str(hdr).lower()
                m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|[-/](1[0-2]|0?[1-9])", text)
                if m:
                    mon = m.group(1)
                    if mon:
                        month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
                        month_num = month_map.get(mon[:3], None)
                    else:
                        try:
                            month_num = int(m.group(2))
                        except Exception:
                            month_num = None
            if month_num is None:
                month_num = 13  # treat as very late in the year
            if latest_idx is None or month_num > latest_month:
                latest_idx = idx
                latest_month = month_num
    if latest_idx is not None and latest_idx < len(row_values):
        return _parse_number(row_values[latest_idx])
    return None


def find_balance_sheet_tab(excel_file: pd.ExcelFile) -> Optional[str]:
    """Find the most appropriate balance sheet tab in the Excel file.
    
    Prioritization:
    1. Sheets matching balance sheet patterns that contain "Cash"
    2. Among those, prefer summary/condensed sheets (higher density, fewer rows, summary keywords)
    3. If multiple still match, choose the one with the shortest name
    4. If no sheets contain "Cash", apply same logic to all balance sheet matches
    
    Returns the sheet name or None if no balance sheet found.
    """
    import re
    
    # Patterns to match balance sheet names
    bs_patterns = [
        r'^balance\s*sheet$',
        r'^bs$',
        r'^bs\d*$',
        r'^bs\s*\(',
        r'balance.*sheet',
        r'^bs[^a-z]*$',  # BS followed by non-letter chars
        r'bs.*monthly',  # BS followed by monthly
        r'monthly.*bs',   # Monthly followed by BS
        r'\bbs$',        # BS at end of name (e.g., "Cons. NFPS BS")
        r'\bbs\b'        # BS as standalone word anywhere in name
    ]
    
    # Find all sheets that match balance sheet patterns
    candidate_sheets = []
    for sheet_name in excel_file.sheet_names:
        normalized_name = sheet_name.lower().strip()
        # Skip navigation/pointer sheets
        if normalized_name.endswith("-->") or normalized_name.endswith("->"):
            continue
        for pattern in bs_patterns:
            if re.search(pattern, normalized_name, re.IGNORECASE):
                candidate_sheets.append(sheet_name)
                break
    
    if not candidate_sheets:
        return None
    
    # Analyze each candidate for cash content and summary characteristics
    sheet_analysis = []
    for sheet_name in candidate_sheets:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            sheet_text = df.fillna('').astype(str).apply(lambda x: ' '.join(x)).str.cat(sep=' ').lower()
            
            # Check for cash content
            has_cash = 'cash' in sheet_text
            
            # Analyze for summary characteristics
            row_count = len(df)
            col_count = len(df.columns)
            non_empty = df.count().sum()
            density = non_empty / (row_count * col_count) if row_count * col_count > 0 else 0
            
            # Check for summary vs detail keywords
            summary_keywords = ['summary', 'condensed', 'total', 'overview']
            detail_keywords = ['detail', 'aging', 'breakdown', 'individual', 'transaction']
            
            summary_score = sum(1 for word in summary_keywords if word in sheet_text)
            detail_score = sum(1 for word in detail_keywords if word in sheet_text)
            
            # Prefer higher density (more concise), fewer rows, more summary keywords
            summary_preference = density * 100 + summary_score * 10 - detail_score * 5 - (row_count / 10)
            
            sheet_analysis.append({
                'name': sheet_name,
                'has_cash': has_cash,
                'summary_preference': summary_preference,
                'row_count': row_count
            })
            
        except Exception:
            continue
    
    if not sheet_analysis:
        return None
    
    # Filter for sheets with cash content first
    cash_sheets = [s for s in sheet_analysis if s['has_cash']]
    final_candidates = cash_sheets if cash_sheets else sheet_analysis
    
    if len(cash_sheets) > 1:
        # Multiple sheets with cash - check for summary/aggregate/monthly keywords
        summary_sheets = []
        for sheet in cash_sheets:
            sheet_name_lower = sheet['name'].lower()
            if 'summary' in sheet_name_lower or 'aggregate' in sheet_name_lower or 'monthly' in sheet_name_lower:
                summary_sheets.append(sheet)
        
        if len(summary_sheets) == 1:
            # Exactly one sheet with summary/aggregate - choose it
            return summary_sheets[0]['name']
        elif len(summary_sheets) > 1:
            # Multiple sheets with summary/aggregate - use shortest name
            summary_sheets.sort(key=lambda x: len(x['name']))
            return summary_sheets[0]['name']
        # If no summary sheets or logic above doesn't apply, fall through to original logic
    
    # Sort by: cash content (True first), summary preference (higher first), shorter name
    final_candidates.sort(key=lambda x: (
        not x['has_cash'],  # False (has cash) comes before True (no cash)
        -x['summary_preference'],  # Higher preference first
        len(x['name'])  # Shorter names first
    ))
    
    return final_candidates[0]['name']


def sheet_to_csv_string(excel_file: pd.ExcelFile, sheet_name: str) -> str:
    """Convert a specific Excel sheet to CSV string format."""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        # Fill NaN values with empty strings for cleaner CSV
        df = df.fillna('')
        return df.to_csv(index=False, header=False)
    except Exception as e:
        logger.error(f"Failed to convert sheet '{sheet_name}' to CSV: {e}")
        return ""


def _sheet_has_meaningful_data(csv_content: str) -> bool:
    """Check if a CSV contains meaningful data (not just navigation/empty cells)."""
    if not csv_content.strip():
        return False
    
    lines = csv_content.split('\n')
    meaningful_lines = 0
    total_words = 0
    
    for line in lines:
        if not line.strip():
            continue
            
        # Count non-empty cells that contain actual words/numbers
        cells = line.split(',')
        line_words = 0
        for cell in cells:
            cell = cell.strip().strip('"')
            if cell and cell not in ['', '0', '0.0']:
                # Check if cell contains letters, numbers, or meaningful content
                if any(c.isalnum() for c in cell):
                    line_words += 1
                    total_words += 1
        
        if line_words >= 2:  # Line has at least 2 meaningful cells
            meaningful_lines += 1
    
    # Sheet has meaningful data if it has multiple lines with content
    # and reasonable word density
    return meaningful_lines >= 3 and total_words >= 10


async def extract_current_assets_with_gemini(csv_content: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Use Gemini to extract current assets, current liabilities, and net working capital from balance sheet CSV."""
    from app.services.vertex_ai import vertex_ai_service
    
    prompt = f"""Analyze this balance sheet CSV and extract current assets, current liabilities, and net working capital.

CSV Data:
{csv_content}

Return JSON in this exact format:
{{
  "current_assets": {{
    "Cash": {{"2023": 123.45, "2024": 456.78}},
    "Accounts Receivable": {{"2023": 123.45, "2024": 456.78}}
  }},
  "current_liabilities": {{
    "Accounts Payable": {{"2023": 123.45, "2024": 456.78}},
    "Accrued Liabilities": {{"2023": 123.45, "2024": 456.78}}
  }},
  "net_working_capital": {{"2023": 123.45, "2024": 456.78}},
  "year_headers": {{"2023": "Dec 2023", "2024": "Jul 2024"}}
}}

Instructions:
- Use December 2023 and the latest available 2024 month
- Include all relevant current asset and liability categories (not individual accounts)  
- For Net Working Capital, look for a row labeled "Net Working Capital", "NWC", "Working Capital", etc.
- If no NWC row exists, set values to null
- Use actual category names from the balance sheet as keys
- In year_headers, capture the actual column headers for 2023 (should be December/Dec) and 2024 (should be the latest month like Jul, July, Aug, etc.)"""

    try:
        response = await vertex_ai_service.generate_text(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            balance_sheet_data = json.loads(json_str)
            return balance_sheet_data
        else:
            logger.warning("No JSON found in Gemini response for balance sheet extraction")
            return {"current_assets": {}, "current_liabilities": {}, "net_working_capital": {}}
            
    except Exception as e:
        logger.error(f"Failed to extract balance sheet data with Gemini: {e}")
        return {"current_assets": {}, "current_liabilities": {}, "net_working_capital": {}}


def extract_net_working_capital_from_excel(file_path: str) -> str:
    """Extract current assets from balance sheet and return HTML table.
    
    New approach:
    1. Find balance sheet tab
    2. Convert to CSV
    3. Use Gemini to extract current assets
    4. Build HTML table
    
    Returns HTML string for the table or error message.
    """
    try:
        excel = pd.ExcelFile(file_path)
        
        # Find balance sheet tab
        balance_sheet_tab = find_balance_sheet_tab(excel)
        if not balance_sheet_tab:
            return '<p><strong>Balance sheet not found</strong></p>'
        
        logger.info(f"Found balance sheet tab: {balance_sheet_tab}")
        
        # Convert to CSV
        csv_content = sheet_to_csv_string(excel, balance_sheet_tab)
        if not csv_content.strip():
            return '<p><strong>Failed to convert balance sheet to CSV</strong></p>'
        
        # This will be called asynchronously from the memo generator
        # For now, return a placeholder that will be replaced
        return f'<p>Processing balance sheet: {balance_sheet_tab}</p>'
        
    except Exception as e:
        logger.error(f"Failed to extract current assets from {file_path}: {e}")
        return '<p><strong>Error processing balance sheet</strong></p>'


def _extract_correct_values_from_csv(csv_content: str, categories: Dict) -> Dict:
    """Extract correct Dec 2023 and latest 2024 values from CSV for the categories identified by Gemini."""
    lines = csv_content.split("\n")
    
    # Find header row to get column positions
    header_row = None
    for line in lines:
        if "dec 2023" in line.lower():
            header_row = line.split(",")
            break
    
    if not header_row:
        return categories  # Fallback to original
    
    # Find the column indices for Dec 2023 and latest 2024
    dec_2023_col = None
    latest_2024_col = None
    
    for i, header in enumerate(header_row):
        if header.strip().lower() == "dec 2023":
            dec_2023_col = i
        elif "2024" in header and header.strip():
            latest_2024_col = i  # Keep updating to get the rightmost one
    
    if dec_2023_col is None:
        return categories  # Fallback
    
    def correct_section(section: Dict[str, Dict]) -> Dict[str, Dict]:
        corrected = {}
        for category_name, year_data in section.items():
            # Find the row that matches this category
            target_row = None
            normalized_category = category_name.lower().replace(" ", "").replace("total", "").replace("current", "").replace("assets", "").replace("liabilities", "")
            
            for line in lines:
                line_lower = line.lower()
                if any(part in line_lower for part in [normalized_category, category_name.lower()]):
                    # Additional checks to avoid false matches
                    if ("bank" in normalized_category and "bank" in line_lower) or \
                       ("cash" in normalized_category and ("cash" in line_lower or "bank" in line_lower)) or \
                       ("current assets" in category_name.lower() and "total current assets" in line_lower) or \
                       ("current liabilities" in category_name.lower() and "total current liabilities" in line_lower) or \
                       (normalized_category in line_lower and len(normalized_category) > 3):
                        target_row = line.split(",")
                        break
            
            if target_row and len(target_row) > max(dec_2023_col, latest_2024_col or 0):
                try:
                    # Extract Dec 2023 value
                    dec_val = target_row[dec_2023_col].strip()
                    dec_2023_value = float(dec_val) if dec_val and dec_val != '' else None
                    
                    # Extract latest 2024 value
                    latest_2024_value = None
                    if latest_2024_col is not None:
                        latest_val = target_row[latest_2024_col].strip()
                        latest_2024_value = float(latest_val) if latest_val and latest_val != '' else None
                    
                    corrected[category_name] = {
                        "2023": dec_2023_value,
                        "2024": latest_2024_value
                    }
                except (ValueError, IndexError):
                    # Fallback to original values
                    corrected[category_name] = year_data
            else:
                # Fallback to original values
                corrected[category_name] = year_data
                
        return corrected
    
    # Handle both structured (current_assets/current_liabilities) and flat formats
    if "current_assets" in categories:
        return {
            "current_assets": correct_section(categories.get("current_assets", {})),
            "current_liabilities": correct_section(categories.get("current_liabilities", {})),
            "net_working_capital": correct_section(categories.get("net_working_capital", {}))
        }
    else:
        return {
            "current_assets": correct_section(categories.get("current_assets", {})),
            "current_liabilities": correct_section(categories.get("current_liabilities", {})),
            "net_working_capital": correct_section(categories.get("net_working_capital", {}))
        }


async def extract_net_working_capital_from_excel_async(file_path: str) -> str:
    """Async version that actually processes with Gemini."""
    try:
        excel = pd.ExcelFile(file_path)
        
        # Find balance sheet tab
        balance_sheet_tab = find_balance_sheet_tab(excel)
        if not balance_sheet_tab:
            return '<p><strong>Balance sheet not found</strong></p>'
        
        logger.info(f"Found balance sheet tab: {balance_sheet_tab}")
        
        # Convert to CSV
        csv_content = sheet_to_csv_string(excel, balance_sheet_tab)
        if not csv_content.strip():
            return '<p><strong>Failed to convert balance sheet to CSV</strong></p>'
        
        # Extract with Gemini
        balance_sheet_data = await extract_current_assets_with_gemini(csv_content)
        
        # If no NWC found in balance sheet, check working capital sheet
        nwc_data = balance_sheet_data.get("net_working_capital", {})
        if not nwc_data or (nwc_data.get("2023") is None and nwc_data.get("2024") is None):
            logger.info("No NWC found in balance sheet, checking working capital sheet")
            
            # Look for working capital sheet with actual data
            wc_sheet = None
            for sheet_name in excel.sheet_names:
                if any(term in sheet_name.lower() for term in ['working capital', 'working cap', 'wc1']):
                    # Check if this sheet contains actual data (not just navigation)
                    try:
                        test_csv = sheet_to_csv_string(excel, sheet_name)
                        if _sheet_has_meaningful_data(test_csv):
                            wc_sheet = sheet_name
                            break
                    except Exception:
                        continue
            
            if wc_sheet:
                logger.info(f"Found working capital sheet: {wc_sheet}")
                wc_csv = sheet_to_csv_string(excel, wc_sheet)
                if wc_csv.strip():
                    # Extract NWC from working capital sheet
                    wc_prompt = f"""Extract Net Working Capital values from this working capital sheet CSV.

CSV Data:
{wc_csv}

Return JSON in this format:
{{"net_working_capital": {{"2023": value_or_null, "2024": value_or_null}}}}

Instructions:
- Use December 2023 and the latest available 2024 month
- Look for rows labeled "Net Working Capital", "NWC", "Working Capital", etc.
- If multiple NWC rows exist, choose the summary/total one"""

                    try:
                        from app.services.vertex_ai import vertex_ai_service
                        response = await vertex_ai_service.generate_text(
                            prompt=wc_prompt,
                            max_tokens=1000,
                            temperature=0.1
                        )
                        
                        import json, re
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            wc_result = json.loads(json_match.group(0))
                            wc_nwc = wc_result.get("net_working_capital", {})
                            if wc_nwc and (wc_nwc.get("2023") is not None or wc_nwc.get("2024") is not None):
                                logger.info(f"Found NWC in working capital sheet: {wc_nwc}")
                                balance_sheet_data["net_working_capital"] = wc_nwc
                    except Exception as e:
                        logger.warning(f"Failed to extract NWC from working capital sheet: {e}")
        
        # Build HTML table
        return build_current_assets_table_html(balance_sheet_data, balance_sheet_tab)
        
    except Exception as e:
        logger.error(f"Failed to extract current assets from {file_path}: {e}")
        return '<p><strong>Error processing balance sheet</strong></p>'


def build_current_assets_table_html(balance_sheet_data: Dict, sheet_name: str = None) -> str:
    """Build HTML table from balance sheet data extracted by Gemini, styled to match the P&L table."""
    current_assets = balance_sheet_data.get("current_assets", {})
    current_liabilities = balance_sheet_data.get("current_liabilities", {})
    net_working_capital = balance_sheet_data.get("net_working_capital", {})
    year_headers = balance_sheet_data.get("year_headers", {})
    
    if not current_assets and not current_liabilities:
        return '<p><strong>No current assets or liabilities found in balance sheet</strong></p>'
    
    def fmt_cell(v: Optional[float]) -> str:
        if v is None:
            return "-"
        # Handle arrays from LLM
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
        return f"${v:,.0f}" if abs(v - int(v)) < 1e-6 else f"${v:,.2f}"
    
    years = ["2023", "2024"]
    # Use actual month headers if available
    year_labels = {
        "2023": year_headers.get("2023", "2023"),
        "2024": year_headers.get("2024", "2024")
    }
    parts = []
    # Use inline styles consistent with build_three_year_pnl_table_html (LLM version)
    parts.append('<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">')
    parts.append('<tr style="background-color: #f8f9fa; font-weight: bold;">')
    parts.append('<td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">Balance Sheet Items</td>')
    for year in years:
        year_label = year_labels.get(year, year)
        parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{year_label}</td>')
    parts.append('</tr>')
    
    # Current Assets section
    if current_assets:
        parts.append('<tr>')
        parts.append('<td style="border: 1px solid #dee2e6; padding: 8px; font-weight: bold; background-color: #f8f9fa;">Current Assets</td>')
        for _ in years:
            parts.append('<td style="border: 1px solid #dee2e6; padding: 8px; background-color: #f8f9fa;"></td>')
        parts.append('</tr>')
        for category, year_data in current_assets.items():
            parts.append('<tr>')
            parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px;">{category}</td>')
            for year in years:
                value = year_data.get(year) if isinstance(year_data, dict) else None
                parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
            parts.append('</tr>')
    
    # Separator row
    if current_assets and current_liabilities:
        # spacer row
        parts.append('<tr>')
        parts.append('<td style="border: 1px solid #dee2e6; padding: 4px; background-color: #fff;"></td>')
        for _ in years:
            parts.append('<td style="border: 1px solid #dee2e6; padding: 4px; background-color: #fff;"></td>')
        parts.append('</tr>')
    
    # Current Liabilities section
    if current_liabilities:
        parts.append('<tr>')
        parts.append('<td style="border: 1px solid #dee2e6; padding: 8px; font-weight: bold; background-color: #f8f9fa;">Current Liabilities</td>')
        for _ in years:
            parts.append('<td style="border: 1px solid #dee2e6; padding: 8px; background-color: #f8f9fa;"></td>')
        parts.append('</tr>')
        for category, year_data in current_liabilities.items():
            parts.append('<tr>')
            parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px;">{category}</td>')
            for year in years:
                value = year_data.get(year) if isinstance(year_data, dict) else None
                parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
            parts.append('</tr>')

    # Net Working Capital row
    if net_working_capital and (net_working_capital.get("2023") is not None or net_working_capital.get("2024") is not None):
        # spacer row
        parts.append('<tr>')
        parts.append('<td style="border: 1px solid #dee2e6; padding: 4px; background-color: #fff;"></td>')
        for _ in years:
            parts.append('<td style="border: 1px solid #dee2e6; padding: 4px; background-color: #fff;"></td>')
        parts.append('</tr>')
        parts.append('<tr>')
        parts.append('<td style="border: 1px solid #dee2e6; padding: 8px;"><b>Net Working Capital</b></td>')
        for year in years:
            value = net_working_capital.get(year)
            parts.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
        parts.append('</tr>')

    parts.append('</table>')
    
    if sheet_name:
        parts.append('<div style="margin-top: 8px; font-size: 9pt;">')
        parts.append('<strong><span class="meta">Source:</span></strong><br/>')
        year_2023_label = year_labels.get("2023", "December 2023")
        year_2024_label = year_labels.get("2024", "latest 2024")
        parts.append(f"• Balance sheet data: {sheet_name} ({year_2023_label} and {year_2024_label} values)<br/>")
        parts.append('</div>')
    
    return ''.join(parts)


def extract_net_working_capital_from_excel_legacy(file_path: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract Net Working Capital components for 2023 and 2024 from the QoE Excel file.
    Strategy:
    - Prefer sheets with 'recast' in the name; otherwise use others.
    - Locate rows by matching label substrings case-insensitively.
    - Columns are monthly; pick the latest month in each year that has a value.
    Returns a dict: {
        'assets': {'cash': v, 'accounts_receivable': v, 'other_current_assets': v},
        'liabilities': {'accounts_payable': v, 'credit_cards_payable': v, 'other_current_liabilities': v},
        'nwc_adjusted': {'2023': v_or_none, '2024': v_or_none}
    }
    """
    result = {
        "assets": {"cash": None, "accounts_receivable": None, "other_current_assets": None},
        "liabilities": {"accounts_payable": None, "credit_cards_payable": None, "other_current_liabilities": None},
        "nwc_adjusted": {"2023": None, "2024": None},
    }
    try:
        excel = pd.ExcelFile(file_path)
        recast = [s for s in excel.sheet_names if "recast" in s.lower()]
        regular = [s for s in excel.sheet_names if "recast" not in s.lower()]
        for sheets in [recast, regular]:
            for sheet in sheets:
                try:
                    # Try multiple header modes
                    for header in [1, 0, None]:
                        df = pd.read_excel(excel, sheet_name=sheet, header=header)
                        if df.empty or df.shape[1] < 2:
                            continue
                        # Normalize first column for label matching
                        first_col = df.iloc[:, 0].astype(str).map(lambda x: _normalize_string(x))
                        # Build header row candidates for months/dates
                        header_row_values = list(df.columns) if header is not None else list(df.iloc[0])

                        def match_index(aliases: List[str]) -> Optional[int]:
                            idxs = first_col[first_col.apply(lambda t: any(a in t for a in aliases))].index
                            return int(idxs[0]) if len(idxs) > 0 else None

                        def fallback_find_row_all_cells(aliases: List[str]) -> Optional[Tuple[int, int]]:
                            arr = df.fillna("").astype(str).values
                            nrows, ncols = arr.shape
                            for rr in range(nrows):
                                for cc in range(ncols):
                                    cell = _normalize_string(arr[rr][cc])
                                    if any(a in cell for a in aliases):
                                        return rr, cc + 1  # start reading one column to the right of label
                            return None

                        def build_headers_for_cols(start_col: int) -> List[object]:
                            headers: List[object] = []
                            # Use declared column headers when possible
                            cols = list(df.columns)
                            # Fallback to first few rows as header candidates
                            for j in range(start_col, df.shape[1]):
                                candidates = []
                                if j < len(cols):
                                    candidates.append(cols[j])
                                # Probe a deeper set of early rows to catch month rows
                                for probe_row in range(min(10, len(df))):
                                    candidates.append(df.iloc[probe_row, j])
                                # Also look directly above the data row if available
                                # choose first non-empty candidate
                                chosen = None
                                # Prefer candidates that clearly map to a year
                                for cand in candidates:
                                    if cand is None or str(cand).strip() == "":
                                        continue
                                    if _extract_year_from_value(cand, prefer_datetime=False) in NWC_YEARS:
                                        chosen = cand
                                        break
                                if chosen is None:
                                    for cand in candidates:
                                        if cand is not None and str(cand).strip() != "":
                                            chosen = cand
                                            break
                                headers.append(chosen)
                            return headers

                        # Extract per label
                        label_to_key = [
                            (NWC_LABELS["cash"], ("assets", "cash")),
                            (NWC_LABELS["accounts_receivable"], ("assets", "accounts_receivable")),
                            (NWC_LABELS["other_current_assets"], ("assets", "other_current_assets")),
                            (NWC_LABELS["accounts_payable"], ("liabilities", "accounts_payable")),
                            (NWC_LABELS["credit_cards_payable"], ("liabilities", "credit_cards_payable")),
                            (NWC_LABELS["other_current_liabilities"], ("liabilities", "other_current_liabilities")),
                        ]

                        for aliases, (bucket, key) in label_to_key:
                            norm_aliases = [_normalize_string(a) for a in aliases]
                            row_idx = match_index(norm_aliases)
                            start_col = 1
                            if row_idx is None:
                                pos = fallback_find_row_all_cells(norm_aliases)
                                if pos is None:
                                    continue
                                row_idx, start_col = pos
                            # Build row and header slices from the discovered start
                            row_vals = list(df.iloc[row_idx, start_col:])
                            hdr_vals = build_headers_for_cols(start_col)
                            # Pick latest month per year; store if not already populated
                            for yr in NWC_YEARS:
                                val = _find_latest_year_value(row_vals, hdr_vals, yr)
                                if isinstance(result[bucket].get(key), dict) is False:
                                    result[bucket][key] = {}
                                if result[bucket][key].get(yr) is None:
                                    result[bucket][key][yr] = val

                        # NWC, adjusted
                        nwc_idx = match_index([_normalize_string(a) for a in NWC_LABELS["nwc_adjusted"]])
                        start_col = 1
                        if nwc_idx is None:
                            pos = fallback_find_row_all_cells([_normalize_string(a) for a in NWC_LABELS["nwc_adjusted"]])
                            if pos is not None:
                                nwc_idx, start_col = pos
                        if nwc_idx is not None:
                            row_vals = list(df.iloc[nwc_idx, start_col:])
                            hdr_vals = build_headers_for_cols(start_col)
                            logger.info(f"NWC DEBUG: Found NWC row at index {nwc_idx}, row_vals={row_vals[:10]}, hdr_vals={hdr_vals[:10]}")
                            for yr in NWC_YEARS:
                                val = _find_latest_year_value(row_vals, hdr_vals, yr)
                                logger.info(f"NWC DEBUG: Year {yr} - extracted value: {val}")
                                if val is not None and result["nwc_adjusted"].get(yr) is None:
                                    result["nwc_adjusted"][yr] = val
                                elif val is None:
                                    logger.warning(f"NWC DEBUG: Failed to find value for year {yr} in headers: {hdr_vals}")
                                    # Fallback: try to find year directly in header values
                                    for idx, h in enumerate(hdr_vals):
                                        if h and str(h).strip() and yr in str(h):
                                            if idx < len(row_vals):
                                                fallback_val = _parse_number(row_vals[idx])
                                                if fallback_val is not None:
                                                    logger.info(f"NWC DEBUG: Fallback found {yr} value: {fallback_val} from header '{h}'")
                                                    result["nwc_adjusted"][yr] = fallback_val
                                                    break

                        # If we have filled a decent amount of data, we can return early on first matching sheet
                        # but we prefer completeness; still, if any NWC adjusted found on this sheet, keep it and continue.
                    # end header modes
                except Exception as se:
                    logger.debug(f"NWC parse failed for sheet '{sheet}' in {file_path}: {se}")
            # After trying recast sheets, if we already found NWC adjusted for both years, break early
            if result["nwc_adjusted"]["2023"] is not None or result["nwc_adjusted"]["2024"] is not None:
                break
        # Normalize component storage to ensure dict structure exists
        for bucket in ["assets", "liabilities"]:
            for key in list(result[bucket].keys()):
                if isinstance(result[bucket][key], dict) is False:
                    result[bucket][key] = {"2023": None, "2024": None}
                else:
                    for yr in NWC_YEARS:
                        result[bucket][key].setdefault(yr, None)
        return result
    except Exception as e:
        logger.warning(f"Failed to extract NWC from Excel {file_path}: {e}")
        return {
            "assets": {"cash": {"2023": None, "2024": None}, "accounts_receivable": {"2023": None, "2024": None}, "other_current_assets": {"2023": None, "2024": None}},
            "liabilities": {"accounts_payable": {"2023": None, "2024": None}, "credit_cards_payable": {"2023": None, "2024": None}, "other_current_liabilities": {"2023": None, "2024": None}},
            "nwc_adjusted": {"2023": None, "2024": None},
        }


def build_net_working_capital_table_html(nwc: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> str:
    """Render the Net Working Capital table for 2023 and 2024 from extracted values.
    If a value for a year is missing, show '-'.
    """
    def fmt_cell(v: Optional[float]) -> str:
        return "-" if v is None else f"${v:,.0f}" if abs(v - int(v)) < 1e-6 else f"${v:,.2f}"

    def get(bkt: str, key: str, yr: str) -> Optional[float]:
        entry = nwc.get(bkt, {}).get(key, {})
        return entry.get(yr)

    years = NWC_YEARS
    parts: List[str] = []
    parts.append('<table class="financial-table">')
    parts.append('<thead>')
    parts.append('<tr><th>Net Working Capital</th>' + ''.join(f'<th>{y}</th>' for y in years) + '</tr>')
    parts.append('</thead>')
    parts.append('<tbody>')
    # Current Assets
    parts.append('<tr><td><b>Current Assets</b></td>' + ''.join('<td></td>' for _ in years) + '</tr>')
    parts.append('<tr><td>Cash</td>' + ''.join(f'<td>{fmt_cell(get("assets","cash", y))}</td>' for y in years) + '</tr>')
    parts.append('<tr><td>Accounts Receivable</td>' + ''.join(f'<td>{fmt_cell(get("assets","accounts_receivable", y))}</td>' for y in years) + '</tr>')
    parts.append('<tr><td>Other</td>' + ''.join(f'<td>{fmt_cell(get("assets","other_current_assets", y))}</td>' for y in years) + '</tr>')
    # Totals for Current Assets
    for y in years:
        ca_vals = [get("assets","cash", y), get("assets","accounts_receivable", y), get("assets","other_current_assets", y)]
        ca_sum = sum(v for v in ca_vals if v is not None) if any(v is not None for v in ca_vals) else None
        parts.append('')
    parts.append('<tr><td><b>Total Current Assets</b></td>' + ''.join(
        f'<td>{fmt_cell((get("assets","cash", y) or 0) + (get("assets","accounts_receivable", y) or 0) + (get("assets","other_current_assets", y) or 0)) if any(v is not None for v in [get("assets","cash", y), get("assets","accounts_receivable", y), get("assets","other_current_assets", y)]) else "-"}</td>'
        for y in years) + '</tr>')
    # Current Liabilities
    parts.append('<tr><td><b>Current Liabilities</b></td>' + ''.join('<td></td>' for _ in years) + '</tr>')
    parts.append('<tr><td>Accounts Payable</td>' + ''.join(f'<td>{fmt_cell(get("liabilities","accounts_payable", y))}</td>' for y in years) + '</tr>')
    parts.append('<tr><td>Credit Cards Payable</td>' + ''.join(f'<td>{fmt_cell(get("liabilities","credit_cards_payable", y))}</td>' for y in years) + '</tr>')
    parts.append('<tr><td>Other</td>' + ''.join(f'<td>{fmt_cell(get("liabilities","other_current_liabilities", y))}</td>' for y in years) + '</tr>')
    parts.append('<tr><td><b>Total Current Liabilities</b></td>' + ''.join(
        f'<td>{fmt_cell((get("liabilities","accounts_payable", y) or 0) + (get("liabilities","credit_cards_payable", y) or 0) + (get("liabilities","other_current_liabilities", y) or 0)) if any(v is not None for v in [get("liabilities","accounts_payable", y), get("liabilities","credit_cards_payable", y), get("liabilities","other_current_liabilities", y)]) else "-"}</td>'
        for y in years) + '</tr>')
    # Net Working Capital (Adjusted)
    parts.append('<tr><td><b>Net Working Capital</b></td>' + ''.join(f'<td>{fmt_cell(nwc.get("nwc_adjusted", {}).get(y))}</td>' for y in years) + '</tr>')
    parts.append('</tbody>')
    parts.append('</table>')
    return ''.join(parts)


# Legacy function removed - replaced with Gemini-based extraction


async def extract_three_year_pnl_with_llm_from_documents(
    document_records: List[Dict[str, str]]
) -> str:
    """Extract P&L data using LLM from Excel documents and return HTML table."""
    # Filter Excel docs (consider either local file_path or storage_path)
    excel_docs = []
    for d in document_records:
        path_candidates = [d.get("file_path") or "", d.get("storage_path") or ""]
        if any(p.lower().endswith((".xlsx", ".xls")) for p in path_candidates):
            excel_docs.append(d)
    
    if not excel_docs:
        return '<p><strong>No Excel files found for P&L extraction</strong></p>'
    
    def score(doc: Dict[str, str]) -> int:
        name = (doc.get("name") or "").lower()
        s = 0
        if "qoe" in name or "quality of earnings" in name:
            s += 10
        if "ebitda" in name or "revenue" in name:
            s += 2
        return s

    ordered = sorted(excel_docs, key=score, reverse=True)

    # Try each ordered Excel until we get meaningful data
    for doc in ordered:
        file_path = doc.get("file_path") or ""
        storage_path = doc.get("storage_path") or ""

        candidate_paths = []
        if file_path:
            candidate_paths.append(file_path)
        if storage_path and storage_path not in candidate_paths:
            candidate_paths.append(storage_path)

        selected_path = None
        for p in candidate_paths:
            try_path = Path(p)
            if try_path.exists():
                selected_path = str(try_path)
                break

        if selected_path:
            logger.info(f"P&L LLM extraction: using local file path: {selected_path}")
            result = await extract_pnl_with_llm_from_excel(selected_path)
            if result and not result.startswith('<p><strong>No'):
                return result

        # If no local file path available, attempt to fetch from GCS
        if storage_path:
            try:
                from app.services.storage import get_storage_service
                storage_service = get_storage_service()
                if hasattr(storage_service, "download_file") and getattr(storage_service, 'bucket', None):
                    logger.info(f"P&L LLM extraction: downloading Excel from storage path: {storage_path}")
                    file_bytes = storage_service.client.bucket(storage_service.bucket_name).blob(storage_path).download_as_bytes()
                    if file_bytes:
                        # Create temp file for processing
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                            tmp_file.write(file_bytes)
                            tmp_file.flush()
                            result = await extract_pnl_with_llm_from_excel(tmp_file.name)
                            Path(tmp_file.name).unlink()  # cleanup
                            if result and not result.startswith('<p><strong>No'):
                                return result
            except Exception as fetch_err:
                logger.warning(f"P&L LLM extraction: failed to download Excel from storage path '{storage_path}': {fetch_err}")

    return '<p><strong>No P&L data found in available Excel files</strong></p>'


async def extract_pnl_with_llm_from_excel(file_path: str) -> str:
    """Extract P&L data using LLM from Excel file and return HTML table."""
    try:
        excel = pd.ExcelFile(file_path)
        
        # Find income statement tab with filtering logic:
        # 1. Filter by "IS" or "Income Statement"
        # 2. Prefer "recast" sheets
        # 3. Choose shorter name if multiple
        income_statement_tab = find_income_statement_tab(excel)
        if not income_statement_tab:
            return '<p><strong>Income statement not found</strong></p>'
        
        logger.info(f"Found income statement tab: {income_statement_tab}")
        
        # Convert to CSV
        csv_content = sheet_to_csv_string(excel, income_statement_tab)
        if not csv_content.strip():
            return '<p><strong>Failed to convert income statement to CSV</strong></p>'
        
        # Extract with Gemini
        pnl_data = await extract_pnl_with_gemini(csv_content)
        
        # Build HTML table
        return build_pnl_table_html(pnl_data, income_statement_tab)
        
    except Exception as e:
        logger.error(f"Failed to extract P&L from {file_path}: {e}")
        return '<p><strong>Error processing income statement</strong></p>'


def find_income_statement_tab(excel: pd.ExcelFile) -> Optional[str]:
    """Find the best income statement tab using the specified filtering logic."""
    # Filter by "IS" or "Income Statement"
    is_sheets = []
    for sheet_name in excel.sheet_names:
        name_lower = sheet_name.lower()
        if "is" in name_lower or "income statement" in name_lower:
            is_sheets.append(sheet_name)
    
    if not is_sheets:
        return None
    
    # Prefer "recast" sheets
    recast_sheets = [s for s in is_sheets if "recast" in s.lower()]
    if recast_sheets:
        # Choose shortest name among recast sheets
        return min(recast_sheets, key=len)
    
    # Choose shortest name among all IS sheets
    return min(is_sheets, key=len)


async def extract_pnl_with_gemini(csv_content: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Use Gemini to extract P&L data from income statement CSV."""
    from app.services.vertex_ai import vertex_ai_service
    
    prompt = f"""Analyze this income statement CSV and extract Revenue/Sales, Gross Profit, and Adjusted EBITDA for all available years.

CSV Data:
{csv_content}

Return JSON in this exact format:
{{
  "revenue": {{"2022": 123.45, "2023": 456.78, "2024": 789.01}},
  "gross_profit": {{"2022": 123.45, "2023": 456.78, "2024": 789.01}},
  "adjusted_ebitda": {{"2022": 123.45, "2023": 456.78, "2024": 789.01}},
  "year_headers": {{"2024": "2024 TTM September"}}
}}

Instructions:
- Extract data for ALL years present in the statement (not just 2022-2024). However, in the final JSON only include years 2022 and later. Exclude 2021 and older.
- Look for "Revenue", "Sales", "Net Revenue", "Net Sales", "Total Revenue" 
- Look for "Gross Profit", "Gross Income"
- Look for "Adjusted EBITDA", "EBITDA", "Reported EBITDA"
- If 2024 data appears multiple ways (e.g., TTM, Revised, Partial, Normal), return only ONE 2024 in the JSON keys. Prefer the full-year 2024 value if clearly labeled; otherwise prefer the latest TTM/partial 2024 and set year_headers["2024"] to a descriptive header like "2024 TTM September". Do not include separate keys for 2024 Revised/TTM/etc.
- Use null for missing values
- Return values in thousands if that's how they appear in the sheet"""

    try:
        response = await vertex_ai_service.generate_text(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            pnl_data = json.loads(json_str)
            return pnl_data
        else:
            logger.warning("No JSON found in Gemini response for P&L extraction")
            return {"revenue": {}, "gross_profit": {}, "adjusted_ebitda": {}, "year_headers": {}}
            
    except Exception as e:
        logger.error(f"Failed to extract P&L data with Gemini: {e}")
        return {"revenue": {}, "gross_profit": {}, "adjusted_ebitda": {}, "year_headers": {}}


def build_pnl_table_html(pnl_data: Dict, sheet_name: str = None) -> str:
    """Build HTML table from P&L data extracted by Gemini."""
    revenue = pnl_data.get("revenue", {})
    gross_profit = pnl_data.get("gross_profit", {})
    adjusted_ebitda = pnl_data.get("adjusted_ebitda", {})
    year_headers = dict(pnl_data.get("year_headers", {}))
    
    if not revenue and not gross_profit and not adjusted_ebitda:
        return '<p><strong>No P&L data found in income statement</strong></p>'
    
    # Normalize years: only include >= 2022 and collapse 2024 variants into a single column
    import re as _re

    def normalize_metric_dict(metric_dict: Dict[str, Optional[float]], metric_name: str) -> Dict[str, Optional[float]]:
        """Return a dict with keys limited to '2022','2023','2024'. For 2024, keep only one value:
        prefer normal full-year; else latest TTM/partial; else revised. Use year_headers['2024'] to label if TTM selected.
        """
        selection: Dict[str, Tuple[int, Optional[float], str]] = {}
        for key, value in metric_dict.items():
            # Extract a 4-digit year from the key
            key_str = str(key)
            m = _re.search(r"(20\d{2})", key_str)
            year = m.group(1) if m else (key_str if str(key_str).isdigit() else None)
            if not year:
                continue
            try:
                y_int = int(year)
            except Exception:
                continue
            if y_int < 2022:
                continue

            # Determine variant priority for 2024
            label_source = year_headers.get(year, key_str)
            label_lower = str(label_source).lower()
            # Priority: normal (2) > ttm/partial (1) > revised (0)
            priority = 2
            if "ttm" in label_lower or "trailing" in label_lower or "partial" in label_lower or _re.search(r"\bsep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug\b", label_lower):
                priority = 1
            if "revised" in label_lower or "rev" in label_lower:
                priority = min(priority, 0)

            current = selection.get(year)
            if current is None or priority > current[0]:
                selection[year] = (priority, value, label_source)

        # Build normalized dict
        normalized: Dict[str, Optional[float]] = {}
        for y in ["2022", "2023", "2024"]:
            if y in selection:
                prio, val, lbl = selection[y]
                normalized[y] = val
                # Update header if we chose a non-normal representation for 2024
                if y == "2024" and prio == 1:
                    year_headers["2024"] = str(lbl) if str(lbl).strip() else "2024 TTM"
                elif y == "2024" and prio == 2:
                    # Ensure clean header for normal 2024
                    year_headers.pop("2024", None)
        return normalized

    revenue_norm = normalize_metric_dict(revenue, "Revenue")
    gp_norm = normalize_metric_dict(gross_profit, "Gross Profit")
    ebitda_norm = normalize_metric_dict(adjusted_ebitda, "Adjusted EBITDA")

    # Compute years present across metrics and cap to last three years
    all_years = {y for y in ["2022", "2023", "2024"] if any(d.get(y) is not None for d in [revenue_norm, gp_norm, ebitda_norm])}
    if not all_years:
        return '<p><strong>No years found in P&L data</strong></p>'
    sorted_years = sorted(all_years, key=lambda y: int(y))
    
    def fmt_cell(v: Optional[float]) -> str:
        if v is None:
            return "-"
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
        return f"${v:,.0f}" if abs(v - int(v)) < 1e-6 else f"${v:,.2f}"
    
    def calculate_sga(gp: Optional[float], ebitda: Optional[float]) -> str:
        """Calculate SG&A as Gross Profit - EBITDA"""
        if gp is None or ebitda is None:
            return "-"
        if isinstance(gp, list) and len(gp) > 0:
            gp = gp[0]
        if isinstance(ebitda, list) and len(ebitda) > 0:
            ebitda = ebitda[0]
        sga = gp - ebitda
        return f"${sga:,.0f}" if abs(sga - int(sga)) < 1e-6 else f"${sga:,.2f}"
    
    # Build HTML table
    html = ['<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">']
    
    # Header row
    html.append('<tr style="background-color: #f8f9fa; font-weight: bold;">')
    html.append('<td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">Three year historical P&L</td>')
    
    for year in sorted_years:
        # Use custom header if available, otherwise just the year
        header = year_headers.get(year, year)
        html.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{header}</td>')
    
    html.append('</tr>')
    
    # Revenue row
    html.append('<tr>')
    html.append('<td style="border: 1px solid #dee2e6; padding: 8px;">Revenue</td>')
    for year in sorted_years:
        value = revenue_norm.get(year)
        html.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
    html.append('</tr>')
    
    # Gross Profit row
    html.append('<tr>')
    html.append('<td style="border: 1px solid #dee2e6; padding: 8px;">Gross Profit</td>')
    for year in sorted_years:
        value = gp_norm.get(year)
        html.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
    html.append('</tr>')
    
    # SG&A row (calculated)
    html.append('<tr>')
    html.append('<td style="border: 1px solid #dee2e6; padding: 8px;">SG&A</td>')
    for year in sorted_years:
        gp_value = gp_norm.get(year)
        ebitda_value = ebitda_norm.get(year)
        sga_value = calculate_sga(gp_value, ebitda_value)
        html.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{sga_value}</td>')
    html.append('</tr>')
    
    # Adjusted EBITDA row
    html.append('<tr>')
    html.append('<td style="border: 1px solid #dee2e6; padding: 8px;">Adjusted EBITDA</td>')
    for year in sorted_years:
        value = ebitda_norm.get(year)
        html.append(f'<td style="border: 1px solid #dee2e6; padding: 8px; text-align: right;">{fmt_cell(value)}</td>')
    html.append('</tr>')
    
    html.append('</table>')
    
    if sheet_name:
        html.append('<div style="margin-top: 8px; font-size: 9pt;">')
        html.append('<strong><span class="meta">Source:</span></strong><br/>')
        html.append(f'• P&L data: {sheet_name}<br/>')
        html.append('</div>')
    
    return '\n'.join(html)


def extract_three_year_pnl_from_documents(
    document_records: List[Dict[str, str]]
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
    """Given document records with at least 'file_path' and 'name', select likely QoE Excel and extract metrics.

    Selection order:
    1) Excel files whose name contains 'qoe' or 'quality of earnings'
    2) Any Excel files
    """
    # Filter Excel docs (consider either local file_path or storage_path)
    excel_docs = []
    for d in document_records:
        path_candidates = [d.get("file_path") or "", d.get("storage_path") or ""]
        if any(p.lower().endswith((".xlsx", ".xls")) for p in path_candidates):
            excel_docs.append(d)
    if not excel_docs:
        return {y: None for y in YEAR_TARGETS}, {y: None for y in YEAR_TARGETS}, {}

    def score(doc: Dict[str, str]) -> int:
        name = (doc.get("name") or "").lower()
        s = 0
        if "qoe" in name or "quality of earnings" in name:
            s += 10
        if "ebitda" in name or "revenue" in name:
            s += 2
        return s

    ordered = sorted(excel_docs, key=score, reverse=True)

    # Try each ordered Excel until we get any values
    for doc in ordered:
        # Prefer an existing local path; otherwise try storage_path (may also be local for local storage)
        file_path = doc.get("file_path") or ""
        storage_path = doc.get("storage_path") or ""

        candidate_paths = []
        if file_path:
            candidate_paths.append(file_path)
        if storage_path and storage_path not in candidate_paths:
            candidate_paths.append(storage_path)

        selected_path = None
        for p in candidate_paths:
            try_path = Path(p)
            if try_path.exists():
                selected_path = str(try_path)
                break

        if selected_path:
            logger.info(f"Financial extraction: using local file path: {selected_path}")
            rev, ebitda, sources = extract_three_year_pnl_from_excel(selected_path)
            if any(v is not None for v in rev.values()) or any(v is not None for v in ebitda.values()):
                return rev, ebitda, sources

        # If no local file path available, attempt to fetch from GCS using storage service
        if storage_path:
            try:
                from app.services.storage import get_storage_service
                storage_service = get_storage_service()
                if hasattr(storage_service, "download_file"):
                    logger.info(f"Financial extraction: downloading Excel from storage path: {storage_path}")
                    file_bytes = storage_service.client.bucket(storage_service.bucket_name).blob(storage_path).download_as_bytes() if getattr(storage_service, 'bucket', None) else None
                    if file_bytes:
                        rev, ebitda, sources = extract_three_year_pnl_from_excel_bytes(file_bytes, file_label=storage_path)
                        if any(v is not None for v in rev.values()) or any(v is not None for v in ebitda.values()):
                            return rev, ebitda, sources
            except Exception as fetch_err:
                logger.warning(f"Financial extraction: failed to download Excel from storage path '{storage_path}': {fetch_err}")

    # As a fallback, aggregate first Excel results even if empty
    first = ordered[0]
    fallback_path = first.get("file_path") or first.get("storage_path") or ""
    if fallback_path and Path(fallback_path).exists():
        rev, ebitda, sources = extract_three_year_pnl_from_excel(fallback_path)
        return rev, ebitda, sources
    # Final fallback: try storage bytes
    try:
        from app.services.storage import get_storage_service
        storage_service = get_storage_service()
        if first.get("storage_path") and getattr(storage_service, 'bucket', None):
            file_bytes = storage_service.client.bucket(storage_service.bucket_name).blob(first["storage_path"]).download_as_bytes()
            rev, ebitda, sources = extract_three_year_pnl_from_excel_bytes(file_bytes, file_label=first["storage_path"])
            return rev, ebitda, sources
    except Exception:
        pass
    return {y: None for y in YEAR_TARGETS}, {y: None for y in YEAR_TARGETS}, {}