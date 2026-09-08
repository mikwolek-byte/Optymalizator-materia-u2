import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SteelOpt - Optymalizator Hutniczy & Stykowanie",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEEL_DENSITY_KG_M3 = 7850.0  # Gęstość objętościowa stali konstrukcyjnej [kg/m³]

EURO_PROFILE_WEIGHTS: Dict[str, float] = {
    # IPE (PN-EN 10034)
    "IPE80": 6.0, "IPE100": 8.1, "IPE120": 10.4, "IPE140": 12.9, "IPE160": 15.8,
    "IPE180": 18.8, "IPE200": 22.4, "IPE220": 26.2, "IPE240": 30.7, "IPE270": 36.1,
    "IPE300": 42.2, "IPE330": 49.1, "IPE360": 57.1, "IPE400": 66.3, "IPE450": 77.6,
    "IPE500": 90.7, "IPE550": 106.0, "IPE600": 122.0,

    # HEA (PN-EN 10034)
    "HEA100": 16.7, "HEA120": 19.9, "HEA140": 24.7, "HEA160": 30.4, "HEA180": 35.5,
    "HEA200": 42.3, "HEA220": 50.5, "HEA240": 60.3, "HEA260": 68.2, "HEA280": 76.4,
    "HEA300": 88.3, "HEA320": 97.6, "HEA340": 105.0, "HEA360": 112.0, "HEA400": 125.0,
    "HEA450": 140.0, "HEA500": 155.0, "HEA550": 166.0, "HEA600": 178.0, "HEA650": 190.0,
    "HEA700": 204.0, "HEA800": 224.0, "HEA900": 252.0, "HEA1000": 272.0,

    # HEB (PN-EN 10034)
    "HEB100": 20.4, "HEB120": 26.7, "HEB140": 33.7, "HEB160": 42.6, "HEB180": 51.2,
    "HEB200": 61.3, "HEB220": 71.5, "HEB240": 83.2, "HEB260": 93.0, "HEB280": 103.0,
    "HEB300": 117.0, "HEB320": 127.0, "HEB340": 134.0, "HEB360": 142.0, "HEB400": 155.0,
    "HEB450": 171.0, "HEB500": 187.0, "HEB550": 199.0, "HEB600": 212.0, "HEB650": 225.0,
    "HEB700": 241.0, "HEB800": 262.0, "HEB900": 291.0, "HEB1000": 314.0,

    # HEM (PN-EN 10034)
    "HEM100": 41.8, "HEM120": 52.1, "HEM140": 63.2, "HEM160": 76.2, "HEM180": 88.9,
    "HEM200": 103.0, "HEM220": 117.0, "HEM240": 157.0, "HEM260": 172.0, "HEM280": 189.0,
    "HEM300": 238.0, "HEM320": 245.0, "HEM340": 248.0, "HEM360": 250.0, "HEM400": 256.0,
    "HEM450": 263.0, "HEM500": 270.0, "HEM550": 277.0, "HEM600": 285.0, "HEM650": 293.0,
    "HEM700": 301.0, "HEM800": 317.0, "HEM900": 333.0, "HEM1000": 349.0,

    # UNP (PN-EN 10279)
    "UNP80": 8.64, "UNP100": 10.6, "UNP120": 13.4, "UNP140": 16.0, "UNP160": 18.8,
    "UNP180": 22.0, "UNP200": 25.3, "UNP220": 29.4, "UNP240": 33.2, "UNP260": 37.9,
    "UNP280": 41.8, "UNP300": 46.2, "UNP320": 59.5, "UNP350": 60.6, "UNP380": 63.1,
    "UNP400": 71.8,
}

@dataclass(frozen=True)
class ProfileGroupKey:
    grade: str
    profile: str

@dataclass(frozen=True)
class PlateGroupKey:
    grade: str
    thickness: float

@dataclass
class Item1D:
    mark: str
    length: float
    profile: str
    grade: str
    quantity: int = 1

@dataclass(frozen=True)
class PlateFormat:
    name: str
    width: float
    length: float

@dataclass
class StockBar:
    bar_id: int
    stock_length: float
    profile: str
    grade: str
    used_length: float
    cuts: List[Tuple[str, float]] = field(default_factory=list)
    kerf_total: float = 0.0
    trim_total: float = 0.0
    scrap_length: float = 0.0
    is_oversized: bool = False

@dataclass
class PlateItem:
    mark: str
    grade: str
    thickness: float
    width: float
    length: float
    quantity: int = 1

@dataclass
class PackedRect:
    mark: str
    x: float
    y: float
    w: float
    h: float

@dataclass
class StockPlate:
    plate_id: int
    grade: str
    thickness: float
    stock_w: float
    stock_l: float
    packed_items: List[PackedRect] = field(default_factory=list)
    used_area: float = 0.0
    scrap_area: float = 0.0
    format_name: str = ""

def get_available_stocks(profile_str: str) -> List[float]:
    """
    Reguła Biznesowa: HEA, HEB, IPE, HEM -> [12100.0, 15100.0]
    Wszystkie inne -> [6000.0, 12000.0]
    """
    p = str(profile_str).upper().strip()
    if any(p.startswith(prefix) for prefix in ["HEA", "HEB", "IPE", "HEM"]):
        return [12100.0, 15100.0]
    return [6000.0, 12000.0]

def get_unit_weight_1d(profile_str: str) -> float:
    """Zwraca masę 1 mb profilu w kg."""
    raw = str(profile_str).upper().replace(" ", "").replace("×", "X").replace("*", "X").replace(",", ".")
    
    m_he = re.match(r"^HE(\d+)([ABM])$", raw)
    if m_he:
        size, variant = m_he.groups()
        raw = f"HE{variant}{size}"

    clean_prof = re.sub(r'[^A-Z0-9]', '', raw)
    for key, weight in EURO_PROFILE_WEIGHTS.items():
        if key == clean_prof or clean_prof.startswith(key):
            return weight

    m_angle = re.search(r'(?:L|KAT|KĄT)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_angle and any(prefix in raw for prefix in ["L", "KAT", "KĄT"]):
        dim1, dim2 = float(m_angle.group(1)), float(m_angle.group(2))
        dim3 = float(m_angle.group(3)) if m_angle.group(3) else None
        a, b, t = (dim1, dim2, dim3) if dim3 is not None else (dim1, dim1, dim2)
        area_mm2 = (a + b - t) * t * 1.012
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    m_rect = re.search(r'(?:CF)?(?:RHS|SHS|RK|RP|PR|PROFIL)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_rect and any(prefix in raw for prefix in ["RHS", "SHS", "RK", "RP", "PROFIL", "CF"]):
        h, b = float(m_rect.group(1)), float(m_rect.group(2))
        t = float(m_rect.group(3)) if m_rect.group(3) else b
        area_mm2 = 2.0 * t * (h + b) - 6.575 * (t ** 2)
        return round(max(area_mm2, 100.0) * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    return 20.0

def normalize_grade(raw_grade: str) -> str:
    if not raw_grade or pd.isna(raw_grade) or str(raw_grade).strip().lower() in ['nan', 'none', '']:
        return "S355J2+N"
    g = str(raw_grade).strip().upper()
    g = re.sub(r'\s+', '', g).replace("–", "-").replace("—", "-")
    if g in ["S235", "S235J", "S235JR", "S235J0", "S235J2"]:
        return "S235JR"
    if g in ["S355", "S355J", "S355JR", "S355J0", "S355J2", "S355J2+N", "S355J2N", "S355K2"]:
        return "S355J2+N"
    return g

def normalize_profile(raw_profile: str) -> str:
    if not raw_profile or pd.isna(raw_profile) or str(raw_profile).strip().lower() in ['nan', 'none', '']:
        return "UNKNOWN"
    p = str(raw_profile).strip().upper().replace(" ", "").replace("×", "X").replace("*", "X").replace(",", ".")
    m_he = re.match(r"^HE(\d+)([ABM])$", p)
    if m_he:
        size, variant = m_he.groups()
        p = f"HE{variant}{size}"
    return p

def optimize_1d_single_group(
    group_key: ProfileGroupKey,
    items: List[Item1D],
    kerf: float = 4.5,
    trim_cut: float = 40.0,
    start_bar_id: int = 1,
    enable_splicing: bool = False
) -> List[StockBar]:
    
    available_stocks = get_available_stocks(group_key.profile)
    sorted_stocks = sorted(available_stocks)
    max_stock_avail = sorted_stocks[-1]

    expanded_cuts: List[Tuple[str, float]] = []
    for it in items:
        for _ in range(it.quantity):
            expanded_cuts.append((it.mark, float(it.length)))
            
    expanded_cuts.sort(key=lambda x: x[1], reverse=True)
    stock_bars: List[StockBar] = []

    if enable_splicing:
        # Algorytm ze stykowaniem: płynne wypełnianie sztang i dzielenie elementów.
        current_bar = None
        for mark, length in expanded_cuts:
            remaining_length = length
            part_idx = 1
            while remaining_length > 0:
                if current_bar is None:
                    current_bar = StockBar(
                        bar_id=start_bar_id + len(stock_bars),
                        stock_length=max_stock_avail,
                        profile=group_key.profile,
                        grade=group_key.grade,
                        used_length=0.0,
                        cuts=[],
                        kerf_total=0.0,
                        trim_total=trim_cut,
                        scrap_length=0.0,
                        is_oversized=False
                    )
                    stock_bars.append(current_bar)

                required_kerf = kerf if len(current_bar.cuts) > 0 else 0.0
                available_space = current_bar.stock_length - current_bar.trim_total - current_bar.used_length

                if available_space <= required_kerf:
                    current_bar.scrap_length = available_space
                    current_bar = None
                    continue

                max_cut_here = available_space - required_kerf

                if remaining_length <= max_cut_here:
                    # Pasuje w całości
                    cut_name = f"{mark}_cz{part_idx}" if part_idx > 1 else mark
                    current_bar.cuts.append((cut_name, remaining_length))
                    current_bar.used_length += remaining_length + required_kerf
                    current_bar.kerf_total += required_kerf
                    remaining_length = 0
                else:
                    # Dzielenie profilu - reszta przechodzi na nową sztangę
                    cut_name = f"{mark}_cz{part_idx}"
                    current_bar.cuts.append((cut_name, max_cut_here))
                    current_bar.used_length += max_cut_here + required_kerf
                    current_bar.kerf_total += required_kerf
                    current_bar.scrap_length = 0.0
                    remaining_length -= max_cut_here
                    part_idx += 1
                    current_bar = None

        if current_bar is not None:
            current_bar.scrap_length = current_bar.stock_length - current_bar.trim_total - current_bar.used_length
            
    else:
        # Klasyczny algorytm FFD (Bez Stykowania)
        for mark, cut_len in expanded_cuts:
            if cut_len + trim_cut > max_stock_avail:
                stock_bars.append(StockBar(
                    bar_id=start_bar_id + len(stock_bars),
                    stock_length=cut_len + trim_cut,
                    profile=group_key.profile, grade=group_key.grade,
                    used_length=cut_len, cuts=[(mark, cut_len)],
                    kerf_total=0.0, trim_total=trim_cut, scrap_length=0.0,
                    is_oversized=True
                ))
                continue

            best_bar_idx = -1
            min_remaining_space = float("inf")

            for i, bar in enumerate(stock_bars):
                if bar.is_oversized: continue
                required_space = cut_len + (kerf if len(bar.cuts) > 0 else 0.0)
                capacity_left = bar.stock_length - (bar.used_length + bar.trim_total)
                if capacity_left >= required_space:
                    if capacity_left - required_space < min_remaining_space:
                        min_remaining_space = capacity_left - required_space
                        best_bar_idx = i

            if best_bar_idx != -1:
                bar = stock_bars[best_bar_idx]
                bar.cuts.append((mark, cut_len))
                bar.used_length += cut_len + kerf
                bar.kerf_total += kerf
                bar.scrap_length = bar.stock_length - bar.used_length - bar.trim_total
            else:
                eligible_stocks = [s for s in sorted_stocks if (s - trim_cut) >= cut_len]
                chosen_stock = eligible_stocks[0] if eligible_stocks else max_stock_avail
                stock_bars.append(StockBar(
                    bar_id=start_bar_id + len(stock_bars),
                    stock_length=chosen_stock,
                    profile=group_key.profile, grade=group_key.grade,
                    used_length=cut_len, cuts=[(mark, cut_len)],
                    kerf_total=0.0, trim_total=trim_cut,
                    scrap_length=chosen_stock - cut_len - trim_cut,
                    is_oversized=False
                ))

    return stock_bars

def pack_single_sheet_shelf(
    plate_id: int, grade: str, thickness: float, fmt: PlateFormat,
    parts: List[Tuple[str, float, float]], kerf_spacing: float, edge_margin: float
) -> Tuple[StockPlate, List[Tuple[str, float, float]]]:
    effective_w, effective_l = fmt.width - 2.0 * edge_margin, fmt.length - 2.0 * edge_margin
    plate = StockPlate(plate_id=plate_id, grade=grade, thickness=thickness, stock_w=fmt.width, stock_l=fmt.length, format_name=fmt.name)

    if effective_w <= 0 or effective_l <= 0 or not parts:
        return plate, list(parts)

    shelves: List[Dict[str, float]] = []
    unplaced_parts: List[Tuple[str, float, float]] = []

    for mark, pw, pl in parts:
        placed = False
        orientations = [(pw, pl), (pl, pw)] if pw != pl else [(pw, pl)]
        for o_w, o_l in orientations:
            if o_w > effective_w or o_l > effective_l: continue
            
            for shelf in shelves:
                if o_l <= shelf["height"] and (shelf["current_x"] + o_w) <= effective_w:
                    plate.packed_items.append(PackedRect(mark=mark, x=edge_margin + shelf["current_x"], y=edge_margin + shelf["y"], w=o_w, h=o_l))
                    shelf["current_x"] += o_w + kerf_spacing
                    plate.used_area += (o_w * o_l)
                    placed = True
                    break
            
            if placed: break

            last_y = shelves[-1]["y"] + shelves[-1]["height"] + kerf_spacing if shelves else 0.0
            if (last_y + o_l) <= effective_l and o_w <= effective_w:
                shelves.append({"y": last_y, "height": o_l, "current_x": o_w + kerf_spacing})
                plate.packed_items.append(PackedRect(mark=mark, x=edge_margin, y=edge_margin + last_y, w=o_w, h=o_l))
                plate.used_area += (o_w * o_l)
                placed = True
                break

        if not placed:
            unplaced_parts.append((mark, pw, pl))

    plate.scrap_area = max(0.0, (fmt.width * fmt.length) - plate.used_area)
    return plate, unplaced_parts

def optimize_2d_single_group(
    group_key: PlateGroupKey, items: List[PlateItem], available_formats: List[PlateFormat],
    kerf_spacing: float = 12.0, edge_margin: float = 20.0, start_plate_id: int = 1
) -> List[StockPlate]:
    parts = [(it.mark, min(it.width, it.length), max(it.width, it.length)) for it in items for _ in range(it.quantity)]
    parts.sort(key=lambda p: (p[1] * p[2]), reverse=True)

    if not parts or not available_formats: return []

    candidate_runs: List[List[StockPlate]] = []
    for fmt in available_formats:
        plates, remaining, curr_id = [], list(parts), start_plate_id
        valid = True
        while remaining:
            plate, unplaced = pack_single_sheet_shelf(curr_id, group_key.grade, group_key.thickness, fmt, remaining, kerf_spacing, edge_margin)
            if not plate.packed_items:
                valid = False
                break
            plates.append(plate)
            curr_id += 1
            remaining = unplaced
        if valid and plates:
            candidate_runs.append(plates)

    if not candidate_runs:
        largest_fmt = max(available_formats, key=lambda f: f.width * f.length)
        plates, remaining, curr_id = [], list(parts), start_plate_id
        while remaining:
            plate, unplaced = pack_single_sheet_shelf(curr_id, group_key.grade, group_key.thickness, largest_fmt, remaining, kerf_spacing, edge_margin)
            if not plate.packed_items:
                mark, pw, pl = remaining.pop(0)
                plates.append(StockPlate(plate_id=curr_id, grade=group_key.grade, thickness=group_key.thickness, stock_w=pw + 2*edge_margin, stock_l=pl + 2*edge_margin, packed_items=[PackedRect(mark=mark, x=edge_margin, y=edge_margin, w=pw, h=pl)], used_area=pw*pl, format_name="Niestandardowy"))
            else:
                plates.append(plate)
                remaining = unplaced
            curr_id += 1
        candidate_runs.append(plates)

    candidate_runs.sort(key=lambda run: (sum(p.scrap_area for p in run), len(run)))
    return candidate_runs[0]

def parse_bom_file(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    prefix = raw_bytes[:1500].strip()

    if b'<?xml' in prefix or b'urn:schemas-microsoft-com:office:spreadsheet' in prefix:
        try:
            root = ET.fromstring(raw_bytes.strip())
            ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
            rows_data = []
            for row in root.findall('.//ss:Row', ns):
                curr_col, row_cells = 0, {}
                for cell in row.findall('ss:Cell', ns):
                    idx = cell.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
                    if idx: curr_col = int(idx) - 1
                    data_elem = cell.find('ss:Data', ns)
                    row_cells[curr_col] = data_elem.text if data_elem is not None else ""
                    curr_col += 1
                if row_cells:
                    rows_data.append([row_cells.get(c, "") for c in range(max(row_cells.keys()) + 1)])
            raw_df = pd.DataFrame(rows_data)
            header_idx = None
            for idx, r in raw_df.iterrows():
                row_str = " ".join([str(v).lower() for v in r if v is not None])
                if "profil" in row_str and any(k in row_str for k in ['długość', 'length', 'materiał', 'pozycja']):
                    header_idx = idx
                    break
            if header_idx is not None:
                raw_df.columns = [str(c).strip() for c in raw_df.iloc[header_idx]]
                return raw_df.iloc[header_idx + 1:].reset_index(drop=True).dropna(how='all')
        except Exception:
            pass

    for engine in ['openpyxl', 'xlrd', None]:
        try: return pd.read_excel(io.BytesIO(raw_bytes), engine=engine) if engine else pd.read_excel(io.BytesIO(raw_bytes))
        except Exception: continue

    for enc in ['utf-8', 'windows-1250', 'iso-8859-2']:
        for sep in [';', ',', '\t']:
            try: return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=enc)
            except Exception: continue

    raise ValueError("Nie udało się odczytać pliku. Sprawdź format skoroszytu lub pliku CSV.")

def map_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c = str(col).lower().strip()
        if any(k in c for k in ['pozycja', 'pos', 'mark', 'nr elementu']) and 'mark' not in col_map.values(): col_map[col] = 'mark'
        elif any(k in c for k in ['profil', 'profile', 'przekrój', 'section']) and 'profile' not in col_map.values(): col_map[col] = 'profile'
        elif any(k in c for k in ['materiał', 'gatunek', 'grade']) and 'grade' not in col_map.values(): col_map[col] = 'grade'
        elif any(k in c for k in ['ilość', 'qty', 'szt']) and 'qty' not in col_map.values(): col_map[col] = 'qty'
        elif any(k in c for k in ['długość', 'length', 'l [mm]']) and 'całk' not in c and 'total' not in c and 'length' not in col_map.values(): col_map[col] = 'length'
        elif any(k in c for k in ['szerokość', 'width', 'b [mm]']) and 'całk' not in c and 'total' not in c and 'width' not in col_map.values(): col_map[col] = 'width'
        elif any(k in c for k in ['grubość', 'thick', 't [mm]']) and 'thick' not in col_map.values(): col_map[col] = 'thick'

    df_ren = df.rename(columns=col_map)
    clean_rows = []
    
    for _, row in df_ren.iterrows():
        m, p, l_v, q_v = str(row.get('mark', '')).strip(), str(row.get('profile', '')).strip(), str(row.get('length', '')).strip(), str(row.get('qty', '1')).strip()
        if 'suma' in m.lower() or 'total' in m.lower() or p in ['', 'None', 'nan']: continue

        try:
            q = int(round(float(q_v.replace(',', '.')))) if q_v not in ['', 'nan', 'None'] else 1
            l = float(l_v.replace(',', '.')) if l_v not in ['', 'nan', 'None'] else 0.0
            w = float(str(row.get('width', '')).strip().replace(',', '.')) if str(row.get('width', '')).strip() not in ['', 'nan', 'None'] else 0.0
            t = float(str(row.get('thick', '')).strip().replace(',', '.')) if str(row.get('thick', '')).strip() not in ['', 'nan', 'None'] else 0.0
            grd = str(row.get('grade', 'S355J2+N')).strip()

            if (w == 0.0 or t == 0.0) and any(sub in p.upper() for sub in ["PL", "BL", "BLACHA", "#", "-"]):
                m_dim = re.search(r'(?:PL|BL|BLACHA|#|-)?\s*(\d+(?:\.\d+)?)\s*[X*x]\s*(\d+(?:\.\d+)?)', p.upper())
                if m_dim: t, w = float(m_dim.group(1)), float(m_dim.group(2))

            if q > 0 and (l > 0 or w > 0):
                clean_rows.append({"mark": m if m not in ['', 'None', 'nan'] else f"P{len(clean_rows)+1}", "profile": p, "grade": grd, "length": l, "width": w, "thick": t, "qty": q})
        except (ValueError, TypeError): continue
    return pd.DataFrame(clean_rows)

def plot_1d_cutting_plan(bars: List[StockBar], title_suffix: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, max(3.2, len(bars) * 0.65)), dpi=120)
    colors = ["#2563EB", "#0D9488", "#EA580C", "#9333EA", "#16A34A", "#4F46E5", "#D97706", "#059669"]
    ax.set_yticks(list(range(len(bars))))
    ax.set_yticklabels([f"Sztanga #{b.bar_id} ({b.stock_length:.0f}mm)" for b in bars], fontsize=8.5, fontweight="bold")
    ax.invert_yaxis()

    for idx, b in enumerate(bars):
        curr_x = 0.0
        if b.trim_total > 0:
            ax.barh(idx, b.trim_total, left=0, color="#64748B", edgecolor="#1E293B", height=0.55, hatch="//")
            curr_x += b.trim_total
        for c_idx, (mark, cut_len) in enumerate(b.cuts):
            color = colors[(c_idx + idx) % len(colors)]
            ax.barh(idx, cut_len, left=curr_x, color=color, edgecolor="#0F172A", height=0.55)
            if cut_len > (b.stock_length * 0.05):
                ax.text(curr_x + cut_len / 2, idx, f"{mark}\n{cut_len:.0f}", va="center", ha="center", color="white", fontsize=7.5, fontweight="bold")
            curr_x += cut_len
            if c_idx < len(b.cuts) - 1:
                ax.barh(idx, 4.5, left=curr_x, color="#0F172A", height=0.55)
                curr_x += 4.5
        rem_scrap = b.stock_length - curr_x
        if rem_scrap > 0:
            ax.barh(idx, rem_scrap, left=curr_x, color="#F1F5F9", edgecolor="#94A3B8", height=0.55)
            if rem_scrap > (b.stock_length * 0.06):
                ax.text(curr_x + rem_scrap / 2, idx, f"Odpad {rem_scrap:.0f}mm", va="center", ha="center", color="#475569", fontsize=7.5)

    ax.set_xlabel("Długość handlowa sztangi [mm]", fontsize=9.5, fontweight="bold")
    ax.set_title(f"Rozkrój Sztang 1D {title_suffix} (Liczba sztang: {len(bars)})", fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_2d_plate_plan(plate: StockPlate) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=120)
    ax.add_patch(patches.Rectangle((0, 0), plate.stock_l, plate.stock_w, linewidth=2.0, edgecolor="#1E293B", facecolor="#F8FAFC", linestyle="--"))
    colors = ["#2563EB", "#0D9488", "#EA580C", "#9333EA", "#16A34A", "#4F46E5", "#D97706", "#059669", "#DC2626"]

    for idx, item in enumerate(plate.packed_items):
        ax.add_patch(patches.Rectangle((item.y, item.x), item.h, item.w, linewidth=1.2, edgecolor="#0F172A", facecolor=colors[idx % len(colors)], alpha=0.88))
        if item.h > 40 and item.w > 40:
            ax.text(item.y + item.h / 2.0, item.x + item.w / 2.0, f"{item.mark}\n{item.h:.0f}×{item.w:.0f}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    eff_pct = (plate.used_area / (plate.stock_w * plate.stock_l) * 100.0) if (plate.stock_w * plate.stock_l) > 0 else 0.0
    ax.set_title(f"Arkusz #{plate.plate_id} ({plate.format_name or f'{plate.stock_w:.0f}×{plate.stock_l:.0f}'}) | Grubość: #{plate.thickness:.0f} mm | Gatunek: {plate.grade} | Wykorzystanie: {eff_pct:.1f}%", fontsize=10.0, fontweight="bold", pad=12)
    ax.set_xlabel("Długość arkusza L [mm]", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Szerokość arkusza B [mm]", fontsize=9.0, fontweight="bold")
    margin_view = max(plate.stock_l, plate.stock_w) * 0.02
    ax.set_xlim(-margin_view, plate.stock_l + margin_view)
    ax.set_ylim(-margin_view, plate.stock_w + margin_view)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig

def build_excel_export(
    procurement_opt1: pd.DataFrame, procurement_opt2: pd.DataFrame,
    cut_1d_opt1: pd.DataFrame, cut_1d_opt2: pd.DataFrame,
    cut_summary_2d: pd.DataFrame,
    profile_stats_opt1: pd.DataFrame, profile_stats_opt2: pd.DataFrame,
    plate_stats_2d: pd.DataFrame, stats_df: pd.DataFrame
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if not stats_df.empty: stats_df.to_excel(writer, sheet_name="Podsumowanie Opcji", index=False)
        if not procurement_opt1.empty: procurement_opt1.to_excel(writer, sheet_name="1A. Zamówienie (Bez Styku)", index=False)
        if not procurement_opt2.empty: procurement_opt2.to_excel(writer, sheet_name="1B. Zamówienie (Ze Stykiem)", index=False)
        if not cut_1d_opt1.empty: cut_1d_opt1.to_excel(writer, sheet_name="2A. Rozkrój 1D (Bez Styku)", index=False)
        if not cut_1d_opt2.empty: cut_1d_opt2.to_excel(writer, sheet_name="2B. Rozkrój 1D (Ze Stykiem)", index=False)
        if not profile_stats_opt1.empty: profile_stats_opt1.to_excel(writer, sheet_name="3A. Odpad 1D (Bez Styku)", index=False)
        if not profile_stats_opt2.empty: profile_stats_opt2.to_excel(writer, sheet_name="3B. Odpad 1D (Ze Stykiem)", index=False)
        if not cut_summary_2d.empty: cut_summary_2d.to_excel(writer, sheet_name="4. Nesting Blach 2D", index=False)
        if not plate_stats_2d.empty: plate_stats_2d.to_excel(writer, sheet_name="5. Odpad Blach 2D", index=False)
    return buffer.getvalue()

with st.sidebar:
    st.header("⚙️ Parametry Technologiczne")

    st.subheader("Parametry Rozkroju 1D (Sztangi)")
    kerf_1d = st.number_input("Szerokość rzazu piły [mm]", min_value=1.0, max_value=12.0, value=4.5, step=0.5)
    trim_1d = st.number_input("Naddatek obcięcia końcówki [mm]", min_value=0.0, max_value=150.0, value=40.0, step=5.0)
    
    st.info("⚠️ **Dostępne długości sztang:** Zgodnie z regułą biznesową aplikacja automatycznie dobierze długości:\n\n• **HEA, HEB, IPE, HEM:** 12.1 m oraz 15.1 m\n• **Pozostałe (Kątowniki, Profile itp.):** 6.0 m oraz 12.0 m")

    st.divider()
    st.subheader("Parametry Rozkroju 2D (Arkusze)")
    kerf_2d = st.number_input("Odstęp cięcia termicznego [mm]", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
    margin_2d = st.number_input("Margines brzegowy arkusza [mm]", min_value=5.0, max_value=50.0, value=20.0, step=5.0)

    PLATE_FORMAT_CATALOG: Dict[str, Tuple[float, float]] = {
        "1500 × 3000 mm": (1500.0, 3000.0), "1500 × 6000 mm": (1500.0, 6000.0),
        "2000 × 6000 mm": (2000.0, 6000.0), "2000 × 12000 mm": (2000.0, 12000.0),
        "2500 × 6000 mm": (2500.0, 6000.0), "2500 × 12000 mm": (2500.0, 12000.0),
    }
    selected_format_labels = st.multiselect("Dostępne formaty arkuszy blach:", options=list(PLATE_FORMAT_CATALOG.keys()), default=["2000 × 6000 mm"])
    available_plate_formats = [PlateFormat(name=lbl, width=PLATE_FORMAT_CATALOG[lbl][0], length=PLATE_FORMAT_CATALOG[lbl][1]) for lbl in (selected_format_labels or ["2000 × 6000 mm"])]

    st.divider()
    st.subheader("💰 Wycena Szacunkowa")
    price_profile_per_kg = st.number_input("Cena stali kształtowej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.60, step=0.10)
    price_plate_per_kg = st.number_input("Cena blachy grubej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.90, step=0.10)
    scrap_price_per_kg = st.number_input("Wartość odkupu złomu [PLN/kg]", min_value=0.2, max_value=5.0, value=1.20, step=0.10)

col_upload, col_sample, col_clear = st.columns([3, 1.2, 0.8])
with col_upload: uploaded_file = st.file_uploader("Wczytaj plik Excel lub CSV z CAD/BIM:", type=["xlsx", "xls", "csv"])
with col_sample: 
    if st.button("🚀 Załaduj Testowy BOM", use_container_width=True):
        st.session_state["bom_data"] = pd.DataFrame([
            {"Pos": "B1", "Profile": "IPE300", "Grade": "S355J2+N", "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
            {"Pos": "C1", "Profile": "HEA240", "Grade": "S355J2+N", "Length_mm": 6250, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
            {"Pos": "K1", "Profile": "L100x100x10", "Grade": "S235JR", "Length_mm": 2400, "Width_mm": 0, "Thick_mm": 0, "Qty": 12},
            {"Pos": "PL1", "Profile": "BLACHA #10", "Grade": "S355J2+N", "Length_mm": 1200, "Width_mm": 800, "Thick_mm": 10, "Qty": 8}
        ])
        st.rerun()
with col_clear: 
    if st.button("🗑️ Wyczyść", use_container_width=True): 
        st.session_state["bom_data"] = None
        st.rerun()

if uploaded_file is not None:
    try:
        st.session_state["bom_data"] = parse_bom_file(uploaded_file)
        st.success(f"Pomyślnie zaimportowano plik `{uploaded_file.name}`.")
    except Exception as err:
        st.error(f"Błąd odczytu pliku: {err}")

df_raw = st.session_state.get("bom_data")

if df_raw is not None and not df_raw.empty:
    df_clean = map_imported_columns(df_raw)

    is_structural_1d = df_clean["profile"].str.contains(r"IPE|HEA|HEB|HEM|UNP|UPE|RHS|SHS|CHS|RO|ROHR|RK|RP|^(?:L|KAT|KĄT|D|FI)", case=False, regex=True)
    is_plate_condition = df_clean["profile"].str.contains(r"PL|BLACHA|PLATE|PŁYT|FORMATKA|#", case=False, regex=True) | ((df_clean["width"] > 0) & (df_clean["thick"] > 0) & (~is_structural_1d))

    raw_items_1d = [Item1D(mark=str(r["mark"]), length=float(r["length"]), profile=str(r["profile"]), grade=str(r["grade"]), quantity=int(r["qty"])) for _, r in df_clean[~is_plate_condition].iterrows() if r["length"] > 0]
    raw_items_2d = [PlateItem(mark=str(r["mark"]), grade=str(r["grade"]), thickness=float(r["thick"]) if float(r["thick"])>0 else 10.0, width=float(r["width"]) if float(r["width"])>0 else 200.0, length=float(r["length"]), quantity=int(r["qty"])) for _, r in df_clean[is_plate_condition].iterrows() if r["length"] > 0 and r["width"] > 0]

    grouped_1d = group_1d_items_by_material(raw_items_1d)
    
    # Przechowywanie wyników dla dwóch opcji
    bars_opt1, bars_opt2 = [], []
    order_items_opt1, order_items_opt2 = [], []
    stats_1d_opt1, stats_1d_opt2 = [], []

    def calculate_1d_scenario(enable_splice: bool, target_bars: list, target_order: list, target_stats: list):
        bar_id_counter = 1
        for group_key, group_items in grouped_1d.items():
            bars = optimize_1d_single_group(group_key, group_items, kerf=kerf_1d, trim_cut=trim_1d, start_bar_id=bar_id_counter, enable_splicing=enable_splice)
            bar_id_counter += len(bars)
            target_bars.extend(bars)

            unit_wt = get_unit_weight_1d(group_key.profile)
            stock_counts, sub_purchased_len = {}, 0.0
            for b in bars:
                stock_counts[b.stock_length] = stock_counts.get(b.stock_length, 0) + 1
                sub_purchased_len += b.stock_length

            for length_mm, qty_bars in stock_counts.items():
                target_order.append({"Kategoria": "Profil hutniczy (1D)", "Asortyment": group_key.profile, "Gatunek Stali": group_key.grade, "Wymiar Handlowy": f"L = {length_mm:.0f} mm", "Ilość Zamawiana [szt.]": qty_bars, "Masa Jednostkowa [kg]": round((length_mm / 1000.0) * unit_wt, 1), "Masa Łączna [kg]": round((length_mm / 1000.0) * unit_wt * qty_bars, 1), "Wymagany Atest": "3.1 wg PN-EN 10204"})

            sub_netto_len = sum(it.length * it.quantity for it in group_items)
            target_stats.append({"Profil": group_key.profile, "Gatunek": group_key.grade, "Masa 1mb [kg]": round(unit_wt, 2), "Liczba sztang [szt.]": len(bars), "Masa netto [kg]": round((sub_netto_len / 1000.0) * unit_wt, 1), "Masa brutto [kg]": round((sub_purchased_len / 1000.0) * unit_wt, 1), "Odpad [kg]": round(max(0.0, ((sub_purchased_len - sub_netto_len) / 1000.0) * unit_wt), 1), "Odpad [%]": round(((sub_purchased_len - sub_netto_len) / sub_purchased_len * 100.0) if sub_purchased_len > 0 else 0.0, 2)})

    # Przeliczanie obu opcji
    calculate_1d_scenario(False, bars_opt1, order_items_opt1, stats_1d_opt1)
    calculate_1d_scenario(True, bars_opt2, order_items_opt2, stats_1d_opt2)

    # Optymalizacja 2D (niezmienna)
    plates_result, order_items_2d, stats_2d = [], [], []
    plate_id_counter = 1
    for group_key, group_items in group_2d_plates_by_material(raw_items_2d).items():
        plates = optimize_2d_single_group(group_key, group_items, available_plate_formats, kerf_spacing=kerf_2d, edge_margin=margin_2d, start_plate_id=plate_id_counter)
        plate_id_counter += len(plates)
        plates_result.extend(plates)

        format_agg = {}
        for pl in plates: format_agg[(pl.stock_w, pl.stock_l, pl.format_name or f"{pl.stock_w:.0f} × {pl.stock_l:.0f} mm")] = format_agg.get((pl.stock_w, pl.stock_l, pl.format_name or f"{pl.stock_w:.0f} × {pl.stock_l:.0f} mm"), 0) + 1

        sub_gross_area_m2 = sum((pl.stock_w * pl.stock_l) / 1_000_000.0 for pl in plates)
        sub_net_area_m2 = sum((it.width * it.length * it.quantity) for it in group_items) / 1_000_000.0

        for (f_w, f_l, f_name), count in format_agg.items():
            single_mass_kg = (f_w * f_l / 1_000_000.0) * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0)
            order_items_2d.append({"Kategoria": "Blacha gruba (2D)", "Asortyment": f"Blacha #{group_key.thickness:.0f} mm", "Gatunek Stali": group_key.grade, "Wymiar Handlowy": f"{f_w:.0f} × {f_l:.0f} mm", "Ilość Zamawiana [szt.]": count, "Masa Jednostkowa [kg]": round(single_mass_kg, 1), "Masa Łączna [kg]": round(count * single_mass_kg, 1), "Wymagany Atest": "3.1 wg PN-EN 10204"})

        stats_2d.append({"Grubość [mm]": group_key.thickness, "Gatunek": group_key.grade, "Liczba arkuszy [szt.]": len(plates), "Masa netto [kg]": round(sub_net_area_m2 * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0), 1), "Masa brutto [kg]": round(sub_gross_area_m2 * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0), 1), "Odpad [%]": round(((sub_gross_area_m2 - sub_net_area_m2) / sub_gross_area_m2 * 100.0) if sub_gross_area_m2 > 0 else 0.0, 2)})

    # Obliczenia kosztowe dla obu opcji (wspólne 2D)
    plate_mass_kg = sum(r["Masa Łączna [kg]"] for r in order_items_2d)
    plate_cost = plate_mass_kg * price_plate_per_kg
    plate_net_mass = sum(r["Masa netto [kg]"] for r in stats_2d)
    plate_waste_kg = max(0.0, plate_mass_kg - plate_net_mass)

    def calc_totals(order_1d, stats_1d):
        m_kg = sum(r["Masa Łączna [kg]"] for r in order_1d)
        c_prof = m_kg * price_profile_per_kg
        n_m = sum(r["Masa netto [kg]"] for r in stats_1d)
        w_kg = max(0.0, m_kg - n_m)
        total_cost = c_prof + plate_cost
        total_scrap = (w_kg + plate_waste_kg) * scrap_price_per_kg
        return m_kg + plate_mass_kg, total_cost, total_cost - total_scrap

    mass1, cost_gross1, cost_net1 = calc_totals(order_items_opt1, stats_1d_opt1)
    mass2, cost_gross2, cost_net2 = calc_totals(order_items_opt2, stats_1d_opt2)

    tab_procure, tab_1d, tab_2d, tab_source = st.tabs(["🛒 Opcje i Zamówienie", "📏 Rozkrój Profili (1D)", "📐 Nesting Blach (2D)", "📋 Zaimportowany BOM"])

    with tab_procure:
        st.markdown("### Porównanie Kosztów: Opcja 1 (Bez Styku) vs Opcja 2 (Ze Stykiem)")
        
        c1, c2 = st.columns(2)
        c1.info(f"**Opcja 1: Klasyczne cięcie (Bez styku)**\n\n• Masa brutto zamówienia: **{mass1/1000.0:.2f} t**\n• Szacowany koszt zakupu: **{cost_gross1:,.2f} PLN**\n• Koszt netto (po odsprzedaży złomu): **{cost_net1:,.2f} PLN**")
        c2.success(f"**Opcja 2: Pełne wykorzystanie (Ze stykiem)**\n\n• Masa brutto zamówienia: **{mass2/1000.0:.2f} t**\n• Szacowany koszt zakupu: **{cost_gross2:,.2f} PLN**\n• Koszt netto (po odsprzedaży złomu): **{cost_net2:,.2f} PLN**")

        st.markdown("#### ✉️ Gotowa treść zapytania ofertowego (E-mail)")
        st.caption("Poniższy szablon nie zawiera już tabeli z profilami – wystarczy go skopiować, a dane załączyć w wygenerowanym Excelu.")
        email_body = (
            "Dzień dobry,\n\n"
            "Proszę o przygotowanie oferty cenowej oraz podanie dostępności dla wyrobów hutniczych, "
            "zgodnie ze szczegółowym zestawieniem (Opcja 1 lub Opcja 2) w załączonym pliku Excel.\n\n"
            "Wymagania dodatkowe:\n"
            "- Atest materiałowy 3.1 (PN-EN 10204) dla wszystkich zamawianych pozycji.\n"
            "- Proszę o uwzględnienie kosztów transportu na nasz zakład.\n\n"
            "Z góry dziękuję za odpowiedź.\n"
            "Pozdrawiam,\n[Twój Podpis]"
        )
        st.text_area("Szablon e-mail:", value=email_body, height=180)

        # Generowanie ramek danych do Excela
        df_opt1, df_opt2 = pd.DataFrame(order_items_opt1 + order_items_2d), pd.DataFrame(order_items_opt2 + order_items_2d)
        df_cut1_rows, df_cut2_rows = [], []
        for b in bars_opt1: df_cut1_rows.append({"Nr": b.bar_id, "Profil": b.profile, "Gatunek": b.grade, "Długość [mm]": b.stock_length, "Rozkrój": " + ".join([f"{m} ({l:.0f}mm)" for m, l in b.cuts]), "Odpad [mm]": round(b.scrap_length, 1)})
        for b in bars_opt2: df_cut2_rows.append({"Nr": b.bar_id, "Profil": b.profile, "Gatunek": b.grade, "Długość [mm]": b.stock_length, "Rozkrój": " + ".join([f"{m} ({l:.0f}mm)" for m, l in b.cuts]), "Odpad [mm]": round(b.scrap_length, 1)})
        
        df_cut2d_rows = [{"Nr": p.plate_id, "Grubość [mm]": p.thickness, "Gatunek": p.grade, "Format": f"{p.stock_w:.0f}×{p.stock_l:.0f}", "Detale": ", ".join([f"{it.mark}" for it in p.packed_items])} for p in plates_result]

        stats_summary = pd.DataFrame([
            {"Parametr": "Masa całkowita zamówienia [t]", "Opcja 1 (Bez Styku)": round(mass1/1000.0, 3), "Opcja 2 (Ze Stykiem)": round(mass2/1000.0, 3)},
            {"Parametr": "Szacowany Koszt Zakupu [PLN]", "Opcja 1 (Bez Styku)": round(cost_gross1, 2), "Opcja 2 (Ze Stykiem)": round(cost_gross2, 2)},
            {"Parametr": "Rzeczywisty Koszt Netto [PLN]", "Opcja 1 (Bez Styku)": round(cost_net1, 2), "Opcja 2 (Ze Stykiem)": round(cost_net2, 2)},
        ])

        excel_buffer = build_excel_export(
            df_opt1, df_opt2, pd.DataFrame(df_cut1_rows), pd.DataFrame(df_cut2_rows), pd.DataFrame(df_cut2d_rows),
            pd.DataFrame(stats_1d_opt1), pd.DataFrame(stats_1d_opt2), pd.DataFrame(stats_2d), stats_summary
        )
        st.download_button(label="📥 Pobierz Raport i Zamówienie z dwoma opcjami (Excel .xlsx)", data=excel_buffer, file_name="Zamowienie_Hutnicze_Opcje.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)

    with tab_1d:
        st.markdown("### Podgląd warsztatowy: Opcja 2 (Ze stykiem) - Maksymalna optymalizacja")
        if bars_opt2: st.pyplot(plot_1d_cutting_plan(bars_opt2[:25]))
        else: st.info("Brak profili.")

    with tab_2d:
        st.markdown("### Podgląd arkuszy (Nesting 2D)")
        if plates_result:
            opts = [f"Arkusz #{p.plate_id} - #{p.thickness:.0f}mm {p.grade} ({p.stock_w:.0f}×{p.stock_l:.0f})" for p in plates_result]
            sel_idx = st.selectbox("Wybierz:", range(len(plates_result)), format_func=lambda i: opts[i])
            st.pyplot(plot_2d_plate_plan(plates_result[sel_idx]))
        else: st.info("Brak blach.")

    with tab_source: st.dataframe(df_clean, use_container_width=True)
else:
    st.info("👈 Wgraj plik z zestawieniem materiałowym (BOM), aby rozpocząć optymalizację dwuwariantową.")
