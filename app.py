import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

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

    # UPE (PN-EN 10279)
    "UPE80": 7.9, "UPE100": 9.82, "UPE120": 12.1, "UPE140": 14.5, "UPE160": 17.0,
    "UPE180": 19.7, "UPE200": 22.8, "UPE220": 26.6, "UPE240": 30.2, "UPE270": 35.2,
    "UPE300": 44.4, "UPE330": 53.2, "UPE360": 61.2, "UPE400": 72.2,
}

PROFILE_GEOMETRY: Dict[str, Tuple[float, float, float, float]] = {
    # (h [mm], b [mm], tw [mm], tf [mm]) - przykładowe wartości nominalne wg PN-EN 10034
    "HEA240": (230.0, 240.0, 7.5, 12.0),
    "HEB600": (600.0, 300.0, 15.5, 30.0),
    "HEB800": (800.0, 300.0, 17.5, 33.0),
    "HEA180": (171.0, 180.0, 6.0, 9.5),
    "HEA220": (210.0, 220.0, 7.0, 11.0),
    "HEA280": (270.0, 280.0, 8.0, 13.0),
    "HEA340": (330.0, 300.0, 9.5, 16.5),
    "HEA450": (440.0, 300.0, 11.5, 21.0),
    "HEB160": (160.0, 160.0, 8.0, 13.0),
    "HEB180": (180.0, 180.0, 8.5, 14.0),
    "HEB280": (280.0, 280.0, 10.5, 18.0),
    "HEB340": (340.0, 300.0, 12.0, 21.5),
    "HEM140": (160.0, 148.0, 15.0, 22.5),
    "HEM180": (200.0, 186.0, 16.0, 25.0),
    "HEM240": (270.0, 205.0, 21.0, 40.0),
    "HEM300": (340.0, 310.0, 21.0, 39.0),
    "IPE200": (200.0, 100.0, 5.6, 8.5),
    "IPE300": (300.0, 150.0, 7.1, 10.7),
}

def get_profile_geometry(profile_str: str) -> Tuple[float, float, float, float]:
    p = str(profile_str).upper().replace(" ", "").replace("×", "X")
    if p in PROFILE_GEOMETRY:
        return PROFILE_GEOMETRY[p]
    # Fallback geometry based on height if string contains numbers
    m = re.search(r'(\d+)', p)
    h_val = float(m.group(1)) if m else 300.0
    return (h_val, h_val * 0.5, h_val * 0.03, h_val * 0.05)

def get_weld_passes(thickness: float) -> int:
    """Tabela doboru liczby ściegów wg grubości materiału (wartości pośrednie zaokrąglane w górę do mniej optymalnego progu)."""
    if thickness <= 5.6: return 2
    if thickness <= 7.9: return 3
    if thickness <= 10.0: return 3
    if thickness <= 15.0: return 4
    if thickness <= 17.0: return 5
    if thickness <= 20.0: return 6
    if thickness <= 25.0: return 8
    if thickness <= 30.0: return 12
    if thickness <= 35.0: return 18
    return 22

def calculate_splicing_cost(profile: str, grade: str, c_rbh: float = 130.0) -> Dict[str, float]:
    h, b, tw, tf = get_profile_geometry(profile)
    nw = get_weld_passes(tw)
    nf = get_weld_passes(tf)

    lw = 2.0 * (h / 1000.0)  # [mb] środnik dwustronnie
    lf = 2.0 * (b / 1000.0)  # [mb] pas górny i dolny
    l_styku = lw + lf         # [mb] sumaryczna długość do składania
    l_ut = (h + 2.0 * b) / 1000.0 # [mb] obrys do badań UT

    t_ciecie = 20.0
    t_skladanie = 15.0 * l_styku
    t_spaw_w = 20.0 * nw * lw
    t_spaw_f = 20.0 * nf * lf
    t_suma = t_ciecie + t_skladanie + t_spaw_w + t_spaw_f

    t_rbh = t_suma / 60.0
    k_rob = t_rbh * c_rbh
    k_ut = l_ut * 70.0
    k_styk = k_rob + k_ut

    return {
        "h": h, "b": b, "tw": tw, "tf": tf,
        "nw": nw, "nf": nf,
        "lw": lw, "lf": lf,
        "l_styku": l_styku, "l_ut": l_ut,
        "t_suma_min": t_suma,
        "t_rbh": t_rbh,
        "k_rob": k_rob,
        "k_ut": k_ut,
        "k_styk": k_styk
    }

def get_unit_weight_1d(profile_str: str) -> float:
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
        dim1 = float(m_angle.group(1))
        dim2 = float(m_angle.group(2))
        dim3 = float(m_angle.group(3)) if m_angle.group(3) else None
        a, b, t = (dim1, dim2, dim3) if dim3 is not None else (dim1, dim1, dim2)
        area_mm2 = (a + b - t) * t * 1.012
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    m_rect = re.search(r'(?:CF)?(?:RHS|SHS|RK|RP|PR|PROFIL)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_rect and any(prefix in raw for prefix in ["RHS", "SHS", "RK", "RP", "PROFIL", "CF"]):
        h = float(m_rect.group(1))
        b = float(m_rect.group(2))
        t = float(m_rect.group(3)) if m_rect.group(3) else b
        area_mm2 = 2.0 * t * (h + b) - 6.575 * (t ** 2)
        return round(max(area_mm2, 100.0) * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    m_pipe = re.search(r'(?:CHS|RO|ROHR|RURA|FI|Ø)\s*(\d+(?:\.\d+)?)[X/](\d+(?:\.\d+)?)', raw)
    if m_pipe:
        d = float(m_pipe.group(1))
        t = float(m_pipe.group(2))
        area_mm2 = math.pi * (d - t) * t
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    return 20.0

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

def group_1d_items_by_material(items: List[Item1D]) -> Dict[ProfileGroupKey, List[Item1D]]:
    grouped: Dict[ProfileGroupKey, List[Item1D]] = {}
    for item in items:
        norm_grd = normalize_grade(item.grade)
        norm_prf = normalize_profile(item.profile)
        key = ProfileGroupKey(grade=norm_grd, profile=norm_prf)
        grouped.setdefault(key, []).append(item)
    return grouped

def group_2d_plates_by_material(items: List[PlateItem]) -> Dict[PlateGroupKey, List[PlateItem]]:
    grouped: Dict[PlateGroupKey, List[PlateItem]] = {}
    for item in items:
        norm_grd = normalize_grade(item.grade)
        norm_thk = round(float(item.thickness), 2)
        key = PlateGroupKey(grade=norm_grd, thickness=norm_thk)
        grouped.setdefault(key, []).append(item)
    return grouped

def optimize_1d_single_group(
    group_key: ProfileGroupKey,
    items: List[Item1D],
    available_stocks: List[float],
    kerf: float = 4.5,
    trim_cut: float = 40.0,
    start_bar_id: int = 1,
) -> List[StockBar]:
    expanded_cuts: List[Tuple[str, float]] = []
    for it in items:
        for _ in range(it.quantity):
            expanded_cuts.append((it.mark, float(it.length)))

    expanded_cuts.sort(key=lambda x: x[1], reverse=True)
    sorted_stocks = sorted(available_stocks)
    max_stock_avail = sorted_stocks[-1]
    stock_bars: List[StockBar] = []

    for mark, cut_len in expanded_cuts:
        if cut_len + trim_cut > max_stock_avail:
            oversized_bar = StockBar(
                bar_id=start_bar_id + len(stock_bars),
                stock_length=cut_len + trim_cut,
                profile=group_key.profile,
                grade=group_key.grade,
                used_length=cut_len,
                cuts=[(mark, cut_len)],
                kerf_total=0.0,
                trim_total=trim_cut,
                scrap_length=0.0,
                is_oversized=True,
            )
            stock_bars.append(oversized_bar)
            continue

        best_bar_idx = -1
        min_remaining_space = float("inf")

        for i, bar in enumerate(stock_bars):
            if bar.is_oversized:
                continue
            required_space = cut_len + (kerf if len(bar.cuts) > 0 else 0.0)
            capacity_left = bar.stock_length - (bar.used_length + bar.trim_total)
            if capacity_left >= required_space:
                remaining_after = capacity_left - required_space
                if remaining_after < min_remaining_space:
                    min_remaining_space = remaining_after
                    best_bar_idx = i

        if best_bar_idx != -1:
            bar = stock_bars[best_bar_idx]
            bar.cuts.append((mark, cut_len))
            bar.used_length += cut_len + kerf
            bar.kerf_total += kerf
            bar.scrap_length = max(0.0, bar.stock_length - bar.used_length - bar.trim_total)
        else:
            eligible_stocks = [s for s in sorted_stocks if (s - trim_cut) >= cut_len]
            chosen_stock = eligible_stocks[0] if eligible_stocks else max_stock_avail
            new_bar = StockBar(
                bar_id=start_bar_id + len(stock_bars),
                stock_length=chosen_stock,
                profile=group_key.profile,
                grade=group_key.grade,
                used_length=cut_len,
                cuts=[(mark, cut_len)],
                kerf_total=0.0,
                trim_total=trim_cut,
                scrap_length=max(0.0, chosen_stock - cut_len - trim_cut),
                is_oversized=False,
            )
            stock_bars.append(new_bar)

    return stock_bars

def pack_single_sheet_shelf(
    plate_id: int,
    grade: str,
    thickness: float,
    fmt: PlateFormat,
    parts: List[Tuple[str, float, float]],
    kerf_spacing: float,
    edge_margin: float,
) -> Tuple[StockPlate, List[Tuple[str, float, float]]]:
    effective_w = fmt.width - 2.0 * edge_margin
    effective_l = fmt.length - 2.0 * edge_margin

    plate = StockPlate(
        plate_id=plate_id,
        grade=grade,
        thickness=thickness,
        stock_w=fmt.width,
        stock_l=fmt.length,
        packed_items=[],
        used_area=0.0,
        scrap_area=0.0,
        format_name=fmt.name,
    )

    if effective_w <= 0 or effective_l <= 0 or not parts:
        return plate, list(parts)

    shelves: List[Dict[str, float]] = []
    unplaced_parts: List[Tuple[str, float, float]] = []

    for mark, pw, pl in parts:
        placed = False
        orientations = [(pw, pl), (pl, pw)] if pw != pl else [(pw, pl)]
        for o_w, o_l in orientations:
            if o_w > effective_w or o_l > effective_l:
                continue
            for shelf in shelves:
                if o_l <= shelf["height"] and (shelf["current_x"] + o_w) <= effective_w:
                    x = edge_margin + shelf["current_x"]
                    y = edge_margin + shelf["y"]
                    plate.packed_items.append(PackedRect(mark=mark, x=x, y=y, w=o_w, h=o_l))
                    shelf["current_x"] += o_w + kerf_spacing
                    plate.used_area += (o_w * o_l)
                    placed = True
                    break
            if placed:
                break

            last_y_end = shelves[-1]["y"] + shelves[-1]["height"] + kerf_spacing if shelves else 0.0
            if (last_y_end + o_l) <= effective_l and o_w <= effective_w:
                new_shelf = {"y": last_y_end, "height": o_l, "current_x": o_w + kerf_spacing}
                shelves.append(new_shelf)
                x = edge_margin
                y = edge_margin + last_y_end
                plate.packed_items.append(PackedRect(mark=mark, x=x, y=y, w=o_w, h=o_l))
                plate.used_area += (o_w * o_l)
                placed = True
                break

        if not placed:
            unplaced_parts.append((mark, pw, pl))

    plate.scrap_area = max(0.0, (fmt.width * fmt.length) - plate.used_area)
    return plate, unplaced_parts

def optimize_2d_single_group(
    group_key: PlateGroupKey,
    items: List[PlateItem],
    available_formats: List[PlateFormat],
    kerf_spacing: float = 12.0,
    edge_margin: float = 20.0,
    start_plate_id: int = 1,
) -> List[StockPlate]:
    parts: List[Tuple[str, float, float]] = []
    for it in items:
        for _ in range(it.quantity):
            w = min(float(it.width), float(it.length))
            l = max(float(it.width), float(it.length))
            parts.append((it.mark, w, l))
    parts.sort(key=lambda p: (p[1] * p[2]), reverse=True)

    if not parts or not available_formats:
        return []

    candidate_runs: List[List[StockPlate]] = []
    for fmt in available_formats:
        plates: List[StockPlate] = []
        remaining = list(parts)
        curr_id = start_plate_id
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
        plates = []
        remaining = list(parts)
        curr_id = start_plate_id
        while remaining:
            plate, unplaced = pack_single_sheet_shelf(curr_id, group_key.grade, group_key.thickness, largest_fmt, remaining, kerf_spacing, edge_margin)
            if not plate.packed_items:
                mark, pw, pl = remaining.pop(0)
                custom_plate = StockPlate(
                    plate_id=curr_id, grade=group_key.grade, thickness=group_key.thickness,
                    stock_w=pw + 2*edge_margin, stock_l=pl + 2*edge_margin,
                    packed_items=[PackedRect(mark=mark, x=edge_margin, y=edge_margin, w=pw, h=pl)],
                    used_area=pw*pl, scrap_area=0.0, format_name="Niestandardowy"
                )
                plates.append(custom_plate)
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
                curr_col = 0
                row_cells = {}
                for cell in row.findall('ss:Cell', ns):
                    idx = cell.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Index')
                    if idx:
                        curr_col = int(idx) - 1
                    data_elem = cell.find('ss:Data', ns)
                    val = data_elem.text if data_elem is not None else ""
                    row_cells[curr_col] = val
                    curr_col += 1
                if row_cells:
                    max_c = max(row_cells.keys())
                    rows_data.append([row_cells.get(c, "") for c in range(max_c + 1)])
            raw_df = pd.DataFrame(rows_data)
            header_idx = None
            for idx, r in raw_df.iterrows():
                row_str = " ".join([str(v).lower() for v in r if v is not None])
                if "profil" in row_str and any(k in row_str for k in ['długość', 'length', 'materiał', 'pozycja']):
                    header_idx = idx
                    break
            if header_idx is not None:
                raw_df.columns = [str(c).strip() for c in raw_df.iloc[header_idx]]
                raw_df = raw_df.iloc[header_idx + 1:].reset_index(drop=True)
            return raw_df.dropna(how='all')
        except Exception:
            pass

    for engine in ['openpyxl', 'xlrd', None]:
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), engine=engine) if engine else pd.read_excel(io.BytesIO(raw_bytes))
        except Exception:
            continue

    for enc in ['utf-8', 'windows-1250', 'iso-8859-2']:
        for sep in [';', ',', '\t']:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=enc)
            except Exception:
                continue

    raise ValueError("Nie udało się odczytać pliku. Sprawdź format skoroszytu.")

def map_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c_clean = str(col).lower().strip()
        if any(k in c_clean for k in ['pozycja', 'position', 'pos', 'mark', 'nr elementu']) and 'mark' not in col_map.values():
            col_map[col] = 'mark'
        elif any(k in c_clean for k in ['profil', 'profile', 'przekrój', 'section']) and 'profile' not in col_map.values():
            col_map[col] = 'profile'
        elif any(k in c_clean for k in ['materiał', 'material', 'gatunek', 'grade']) and 'grade' not in col_map.values():
            col_map[col] = 'grade'
        elif any(k in c_clean for k in ['ilość', 'ilosc', 'quantity', 'qty', 'szt']) and 'qty' not in col_map.values():
            col_map[col] = 'qty'
        elif any(k in c_clean for k in ['długość', 'dlugosc', 'length', 'l [mm]']) and not any(k in c_clean for k in ['całk', 'total']) and 'length' not in col_map.values():
            col_map[col] = 'length'
        elif any(k in c_clean for k in ['szerokość', 'szerokosc', 'width', 'b [mm]']) and not any(k in c_clean for k in ['całk', 'total']) and 'width' not in col_map.values():
            col_map[col] = 'width'
        elif any(k in c_clean for k in ['grubość', 'grubosc', 'thick', 't [mm]']) and 'thick' not in col_map.values():
            col_map[col] = 'thick'

    df_ren = df.rename(columns=col_map)
    clean_rows = []
    for _, row in df_ren.iterrows():
        mark_val = str(row.get('mark', '')).strip()
        prof_val = str(row.get('profile', '')).strip()
        len_val = str(row.get('length', '')).strip()
        qty_val = str(row.get('qty', '1')).strip()

        if any(w in mark_val.lower() for w in ['suma', 'total']) or any(w in len_val.lower() for w in ['suma', 'total']):
            continue
        if prof_val in ['', 'None', 'nan']:
            continue

        try:
            q = int(round(float(str(qty_val).replace(',', '.'))))
            l = float(str(len_val).replace(',', '.'))
            w = float(str(row.get('width', 0.0)).replace(',', '.')) if pd.notna(row.get('width')) else 0.0
            t = float(str(row.get('thick', 0.0)).replace(',', '.')) if pd.notna(row.get('thick')) else 0.0
            grd = str(row.get('grade', 'S355J2+N')).strip()

            if (w == 0.0 or t == 0.0) and any(p_sub in prof_val.upper() for p_sub in ["PL", "BL", "BLACHA", "#", "-"]):
                m_dim = re.search(r'(?:PL|BL|BLACHA|#|-)?\s*(\d+(?:\.\d+)?)\s*[X*x]\s*(\d+(?:\.\d+)?)', prof_val.upper())
                if m_dim:
                    t = float(m_dim.group(1))
                    w = float(m_dim.group(2))

            if q > 0 and l > 0:
                clean_rows.append({
                    "mark": mark_val if mark_val not in ['', 'None', 'nan'] else f"P{len(clean_rows)+1}",
                    "profile": prof_val,
                    "grade": grd,
                    "length": l,
                    "width": w,
                    "thick": t,
                    "qty": q,
                })
        except (ValueError, TypeError):
            continue
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
    sheet_rect = patches.Rectangle((0, 0), plate.stock_l, plate.stock_w, linewidth=2.0, edgecolor="#1E293B", facecolor="#F8FAFC", linestyle="--")
    ax.add_patch(sheet_rect)
    colors = ["#2563EB", "#0D9488", "#EA580C", "#9333EA", "#16A34A", "#4F46E5", "#D97706", "#059669", "#DC2626"]

    for idx, item in enumerate(plate.packed_items):
        color = colors[idx % len(colors)]
        part_rect = patches.Rectangle((item.y, item.x), item.h, item.w, linewidth=1.2, edgecolor="#0F172A", facecolor=color, alpha=0.88)
        ax.add_patch(part_rect)
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

def build_excel_export(procurement_df, cut_summary_1d, cut_summary_2d, profile_stats_1d, plate_stats_2d, splicing_df, stats_df) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if procurement_df is not None and not procurement_df.empty:
            procurement_df.to_excel(writer, sheet_name="1. Do Zamówienia (RFQ)", index=False)
        if cut_summary_1d is not None and not cut_summary_1d.empty:
            cut_summary_1d.to_excel(writer, sheet_name="2. Rozkrój Warsztat 1D", index=False)
        if cut_summary_2d is not None and not cut_summary_2d.empty:
            cut_summary_2d.to_excel(writer, sheet_name="3. Nesting Blach 2D", index=False)
        if profile_stats_1d is not None and not profile_stats_1d.empty:
            profile_stats_1d.to_excel(writer, sheet_name="4. Odpad wg Profili 1D", index=False)
        if plate_stats_2d is not None and not plate_stats_2d.empty:
            plate_stats_2d.to_excel(writer, sheet_name="5. Odpad wg Blach 2D", index=False)
        if splicing_df is not None and not splicing_df.empty:
            splicing_df.to_excel(writer, sheet_name="6. Analiza Stykowania", index=False)
        if stats_df is not None and not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="7. Podsumowanie Kosztów", index=False)
    return buffer.getvalue()

st.title("🏗️ SteelOpt: Optymalizator Rozkroju i Generator RFQ + Stykowanie")
st.caption("Zaawansowane planowanie hutnicze | Izolacja gatunków stali | Opcjonalne Stykowanie Profili")

if "bom_data" not in st.session_state:
    st.session_state["bom_data"] = None

with st.sidebar:
    st.header("⚙️ Parametry Technologiczne")

    st.subheader("Parametry Rozkroju 1D (Sztangi)")
    kerf_1d = st.number_input("Szerokość rzazu piły [mm]", min_value=1.0, max_value=12.0, value=4.5, step=0.5)
    trim_1d = st.number_input("Naddatek obcięcia końcówki [mm]", min_value=0.0, max_value=150.0, value=40.0, step=5.0)
    stock_options_1d = st.multiselect(
        "Dostępne sztangi handlowe [mm]:",
        options=[6000.0, 12000.0, 12100.0, 14000.0, 15000.0, 15100.0, 18000.0],
        default=[6000.0, 12100.0, 15100.0]
    )
    if not stock_options_1d:
        stock_options_1d = [12100.0]

    st.divider()
    st.subheader("🔗 Moduł Opcjonalnego Stykowania Profili")
    enable_splicing = st.checkbox(
        "Zezwalaj na stykowanie profili (Weld Splicing)",
        value=False,
        help="Gdy aktywne, system analizuje opłacalność łączenia krótszych odcinków spawaniem doczołowym zamiast zakupu nowych sztang."
    )
    splicing_rbh_rate = st.number_input("Stawka roboczogodziny spawania [PLN/h]", min_value=50.0, max_value=300.0, value=130.0, step=10.0)

    st.divider()
    st.subheader("Parametry Rozkroju 2D (Arkusze)")
    kerf_2d = st.number_input("Odstęp cięcia termicznego [mm]", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
    margin_2d = st.number_input("Margines brzegowy arkusza [mm]", min_value=5.0, max_value=50.0, value=20.0, step=5.0)

    PLATE_FORMAT_CATALOG: Dict[str, Tuple[float, float]] = {
        "1500 × 3000 mm": (1500.0, 3000.0),
        "1500 × 6000 mm": (1500.0, 6000.0),
        "2000 × 6000 mm": (2000.0, 6000.0),
        "2000 × 12000 mm": (2000.0, 12000.0),
        "2500 × 6000 mm": (2500.0, 6000.0),
        "2500 × 12000 mm": (2500.0, 12000.0),
    }
    selected_format_labels = st.multiselect(
        "Dostępne formaty arkuszy blach:",
        options=list(PLATE_FORMAT_CATALOG.keys()),
        default=["1500 × 3000 mm", "1500 × 6000 mm", "2000 × 6000 mm", "2000 × 12000 mm"]
    )
    if not selected_format_labels:
        selected_format_labels = ["2000 × 6000 mm"]

    available_plate_formats = [
        PlateFormat(name=lbl, width=PLATE_FORMAT_CATALOG[lbl][0], length=PLATE_FORMAT_CATALOG[lbl][1])
        for lbl in selected_format_labels
    ]

    st.divider()
    st.subheader("💰 Wycena Szacunkowa")
    price_profile_per_kg = st.number_input("Cena stali kształtowej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.60, step=0.10)
    price_plate_per_kg = st.number_input("Cena blachy grubej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.90, step=0.10)
    scrap_price_per_kg = st.number_input("Wartość odkupu złomu [PLN/kg]", min_value=0.2, max_value=5.0, value=1.20, step=0.10)

st.subheader("📥 1. Dane Wejściowe: Zestawienie Materiałowe (BOM)")
col_upload, col_sample, col_clear = st.columns([3, 1.2, 0.8])

with col_upload:
    uploaded_file = st.file_uploader("Wczytaj plik Excel (.xlsx, .xls) lub CSV z CAD/BIM:", type=["xlsx", "xls", "csv"])

with col_sample:
    st.write("Szybki test:")
    if st.button("🚀 Załaduj Testowy BOM", use_container_width=True):
        st.session_state["bom_data"] = pd.DataFrame([
            {"Pos": "B1_355", "Profile": "IPE300", "Grade": "S355J2+N", "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
            {"Pos": "B2_235", "Profile": "IPE300", "Grade": "S235JR", "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
            {"Pos": "C1_355", "Profile": "HEA240", "Grade": "S355J2+N", "Length_mm": 6250, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
            {"Pos": "C2_235", "Profile": "HEA240", "Grade": "S235JR", "Length_mm": 4150, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
            {"Pos": "K1_235", "Profile": "L100x100x10", "Grade": "S235JR", "Length_mm": 2400, "Width_mm": 0, "Thick_mm": 0, "Qty": 12},
            {"Pos": "PL1_355_10", "Profile": "BLACHA #10", "Grade": "S355J2+N", "Length_mm": 1200, "Width_mm": 800, "Thick_mm": 10, "Qty": 8},
            {"Pos": "PL2_235_10", "Profile": "BLACHA #10", "Grade": "S235JR", "Length_mm": 1200, "Width_mm": 800, "Thick_mm": 10, "Qty": 6},
            {"Pos": "PL3_355_20", "Profile": "BLACHA #20", "Grade": "S355J2+N", "Length_mm": 650, "Width_mm": 450, "Thick_mm": 20, "Qty": 16},
        ])
        st.rerun()

with col_clear:
    st.write("Reset:")
    if st.button("🗑️ Wyczyść", use_container_width=True):
        st.session_state["bom_data"] = None
        st.rerun()

if uploaded_file is not None:
    try:
        loaded_df = parse_bom_file(uploaded_file)
        st.session_state["bom_data"] = loaded_df
        st.success(f"Pomyślnie zaimportowano plik `{uploaded_file.name}` ({len(loaded_df)} wierszy).")
    except Exception as err:
        st.error(f"Błąd odczytu pliku: {err}")

df_raw = st.session_state.get("bom_data")

if df_raw is not None and not df_raw.empty:
    df_clean = map_imported_columns(df_raw)

    is_structural_1d = df_clean["profile"].str.contains(
        r"IPE|HEA|HEB|HEM|UNP|UPE|RHS|SHS|CHS|RO|ROHR|RK|RP|^(?:L|KAT|KĄT|D|FI)", case=False, regex=True
    )
    is_plate_condition = (
        df_clean["profile"].str.contains(r"PL|BLACHA|PLATE|PŁYT|FORMATKA|#", case=False, regex=True)
        | ((df_clean["width"] > 0) & (df_clean["thick"] > 0) & (~is_structural_1d))
    )

    df_1d = df_clean[~is_plate_condition].copy()
    df_2d = df_clean[is_plate_condition].copy()

    raw_items_1d = [
        Item1D(mark=str(r["mark"]), length=float(r["length"]), profile=str(r["profile"]), grade=str(r["grade"]), quantity=int(r["qty"]))
        for _, r in df_1d.iterrows()
    ]
    raw_items_2d = [
        PlateItem(mark=str(r["mark"]), grade=str(r["grade"]), thickness=float(r["thick"]) if float(r["thick"]) > 0 else 10.0, width=float(r["width"]) if float(r["width"]) > 0 else 200.0, length=float(r["length"]), quantity=int(r["qty"]))
        for _, r in df_2d.iterrows()
    ]

    grouped_1d_dict = group_1d_items_by_material(raw_items_1d)
    grouped_2d_dict = group_2d_plates_by_material(raw_items_2d)

    bars_result_all: List[StockBar] = []
    bars_by_group: Dict[ProfileGroupKey, List[StockBar]] = {}
    order_items_unified: List[Dict] = []
    profile_waste_summary_1d: List[Dict] = []
    splicing_analysis_rows: List[Dict] = []

    bar_id_counter = 1
    for group_key, group_items in grouped_1d_dict.items():
        bars = optimize_1d_single_group(group_key, group_items, stock_options_1d, kerf=kerf_1d, trim_cut=trim_1d, start_bar_id=bar_id_counter)
        bar_id_counter += len(bars)
        bars_result_all.extend(bars)
        bars_by_group[group_key] = bars

        unit_wt = get_unit_weight_1d(group_key.profile)
        stock_counts: Dict[float, int] = {}
        sub_purchased_len = 0.0
        for b in bars:
            stock_counts[b.stock_length] = stock_counts.get(b.stock_length, 0) + 1
            sub_purchased_len += b.stock_length

        for length_mm, qty_bars in stock_counts.items():
            tot_mass = (length_mm / 1000.0) * unit_wt * qty_bars
            order_items_unified.append({
                "Kategoria": "Profil hutniczy (1D)",
                "Asortyment": group_key.profile,
                "Gatunek Stali": group_key.grade,
                "Wymiar Handlowy": f"L = {length_mm:.0f} mm",
                "Ilość Zamawiana [szt.]": qty_bars,
                "Masa Jednostkowa [kg]": round((length_mm / 1000.0) * unit_wt, 1),
                "Masa Łączna [kg]": round(tot_mass, 1),
                "Wymagany Atest": "3.1 wg PN-EN 10204",
            })

        sub_netto_len = sum(it.length * it.quantity for it in group_items)
        sub_netto_mass = (sub_netto_len / 1000.0) * unit_wt
        sub_purchased_mass = (sub_purchased_len / 1000.0) * unit_wt
        sub_scrap_len_m = max(0.0, (sub_purchased_len - sub_netto_len) / 1000.0)
        sub_scrap_mass_kg = sub_scrap_len_m * unit_wt
        sub_scrap_pct = ((sub_purchased_len - sub_netto_len) / sub_purchased_len * 100.0) if sub_purchased_len > 0 else 0.0

        profile_waste_summary_1d.append({
            "Profil": group_key.profile,
            "Gatunek": group_key.grade,
            "Masa 1mb [kg]": round(unit_wt, 2),
            "Liczba sztang [szt.]": len(bars),
            "Długość netto [m]": round(sub_netto_len / 1000.0, 2),
            "Długość brutto [m]": round(sub_purchased_len / 1000.0, 2),
            "Odpad [kg]": round(sub_scrap_mass_kg, 1),
            "Odpad [%]": round(sub_scrap_pct, 2),
        })

        # Splicing analysis for profile group
        if enable_splicing:
            s_cost = calculate_splicing_cost(group_key.profile, group_key.grade, c_rbh=splicing_rbh_rate)
            delta_c_mat = price_profile_per_kg - scrap_price_per_kg
            bep_kg = s_cost["k_styk"] / delta_c_mat if delta_c_mat > 0 else 0.0
            bep_m = bep_kg / unit_wt if unit_wt > 0 else 0.0
            splicing_analysis_rows.append({
                "Profil": group_key.profile,
                "Gatunek": group_key.grade,
                "Masa 1mb [kg/m]": round(unit_wt, 2),
                "Czas styku [min]": round(s_cost["t_suma_min"], 1),
                "Koszt robocizny [PLN]": round(s_cost["k_rob"], 2),
                "Koszt badań UT [PLN]": round(s_cost["k_ut"], 2),
                "Koszt całkowity styku [PLN]": round(s_cost["k_styk"], 2),
                "Próg opłacalności (BEP) [kg]": round(bep_kg, 1),
                "Próg opłacalności (BEP) [m]": round(bep_m, 2),
            })

    plates_result_all: List[StockPlate] = []
    plates_by_group: Dict[PlateGroupKey, List[StockPlate]] = {}
    plate_waste_summary_2d: List[Dict] = []
    plate_id_counter = 1

    for group_key, group_items in grouped_2d_dict.items():
        plates = optimize_2d_single_group(group_key, group_items, available_plate_formats, kerf_spacing=kerf_2d, edge_margin=margin_2d, start_plate_id=plate_id_counter)
        plate_id_counter += len(plates)
        plates_result_all.extend(plates)
        plates_by_group[group_key] = plates

        format_aggregation: Dict[Tuple[float, float, str], int] = {}
        for pl in plates:
            fmt_key = (pl.stock_w, pl.stock_l, pl.format_name or f"{pl.stock_w:.0f} × {pl.stock_l:.0f} mm")
            format_aggregation[fmt_key] = format_aggregation.get(fmt_key, 0) + 1

        sub_gross_area_m2 = sum((pl.stock_w * pl.stock_l) / 1_000_000.0 for pl in plates)
        sub_net_area_m2 = sum((it.width * it.length * it.quantity) for it in group_items) / 1_000_000.0

        for (f_w, f_l, f_name), count_sheets in format_aggregation.items():
            single_sheet_area = (f_w * f_l) / 1_000_000.0
            single_mass_kg = single_sheet_area * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0)
            tot_format_mass_kg = count_sheets * single_mass_kg
            order_items_unified.append({
                "Kategoria": "Blacha gruba (2D)",
                "Asortyment": f"Blacha #{group_key.thickness:.0f} mm",
                "Gatunek Stali": group_key.grade,
                "Wymiar Handlowy": f"{f_w:.0f} × {f_l:.0f} mm",
                "Ilość Zamawiana [szt.]": count_sheets,
                "Masa Jednostkowa [kg]": round(single_mass_kg, 1),
                "Masa Łączna [kg]": round(tot_format_mass_kg, 1),
                "Wymagany Atest": "3.1 wg PN-EN 10204",
            })

        sub_gross_mass_kg = sub_gross_area_m2 * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0)
        sub_net_mass_kg = sub_net_area_m2 * group_key.thickness * (STEEL_DENSITY_KG_M3 / 1000.0)
        sub_waste_mass_kg = max(0.0, sub_gross_mass_kg - sub_net_mass_kg)
        sub_waste_pct = ((sub_gross_area_m2 - sub_net_area_m2) / sub_gross_area_m2 * 100.0) if sub_gross_area_m2 > 0 else 0.0
        format_summary_str = ", ".join([f"{cnt}× ({f_w:.0f}×{f_l:.0f})" for (f_w, f_l, _), cnt in format_aggregation.items()])

        plate_waste_summary_2d.append({
            "Grubość [mm]": group_key.thickness,
            "Gatunek": group_key.grade,
            "Liczba arkuszy [szt.]": len(plates),
            "Dobrane formaty": format_summary_str,
            "Powierzchnia netto [m²]": round(sub_net_area_m2, 2),
            "Powierzchnia brutto [m²]": round(sub_gross_area_m2, 2),
            "Odpad [kg]": round(sub_waste_mass_kg, 1),
            "Odpad [%]": round(sub_waste_pct, 2),
        })

    tab_procure, tab_1d_view, tab_2d_view, tab_splice, tab_source = st.tabs([
        "🛒 Generator Zestawienia do Zakupu (RFQ)",
        "📏 Rozkrój Profili (1D)",
        "📐 Nesting Blach (2D)",
        "🔗 Analiza Stykowania Profili",
        "📋 Zaimportowany BOM",
    ])

    with tab_procure:
        st.markdown("### 🛒 Oficjalne Zestawienie Zakupowe (Handlowe RFQ)")
        df_order = pd.DataFrame(order_items_unified)
        if not df_order.empty:
            total_mass_kg = df_order["Masa Łączna [kg]"].sum()
            total_mass_t = total_mass_kg / 1000.0
            prof_mass_kg = sum(r["Masa Łączna [kg]"] for r in order_items_unified if "1D" in r["Kategoria"])
            plate_mass_kg = sum(r["Masa Łączna [kg]"] for r in order_items_unified if "2D" in r["Kategoria"])

            cost_prof = prof_mass_kg * price_profile_per_kg
            cost_plate = plate_mass_kg * price_plate_per_kg
            total_material_cost = cost_prof + cost_plate

            net_mass_1d = sum(p["Masa netto [kg]"] for p in profile_waste_summary_1d) if profile_waste_summary_1d else 0.0
            net_mass_2d = sum(p["Masa netto [kg]"] for p in plate_waste_summary_2d) if plate_waste_summary_2d else 0.0
            total_net_mass = net_mass_1d + net_mass_2d
            total_waste_kg = max(0.0, total_mass_kg - total_net_mass)
            scrap_revenue = total_waste_kg * scrap_price_per_kg
            net_steel_cost = total_material_cost - scrap_revenue

            c_q1, c_q2, c_q3, c_q4 = st.columns(4)
            c_q1.metric("Łączna Masa Zakupu", f"{total_mass_t:.2f} t", f"{total_mass_kg:,.0f} kg")
            c_q2.metric("Szacowany Koszt Stali", f"{total_material_cost:,.2f} PLN")
            c_q3.metric("Masa Odpadu Hutniczego", f"{total_waste_kg:,.1f} kg")
            c_q4.metric("Odzysk ze Złomu", f"{scrap_revenue:,.2f} PLN", f"netto: {net_steel_cost:,.2f} PLN")

            st.markdown("#### 📑 Specyfikacja Pozycji do Zamówienia:")
            st.dataframe(df_order.style.format({"Masa Jednostkowa [kg]": "{:,.1f} kg", "Masa Łączna [kg]": "{:,.1f} kg"}), use_container_width=True)

            workshop_1d_rows = []
            for b in bars_result_all:
                cuts_desc = " + ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts])
                workshop_1d_rows.append({"Nr Sztangi": b.bar_id, "Profil": b.profile, "Gatunek": b.grade, "Długość Handlowa [mm]": b.stock_length, "Rozkrój": cuts_desc, "Odpad [mm]": round(b.scrap_length, 1)})
            df_workshop_1d = pd.DataFrame(workshop_1d_rows)

            workshop_2d_rows = []
            for p in plates_result_all:
                items_desc = ", ".join([f"{it.mark} ({it.w:.0f}×{it.h:.0f})" for it in p.packed_items])
                workshop_2d_rows.append({"Nr Arkusza": p.plate_id, "Grubość [mm]": p.thickness, "Gatunek": p.grade, "Format": f"{p.stock_w:.0f}×{p.stock_l:.0f}", "Detale": items_desc})
            df_workshop_2d = pd.DataFrame(workshop_2d_rows)

            stats_summary_rows = [
                {"Segment": "Profile Hutnicze (1D)", "Masa [t]": round(prof_mass_kg / 1000.0, 3), "Koszt [PLN]": round(cost_prof, 2)},
                {"Segment": "Blachy Grube (2D)", "Masa [t]": round(plate_mass_kg / 1000.0, 3), "Koszt [PLN]": round(cost_plate, 2)},
                {"Segment": "SUMA", "Masa [t]": round(total_mass_t, 3), "Koszt [PLN]": round(total_material_cost, 2)},
            ]
            df_stats = pd.DataFrame(stats_summary_rows)

            excel_buffer = build_excel_export(
                procurement_df=df_order,
                cut_summary_1d=df_workshop_1d,
                cut_summary_2d=df_workshop_2d,
                profile_stats_1d=pd.DataFrame(profile_waste_summary_1d),
                plate_stats_2d=pd.DataFrame(plate_waste_summary_2d),
                splicing_df=pd.DataFrame(splicing_analysis_rows) if splicing_analysis_rows else None,
                stats_df=df_stats
            )

            st.download_button(
                label="📥 Pobierz Kompletny Pakiet Handlowy (Excel .xlsx)",
                data=excel_buffer,
                file_name="Zamowienie_Hutnicze_SteelOpt.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("Brak pozycji do zamówienia.")

    with tab_1d_view:
        st.markdown("### 📏 Wizualna Prezentacja Rozkroju Profili Hutniczych (1D)")
        if not bars_result_all:
            st.info("Brak profili 1D.")
        else:
            group_options = ["Wszystkie grupy"] + [f"{k.profile} | {k.grade}" for k in bars_by_group.keys()]
            selected_grp_str = st.selectbox("Wybierz grupę do analizy:", group_options)
            bars_filtered = bars_result_all if selected_grp_str == "Wszystkie grupy" else bars_by_group[next(k for k in bars_by_group.keys() if f"{k.profile} | {k.grade}" == selected_grp_str)]
            
            fig_1d = plot_1d_cutting_plan(bars_filtered[:25], title_suffix=selected_grp_str)
            st.pyplot(fig_1d)
            plt.close(fig_1d)

    with tab_2d_view:
        st.markdown("### 📐 Wizualna Prezentacja Rozkroju Arkuszy Blach (2D Nesting)")
        if not plates_result_all:
            st.info("Brak formatek blach.")
        else:
            plate_choices = [f"Arkusz #{p.plate_id} - #{p.thickness:.0f}mm {p.grade} ({p.stock_w:.0f}×{p.stock_l:.0f})" for p in plates_result_all]
            selected_p_idx = st.selectbox("Wybierz arkusz do podglądu:", range(len(plates_result_all)), format_func=lambda i: plate_choices[i])
            fig_single = plot_2d_plate_plan(plates_result_all[selected_p_idx])
            st.pyplot(fig_single)
            plt.close(fig_single)

    with tab_splice:
        st.markdown("### 🔗 Analiza Opłacalności Stykowania Profili (Weld Splicing)")
        if not enable_splicing:
            st.warning("⚠️ Moduł stykowania jest obecnie wyłączony. Włącz opcję **'Zezwalaj na stykowanie profili'** w panelu bocznym, aby uruchomić kalkulację złączy.")
        else:
            if splicing_analysis_rows:
                st.info("Poniższa tabela przedstawia inżynierską kalkulację kosztów wykonania styku spawanego (cięcie, składanie, spawanie wielościegowe, badania UT) oraz minimalny próg opłacalności (BEP) dla poszczególnych profili.")
                st.dataframe(pd.DataFrame(splicing_analysis_rows), use_container_width=True)
            else:
                st.info("Brak profili do analizy stykowania.")

    with tab_source:
        st.markdown("### 📋 Zaimportowany BOM i Weryfikacja")
        st.dataframe(df_clean, use_container_width=True)
else:
    st.info("👈 Wgraj plik z zestawieniem materiałowym (BOM) lub kliknij **'🚀 Załaduj Testowy BOM'**, aby uruchomić optymalizację.")
