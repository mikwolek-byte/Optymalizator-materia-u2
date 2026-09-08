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
    page_title="SteelOpt - Profesjonalny Optymalizator Hutniczy & RFQ",
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

def get_unit_weight_1d(profile_str: str) -> float:
    """Oblicza lub pobiera masę 1 mb profilu stalowego [kg/m] wg norm europejskich."""
    raw = str(profile_str).upper().replace(" ", "").replace("×", "X").replace("*", "X").replace(",", ".")

    m_he = re.match(r"^HE(\d+)([ABM])$", raw)
    if m_he:
        size, variant = m_he.groups()
        raw = f"HE{variant}{size}"

    clean_prof = re.sub(r'[^A-Z0-9]', '', raw)
    for key, weight in EURO_PROFILE_WEIGHTS.items():
        if key == clean_prof or clean_prof.startswith(key):
            return weight

    # Kątowniki równoramienne i nierównoramienne (L)
    m_angle = re.search(r'(?:L|KAT|KĄT|KATOWNIK|KĄTOWNIK)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_angle and any(prefix in raw for prefix in ["L", "KAT", "KĄT"]):
        dim1 = float(m_angle.group(1))
        dim2 = float(m_angle.group(2))
        dim3 = float(m_angle.group(3)) if m_angle.group(3) else None
        if dim3 is not None:
            a, b, t = dim1, dim2, dim3
        else:
            a, b, t = dim1, dim1, dim2
        area_mm2 = (a + b - t) * t * 1.012
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Profile zamknięte prostokątne / kwadratowe (RHS, SHS, CFRHS, RK, RP)
    m_rect = re.search(r'(?:CF)?(?:RHS|SHS|RK|RP|PR|PROFIL)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_rect and any(prefix in raw for prefix in ["RHS", "SHS", "RK", "RP", "PROFIL", "CF"]):
        h = float(m_rect.group(1))
        b = float(m_rect.group(2))
        t = float(m_rect.group(3)) if m_rect.group(3) else b
        if m_rect.group(3) is None:
            b = h
        area_mm2 = 2.0 * t * (h + b) - (4.0 + 2.575) * (t ** 2)
        if area_mm2 <= 0:
            area_mm2 = 2.0 * (h + b) * t - 4.0 * (t ** 2)
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Rury okrągłe CHS
    m_pipe = re.search(r'(?:CHS|RO|ROHR|RURA|FI|Ø)\s*(\d+(?:\.\d+)?)[X/](\d+(?:\.\d+)?)', raw)
    if m_pipe:
        d = float(m_pipe.group(1))
        t = float(m_pipe.group(2))
        area_mm2 = math.pi * (d - t) * t
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Pręty okrągłe pełne
    m_bar = re.search(r'^(?:D|FI|Ø)(\d+(?:\.\d+)?)$', raw)
    if m_bar:
        d = float(m_bar.group(1))
        area_mm2 = math.pi * ((d / 2.0) ** 2)
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Płaskowniki (FL, PD)
    m_flat = re.search(r'^(?:FL|PD|PL|BL)(\d+(?:\.\d+)?)[X*](\d+(?:\.\d+)?)$', raw)
    if m_flat:
        t = float(m_flat.group(1))
        w = float(m_flat.group(2))
        return round(t * w * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

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
    g = re.sub(r'\s+', '', g)
    g = g.replace("–", "-").replace("—", "-")

    if g in ["S235", "S235J", "S235JR", "S235J0", "S235J2"]:
        return "S235JR"
    if g in ["S355", "S355J", "S355JR", "S355J0", "S355J2", "S355J2+N", "S355J2N", "S355K2"]:
        return "S355J2+N"
    if g in ["S355H", "S355J2H", "S355NH"]:
        return "S355J2H"
    if g in ["S275", "S275JR", "S275J2"]:
        return "S275JR"

    return g

def normalize_profile(raw_profile: str) -> str:
    if not raw_profile or pd.isna(raw_profile) or str(raw_profile).strip().lower() in ['nan', 'none', '']:
        return "UNKNOWN"

    p = str(raw_profile).strip().upper()
    p = p.replace(" ", "").replace("×", "X").replace("*", "X").replace(",", ".")

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
        grouped.setdefault(key, []).append(
            Item1D(
                mark=item.mark,
                length=float(item.length),
                profile=norm_prf,
                grade=norm_grd,
                quantity=int(item.quantity),
            )
        )
    return grouped

def group_2d_plates_by_material(items: List[PlateItem]) -> Dict[PlateGroupKey, List[PlateItem]]:
    grouped: Dict[PlateGroupKey, List[PlateItem]] = {}
    for item in items:
        norm_grd = normalize_grade(item.grade)
        norm_thk = round(float(item.thickness), 2)
        key = PlateGroupKey(grade=norm_grd, thickness=norm_thk)
        grouped.setdefault(key, []).append(
            PlateItem(
                mark=item.mark,
                grade=norm_grd,
                thickness=norm_thk,
                width=float(item.width),
                length=float(item.length),
                quantity=int(item.quantity),
            )
        )
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
    """Próbuje upakować jak najwięcej części na pojedynczym arkuszu o zadanym formacie."""
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

            # Sprawdzenie istniejących półek nestingowych
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

            # Otwarcie nowej półki na arkuszu
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

    total_sheet_area = fmt.width * fmt.length
    plate.scrap_area = max(0.0, total_sheet_area - plate.used_area)
    return plate, unplaced_parts

def simulate_uniform_format_nesting(
    group_key: PlateGroupKey,
    parts_list: List[Tuple[str, float, float]],
    fmt: PlateFormat,
    kerf_spacing: float,
    edge_margin: float,
    start_plate_id: int,
) -> Optional[List[StockPlate]]:
    """Symuluje rozkrój całej grupy przy użyciu wyłącznie jednego zadanego formatu arkusza."""
    effective_w = fmt.width - 2.0 * edge_margin
    effective_l = fmt.length - 2.0 * edge_margin

    # Weryfikacja wykonalności: żaden detal nie może przekraczać wymiaru efektywnego
    for _, pw, pl in parts_list:
        fits_normal = (pw <= effective_w and pl <= effective_l)
        fits_rotated = (pl <= effective_w and pw <= effective_l)
        if not (fits_normal or fits_rotated):
            return None

    plates: List[StockPlate] = []
    remaining = list(parts_list)
    curr_id = start_plate_id

    while remaining:
        plate, unplaced = pack_single_sheet_shelf(
            plate_id=curr_id,
            grade=group_key.grade,
            thickness=group_key.thickness,
            fmt=fmt,
            parts=remaining,
            kerf_spacing=kerf_spacing,
            edge_margin=edge_margin,
        )
        if not plate.packed_items:
            return None
        plates.append(plate)
        curr_id += 1
        remaining = unplaced

    return plates

def simulate_adaptive_mixed_nesting(
    group_key: PlateGroupKey,
    parts_list: List[Tuple[str, float, float]],
    candidate_formats: List[PlateFormat],
    kerf_spacing: float,
    edge_margin: float,
    start_plate_id: int,
) -> List[StockPlate]:
    """Dynamicznie dobiera najlepszy format arkusza na każdym kroku rozkroju."""
    plates: List[StockPlate] = []
    remaining = list(parts_list)
    curr_id = start_plate_id

    while remaining:
        best_plate: Optional[StockPlate] = None
        best_unplaced: List[Tuple[str, float, float]] = []
        best_score = -float("inf")

        # Sprawdzenie czy wszystkie pozostałe detale mieszczą się na którymś arkuszu (faza końcowa)
        candidate_results = []
        for fmt in candidate_formats:
            p, unplaced = pack_single_sheet_shelf(
                plate_id=curr_id,
                grade=group_key.grade,
                thickness=group_key.thickness,
                fmt=fmt,
                parts=remaining,
                kerf_spacing=kerf_spacing,
                edge_margin=edge_margin,
            )
            if p.packed_items:
                candidate_results.append((fmt, p, unplaced))

        if not candidate_results:
            # Sytuacja awaryjna: detal ponadgabarytowy
            largest_fmt = max(candidate_formats, key=lambda f: f.width * f.length)
            mark, pw, pl = remaining.pop(0)
            custom_plate = StockPlate(
                plate_id=curr_id,
                grade=group_key.grade,
                thickness=group_key.thickness,
                stock_w=max(pw + 2 * edge_margin, largest_fmt.width),
                stock_l=max(pl + 2 * edge_margin, largest_fmt.length),
                packed_items=[PackedRect(mark=mark, x=edge_margin, y=edge_margin, w=pw, h=pl)],
                used_area=pw * pl,
                scrap_area=0.0,
                format_name="Arkusz Niestandardowy (Ponadgabaryt)",
            )
            custom_plate.scrap_area = max(0.0, (custom_plate.stock_w * custom_plate.stock_l) - custom_plate.used_area)
            plates.append(custom_plate)
            curr_id += 1
            continue

        # Kryterium wyboru:
        # 1. Jeśli arkusz mieści WSZYSTKIE pozostałe detale, wybieramy ten o najmniejszym odpadzie (najmniejszy dopasowany arkusz).
        finishing_candidates = [c for c in candidate_results if len(c[2]) == 0]
        if finishing_candidates:
            # Minimalizacja odpadu bezwzględnego [m²] na ostatnim arkuszu
            finishing_candidates.sort(key=lambda c: c[1].scrap_area)
            chosen_fmt, chosen_plate, chosen_unplaced = finishing_candidates[0]
        else:
            # 2. Jeśli detale nie mieszczą się w całości, wybieramy arkusz o najwyższym wskaźniku upakowania (yield)
            candidate_results.sort(
                key=lambda c: (c[1].used_area / (c[0].width * c[0].length)),
                reverse=True
            )
            chosen_fmt, chosen_plate, chosen_unplaced = candidate_results[0]

        plates.append(chosen_plate)
        curr_id += 1
        remaining = chosen_unplaced

    return plates

def optimize_2d_single_group(
    group_key: PlateGroupKey,
    items: List[PlateItem],
    available_formats: List[PlateFormat],
    kerf_spacing: float = 12.0,
    edge_margin: float = 20.0,
    allow_mixed_formats: bool = True,
    start_plate_id: int = 1,
) -> List[StockPlate]:
    """Wielowariantowa symulacja i selekcja optymalnego planu rozkroju 2D pod kątem min. odpadu."""
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

    # 1. Przetestowanie każdego wybranego formatu jednorodnego
    for fmt in available_formats:
        run_res = simulate_uniform_format_nesting(
            group_key=group_key,
            parts_list=parts,
            fmt=fmt,
            kerf_spacing=kerf_spacing,
            edge_margin=edge_margin,
            start_plate_id=start_plate_id,
        )
        if run_res is not None:
            candidate_runs.append(run_res)

    # 2. Przetestowanie strategii adaptacyjnej (miksowanie formatów)
    if allow_mixed_formats and len(available_formats) > 1:
        mixed_run = simulate_adaptive_mixed_nesting(
            group_key=group_key,
            parts_list=parts,
            candidate_formats=available_formats,
            kerf_spacing=kerf_spacing,
            edge_margin=edge_margin,
            start_plate_id=start_plate_id,
        )
        if mixed_run:
            candidate_runs.append(mixed_run)

    if not candidate_runs:
        # Fallback na największy dostępny format
        largest_fmt = max(available_formats, key=lambda f: f.width * f.length)
        return simulate_adaptive_mixed_nesting(
            group_key=group_key,
            parts_list=parts,
            candidate_formats=[largest_fmt],
            kerf_spacing=kerf_spacing,
            edge_margin=edge_margin,
            start_plate_id=start_plate_id,
        )

    # Wybór planu o bezwzględnie najmniejszym odpadzie powierzchniowym (min scrap_area)
    def evaluate_run(run: List[StockPlate]) -> Tuple[float, float, int]:
        total_scrap = sum(p.scrap_area for p in run)
        total_gross = sum(p.stock_w * p.stock_l for p in run)
        yield_pct = (sum(p.used_area for p in run) / total_gross * 100.0) if total_gross > 0 else 0.0
        return (total_scrap, -yield_pct, len(run))

    candidate_runs.sort(key=evaluate_run)
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
                if "profil" in row_str and any(k in row_str for k in ["długość", "length", "materiał", "pozycja"]):
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

    raise ValueError("Nie udało się odczytać pliku. Sprawdź format skoroszytu Excel lub CSV.")

def map_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c_clean = str(col).lower().strip()
        if any(k in c_clean for k in ['pozycja', 'position', 'pos', 'mark', 'nr elementu', 'nr części']) and 'mark' not in col_map.values():
            col_map[col] = 'mark'
        elif any(k in c_clean for k in ['profil', 'profile', 'przekrój', 'section', 'asortyment']) and 'profile' not in col_map.values():
            col_map[col] = 'profile'
        elif any(k in c_clean for k in ['materiał', 'material', 'gatunek', 'grade', 'stal']) and 'grade' not in col_map.values():
            col_map[col] = 'grade'
        elif any(k in c_clean for k in ['ilość', 'ilosc', 'quantity', 'qty', 'szt', 'liczba']) and 'qty' not in col_map.values():
            col_map[col] = 'qty'
        elif any(k in c_clean for k in ['długość', 'dlugosc', 'length', 'l [mm]', 'len']) and not any(k in c_clean for k in ['całk', 'total', 'suma']) and 'length' not in col_map.values():
            col_map[col] = 'length'
        elif any(k in c_clean for k in ['szerokość', 'szerokosc', 'width', 'b [mm]']) and not any(k in c_clean for k in ['całk', 'total', 'suma']) and 'width' not in col_map.values():
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

            # Autodetekcja wymiarów blach z nazwy profilu jeśli kolumny były puste (np. PL10x200, BL12*300)
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
                ax.text(curr_x + cut_len / 2, idx, f"{mark}\n{cut_len:.0f}",
                        va="center", ha="center", color="white", fontsize=7.5, fontweight="bold")
            curr_x += cut_len
            if c_idx < len(b.cuts) - 1:
                ax.barh(idx, 4.5, left=curr_x, color="#0F172A", height=0.55)
                curr_x += 4.5

        rem_scrap = b.stock_length - curr_x
        if rem_scrap > 0:
            ax.barh(idx, rem_scrap, left=curr_x, color="#F1F5F9", edgecolor="#94A3B8", height=0.55)
            if rem_scrap > (b.stock_length * 0.06):
                ax.text(curr_x + rem_scrap / 2, idx, f"Odpad {rem_scrap:.0f}mm",
                        va="center", ha="center", color="#475569", fontsize=7.5)

    ax.set_xlabel("Długość handlowa sztangi [mm]", fontsize=9.5, fontweight="bold")
    title = f"Rozkrój Sztang 1D {title_suffix} (Liczba sztang: {len(bars)})"
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_2d_plate_plan(plate: StockPlate) -> plt.Figure:
    """Rysuje techniczny rzut 2D pojedynczego arkusza blachy z naniesionymi formatkami i marginesami CNC."""
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=120)

    # Arkusz bazowy: Oś X = Długość L (stock_l), Oś Y = Szerokość W (stock_w)
    sheet_rect = patches.Rectangle(
        (0, 0), plate.stock_l, plate.stock_w,
        linewidth=2.0, edgecolor="#1E293B", facecolor="#F8FAFC", linestyle="--"
    )
    ax.add_patch(sheet_rect)

    colors = [
        "#2563EB", "#0D9488", "#EA580C", "#9333EA", "#16A34A",
        "#4F46E5", "#D97706", "#059669", "#DC2626", "#0891B2"
    ]

    for idx, item in enumerate(plate.packed_items):
        color = colors[idx % len(colors)]
        # item.y = współrzędna wzdłuż długości L (oś X)
        # item.x = współrzędna wzdłuż szerokości W (oś Y)
        # item.h = wymiar wzdłuż L (szerokość na osi X)
        # item.w = wymiar wzdłuż W (wysokość na osi Y)
        part_rect = patches.Rectangle(
            (item.y, item.x), item.h, item.w,
            linewidth=1.2, edgecolor="#0F172A", facecolor=color, alpha=0.88
        )
        ax.add_patch(part_rect)

        # Czytelna etykieta detalu
        if item.h > (plate.stock_l * 0.035) and item.w > (plate.stock_w * 0.04):
            font_sz = 7.5 if (item.h > 400 and item.w > 200) else 6.0
            label_text = f"{item.mark}\n{item.h:.0f}×{item.w:.0f}"
            ax.text(
                item.y + item.h / 2.0, item.x + item.w / 2.0,
                label_text,
                ha="center", va="center", color="white",
                fontsize=font_sz, fontweight="bold"
            )

    eff_pct = (plate.used_area / (plate.stock_w * plate.stock_l) * 100.0) if (plate.stock_w * plate.stock_l) > 0 else 0.0
    title_str = (
        f"Arkusz #{plate.plate_id} ({plate.format_name or f'{plate.stock_w:.0f}×{plate.stock_l:.0f}'}) | "
        f"Grubość: #{plate.thickness:.0f} mm | Gatunek: {plate.grade}\n"
        f"Format: {plate.stock_w:.0f} × {plate.stock_l:.0f} mm | Detali: {len(plate.packed_items)} szt. | "
        f"Efektywność nestingu: {eff_pct:.1f}% (Odpad: {(100.0 - eff_pct):.1f}%)"
    )
    ax.set_title(title_str, fontsize=10.0, fontweight="bold", pad=12)
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
    procurement_df: pd.DataFrame,
    cut_summary_1d: pd.DataFrame,
    cut_summary_2d: pd.DataFrame,
    stats_df: pd.DataFrame,
    profile_stats_1d: Optional[pd.DataFrame] = None,
    plate_stats_2d: Optional[pd.DataFrame] = None,
) -> bytes:
    """Buduje wielozakładkowy arkusz Excel (.xlsx) ze specyfikacją zamówieniową i kartami warsztatowymi."""
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
        if stats_df is not None and not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="6. Podsumowanie Kosztów", index=False)
    return buffer.getvalue()

st.title("🏗️ SteelOpt: Optymalizator Rozkroju i Generator RFQ")
st.caption("Precyzyjne planowanie cięcia hutniczego | Izolacja gatunków stali | Generator Zamówień i Zapytania Ofertowego")

if "bom_data" not in st.session_state:
    st.session_state["bom_data"] = None

with st.sidebar:
    st.header("⚙️ Parametry Technologiczne")

    st.subheader("Parametry Rozkroju 1D (Sztangi)")
    kerf_1d = st.number_input("Szerokość rzazu piły [mm]", min_value=1.0, max_value=12.0, value=4.5, step=0.5)
    trim_1d = st.number_input("Naddatek obcięcia końcówki [mm]", min_value=0.0, max_value=150.0, value=40.0, step=5.0)
    stock_options_1d = st.multiselect(
        "Dostępne sztangi handlowe [mm]:",
        options=[6000.0, 10000.0, 12000.0, 12100.0, 14000.0, 15000.0, 15100.0, 18000.0],
        default=[6000.0, 12100.0, 15100.0],
        help="Standardowe długości hutnicze: 6.0m (dla mniejszych profili/ceowników/kątowników), 12.1m oraz 15.1m (belki główne)."
    )
    if not stock_options_1d:
        stock_options_1d = [12100.0]

    st.divider()
    st.subheader("Parametry Rozkroju 2D (Arkusze)")
    kerf_2d = st.number_input("Odstęp termiczny palnika [mm]", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
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
        "Dostępne formaty arkuszy handlowych:",
        options=list(PLATE_FORMAT_CATALOG.keys()),
        default=["1500 × 3000 mm", "1500 × 6000 mm", "2000 × 6000 mm", "2000 × 12000 mm"],
        help="Zaznacz wszystkie formaty, jakimi dysponuje lub które może dostarczyć huta/dystrybutor. Algorytm dokona doboru optymalnego."
    )
    if not selected_format_labels:
        selected_format_labels = ["2000 × 6000 mm"]

    allow_mixed_formats = st.checkbox(
        "Zezwalaj na miksowanie formatów w grupie (optymalizacja hybrydowa)",
        value=True,
        help="Pozwala dobrać duży arkusz dla głównej partii i mniejszy na domiar, minimalizując zakup pustych powierzchni."
    )
    
    available_plate_formats = [
        PlateFormat(name=lbl, width=PLATE_FORMAT_CATALOG[lbl][0], length=PLATE_FORMAT_CATALOG[lbl][1])
        for lbl in selected_format_labels
    ]

    st.divider()
    st.subheader("💰 Wycena Szacunkowa (Koszty)")
    price_profile_per_kg = st.number_input("Cena stali kształtowej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.60, step=0.10)
    price_plate_per_kg = st.number_input("Cena blachy grubej [PLN/kg]", min_value=1.0, max_value=25.0, value=4.90, step=0.10)
    scrap_price_per_kg = st.number_input("Wartość odkupu złomu [PLN/kg]", min_value=0.2, max_value=5.0, value=1.20, step=0.10)

st.subheader("📥 1. Dane Wejściowe: Zestawienie Materiałowe (BOM)")
col_upload, col_sample, col_clear = st.columns([3, 1.2, 0.8])

with col_upload:
    uploaded_file = st.file_uploader(
        "Wczytaj plik Excel (.xlsx, .xls) lub CSV z CAD/BIM:",
        type=["xlsx", "xls", "csv"],
        help="Obsługuje pliki Tekla Structures, Advance Steel, Bocad oraz CSV."
    )

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
        r"IPE|HEA|HEB|HEM|UNP|UPE|RHS|SHS|CHS|RO|ROHR|RK|RP|^(?:L|KAT|KĄT|D|FI)",
        case=False,
        regex=True
    )

    is_plate_condition = (
        df_clean["profile"].str.contains(r"PL|BLACHA|PLATE|PŁYT|FORMATKA|#", case=False, regex=True)
        | ((df_clean["width"] > 0) & (df_clean["thick"] > 0) & (~is_structural_1d))
    )

    df_1d = df_clean[~is_plate_condition].copy()
    df_2d = df_clean[is_plate_condition].copy()

    raw_items_1d = [
        Item1D(
            mark=str(r["mark"]),
            length=float(r["length"]),
            profile=str(r["profile"]),
            grade=str(r["grade"]),
            quantity=int(r["qty"]),
        )
        for _, r in df_1d.iterrows()
    ]

    raw_items_2d = [
        PlateItem(
            mark=str(r["mark"]),
            grade=str(r["grade"]),
            thickness=float(r["thick"]) if float(r["thick"]) > 0 else 10.0,
            width=float(r["width"]) if float(r["width"]) > 0 else 200.0,
            length=float(r["length"]),
            quantity=int(r["qty"]),
        )
        for _, r in df_2d.iterrows()
    ]

    grouped_1d_dict = group_1d_items_by_material(raw_items_1d)
    grouped_2d_dict = group_2d_plates_by_material(raw_items_2d)

    bars_result_all: List[StockBar] = []
    bars_by_group: Dict[ProfileGroupKey, List[StockBar]] = {}
    order_items_unified: List[Dict] = []
    profile_waste_summary_1d: List[Dict] = []

    bar_id_counter = 1
    for group_key, group_items in grouped_1d_dict.items():
        bars = optimize_1d_single_group(
            group_key=group_key,
            items=group_items,
            available_stocks=stock_options_1d,
            kerf=kerf_1d,
            trim_cut=trim_1d,
            start_bar_id=bar_id_counter,
        )
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
            "Masa netto [kg]": round(sub_netto_mass, 1),
            "Masa brutto [kg]": round(sub_purchased_mass, 1),
            "Odpad [kg]": round(sub_scrap_mass_kg, 1),
            "Odpad [%]": round(sub_scrap_pct, 2),
        })

    plates_result_all: List[StockPlate] = []
    plates_by_group: Dict[PlateGroupKey, List[StockPlate]] = {}
    plate_waste_summary_2d: List[Dict] = []

    plate_id_counter = 1

    for group_key, group_items in grouped_2d_dict.items():
        plates = optimize_2d_single_group(
            group_key=group_key,
            items=group_items,
            available_formats=available_plate_formats,
            kerf_spacing=kerf_2d,
            edge_margin=margin_2d,
            allow_mixed_formats=allow_mixed_formats,
            start_plate_id=plate_id_counter,
        )
        plate_id_counter += len(plates)
        plates_result_all.extend(plates)
        plates_by_group[group_key] = plates

        # Agregacja zamówienia z podziałem na dobrane formaty handlowe
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
            "Masa netto [kg]": round(sub_net_mass_kg, 1),
            "Masa brutto [kg]": round(sub_gross_mass_kg, 1),
            "Odpad [kg]": round(sub_waste_mass_kg, 1),
            "Odpad [%]": round(sub_waste_pct, 2),
        })

    tab_procure, tab_1d_view, tab_2d_view, tab_source = st.tabs([
        "🛒 Generator Zestawienia do Zakupu (RFQ)",
        "📏 Prezentacja Rozkroju Profili (1D)",
        "📐 Prezentacja Rozkroju Blach (2D)",
        "📋 Zaimportowany BOM & Grupowanie",
    ])

    with tab_procure:
        st.markdown("### 🛒 Oficjalne Zestawienie Zakupowe (Handlowe RFQ)")
        st.caption("Gotowa specyfikacja dla hut i dystrybutorów stali (m.in. Thyssenkrupp, Bowim, Konsorcjum Stali, ArcelorMittal).")

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
            c_q3.metric("Masa Odpadu Hutniczego", f"{total_waste_kg:,.1f} kg", f"{(total_waste_kg/total_mass_kg*100.0):.1f}% zakupu" if total_mass_kg > 0 else "")
            c_q4.metric("Odzysk ze Złomu", f"{scrap_revenue:,.2f} PLN", f"netto: {net_steel_cost:,.2f} PLN")

            st.markdown("#### 📑 Specyfikacja Pozycji do Zamówienia:")
            st.dataframe(
                df_order.style.format({
                    "Masa Jednostkowa [kg]": "{:,.1f} kg",
                    "Masa Łączna [kg]": "{:,.1f} kg",
                }),
                use_container_width=True
            )

            # Generator gotowej treści e-maila RFQ
            st.markdown("#### ✉️ Generator E-maila Zapytania Ofertowego do Handlowca:")
            email_lines = [
                "Dzień dobry,",
                "",
                "Zwracam się z uprzejmą prośbą o przedstawienie oferty cenowej oraz terminu realizacji na poniższy asortyment hutniczy:",
                "Wymagane atesty hutnicze: 3.1 wg PN-EN 10204 dla wszystkich pozycji.",
                "",
                "Zestawienie materiałowe:",
                "-----------------------------------------------------------------------------------------",
                f"{'Lp.':<4} | {'Asortyment':<16} | {'Gatunek':<12} | {'Wymiar handlowy':<18} | {'Ilość':<8} | {'Waga [kg]'}",
                "-----------------------------------------------------------------------------------------",
            ]
            for idx, r in enumerate(order_items_unified, 1):
                email_lines.append(
                    f"{idx:<4} | {r['Asortyment']:<16} | {r['Gatunek Stali']:<12} | {r['Wymiar Handlowy']:<18} | {str(r['Ilość Zamawiana [szt.]']) + ' szt.':<8} | {r['Masa Łączna [kg]']:>8.1f} kg"
                )
            email_lines.extend([
                "-----------------------------------------------------------------------------------------",
                f"ŁĄCZNA WAGA ZAMÓWIENIA: {total_mass_t:.2f} ton ({total_mass_kg:,.1f} kg)",
                "",
                "Proszę o uwzględnienie kosztu transportu na plac budowy / warsztat wytwórni konstrukcji.",
                "Z poważaniem,",
                "Dział Zaopatrzenia i Zakupów",
            ])
            email_text = "\n".join(email_lines)

            st.text_area("Gotowy tekst do wklejenia w programie pocztowym (Outlook / Thunderbird):", value=email_text, height=220)

            # Eksport do Excela
            workshop_1d_rows = []
            for b in bars_result_all:
                cuts_desc = " + ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts])
                workshop_1d_rows.append({
                    "Nr Sztangi": b.bar_id,
                    "Profil": b.profile,
                    "Gatunek": b.grade,
                    "Długość Handlowa [mm]": b.stock_length,
                    "Kolejność Rozkroju": cuts_desc,
                    "Długość Netto [mm]": sum(c[1] for c in b.cuts),
                    "Odpad [mm]": round(b.scrap_length, 1),
                    "Wykorzystanie [%]": round(((b.stock_length - b.scrap_length) / b.stock_length) * 100.0, 1),
                    "Ponadgabaryt": "TAK" if b.is_oversized else "NIE",
                })
            df_workshop_1d = pd.DataFrame(workshop_1d_rows)

            workshop_2d_rows = []
            for p in plates_result_all:
                items_desc = ", ".join([f"{it.mark} ({it.w:.0f}×{it.h:.0f})" for it in p.packed_items])
                workshop_2d_rows.append({
                    "Nr Arkusza": p.plate_id,
                    "Grubość [mm]": p.thickness,
                    "Gatunek": p.grade,
                    "Format Arkusza [mm]": f"{p.stock_w:.0f} × {p.stock_l:.0f}",
                    "Ilość Formatek": len(p.packed_items),
                    "Wykorzystanie [%]": round((p.used_area / (p.stock_w * p.stock_l)) * 100.0, 1),
                    "Rozmieszczone Pozycje": items_desc,
                })
            df_workshop_2d = pd.DataFrame(workshop_2d_rows)

            stats_summary_rows = [
                {"Segment": "Profile Hutnicze (1D)", "Masa zamówienia [t]": round(prof_mass_kg / 1000.0, 3), "Koszt [PLN]": round(cost_prof, 2)},
                {"Segment": "Blachy Grube (2D)", "Masa zamówienia [t]": round(plate_mass_kg / 1000.0, 3), "Koszt [PLN]": round(cost_plate, 2)},
                {"Segment": "SUMA ZAKUPU", "Masa zamówienia [t]": round(total_mass_t, 3), "Koszt [PLN]": round(total_material_cost, 2)},
            ]
            df_stats = pd.DataFrame(stats_summary_rows)

            excel_buffer = build_excel_export(
                procurement_df=df_order,
                cut_summary_1d=df_workshop_1d,
                cut_summary_2d=df_workshop_2d,
                stats_df=df_stats,
                profile_stats_1d=pd.DataFrame(profile_waste_summary_1d) if profile_waste_summary_1d else None,
                plate_stats_2d=pd.DataFrame(plate_waste_summary_2d) if plate_waste_summary_2d else None,
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
            st.info("Brak profili 1D w zaimportowanym zestawie.")
        else:
            group_options = ["Wszystkie grupy profili"] + [f"{k.profile} | {k.grade}" for k in bars_by_group.keys()]
            selected_grp_str = st.selectbox("Wybierz grupę do analizy graficznej:", group_options)

            if selected_grp_str == "Wszystkie grupy profili":
                bars_filtered = bars_result_all
                suffix = "- Wszystkie"
            else:
                prof_sel, grd_sel = [s.strip() for s in selected_grp_str.split("|")]
                matching_key = next((k for k in bars_by_group.keys() if k.profile == prof_sel and k.grade == grd_sel), None)
                bars_filtered = bars_by_group[matching_key] if matching_key else []
                suffix = f"- {selected_grp_str}"

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            tot_filt_len = sum(b.stock_length for b in bars_filtered) / 1000.0
            tot_used_len = sum(sum(c[1] for c in b.cuts) for b in bars_filtered) / 1000.0
            eff_1d = (tot_used_len / tot_filt_len * 100.0) if tot_filt_len > 0 else 0.0

            col_stat1.metric("Liczba sztang w widoku", f"{len(bars_filtered)} szt.")
            col_stat2.metric("Łączna długość handlowa", f"{tot_filt_len:,.1f} mb")
            col_stat3.metric("Wykorzystanie materiału", f"{eff_1d:.1f}%", f"Odpad: {(100.0 - eff_1d):.1f}%")

            st.markdown("#### Wykres rozkroju sztang:")
            # Paginacja lub ograniczenie widoku dla zachowania czytelności
            bars_to_draw = bars_filtered[:20]
            fig_1d = plot_1d_cutting_plan(bars_to_draw, title_suffix=suffix)
            st.pyplot(fig_1d)
            plt.close(fig_1d)

            if len(bars_filtered) > 20:
                st.caption(f"ℹ️ Na wykresie wyświetlono pierwsze 20 z {len(bars_filtered)} sztang. Pełna lista znajduje się w tabeli poniżej.")

            st.markdown("#### 📋 Szczegółowe Karty Cięcia dla Operatora Piły:")
            cut_cards_data = []
            for b in bars_filtered:
                cut_cards_data.append({
                    "Nr Sztangi": b.bar_id,
                    "Profil": b.profile,
                    "Gatunek": b.grade,
                    "Długość Handlowa [mm]": b.stock_length,
                    "Liczba cięć": len(b.cuts),
                    "Rozkrój [Poz (długość)]": " | ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts]),
                    "Odpad resztkowy [mm]": round(b.scrap_length, 1),
                    "Efektywność [%]": f"{((b.stock_length - b.scrap_length) / b.stock_length * 100.0):.1f}%",
                    "Uwagi": "⚠️ Ponadgabaryt!" if b.is_oversized else "OK",
                })
            st.dataframe(pd.DataFrame(cut_cards_data), use_container_width=True)

    with tab_2d_view:
        st.markdown("### 📐 Wizualna Prezentacja Rozkroju Arkuszy Blach (2D Nesting)")
        if not plates_result_all:
            st.info("Brak formatek blach w zaimportowanym zestawie.")
        else:
            c_s1, c_s2, c_s3 = st.columns(3)
            tot_sheets = len(plates_result_all)
            tot_gross_m2 = sum((p.stock_w * p.stock_l) / 1_000_000.0 for p in plates_result_all)
            tot_used_m2 = sum(p.used_area / 1_000_000.0 for p in plates_result_all)
            avg_eff = (tot_used_m2 / tot_gross_m2 * 100.0) if tot_gross_m2 > 0 else 0.0
            c_s1.metric("Liczba doborowych arkuszy", f"{tot_sheets} szt.")
            c_s2.metric("Powierzchnia handlowa brutto", f"{tot_gross_m2:.2f} m²")
            c_s3.metric("Wykorzystanie arkuszy (Yield)", f"{avg_eff:.1f}%", f"Odpad: {(100.0 - avg_eff):.1f}%")

            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                plate_choices = [f"Arkusz #{p.plate_id} - #{p.thickness:.0f}mm {p.grade} ({p.stock_w:.0f}×{p.stock_l:.0f} mm)" for p in plates_result_all]
                selected_plate_idx = st.selectbox("Wybierz arkusz blachy do podglądu CNC:", range(len(plates_result_all)), format_func=lambda i: plate_choices[i])
            with col_sel2:
                show_all = st.checkbox("Pokaż wszystkie arkusze jeden pod drugim", value=False)

            if show_all:
                max_show = min(len(plates_result_all), 20)
                for p in plates_result_all[:max_show]:
                    fig_p = plot_2d_plate_plan(p)
                    st.pyplot(fig_p)
                    plt.close(fig_p)
                if len(plates_result_all) > max_show:
                    st.caption(f"ℹ️ Wyświetlono pierwsze {max_show} z {len(plates_result_all)} arkuszy. Pojedyncze arkusze możesz przeglądać wybierając je z listy.")
            else:
                curr_plate = plates_result_all[selected_plate_idx]
                fig_single = plot_2d_plate_plan(curr_plate)
                st.pyplot(fig_single)
                plt.close(fig_single)

                st.markdown(f"#### 📐 Współrzędne Wycinania CNC dla Arkusza #{curr_plate.plate_id}:")
                coords_data = []
                for idx, item in enumerate(curr_plate.packed_items, 1):
                    coords_data.append({
                        "Nr Detalu": idx,
                        "Pozycja (Mark)": item.mark,
                        "Współrzędna X [mm]": round(item.y, 1),
                        "Współrzędna Y [mm]": round(item.x, 1),
                        "Długość L [mm]": round(item.h, 1),
                        "Szerokość B [mm]": round(item.w, 1),
                        "Pole detalu [m²]": round((item.w * item.h) / 1_000_000.0, 3),
                    })
                st.dataframe(pd.DataFrame(coords_data), use_container_width=True)

    with tab_source:
        st.markdown("### 📋 Zaimportowany BOM i Weryfikacja Separacji Gatunkowej")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Wszystkie wiersze", len(df_clean))
        col_m2.metric("Suma sztuk detali", int(df_clean["qty"].sum()))
        col_m3.metric("Grupy 1D (Profil + Gatunek)", len(grouped_1d_dict))
        col_m4.metric("Grupy 2D (Grubość + Gatunek)", len(grouped_2d_dict))

        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.markdown("#### Grupy profili 1D:")
            df_g1 = pd.DataFrame([{"Gatunek": k.grade, "Profil": k.profile, "Pozycji": len(v), "Łącznie sztuk": sum(x.quantity for x in v)} for k, v in grouped_1d_dict.items()])
            st.dataframe(df_g1, use_container_width=True)
        with c_g2:
            st.markdown("#### Grupy blach 2D:")
            df_g2 = pd.DataFrame([{"Gatunek": k.grade, "Grubość [mm]": k.thickness, "Pozycji": len(v), "Łącznie sztuk": sum(x.quantity for x in v)} for k, v in grouped_2d_dict.items()])
            st.dataframe(df_g2, use_container_width=True)

        st.markdown("#### Podgląd przetworzonych wierszy wejściowych:")
        st.dataframe(df_clean, use_container_width=True)
else:
    st.info("👈 Wgraj plik z zestawieniem materiałowym (BOM) lub kliknij **'🚀 Załaduj Testowy BOM'**, aby uruchomić optymalizację i wygenerować zapytanie ofertowe.")
