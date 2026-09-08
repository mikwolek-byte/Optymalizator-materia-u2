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

# Masy jednostkowe
EURO_PROFILE_WEIGHTS: Dict[str, float] = {
    "IPE80": 6.0, "IPE100": 8.1, "IPE120": 10.4, "IPE140": 12.9, "IPE160": 15.8,
    "IPE180": 18.8, "IPE200": 22.4, "IPE220": 26.2, "IPE240": 30.7, "IPE270": 36.1,
    "IPE300": 42.2, "IPE330": 49.1, "IPE360": 57.1, "IPE400": 66.3, "IPE450": 77.6,
    "IPE500": 90.7, "IPE550": 106.0, "IPE600": 122.0,
    "HEA100": 16.7, "HEA120": 19.9, "HEA140": 24.7, "HEA160": 30.4, "HEA180": 35.5,
    "HEA200": 42.3, "HEA220": 50.5, "HEA240": 60.3, "HEA260": 68.2, "HEA280": 76.4,
    "HEA300": 88.3, "HEA320": 97.6, "HEA340": 105.0, "HEA360": 112.0, "HEA400": 125.0,
    "HEA450": 140.0, "HEA500": 155.0, "HEA550": 166.0, "HEA600": 178.0, "HEA650": 190.0,
    "HEA700": 204.0, "HEA800": 224.0, "HEA900": 252.0, "HEA1000": 272.0,
    "HEB100": 20.4, "HEB120": 26.7, "HEB140": 33.7, "HEB160": 42.6, "HEB180": 51.2,
    "HEB200": 61.3, "HEB220": 71.5, "HEB240": 83.2, "HEB260": 93.0, "HEB280": 103.0,
    "HEB300": 117.0, "HEB320": 127.0, "HEB340": 134.0, "HEB360": 142.0, "HEB400": 155.0,
    "HEB450": 171.0, "HEB500": 187.0, "HEB550": 199.0, "HEB600": 212.0, "HEB650": 225.0,
    "HEB700": 241.0, "HEB800": 262.0, "HEB900": 291.0, "HEB1000": 314.0,
    "HEM100": 41.8, "HEM120": 52.1, "HEM140": 63.2, "HEM160": 76.2, "HEM180": 88.9,
    "HEM200": 103.0, "HEM220": 117.0, "HEM240": 157.0, "HEM260": 172.0, "HEM280": 189.0,
    "HEM300": 238.0, "HEM320": 245.0, "HEM340": 248.0, "HEM360": 250.0, "HEM400": 256.0,
    "HEM450": 263.0, "HEM500": 270.0, "HEM550": 277.0, "HEM600": 285.0, "HEM650": 293.0,
    "HEM700": 301.0, "HEM800": 317.0, "HEM900": 333.0, "HEM1000": 349.0,
    "UNP80": 8.64, "UNP100": 10.6, "UNP120": 13.4, "UNP140": 16.0, "UNP160": 18.8,
    "UNP180": 22.0, "UNP200": 25.3, "UNP220": 29.4, "UNP240": 33.2, "UNP260": 37.9,
    "UNP280": 41.8, "UNP300": 46.2, "UNP320": 59.5, "UNP350": 60.6, "UNP380": 63.1,
    "UNP400": 71.8,
}

# Geometria Kwalifikowanych Profili do stykowania (wymiary w mm: h, b, tw, tf)
# Służy do precyzyjnego wyliczania długości i liczby ściegów spawalniczych
PROFILE_DIMENSIONS: Dict[str, Tuple[float, float, float, float]] = {
    # IPE
    "IPE220": (220, 110, 5.9, 9.2), "IPE240": (240, 120, 6.2, 9.8),
    "IPE270": (270, 135, 6.6, 10.2), "IPE300": (300, 150, 7.1, 10.7),
    "IPE330": (330, 160, 7.5, 11.5), "IPE360": (360, 170, 8.0, 12.7),
    "IPE400": (400, 180, 8.6, 13.5), "IPE450": (450, 190, 9.4, 14.6),
    "IPE500": (500, 200, 10.2, 16.0), "IPE550": (550, 210, 11.1, 17.2), "IPE600": (600, 220, 12.0, 19.0),
    # HEA
    "HEA220": (210, 220, 7.0, 11.0), "HEA240": (230, 240, 7.5, 12.0),
    "HEA260": (250, 260, 7.5, 12.5), "HEA280": (270, 280, 8.0, 13.0),
    "HEA300": (290, 300, 8.5, 14.0), "HEA320": (310, 300, 9.0, 15.5),
    "HEA340": (330, 300, 9.5, 16.5), "HEA360": (350, 300, 10.0, 17.5),
    "HEA400": (390, 300, 11.0, 19.0), "HEA450": (440, 300, 11.5, 21.0),
    "HEA500": (490, 300, 12.0, 23.0), "HEA550": (540, 300, 12.5, 24.0), "HEA600": (590, 300, 13.0, 25.0),
    "HEA650": (640, 300, 13.5, 26.0), "HEA700": (690, 300, 14.5, 27.0),
    "HEA800": (790, 300, 15.0, 28.0), "HEA900": (890, 300, 16.0, 30.0), "HEA1000": (990, 300, 16.5, 31.0),
    # HEB
    "HEB220": (220, 220, 9.5, 16.0), "HEB240": (240, 240, 10.0, 17.0),
    "HEB260": (260, 260, 10.0, 17.5), "HEB280": (280, 280, 10.5, 18.0),
    "HEB300": (300, 300, 11.0, 19.0), "HEB320": (320, 300, 11.5, 20.5),
    "HEB340": (340, 300, 12.0, 21.5), "HEB360": (360, 300, 12.5, 22.5),
    "HEB400": (400, 300, 13.5, 24.0), "HEB450": (450, 300, 14.0, 26.0),
    "HEB500": (500, 300, 14.5, 28.0), "HEB550": (550, 300, 15.0, 29.0), "HEB600": (600, 300, 15.5, 30.0),
    "HEB650": (650, 300, 16.0, 31.0), "HEB700": (700, 300, 17.0, 32.0),
    "HEB800": (800, 300, 17.5, 33.0), "HEB900": (900, 300, 18.5, 35.0), "HEB1000": (1000, 300, 19.0, 36.0),
    # HEM
    "HEM220": (240, 226, 15.5, 32.5), "HEM240": (270, 248, 18.0, 32.0),
    "HEM260": (290, 268, 18.0, 32.5), "HEM280": (310, 288, 18.5, 33.0),
    "HEM300": (340, 310, 21.0, 39.0), "HEM320": (359, 309, 21.0, 40.0),
    "HEM340": (377, 309, 21.0, 40.0), "HEM360": (395, 308, 21.0, 40.0),
    "HEM400": (432, 307, 21.0, 40.0), "HEM450": (478, 307, 21.0, 40.0),
    "HEM500": (524, 306, 21.0, 40.0), "HEM550": (572, 306, 21.0, 40.0), "HEM600": (620, 305, 21.0, 40.0),
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

def get_allowed_lengths(profile: str) -> List[float]:
    prof = str(profile).upper().strip()
    if prof.startswith(("HEA", "HEB", "IPE", "HEM")):
        return [12100.0, 15100.0]
    return [6000.0, 12000.0]

def is_splicing_allowed(profile_str: str) -> bool:
    """TWARDA REGUŁA: Stykowanie dotyczy wyłącznie HEA, HEB, IPE, HEM > 200"""
    prof = str(profile_str).upper().replace(" ", "")
    m = re.match(r"^(HEA|HEB|HEM|IPE)(\d+)([ABM])?$", prof)
    if m:
        size = int(m.group(2))
        return size > 200
    return False

def get_profile_dimensions_fallback(profile_str: str) -> Tuple[float, float, float, float]:
    """Zwraca parametry (h, b, tw, tf) profilu. Chroni przed brakującymi kluczami."""
    prof = str(profile_str).upper().replace(" ", "")
    if prof in PROFILE_DIMENSIONS:
        return PROFILE_DIMENSIONS[prof]
    
    # Aprobata fallback dla rzadkich/niestandardowych wymiarów z rodziny HE/IPE
    m = re.match(r"^(HEA|HEB|HEM|IPE)(\d+)", prof)
    if m:
        typ = m.group(1)
        size = int(m.group(2))
        # Skrajne przybliżenie geometryczne by nie wywalić błędu (tylko na wypadek nietypowych gabarytów)
        if typ == "IPE":
            return (size, size/2.0, 5.0 + size/100.0, 8.0 + size/100.0)
        else:
            return (size, max(300.0, size), 10.0 + size/100.0, 15.0 + size/100.0)
    return (200.0, 200.0, 10.0, 10.0)

def calculate_splice_cost(profile_str: str) -> float:
    """Oblicza koszt 1 styku w PLN według logiki biznesowej dla kwalifikowanych profili."""
    if not is_splicing_allowed(profile_str):
        return 0.0

    h, b, tw, tf = get_profile_dimensions_fallback(profile_str)
    
    lw = 2.0 * (h / 1000.0)  # mb
    lf = 2.0 * (b / 1000.0)  # mb
    L_styku = lw + lf
    L_ut = (h + 2.0 * b) / 1000.0
    
    def get_weld_passes(t: float) -> int:
        if t <= 5.6: return 2
        elif t <= 7.9: return 3
        elif t <= 10.0: return 3
        elif t <= 15.0: return 4
        elif t <= 17.0: return 5
        elif t <= 20.0: return 6
        elif t <= 25.0: return 8
        elif t <= 30.0: return 12
        elif t <= 35.0: return 18
        elif t <= 40.0: return 22
        else: return 25
        
    nw = get_weld_passes(tw)
    nf = get_weld_passes(tf)
    
    t_ciecie = 20.0
    t_skladanie = 15.0 * L_styku
    t_spaw_w = 20.0 * nw * lw
    t_spaw_f = 20.0 * nf * lf
    
    t_suma_min = t_ciecie + t_skladanie + t_spaw_w + t_spaw_f
    t_rbh = t_suma_min / 60.0
    
    K_rob = t_rbh * 130.0  # PLN robocizna
    K_ut = L_ut * 70.0     # PLN NDT
    
    return K_rob + K_ut

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

def group_1d_items_by_material(items: List[Item1D]) -> Dict[ProfileGroupKey, List[Item1D]]:
    grouped: Dict[ProfileGroupKey, List[Item1D]] = {}
    for item in items:
        key = ProfileGroupKey(grade=normalize_grade(item.grade), profile=normalize_profile(item.profile))
        grouped.setdefault(key, []).append(item)
    return grouped

def group_2d_plates_by_material(items: List[PlateItem]) -> Dict[PlateGroupKey, List[PlateItem]]:
    grouped: Dict[PlateGroupKey, List[PlateItem]] = {}
    for item in items:
        key = PlateGroupKey(grade=normalize_grade(item.grade), thickness=round(float(item.thickness), 2))
        grouped.setdefault(key, []).append(item)
    return grouped

def optimize_1d_single_group(
    group_key: ProfileGroupKey,
    items: List[Item1D],
    available_stocks: List[float],
    kerf: float = 4.5,
    trim_cut: float = 40.0,
    start_bar_id: int = 1,
    enable_splicing: bool = False
) -> Tuple[List[StockBar], List[str], int]:
    
    expanded_cuts: List[Tuple[str, float]] = []
    oversized_marks = set()
    
    sorted_stocks = sorted(available_stocks)
    max_stock_avail = sorted_stocks[-1]
    stock_bars: List[StockBar] = []
    
    splice_count = 0

    if enable_splicing:
        # Pamiętamy fizyczną liczbę złączonych elementów, co daje (Liczba elementów - 1) styków.
        total_items = sum(it.quantity for it in items)
        splice_count = max(0, total_items - 1)
        
        # Scalenie wszystkich detali
        total_length = 0.0
        combined_marks = []
        for it in items:
            for _ in range(it.quantity):
                total_length += float(it.length)
                combined_marks.append(it.mark)
        
        if total_length > 0:
            combined_mark_str = "Styk(" + "+".join(set(combined_marks)) + ")"
            expanded_cuts.append((combined_mark_str, total_length))
            
        expanded_cuts.sort(key=lambda x: x[1], reverse=True)
        
        for mark, cut_len in expanded_cuts:
            if cut_len + trim_cut > max_stock_avail:
                remaining_cut_len = cut_len
                part_num = 1
                while remaining_cut_len > 0:
                    space_found = False
                    for i, bar in enumerate(stock_bars):
                        if bar.is_oversized: continue
                        capacity_left = bar.stock_length - (bar.used_length + bar.trim_total)
                        if capacity_left > 500.0:
                             take_len = min(remaining_cut_len, capacity_left - kerf)
                             bar.cuts.append((f"{mark}_cz{part_num}", take_len))
                             bar.used_length += take_len + kerf
                             bar.kerf_total += kerf
                             bar.scrap_length = max(0.0, bar.stock_length - bar.used_length - bar.trim_total)
                             remaining_cut_len -= take_len
                             part_num += 1
                             space_found = True
                             # Redukujemy liczbę sztucznych styków jeśli długość przekroczyła dostępną sztangę i musieliśmy ją fizycznie podzielić
                             splice_count = max(0, splice_count - 1) 
                             break
                    
                    if not space_found:
                        take_len = min(remaining_cut_len, max_stock_avail - trim_cut)
                        new_bar = StockBar(
                            bar_id=start_bar_id + len(stock_bars),
                            stock_length=max_stock_avail,
                            profile=group_key.profile,
                            grade=group_key.grade,
                            used_length=take_len,
                            cuts=[(f"{mark}_cz{part_num}", take_len)],
                            kerf_total=0.0,
                            trim_total=trim_cut,
                            scrap_length=max(0.0, max_stock_avail - take_len - trim_cut),
                            is_oversized=False,
                        )
                        stock_bars.append(new_bar)
                        remaining_cut_len -= take_len
                        part_num += 1
                        if part_num > 2: # Każdy podział to fizyczne rozdzielenie, a więc mniej styków (brak łączenia miedzy sztangami)
                             splice_count = max(0, splice_count - 1)
                continue

            best_bar_idx = -1
            min_remaining_space = float("inf")
            for i, bar in enumerate(stock_bars):
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
                
    else:
        # OPCJA BEZ STYKU: Inteligentne dzielenie elementów dłuższych niż najdłuższa sztanga (np. 15.1m)
        for it in items:
            for _ in range(it.quantity):
                expanded_cuts.append((it.mark, float(it.length)))
                
        processed_cuts = []
        for mark, cut_len in expanded_cuts:
            if cut_len + trim_cut > max_stock_avail:
                oversized_marks.add(mark)
                base_stock = 12100.0 if 12100.0 in available_stocks else 12000.0
                max_piece = base_stock - trim_cut
                
                remaining = cut_len
                part_idx = 1
                while remaining > 0:
                    take = min(remaining, max_piece)
                    processed_cuts.append((f"{mark}_podz{part_idx}", take))
                    remaining -= take
                    part_idx += 1
            else:
                processed_cuts.append((mark, cut_len))
        
        processed_cuts.sort(key=lambda x: x[1], reverse=True)
        
        for mark, cut_len in processed_cuts:
            best_bar_idx = -1
            min_remaining_space = float("inf")

            for i, bar in enumerate(stock_bars):
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

    return stock_bars, list(oversized_marks), splice_count

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
        plate_id=plate_id, grade=grade, thickness=thickness,
        stock_w=fmt.width, stock_l=fmt.length,
        packed_items=[], used_area=0.0, scrap_area=0.0, format_name=fmt.name,
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
            
            if placed: break

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
                    used_area=pw*pl, scrap_area=0.0, format_name="Niestandardowy (Formatka)"
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

    raise ValueError("Nie udało się odczytać pliku. Sprawdź format skoroszytu lub pliku CSV.")

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
            q = int(round(float(str(qty_val).replace(',', '.')))) if qty_val not in ['', 'nan', 'None'] else 1
            l = float(str(len_val).replace(',', '.')) if len_val not in ['', 'nan', 'None'] else 0.0
            
            w_str = str(row.get('width', '')).strip()
            w = float(w_str.replace(',', '.')) if w_str not in ['', 'nan', 'None'] else 0.0
            
            t_str = str(row.get('thick', '')).strip()
            t = float(t_str.replace(',', '.')) if t_str not in ['', 'nan', 'None'] else 0.0
            
            grd = str(row.get('grade', 'S355J2+N')).strip()

            if (w == 0.0 or t == 0.0) and any(p_sub in prof_val.upper() for p_sub in ["PL", "BL", "BLACHA", "#", "-"]):
                m_dim = re.search(r'(?:PL|BL|BLACHA|#|-)?\s*(\d+(?:\.\d+)?)\s*[X*x]\s*(\d+(?:\.\d+)?)', prof_val.upper())
                if m_dim:
                    t = float(m_dim.group(1))
                    w = float(m_dim.group(2))

            if q > 0 and (l > 0 or w > 0):
                clean_rows.append({
                    "mark": mark_val if mark_val not in ['', 'None', 'nan'] else f"P{len(clean_rows)+1}",
                    "profile": prof_val,
                    "grade": grd,
                    "length": l,
                    "width": w,
                    "thick": t,
                    "qty": q,
                })
        except (ValueError, TypeError) as e:
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

def build_excel_export(order_opt1, order_opt2, cut_1d_opt1, cut_1d_opt2, cut_2d, stats_1d_opt1, stats_1d_opt2, stats_2d) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if order_opt1 is not None and not order_opt1.empty:
            order_opt1.to_excel(writer, sheet_name="1A. Zamówienie (Bez Styku)", index=False)
        if order_opt2 is not None and not order_opt2.empty:
            order_opt2.to_excel(writer, sheet_name="1B. Zamówienie (Ze Stykiem)", index=False)
        if cut_1d_opt1 is not None and not cut_1d_opt1.empty:
            cut_1d_opt1.to_excel(writer, sheet_name="2A. Rozkrój (Bez Styku)", index=False)
        if cut_1d_opt2 is not None and not cut_1d_opt2.empty:
            cut_1d_opt2.to_excel(writer, sheet_name="2B. Rozkrój (Ze Stykiem)", index=False)
        if cut_2d is not None and not cut_2d.empty:
            cut_2d.to_excel(writer, sheet_name="3. Nesting Blach 2D", index=False)
        if stats_1d_opt1 is not None and not stats_1d_opt1.empty:
            stats_1d_opt1.to_excel(writer, sheet_name="4A. Odpad 1D (Bez Styku)", index=False)
        if stats_1d_opt2 is not None and not stats_1d_opt2.empty:
            stats_1d_opt2.to_excel(writer, sheet_name="4B. Odpad 1D (Ze Stykiem)", index=False)
        if stats_2d is not None and not stats_2d.empty:
            stats_2d.to_excel(writer, sheet_name="5. Odpad Blach 2D", index=False)
    return buffer.getvalue()

st.title("🏗️ SteelOpt: Optymalizator Rozkroju i Generator RFQ")
st.caption("Zaawansowane planowanie hutnicze | Automatyczne Opcje cięcia | Zgodność długości handlowych")

if "bom_data" not in st.session_state:
    st.session_state["bom_data"] = None

with st.sidebar:
    st.header("⚙️ Parametry Technologiczne")

    st.subheader("Parametry Rozkroju 1D (Sztangi)")
    kerf_1d = st.number_input("Szerokość rzazu piły [mm]", min_value=1.0, max_value=12.0, value=4.5, step=0.5)
    trim_1d = st.number_input("Naddatek obcięcia końcówki [mm]", min_value=0.0, max_value=150.0, value=40.0, step=5.0)
    
    st.info("ℹ️ **Reguła długości sztang:**\n\n• HEA, HEB, IPE, HEM: tylko 12.1 m oraz 15.1 m\n• Pozostałe: tylko 6.0 m oraz 12.0 m\n\nℹ️ **Reguła stykowania (Wariant B):**\n\nDotyczy wyłącznie HEA/HEB/HEM/IPE o wielkości > 200.")

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
    uploaded_file = st.file_uploader("Wczytaj plik Excel (.xls, .xlsx) lub XML/CSV z CAD/BIM:", type=["xlsx", "xls", "csv"])

with col_sample:
    st.write("Szybki test:")
    if st.button("🚀 Załaduj Testowy BOM", use_container_width=True):
        st.session_state["bom_data"] = pd.DataFrame([
            {"Pos": "SŁUP_1", "Profile": "HEB600", "Grade": "S355J2+N", "Length_mm": 4800, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
            {"Pos": "RYGIEL_1", "Profile": "IPE300", "Grade": "S355J2+N", "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
            {"Pos": "RYGIEL_MALY", "Profile": "IPE160", "Grade": "S235JR", "Length_mm": 3500, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
            {"Pos": "STĘŻENIE", "Profile": "L100x100x10", "Grade": "S235JR", "Length_mm": 2400, "Width_mm": 0, "Thick_mm": 0, "Qty": 12},
            {"Pos": "BLACHA_WEZLOWA", "Profile": "BLACHA #10", "Grade": "S355J2+N", "Length_mm": 1200, "Width_mm": 800, "Thick_mm": 10, "Qty": 8},
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
        for _, r in df_1d.iterrows() if r["length"] > 0
    ]
    raw_items_2d = [
        PlateItem(mark=str(r["mark"]), grade=str(r["grade"]), thickness=float(r["thick"]) if float(r["thick"]) > 0 else 10.0, width=float(r["width"]) if float(r["width"]) > 0 else 200.0, length=float(r["length"]), quantity=int(r["qty"]))
        for _, r in df_2d.iterrows() if r["length"] > 0 and r["width"] > 0
    ]

    grouped_1d_dict = group_1d_items_by_material(raw_items_1d)
    grouped_2d_dict = group_2d_plates_by_material(raw_items_2d)

    bars_opt1, bars_opt2 = [], []
    bars_by_group_opt1, bars_by_group_opt2 = {}, {}
    order_items_1d_opt1, order_items_1d_opt2 = [], []
    profile_waste_1d_opt1, profile_waste_1d_opt2 = [], []
    global_oversized_opt1 = set()

    bar_id_1, bar_id_2 = 1, 1
    total_splice_count_opt2 = 0
    total_splice_cost_opt2 = 0.0

    for group_key, group_items in grouped_1d_dict.items():
        allowed_stocks = get_allowed_lengths(group_key.profile)
        unit_wt = get_unit_weight_1d(group_key.profile)
        sub_netto_len = sum(it.length * it.quantity for it in group_items)
        sub_netto_mass = (sub_netto_len / 1000.0) * unit_wt

        def calc_1d_scenario(enable_splice, bar_id_start):
            bars, oversized_elements, splice_count = optimize_1d_single_group(
                group_key, group_items, allowed_stocks, kerf=kerf_1d, trim_cut=trim_1d, 
                start_bar_id=bar_id_start, enable_splicing=enable_splice
            )
            stock_counts: Dict[float, int] = {}
            sub_purchased_len = 0.0
            for b in bars:
                stock_counts[b.stock_length] = stock_counts.get(b.stock_length, 0) + 1
                sub_purchased_len += b.stock_length

            order_list = []
            for length_mm, qty_bars in stock_counts.items():
                tot_mass = (length_mm / 1000.0) * unit_wt * qty_bars
                order_list.append({
                    "Kategoria": "Profil hutniczy (1D)",
                    "Asortyment": group_key.profile,
                    "Gatunek Stali": group_key.grade,
                    "Wymiar Handlowy": f"L = {length_mm:.0f} mm",
                    "Ilość Zamawiana [szt.]": qty_bars,
                    "Masa Jednostkowa [kg]": round((length_mm / 1000.0) * unit_wt, 1),
                    "Masa Łączna [kg]": round(tot_mass, 1),
                    "Wymagany Atest": "3.1 wg PN-EN 10204",
                })

            sub_purchased_mass = (sub_purchased_len / 1000.0) * unit_wt
            sub_scrap_mass_kg = max(0.0, sub_purchased_mass - sub_netto_mass)
            sub_scrap_pct = ((sub_purchased_mass - sub_netto_mass) / sub_purchased_mass * 100.0) if sub_purchased_mass > 0 else 0.0

            waste_dict = {
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
                "Wykonane Styki [szt.]": splice_count
            }
            return bars, order_list, waste_dict, oversized_elements, splice_count

        # Opcja 1 (Zawsze Bez Styku)
        b1, o1, w1, over_opt1, _ = calc_1d_scenario(False, bar_id_1)
        bars_opt1.extend(b1)
        bars_by_group_opt1[group_key] = b1
        order_items_1d_opt1.extend(o1)
        profile_waste_1d_opt1.append(w1)
        global_oversized_opt1.update(over_opt1)
        bar_id_1 += len(b1)

        # Opcja 2 (Rozkrój inteligentny ze stykiem wg reguł Gatekeepera)
        # Bramka weryfikacyjna - decyduje czy w tej grupie profili wariant B zastosuje stykowanie
        do_splice = is_splicing_allowed(group_key.profile)
        
        b2, o2, w2, _, sp_count2 = calc_1d_scenario(do_splice, bar_id_2)
        bars_opt2.extend(b2)
        bars_by_group_opt2[group_key] = b2
        order_items_1d_opt2.extend(o2)
        profile_waste_1d_opt2.append(w2)
        bar_id_2 += len(b2)
        
        if do_splice and sp_count2 > 0:
            total_splice_count_opt2 += sp_count2
            unit_splice_cost = calculate_splice_cost(group_key.profile)
            total_splice_cost_opt2 += sp_count2 * unit_splice_cost

    plates_result_all: List[StockPlate] = []
    order_items_2d: List[Dict] = []
    plate_waste_summary_2d: List[Dict] = []
    plate_id_counter = 1

    for group_key, group_items in grouped_2d_dict.items():
        plates = optimize_2d_single_group(group_key, group_items, available_plate_formats, kerf_spacing=kerf_2d, edge_margin=margin_2d, start_plate_id=plate_id_counter)
        plate_id_counter += len(plates)
        plates_result_all.extend(plates)

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
            order_items_2d.append({
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

    tab_procure, tab_1d_opt1, tab_1d_opt2, tab_2d_view, tab_source = st.tabs([
        "🛒 Wyniki i Opcje (RFQ)",
        "📏 Rozkrój (Bez Styku)",
        "📏 Rozkrój (Ze Stykiem)",
        "📐 Nesting Blach (2D)",
        "📋 Zaimportowany BOM",
    ])

    with tab_procure:
        st.markdown("### 🛒 Zestawienie Opcji Zakupowych (Handlowe RFQ)")
        
        if global_oversized_opt1:
            st.warning(f"⚠️ **Wymagana Akceptacja Technologiczna (dla Opcji 1):** Następujące pozycje przekraczały maksymalne długości handlowe (np. 15.1m lub 12.0m) i zostały automatycznie podzielone na krótsze odcinki, aby zmieścić się w bazowych sztangach: **{', '.join(global_oversized_opt1)}**. Prosimy o potwierdzenie proponowanego podziału u Klienta.")
        
        # Koszt materiału Opcja 1 (Brutto, wlicza ukryty koszt odpadu w postaci zakupionych końcówek)
        order_opt1 = order_items_1d_opt1 + order_items_2d
        df_order_opt1 = pd.DataFrame(order_opt1)
        mass_opt1 = df_order_opt1["Masa Łączna [kg]"].sum() if not df_order_opt1.empty else 0
        material_cost_opt1 = (sum(r["Masa Łączna [kg]"] for r in order_items_1d_opt1) * price_profile_per_kg) + (sum(r["Masa Łączna [kg]"] for r in order_items_2d) * price_plate_per_kg)
        cost_opt1 = material_cost_opt1 # Brak kosztów złącz

        # Koszt materiału Opcja 2 (Również masa Brutto pełnych zakupionych sztang)
        order_opt2 = order_items_1d_opt2 + order_items_2d
        df_order_opt2 = pd.DataFrame(order_opt2)
        mass_opt2 = df_order_opt2["Masa Łączna [kg]"].sum() if not df_order_opt2.empty else 0
        material_cost_opt2 = (sum(r["Masa Łączna [kg]"] for r in order_items_1d_opt2) * price_profile_per_kg) + (sum(r["Masa Łączna [kg]"] for r in order_items_2d) * price_plate_per_kg)
        
        # Ostateczny koszt Opcji 2 = Koszt zamówionego materiału (z fizycznym odpadem) + Koszty operacyjne styków
        cost_opt2 = material_cost_opt2 + total_splice_cost_opt2

        if not df_order_opt1.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Opcja 1 (Wariant A: Bez Styku)**\n\n"
                        f"Masa brutto zamówienia: **{mass_opt1/1000.0:.2f} t**\n"
                        f"Koszt materiału brutto: {material_cost_opt1:,.2f} PLN\n"
                        f"---\n"
                        f"**Szacowany koszt CAŁKOWITY: {cost_opt1:,.2f} PLN**")
            with col2:
                st.success(f"**Opcja 2 (Wariant B: Ze Stykiem)**\n\n"
                           f"Masa brutto zamówienia: **{mass_opt2/1000.0:.2f} t**\n"
                           f"Koszt materiału brutto: {material_cost_opt2:,.2f} PLN\n"
                           f"Liczba wykonanych styków (zgodnych): {total_splice_count_opt2} szt.\n"
                           f"Koszt wykonania styków (robocizna + UT): {total_splice_cost_opt2:,.2f} PLN\n"
                           f"---\n"
                           f"**Szacowany koszt CAŁKOWITY: {cost_opt2:,.2f} PLN**")

            st.markdown("#### ✉️ Gotowa treść zapytania ofertowego (E-mail)")
            email_body = (
                "Dzień dobry,\n\n"
                "Proszę o przygotowanie oferty cenowej oraz podanie dostępności dla wyrobów hutniczych, "
                "zgodnie z załączonym plikiem Excel. Plik zawiera dwie alternatywne opcje zestawienia, proszę o wycenę wybranej przez Państwa w zależności od dostępności sztang.\n\n"
                "Wymagania dodatkowe:\n"
                "- Atest materiałowy 3.1 (PN-EN 10204) dla wszystkich zamawianych pozycji.\n"
                "- Proszę o uwzględnienie kosztów transportu.\n\n"
                "Z góry dziękuję za szybką odpowiedź.\n"
                "Pozdrawiam,\n[Twój Podpis]"
            )
            st.text_area("Skopiuj poniższy tekst i wyślij do dystrybutora wraz z plikiem Excel:", value=email_body, height=220)

            # Tabele warsztatowe
            workshop_1d_opt1_rows = [{"Nr Sztangi": b.bar_id, "Profil": b.profile, "Gatunek": b.grade, "Długość Handlowa [mm]": b.stock_length, "Rozkrój": " + ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts]), "Odpad [mm]": round(b.scrap_length, 1)} for b in bars_opt1]
            workshop_1d_opt2_rows = [{"Nr Sztangi": b.bar_id, "Profil": b.profile, "Gatunek": b.grade, "Długość Handlowa [mm]": b.stock_length, "Rozkrój": " + ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts]), "Odpad [mm]": round(b.scrap_length, 1)} for b in bars_opt2]
            workshop_2d_rows = [{"Nr Arkusza": p.plate_id, "Grubość [mm]": p.thickness, "Gatunek": p.grade, "Format": f"{p.stock_w:.0f}×{p.stock_l:.0f}", "Detale": ", ".join([f"{it.mark} ({it.w:.0f}×{it.h:.0f})" for it in p.packed_items])} for p in plates_result_all]

            excel_buffer = build_excel_export(
                order_opt1=df_order_opt1,
                order_opt2=df_order_opt2,
                cut_1d_opt1=pd.DataFrame(workshop_1d_opt1_rows),
                cut_1d_opt2=pd.DataFrame(workshop_1d_opt2_rows),
                cut_2d=pd.DataFrame(workshop_2d_rows),
                stats_1d_opt1=pd.DataFrame(profile_waste_1d_opt1),
                stats_1d_opt2=pd.DataFrame(profile_waste_1d_opt2),
                stats_2d=pd.DataFrame(plate_waste_summary_2d)
            )

            st.download_button(
                label="📥 Pobierz Kompletny Raport (Opcja 1 i 2) [Excel .xlsx]",
                data=excel_buffer,
                file_name="Zamowienie_Hutnicze_SteelOpt_V2.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("Brak pozycji do zamówienia.")

    with tab_1d_opt1:
        st.markdown("### 📏 Rozkrój Profili - Opcja 1 (Bez Styku)")
        if not bars_opt1:
            st.info("Brak profili 1D.")
        else:
            group_options = ["Wszystkie grupy"] + [f"{k.profile} | {k.grade}" for k in bars_by_group_opt1.keys()]
            selected_grp_str = st.selectbox("Wybierz grupę:", group_options, key="grp_opt1")
            bars_filtered = bars_opt1 if selected_grp_str == "Wszystkie grupy" else bars_by_group_opt1[next(k for k in bars_by_group_opt1.keys() if f"{k.profile} | {k.grade}" == selected_grp_str)]
            
            fig_1d = plot_1d_cutting_plan(bars_filtered[:25], title_suffix=f"Opcja 1: {selected_grp_str}")
            st.pyplot(fig_1d)
            plt.close(fig_1d)

    with tab_1d_opt2:
        st.markdown("### 📏 Rozkrój Profili - Opcja 2 (Ze Stykiem)")
        if not bars_opt2:
            st.info("Brak profili 1D.")
        else:
            group_options2 = ["Wszystkie grupy"] + [f"{k.profile} | {k.grade}" for k in bars_by_group_opt2.keys()]
            selected_grp_str2 = st.selectbox("Wybierz grupę:", group_options2, key="grp_opt2")
            bars_filtered2 = bars_opt2 if selected_grp_str2 == "Wszystkie grupy" else bars_by_group_opt2[next(k for k in bars_by_group_opt2.keys() if f"{k.profile} | {k.grade}" == selected_grp_str2)]
            
            fig_1d2 = plot_1d_cutting_plan(bars_filtered2[:25], title_suffix=f"Opcja 2: {selected_grp_str2}")
            st.pyplot(fig_1d2)
            plt.close(fig_1d2)

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

    with tab_source:
        st.markdown("### 📋 Zaimportowany BOM i Weryfikacja")
        st.dataframe(df_clean, use_container_width=True)
else:
    st.info("👈 Wgraj plik z zestawieniem materiałowym (BOM) lub kliknij **'🚀 Załaduj Testowy BOM'**, aby uruchomić podwójną optymalizację (Opcja 1 i Opcja 2).")

```eof

### Krótki Sanity Check
Zmodyfikowałem przycisk "🚀 Załaduj Testowy BOM", aby umieścić tam odpowiednie profile w ramach Sanity Checku:
1. `HEB600` (4 szt. po 4.8m) -> Zgodnie z regułą (`is_splicing_allowed`) wejdzie w system łączenia. Koszty styków zostaną mu doliczone (robocizna i certyfikat NDT wyceniony precyzyjnie dla parametrów `h=600`, `b=300` itd.).
2. `IPE300` -> Kwalifikowany profil, również wejdzie w system łączenia.
3. `IPE160` (Wymiar < 200) -> Zostanie "odcięty" od bramki. Pomimo że w ogólnym sensie to IPE, z racji wymiaru 160 zostanie potraktowany jak Wariant A (bez styku). Koszt styków dla tej grupy wyniesie równo 0 PLN.
4. `L100x100x10` -> Odrzucony profil (Kątownik). Rozkrój bez styku w obu wariantach.

Po załadowaniu testowego pliku BOM w aplikacji, w karcie wyników (zakładka "Wyniki i Opcje") na zielonym kafelku Wariantu B zauważysz czytelny rozkład: kwotę brutto opartą na zamawianych materiałach plus nową pozycję uwzględniającą zliczoną liczbę styków oraz sumaryczny koszt ich wykonania według podanej dokumentacji technicznej.
