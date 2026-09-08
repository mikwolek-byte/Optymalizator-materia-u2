import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SteelOpt - Optymalizator Rozkroju i Zamówień Hutniczych",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEEL_DENSITY_KG_M3 = 7850.0  # kg/m3

# Baza mas jednostkowych najpopularniejszych profili wg norm europejskich (kg/m)
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

    # HEM - profile wzmocnione / słupowe (PN-EN 10034)
    "HEM100": 41.8, "HEM120": 52.1, "HEM140": 63.2, "HEM160": 76.2, "HEM180": 88.9,
    "HEM200": 103.0, "HEM220": 117.0, "HEM240": 157.0, "HEM260": 172.0, "HEM280": 189.0,
    "HEM300": 238.0, "HEM320": 245.0, "HEM340": 248.0, "HEM360": 250.0, "HEM400": 256.0,
    "HEM450": 263.0, "HEM500": 270.0, "HEM550": 278.0, "HEM600": 285.0, "HEM650": 293.0,
    "HEM700": 301.0, "HEM800": 317.0, "HEM900": 333.0, "HEM1000": 349.0,

    # UNP - cechowniki standardowe (PN-EN 10279)
    "UNP80": 8.64, "UNP100": 10.6, "UNP120": 13.4, "UNP140": 16.0, "UNP160": 18.8,
    "UNP180": 22.0, "UNP200": 25.3, "UNP220": 29.4, "UNP240": 33.2, "UNP260": 37.9,
    "UNP280": 41.8, "UNP300": 46.2, "UNP320": 59.5, "UNP350": 60.6, "UNP380": 63.1,
    "UNP400": 71.8,

    # UPE - cechowniki o półkach równoległych
    "UPE80": 7.9, "UPE100": 9.82, "UPE120": 12.1, "UPE140": 14.5, "UPE160": 17.0,
    "UPE180": 19.7, "UPE200": 22.8, "UPE220": 26.6, "UPE240": 30.2, "UPE270": 35.2,
    "UPE300": 44.4, "UPE330": 53.2, "UPE360": 61.2, "UPE400": 72.2,
}

def get_unit_weight_1d(profile_str: str) -> float:
    """
    Oblicza lub odnajduje masę 1 mb profilu hutniczego [kg/m].
    Wspiera: IPE, HEA, HEB, HEM, UNP, UPE, Kątowniki (L), Rury prostokątne/kwadratowe (RHS/SHS/RK/RP)
    oraz Rury okrągłe (CHS/RO).
    """
    raw_str = str(profile_str).upper().replace(" ", "").replace("×", "X").replace("*", "X")
    
    # 1. Normalizacja dialektów niemieckich / Tekla (np. HE240A -> HEA240, HE300M -> HEM300)
    he_match = re.match(r"^HE(\d+)([ABM])$", raw_str)
    if he_match:
        size, variant = he_match.groups()
        raw_str = f"HE{variant}{size}"

    # 2. Wyszukiwanie w tabeli walcowanych profili otwartych (IPE, HEA, HEB, HEM, UNP, UPE)
    clean_prof = re.sub(r'[^A-Z0-9]', '', raw_str)
    for key, weight in EURO_PROFILE_WEIGHTS.items():
        if key == clean_prof or clean_prof.startswith(key):
            return weight

    # 3. KĄTOWNIKI (L) - np. L100x100x10, L120x80x8, L100x10, KAT 50x5
    # Wariant trójwymiarowy (L a x b x t)
    angle_3d = re.search(r'(?:L|KAT|KATOWNIK|KĄTOWNIK)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)', raw_str)
    if angle_3d and ("L" in raw_str or "KAT" in raw_str):
        a, b, t = map(float, angle_3d.groups())
        # Masa z uwzględnieniem zaokrąglenia wewnętrznego (mnożnik 1.012)
        area_mm2 = (a + b - t) * t * 1.012
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Wariant dwuwymiarowy (równoramienny L a x t) np. L100x10
    angle_2d = re.search(r'(?:L|KAT|KATOWNIK|KĄTOWNIK)\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)', raw_str)
    if angle_2d:
        a, t = map(float, angle_2d.groups())
        area_mm2 = (2 * a - t) * t * 1.012
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # 4. PROFILE ZAMKNIĘTE PROSTOKĄTNE I KWADRATOWE (RHS, SHS, RK, RP, PROFIL)
    # Wariant trójwymiarowy np. RHS 120x80x5, 100x100x6, RK 80x80x4
    tube_rect = re.search(r'(?:RHS|SHS|RK|RP|PR|PROFIL)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)', raw_str)
    if tube_rect:
        h, b, t = map(float, tube_rect.groups())
        # Norma PN-EN 10219-2 dla profili formowanych na zimno z narożami ro=2t:
        area_mm2 = 2 * t * (h + b) - (4.0 + 2.575) * (t ** 2)
        if area_mm2 <= 0:
            area_mm2 = 2 * (h + b) * t - 4 * (t ** 2)
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Wariant kwadratowy dwuwymiarowy np. SHS 100x5, RK 80x4
    tube_sq = re.search(r'(?:SHS|RK)\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)', raw_str)
    if tube_sq:
        h, t = map(float, tube_sq.groups())
        area_mm2 = 4 * t * h - (4.0 + 2.575) * (t ** 2)
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # 5. RURY OKRĄGŁE (CHS, RO, RURA, FI) np. RO 88.9x4, CHS 114.3x5, RURA 60.3x3.2
    pipe_match = re.search(r'(?:CHS|RO|RURA|FI|Ø)\s*(\d+(?:\.\d+)?)[X/](\d+(?:\.\d+)?)', raw_str)
    if pipe_match:
        d, t = map(float, pipe_match.groups())
        area_mm2 = math.pi * (d - t) * t
        return round(area_mm2 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)

    # Domyślna bezpieczna waga zastępcza, jeśli profil niestandardowy
    return 25.0

def optimize_1d_cutting(
    items: List[Item1D],
    available_stocks: List[float],
    kerf: float = 4.0,
    trim_cut: float = 30.0,
) -> List[StockBar]:
    """
    Optymalizacja 1D rozkroju sztangowego z uwzględnieniem rzazu i podwójnego/pojedynczego naddatku.
    Stosuje heurystykę Best-Fit Decreasing z minimalizacją naddatku i doborem optymalnego formatu handlowego.
    """
    # Rozwinięcie listy detali
    expanded_cuts: List[Tuple[str, float, str, str]] = []
    for it in items:
        for _ in range(it.quantity):
            expanded_cuts.append((it.mark, float(it.length), it.profile, it.grade))

    # Sortowanie malejąco wg długości (priorytet największych elementów)
    expanded_cuts.sort(key=lambda x: x[1], reverse=True)

    sorted_stocks = sorted(available_stocks)
    stock_bars: List[StockBar] = []

    for mark, cut_len, profile, grade in expanded_cuts:
        best_bar_idx = -1
        min_remaining_space = float("inf")

        # 1. Próba upakowania w już otwartą sztangę (Best-Fit)
        for i, bar in enumerate(stock_bars):
            required_space = cut_len + (kerf if len(bar.cuts) > 0 else 0.0)
            capacity_left = bar.stock_length - (bar.used_length + trim_cut)
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
            # 2. Dobór nowej sztangi o najkrótszym pasującym wymiarze handlowym
            eligible_lengths = [
                s for s in sorted_stocks if s >= (cut_len + trim_cut)
            ]
            if not eligible_lengths:
                # Jeśli detal dłuższy niż największa sztanga - przypisz największą z informacją
                selected_stock = sorted_stocks[-1]
            else:
                selected_stock = eligible_lengths[0]

            new_bar = StockBar(
                bar_id=len(stock_bars) + 1,
                stock_length=selected_stock,
                profile=profile,
                grade=grade,
                used_length=cut_len,
                cuts=[(mark, cut_len)],
                kerf_total=0.0,
                trim_total=trim_cut,
                scrap_length=max(0.0, selected_stock - cut_len - trim_cut),
            )
            stock_bars.append(new_bar)

    return stock_bars

def optimize_2d_plates(
    items: List[PlateItem],
    stock_w: float,
    stock_l: float,
    kerf_spacing: float = 12.0,
    edge_margin: float = 20.0,
) -> List[StockPlate]:
    """
    Algorytm rozkroju 2D (Guillotine Shelf First-Fit Decreasing) dla blach grubych i formatek.
    Uwzględnia marginesy technologiczne od brzegów oraz odstęp między detalami (palnik/plazma).
    """
    # Rozwinięcie elementów z możliwością rotacji o 90 stopni (dłuższy bok jako długość)
    parts: List[Tuple[str, float, float]] = []
    grade = items[0].grade if items else "S355J2+N"
    thickness = items[0].thickness if items else 10.0

    for it in items:
        for _ in range(it.quantity):
            # Normalizacja orientacji: w mniejszy, l większy
            w, l = min(it.width, it.length), max(it.width, it.length)
            parts.append((it.mark, w, l))

    # Sortowanie wg wysokości (lub powierzchni) malejąco
    parts.sort(key=lambda p: (p[1], p[2]), reverse=True)

    effective_w = stock_w - 2 * edge_margin
    effective_l = stock_l - 2 * edge_margin

    plates: List[StockPlate] = []
    if effective_w <= 0 or effective_l <= 0:
        return plates

    # Każda płyta przechowuje stan półek (shelves): y, shelf_height, current_x
    plate_shelves: List[List[Dict[str, float]]] = []

    for mark, pw, pl in parts:
        placed = False

        # Sprawdzenie obydwu orientacji detalu z uwzględnieniem kerfu
        orientations = [(pw, pl), (pl, pw)]

        for p_idx, plate in enumerate(plates):
            shelves = plate_shelves[p_idx]
            for o_w, o_l in orientations:
                # 1. Próba zmieszczenia w istniejącej półce
                for shelf in shelves:
                    if o_l <= shelf["height"] and (shelf["current_x"] + o_w + kerf_spacing) <= effective_w:
                        x = edge_margin + shelf["current_x"]
                        y = edge_margin + shelf["y"]
                        plate.packed_items.append(PackedRect(mark=mark, x=x, y=y, w=o_w, h=o_l))
                        shelf["current_x"] += o_w + kerf_spacing
                        plate.used_area += (o_w * o_l)
                        placed = True
                        break
                if placed:
                    break

                # 2. Próba otwarcia nowej półki na tej samej blasze
                last_y_end = shelves[-1]["y"] + shelves[-1]["height"] + kerf_spacing if shelves else 0.0
                if not placed and (last_y_end + o_l) <= effective_l and (o_w <= effective_w):
                    new_shelf = {"y": last_y_end, "height": o_l, "current_x": o_w + kerf_spacing}
                    shelves.append(new_shelf)
                    x = edge_margin
                    y = edge_margin + last_y_end
                    plate.packed_items.append(PackedRect(mark=mark, x=x, y=y, w=o_w, h=o_l))
                    plate.used_area += (o_w * o_l)
                    placed = True
                    break
            if placed:
                break

        # 3. Dodanie nowego arkusza handlowego blachy
        if not placed:
            best_o_w, best_o_l = (pw, pl) if pw <= effective_w and pl <= effective_l else (pl, pw)
            new_plate = StockPlate(
                plate_id=len(plates) + 1,
                grade=grade,
                thickness=thickness,
                stock_w=stock_w,
                stock_l=stock_l,
                packed_items=[PackedRect(mark=mark, x=edge_margin, y=edge_margin, w=best_o_w, h=best_o_l)],
                used_area=(best_o_w * best_o_l),
                scrap_area=0.0,
            )
            plates.append(new_plate)
            plate_shelves.append([{"y": 0.0, "height": best_o_l, "current_x": best_o_w + kerf_spacing}])

    # Przeliczenie odpadu i statystyk arkuszy
    for pl in plates:
        total_plate_area = pl.stock_w * pl.stock_l
        pl.scrap_area = max(0.0, total_plate_area - pl.used_area)

    return plates

def generate_sample_bom_df() -> pd.DataFrame:
    """Tworzy realistyczne zestawienie materiałowe (BOM) ze wszystkimi typami profili."""
    data = [
        {"Pos": "B1", "Profile": "IPE300", "Grade": "S355J2+N", "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
        {"Pos": "B2", "Profile": "HEA240", "Grade": "S355J2+N", "Length_mm": 6250, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
        {"Pos": "C1", "Profile": "HEB200", "Grade": "S355J2+N", "Length_mm": 4150, "Width_mm": 0, "Thick_mm": 0, "Qty": 4},
        {"Pos": "C2", "Profile": "HEM160", "Grade": "S355J2+N", "Length_mm": 3800, "Width_mm": 0, "Thick_mm": 0, "Qty": 3},
        {"Pos": "U1", "Profile": "UNP180", "Grade": "S235JR", "Length_mm": 2950, "Width_mm": 0, "Thick_mm": 0, "Qty": 8},
        {"Pos": "K1", "Profile": "L100x100x10", "Grade": "S235JR", "Length_mm": 2400, "Width_mm": 0, "Thick_mm": 0, "Qty": 10},
        {"Pos": "K2", "Profile": "L80x8", "Grade": "S235JR", "Length_mm": 1800, "Width_mm": 0, "Thick_mm": 0, "Qty": 12},
        {"Pos": "RK1", "Profile": "SHS120x120x6", "Grade": "S355J2H", "Length_mm": 4500, "Width_mm": 0, "Thick_mm": 0, "Qty": 6},
        {"Pos": "RP1", "Profile": "RHS160x80x5", "Grade": "S355J2H", "Length_mm": 3200, "Width_mm": 0, "Thick_mm": 0, "Qty": 5},
        {"Pos": "RO1", "Profile": "RO88.9x4", "Grade": "S355J2H", "Length_mm": 2800, "Width_mm": 0, "Thick_mm": 0, "Qty": 8},
        {"Pos": "PL1", "Profile": "BLACHA", "Grade": "S355J2+N", "Length_mm": 650, "Width_mm": 450, "Thick_mm": 20, "Qty": 16},
        {"Pos": "PL2", "Profile": "BLACHA", "Grade": "S355J2+N", "Length_mm": 350, "Width_mm": 250, "Thick_mm": 20, "Qty": 32},
        {"Pos": "PL3", "Profile": "BLACHA", "Grade": "S235JR", "Length_mm": 1200, "Width_mm": 800, "Thick_mm": 10, "Qty": 6},
    ]
    return pd.DataFrame(data)

def map_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rozpoznaje nazwy kolumn z różnych dialektów CAD (Tekla, Bocad, Advance Steel, polski/angielski)."""
    mapping = {}
    cols = df.columns.astype(str)

    patterns = {
        "mark": r"(pos|mark|pozycja|nr|oznaczenie|element|id)",
        "profile": r"(profile|profil|przekrój|section|asortyment)",
        "grade": r"(grade|gatunek|materiał|stal|material)",
        "length": r"(length|długość|dlugosc|len|l_mm|dł)",
        "width": r"(width|szerokość|szerokosc|w_mm|szer)",
        "thick": r"(thick|grubość|grubosc|thk|t_mm|gr)",
        "qty": r"(qty|quantity|ilość|ilosc|szt|liczba|count)",
    }

    used_cols = set()
    for target_key, pattern in patterns.items():
        for col in cols:
            if col not in used_cols and re.search(pattern, col, re.IGNORECASE):
                mapping[target_key] = col
                used_cols.add(col)
                break

    clean_df = pd.DataFrame()
    clean_df["mark"] = df[mapping["mark"]].astype(str) if "mark" in mapping else [f"P{i+1}" for i in range(len(df))]
    clean_df["profile"] = df[mapping["profile"]].astype(str) if "profile" in mapping else "UNKNOWN"
    clean_df["grade"] = df[mapping["grade"]].astype(str).str.strip().str.upper() if "grade" in mapping else "S355J2+N"
    clean_df["length"] = pd.to_numeric(df[mapping["length"]], errors="coerce").fillna(0.0) if "length" in mapping else 0.0
    clean_df["width"] = pd.to_numeric(df[mapping["width"]], errors="coerce").fillna(0.0) if "width" in mapping else 0.0
    clean_df["thick"] = pd.to_numeric(df[mapping["thick"]], errors="coerce").fillna(0.0) if "thick" in mapping else 0.0
    clean_df["qty"] = pd.to_numeric(df[mapping["qty"]], errors="coerce").fillna(1).astype(int) if "qty" in mapping else 1

    # Podstawowa walidacja i oczyszczanie wartości
    clean_df = clean_df[(clean_df["qty"] > 0) & (clean_df["length"] > 0)].copy()
    return clean_df

def plot_1d_cutting_plan(bars: List[StockBar], max_display: int = 15) -> plt.Figure:
    """Generuje wykres warsztatowy rozkroju sztang za pomocą Matplotlib."""
    bars_to_plot = bars[:max_display]
    fig, ax = plt.subplots(figsize=(12, max(3.5, len(bars_to_plot) * 0.55)))

    colors = ["#2b5c8f", "#3e8e7e", "#d97736", "#c0392b", "#8e44ad", "#16a085", "#2980b9"]

    y_positions = list(range(len(bars_to_plot)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Sztanga #{b.bar_id} ({b.stock_length:.0f} mm)" for b in bars_to_plot], fontsize=9)
    ax.invert_yaxis()

    for idx, b in enumerate(bars_to_plot):
        curr_x = b.trim_total / 2.0  # Rozpoczęcie po pierwszym obcięciu
        # Naddatek wstępny
        if b.trim_total > 0:
            ax.barh(idx, curr_x, left=0, color="#7f8c8d", edgecolor="black", height=0.6, hatch="//")

        for c_idx, (mark, cut_len) in enumerate(b.cuts):
            color = colors[(c_idx + idx) % len(colors)]
            ax.barh(idx, cut_len, left=curr_x, color=color, edgecolor="black", height=0.6)
            if cut_len > (b.stock_length * 0.05):
                ax.text(curr_x + cut_len / 2, idx, f"{mark}\n{cut_len:.0f}",
                        va="center", ha="center", color="white", fontsize=7.5, fontweight="bold")
            curr_x += cut_len
            # Rzaz piły
            ax.barh(idx, 4.0, left=curr_x, color="#2c3e50", height=0.6)
            curr_x += 4.0

        # Odpad końcowy
        rem_scrap = b.stock_length - curr_x
        if rem_scrap > 0:
            ax.barh(idx, rem_scrap, left=curr_x, color="#bdc3c7", edgecolor="black", height=0.6, alpha=0.7)
            if rem_scrap > (b.stock_length * 0.06):
                ax.text(curr_x + rem_scrap / 2, idx, f"Odpad: {rem_scrap:.0f}",
                        va="center", ha="center", color="#333333", fontsize=7, style="italic")

    ax.set_xlabel("Długość [mm]", fontsize=10)
    ax.set_title(f"Warsztatowy Plan Rozkroju 1D (Wyświetlono {len(bars_to_plot)} z {len(bars)} sztang)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_2d_plate_plan(plate: StockPlate) -> plt.Figure:
    """Wizualizuje rozkrój arkusza blachy z pozycjami formatek."""
    fig, ax = plt.subplots(figsize=(10, 5))
    # Obrys arkusza
    sheet_rect = patches.Rectangle((0, 0), plate.stock_l, plate.stock_w,
                                   linewidth=1.5, edgecolor="#2c3e50", facecolor="#ecf0f1")
    ax.add_patch(sheet_rect)

    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#f1c40f", "#e74c3c", "#1abc9c"]
    for idx, item in enumerate(plate.packed_items):
        color = colors[idx % len(colors)]
        r = patches.Rectangle((item.y, item.x), item.h, item.w,
                              linewidth=1, edgecolor="black", facecolor=color, alpha=0.85)
        ax.add_patch(r)
        if item.h > 150 and item.w > 100:
            ax.text(item.y + item.h / 2, item.x + item.w / 2,
                    f"{item.mark}\n{item.h:.0f}x{item.w:.0f}",
                    color="white", fontsize=7, ha="center", va="center", weight="bold")

    ax.set_xlim(-100, plate.stock_l + 100)
    ax.set_ylim(-100, plate.stock_w + 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Długość arkusza [mm]")
    ax.set_ylabel("Szerokość arkusza [mm]")
    ax.set_title(f"Arkusz #{plate.plate_id} - Blacha #{plate.thickness:.0f}mm {plate.grade} ({plate.stock_w:.0f}x{plate.stock_l:.0f}mm) - Wykorzystanie: {(plate.used_area / (plate.stock_w * plate.stock_l) * 100):.1f}%", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig

def build_excel_export(
    procurement_df: pd.DataFrame,
    cut_summary_1d: pd.DataFrame,
    cut_summary_2d: pd.DataFrame,
    stats_df: pd.DataFrame,
    profile_stats_1d: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    """Tworzy sformatowany plik Excel ze wszystkimi zestawieniami i planami zamówień."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        procurement_df.to_excel(writer, sheet_name="Do Zamówienia", index=False)
        if not cut_summary_1d.empty:
            cut_summary_1d.to_excel(writer, sheet_name="Rozkroj Profile 1D", index=False)
        if profile_stats_1d is not None and not profile_stats_1d.empty:
            profile_stats_1d.to_excel(writer, sheet_name="Odpad wg Profili 1D", index=False)
        if not cut_summary_2d.empty:
            cut_summary_2d.to_excel(writer, sheet_name="Rozkroj Blachy 2D", index=False)
        stats_df.to_excel(writer, sheet_name="Statystyka Odpadu", index=False)
    output.seek(0)
    return output

st.title("🏗️ SteelOpt: Optymalizator Rozkroju Stali i Zamówień Hutniczych")
st.caption("Zaawansowane mapowanie BOM (Tekla, Advance Steel, Bocad) | Algorytmy 1D/2D CSP | Generator Zakupowy")

with st.sidebar:
    st.header("⚙️ Parametry Technologiczne")

    st.subheader("Parametry Rozkroju 1D (Profile)")
    kerf_1d = st.number_input("Szerokość rzazu piły [mm]", min_value=1.0, max_value=12.0, value=4.5, step=0.5)
    trim_1d = st.number_input("Obcięcie końcówki (Trim Cut) [mm]", min_value=0.0, max_value=150.0, value=40.0, step=5.0)
    stock_options_1d = st.multiselect(
        "Dostępne sztangi handlowe [mm]:",
        options=[10000.0, 12000.0, 12100.0, 14000.0, 15000.0, 18000.0],
        default=[12000.0, 12100.0, 14000.0, 15000.0]
    )
    if not stock_options_1d:
        stock_options_1d = [12000.0]

    st.divider()
    st.subheader("Parametry Rozkroju 2D (Blachy)")
    kerf_2d = st.number_input("Odstęp termiczny (plazma/tlen) [mm]", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
    margin_2d = st.number_input("Margines brzegowy arkusza [mm]", min_value=5.0, max_value=50.0, value=20.0, step=5.0)
    plate_formats = {
        "1500 x 3000 mm": (1500.0, 3000.0),
        "1500 x 6000 mm": (1500.0, 6000.0),
        "2000 x 6000 mm": (2000.0, 6000.0),
        "2000 x 12000 mm": (2000.0, 12000.0),
    }
    selected_format_label = st.selectbox("Format handlowy arkusza blachy:", list(plate_formats.keys()), index=1)
    sel_plate_w, sel_plate_l = plate_formats[selected_format_label]

st.subheader("📥 1. Import Zestawienia Materiałowego (BOM)")
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Wczytaj plik Excel (.xlsx, .xls) lub CSV z CAD/BIM:",
        type=["xlsx", "xls", "csv"],
        help="Obsługuje eksporty z Tekla Structures, Bocad, Autodesk Advance Steel itp."
    )

with col_sample:
    st.write("Szybki test:")
    use_sample = st.button("🚀 Załaduj Przykładowy BOM", use_container_width=True)

df_raw: Optional[pd.DataFrame] = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.success(f"Wczytano plik `{uploaded_file.name}` ({len(df_raw)} wierszy).")
    except Exception as err:
        st.error(f"Błąd odczytu pliku: {err}")
elif use_sample:
    df_raw = generate_sample_bom_df()
    st.info("Załadowano przykładowy zestaw testowy zawierający profile IPE/HEA/UNP oraz blachy węzłowe.")

if df_raw is not None and not df_raw.empty:
    with st.expander("Podgląd surowych danych przed przetworzeniem", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)

    df_clean = map_imported_columns(df_raw)

    # Klasyfikacja elementów na 1D (profile) oraz 2D (blachy).
    # Zabezpieczenie: Kątowniki (L), dwuteowniki i profile rurowe posiadające wymiar w CAD nie mogą wpaść do blach.
    is_structural_1d = df_clean["profile"].str.contains(
        r"IPE|HEA|HEB|HEM|UNP|UPE|RHS|SHS|CHS|RO|RK|RP|^(?:L|KAT|KĄT)",
        case=False,
        regex=True
    )

    is_plate_condition = (
        df_clean["profile"].str.contains(r"PL|BLACHA|PLATE|BL|FLAT|PŁYT", case=False, regex=True)
        | ((df_clean["width"] > 0) & (~is_structural_1d))
        | ((df_clean["thick"] > 0) & (~is_structural_1d))
    )

    df_1d = df_clean[~is_plate_condition].copy()
    df_2d = df_clean[is_plate_condition].copy()

    tab_overview, tab_1d, tab_2d, tab_order = st.tabs([
        "📊 Podsumowanie Asortymentu",
        "📏 Rozkrój Profili (1D)",
        "📐 Rozkrój Blach (2D)",
        "🛒 Zamówienie Hutnicze & Eksport",
    ])

    with tab_overview:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Pozycje w BOM", len(df_clean))
        col_m2.metric("Suma elementów (szt.)", int(df_clean["qty"].sum()))
        col_m3.metric("Elementy 1D (Profile/Rury)", int(df_1d["qty"].sum()) if not df_1d.empty else 0)
        col_m4.metric("Elementy 2D (Formatki/Węzłówki)", int(df_2d["qty"].sum()) if not df_2d.empty else 0)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Wykryte Profile Hutnicze (1D)")
            if not df_1d.empty:
                st.dataframe(df_1d[["mark", "profile", "grade", "length", "qty"]], use_container_width=True)
            else:
                st.write("Brak pozycji profili hutniczych.")

        with c2:
            st.markdown("#### Wykryte Blachy i Płaskowniki (2D)")
            if not df_2d.empty:
                st.dataframe(df_2d[["mark", "grade", "thick", "length", "width", "qty"]], use_container_width=True)
            else:
                st.write("Brak formatek blach w zestawieniu.")

    bars_result_all: List[StockBar] = []
    order_items_1d: List[Dict] = []
    profile_waste_summary_1d: List[Dict] = []

    with tab_1d:
        st.markdown("### ⚙️ Wyniki Optymalizacji 1D (Profile & Rury)")
        if df_1d.empty:
            st.info("Brak profili do rozkroju.")
        else:
            # Grupowanie po profilu i gatunku
            grouped_1d = df_1d.groupby(["profile", "grade"])

            total_theoretical_len = 0.0
            total_purchased_len = 0.0
            total_theoretical_mass = 0.0
            total_purchased_mass = 0.0

            for (prof, gr), group in grouped_1d:
                items_1d = [
                    Item1D(
                        mark=row["mark"],
                        profile=prof,
                        grade=gr,
                        length=row["length"],
                        quantity=int(row["qty"]),
                    )
                    for _, row in group.iterrows()
                ]

                bars = optimize_1d_cutting(
                    items=items_1d,
                    available_stocks=stock_options_1d,
                    kerf=kerf_1d,
                    trim_cut=trim_1d,
                )
                bars_result_all.extend(bars)

                unit_wt = get_unit_weight_1d(prof)

                # Zliczanie zamówienia oraz długości zakupionej dla profilu
                stock_counts = {}
                sub_purchased_len = 0.0
                for b in bars:
                    stock_counts[b.stock_length] = stock_counts.get(b.stock_length, 0) + 1
                    total_purchased_len += b.stock_length
                    sub_purchased_len += b.stock_length
                    total_purchased_mass += (b.stock_length / 1000.0) * unit_wt

                for length_mm, qty_bars in stock_counts.items():
                    order_items_1d.append({
                        "Typ": "Profil hutniczy",
                        "Asortyment": prof,
                        "Gatunek": gr,
                        "Wymiar handlowy": f"L = {length_mm:.0f} mm",
                        "Ilość [szt.]": qty_bars,
                        "Masa jedn. [kg/m]": round(unit_wt, 2),
                        "Masa łączna [kg]": round((length_mm / 1000.0) * unit_wt * qty_bars, 1),
                    })

                sub_netto_len = sum(it.length * it.quantity for it in items_1d)
                sub_netto_mass = (sub_netto_len / 1000.0) * unit_wt
                sub_purchased_mass = (sub_purchased_len / 1000.0) * unit_wt
                sub_scrap_len_m = max(0.0, (sub_purchased_len - sub_netto_len) / 1000.0)
                sub_scrap_mass_kg = sub_scrap_len_m * unit_wt
                sub_scrap_pct = ((sub_purchased_len - sub_netto_len) / sub_purchased_len * 100.0) if sub_purchased_len > 0 else 0.0

                for it in items_1d:
                    total_theoretical_len += it.length * it.quantity
                    total_theoretical_mass += (it.length / 1000.0) * unit_wt * it.quantity

                profile_waste_summary_1d.append({
                    "Profil": prof,
                    "Gatunek": gr,
                    "Masa jedn. [kg/m]": round(unit_wt, 2),
                    "Ilość sztang [szt.]": len(bars),
                    "Długość netto [m]": round(sub_netto_len / 1000.0, 2),
                    "Długość brutto [m]": round(sub_purchased_len / 1000.0, 2),
                    "Masa netto [kg]": round(sub_netto_mass, 1),
                    "Masa brutto [kg]": round(sub_purchased_mass, 1),
                    "Odpad [m]": round(sub_scrap_len_m, 2),
                    "Odpad [kg]": round(sub_scrap_mass_kg, 1),
                    "Odpad [%]": round(sub_scrap_pct, 2),
                })

            # Metryki 1D
            scrap_1d_pct = ((total_purchased_len - total_theoretical_len) / total_purchased_len * 100) if total_purchased_len > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Łączna liczba sztang handlowych", len(bars_result_all))
            c2.metric("Masa do zamówienia (1D)", f"{total_purchased_mass / 1000.0:.2f} t")
            c3.metric("Średni odpad technologiczny 1D", f"{scrap_1d_pct:.2f}%", delta=f"-{scrap_1d_pct:.1f}%", delta_color="inverse")

            st.markdown("#### 📊 Bilans Odpadu w Rozbiciu na Poszczególne Profile")
            df_profile_waste = pd.DataFrame(profile_waste_summary_1d)
            st.dataframe(
                df_profile_waste.style.format({
                    "Odpad [%]": "{:.2f}%",
                    "Odpad [kg]": "{:.1f} kg",
                    "Odpad [m]": "{:.2f} m",
                    "Masa netto [kg]": "{:.1f}",
                    "Masa brutto [kg]": "{:.1f}",
                    "Długość netto [m]": "{:.2f}",
                    "Długość brutto [m]": "{:.2f}",
                }),
                use_container_width=True
            )

            st.pyplot(plot_1d_cutting_plan(bars_result_all))

            # Tabela rozkroju warsztatowego
            workshop_1d_data = []
            for b in bars_result_all:
                cuts_repr = " + ".join([f"{mark} ({l:.0f}mm)" for mark, l in b.cuts])
                workshop_1d_data.append({
                    "Nr Sztangi": b.bar_id,
                    "Profil": b.profile,
                    "Gatunek": b.grade,
                    "Długość sztangi [mm]": b.stock_length,
                    "Pozycje cięcia": cuts_repr,
                    "Suma netto [mm]": sum(c[1] for c in b.cuts),
                    "Odpad [mm]": round(b.scrap_length, 1),
                    "Wskaźnik odpadu [%]": round((b.scrap_length / b.stock_length) * 100, 1),
                })
            df_workshop_1d = pd.DataFrame(workshop_1d_data)
            st.markdown("#### Plan cięcia warsztatowego (Karty cięcia):")
            st.dataframe(df_workshop_1d, use_container_width=True)

    plates_result_all: List[StockPlate] = []
    order_items_2d: List[Dict] = []

    with tab_2d:
        st.markdown("### 📐 Wyniki Optymalizacji 2D (Blachy & Węzłówki)")
        if df_2d.empty:
            st.info("Brak formatek blach w zaimportowanym pliku.")
        else:
            # Grupowanie po grubości i gatunku
            grouped_2d = df_2d.groupby(["thick", "grade"])

            total_purchased_plate_area = 0.0
            total_used_plate_area = 0.0

            for (thk, gr), group in grouped_2d:
                plate_items = [
                    PlateItem(
                        mark=row["mark"],
                        grade=gr,
                        thickness=float(thk) if float(thk) > 0 else 10.0,
                        width=float(row["width"]),
                        length=float(row["length"]),
                        quantity=int(row["qty"]),
                    )
                    for _, row in group.iterrows()
                ]

                plates = optimize_2d_plates(
                    items=plate_items,
                    stock_w=sel_plate_w,
                    stock_l=sel_plate_l,
                    kerf_spacing=kerf_2d,
                    edge_margin=margin_2d,
                )
                plates_result_all.extend(plates)

                # Masa arkusza = Powierzchnia (m2) * grubość (mm) * 7.85 kg/m2/mm
                plate_area_m2 = (sel_plate_w * sel_plate_l) / 1_000_000.0
                plate_weight_single = plate_area_m2 * float(thk) * (STEEL_DENSITY_KG_M3 / 1000.0)

                order_items_2d.append({
                    "Typ": "Blacha gruba",
                    "Asortyment": f"Blacha #{thk:.0f} mm",
                    "Gatunek": gr,
                    "Wymiar handlowy": f"{sel_plate_w:.0f} x {sel_plate_l:.0f} mm",
                    "Ilość [szt.]": len(plates),
                    "Masa jedn. [kg/szt]": round(plate_weight_single, 1),
                    "Masa łączna [kg]": round(plate_weight_single * len(plates), 1),
                })

                for p in plates:
                    total_purchased_plate_area += (p.stock_w * p.stock_l)
                    total_used_plate_area += p.used_area

            scrap_2d_pct = ((total_purchased_plate_area - total_used_plate_area) / total_purchased_plate_area * 100) if total_purchased_plate_area > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Liczba arkuszy hutniczych", len(plates_result_all))
            total_plate_mass_kg = sum(x["Masa łączna [kg]"] for x in order_items_2d)
            c2.metric("Masa blach do zamówienia", f"{total_plate_mass_kg / 1000.0:.2f} t")
            c3.metric("Średni odpad nestingowy 2D", f"{scrap_2d_pct:.2f}%", delta=f"-{scrap_2d_pct:.1f}%", delta_color="inverse")

            # Wykres pierwszych 3 arkuszy
            st.markdown("#### Wizualizacja Map Rozkroju Arkuszy:")
            for p in plates_result_all[:3]:
                st.pyplot(plot_2d_plate_plan(p))

            # Tabela rozkroju 2D
            workshop_2d_data = []
            for p in plates_result_all:
                items_summary = ", ".join([f"{it.mark} ({it.w:.0f}x{it.h:.0f})" for it in p.packed_items])
                workshop_2d_data.append({
                    "Nr Arkusza": p.plate_id,
                    "Grubość [mm]": p.thickness,
                    "Gatunek": p.grade,
                    "Format [mm]": f"{p.stock_w:.0f} x {p.stock_l:.0f}",
                    "Liczba detali": len(p.packed_items),
                    "Wskaźnik wykorzystania [%]": round((p.used_area / (p.stock_w * p.stock_l)) * 100, 1),
                    "Zawarte pozycje": items_summary,
                })
            df_workshop_2d = pd.DataFrame(workshop_2d_data)
            st.dataframe(df_workshop_2d, use_container_width=True)

    with tab_order:
        st.markdown("### 🛒 Zestawienie Zbiorcze pod Zamówienie Handlowe do Huty")

        all_procurement = order_items_1d + order_items_2d
        df_order = pd.DataFrame(all_procurement)

        if not df_order.empty:
            total_steel_mass_t = df_order["Masa łączna [kg]"].sum() / 1000.0
            st.success(f"📦 Łączna masa zamówienia handlowego stali: **{total_steel_mass_t:.2f} ton**")
            st.dataframe(df_order, use_container_width=True)

            # Tabela podsumowania odpadu
            stats_data = [
                {"Segment": "Profile 1D", "Masa zamówienia [t]": round(sum(x["Masa łączna [kg]"] for x in order_items_1d) / 1000.0, 3) if order_items_1d else 0, "Średni Odpad [%]": round(scrap_1d_pct, 2) if 'scrap_1d_pct' in locals() else 0},
                {"Segment": "Blachy 2D", "Masa zamówienia [t]": round(sum(x["Masa łączna [kg]"] for x in order_items_2d) / 1000.0, 3) if order_items_2d else 0, "Średni Odpad [%]": round(scrap_2d_pct, 2) if 'scrap_2d_pct' in locals() else 0},
            ]
            df_stats = pd.DataFrame(stats_data)

            excel_buffer = build_excel_export(
                procurement_df=df_order,
                cut_summary_1d=df_workshop_1d if 'df_workshop_1d' in locals() else pd.DataFrame(),
                cut_summary_2d=df_workshop_2d if 'df_workshop_2d' in locals() else pd.DataFrame(),
                stats_df=df_stats,
                profile_stats_1d=df_profile_waste if 'df_profile_waste' in locals() else None,
            )

            st.download_button(
                label="📥 Pobierz Gotowe Zamówienie i Karty Cięcia (Excel .xlsx)",
                data=excel_buffer,
                file_name="Zamowienie_Stali_i_Rozkroj.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
        else:
            st.warning("Brak danych do wygenerowania zamówienia handlowego.")
else:
    st.info("👈 Wgraj plik z zestawieniem materiałowym (BOM) lub kliknij **'Załaduj Przykładowy BOM'** po lewej stronie, aby uruchomić silnik optymalizacji.")