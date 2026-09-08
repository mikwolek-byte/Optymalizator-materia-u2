import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# 1. MODELE DANYCH
# ==============================================================================

@dataclass(frozen=True)
class ProfileGroupKey:
    grade: str
    profile: str

@dataclass(frozen=True)
class PlateGroupKey:
    grade: str
    thickness: float

@dataclass
class CutPiece1D:
    id: str
    pos: str
    length: float
    profile: str
    grade: str

@dataclass
class StockBar1D:
    bar_id: int
    stock_length: float
    cuts: List[CutPiece1D]
    cuts_len: float
    kerf: float
    trim_cut: float
    profile: str
    grade: str

    @property
    def free_length(self) -> float:
        if not self.cuts:
            return self.stock_length - self.trim_cut
        return self.stock_length - self.trim_cut - (self.cuts_len + len(self.cuts) * self.kerf)

    @property
    def waste_length(self) -> float:
        return max(0.0, self.free_length)

    @property
    def efficiency(self) -> float:
        return (self.cuts_len / self.stock_length) * 100.0 if self.stock_length > 0 else 0.0

@dataclass
class PlateItem:
    pos: str
    grade: str
    thickness: float
    width: float
    length: float

@dataclass
class PackedRect:
    pos: str
    x: float
    y: float
    w: float
    h: float

@dataclass
class StockPlate:
    sheet_id: int
    grade: str
    thickness: float
    stock_w: float
    stock_l: float
    format_name: str
    packed_items: List[PackedRect] = field(default_factory=list)
    used_area: float = 0.0

    @property
    def total_area(self) -> float:
        return self.stock_w * self.stock_l

    @property
    def scrap_area(self) -> float:
        return max(0.0, self.total_area - self.used_area)

    @property
    def efficiency(self) -> float:
        return (self.used_area / self.total_area * 100.0) if self.total_area > 0 else 0.0


# ==============================================================================
# 2. BAZA WIEDZY I NORMALIZACJA
# ==============================================================================

PROFILE_WEIGHTS: Dict[str, float] = {
    "IPE300": 42.2, "HEA240": 60.3, "HEB600": 212.0, "HEB800": 262.0,
    "HEA180": 35.5, "HEA220": 50.5, "HEA260": 68.2, "HEA280": 76.4, "HEA340": 105.0,
    "HEB160": 42.6, "HEB180": 51.2, "HEB280": 103.0, "HEB340": 134.0,
    "HEM140": 63.2, "HEM180": 88.9, "HEM240": 157.0, "HEM300": 238.0
}

def get_unit_weight(profile_str: str) -> float:
    p = str(profile_str).upper().replace(" ", "").replace(",", ".")
    if p in PROFILE_WEIGHTS: return PROFILE_WEIGHTS[p]
    
    m_he = re.match(r'^HE(\d+)([ABM])$', p)
    if m_he:
        key = f"HE{m_he.group(2)}{m_he.group(1)}"
        if key in PROFILE_WEIGHTS: return PROFILE_WEIGHTS[key]
    
    m_angle = re.match(r'^L(\d+)(?:[X*](\d+))?[X*](\d+(?:\.\d+)?)$', p)
    if m_angle:
        w = float(m_angle.group(1))
        h = float(m_angle.group(2)) if m_angle.group(2) else w
        t = float(m_angle.group(3))
        return round(((w + h - t) * t) * 7.85e-3, 2)
    
    m_rhs = re.match(r'^(?:CF)?(?:RHS|SHS|RK|RP|RKR)(\d+)[X*](\d+)(?:[X*](\d+(?:\.\d+)?))?$', p)
    if m_rhs:
        h, w = float(m_rhs.group(1)), float(m_rhs.group(2))
        t = float(m_rhs.group(3)) if m_rhs.group(3) else (w if not m_rhs.group(3) else 0.0)
        return round((2.0 * t * (h + w - 2.0 * t)) * 7.85e-3, 2)
    
    m_chs = re.match(r'^(?:CHS|RO|ROHR)(\d+(?:\.\d+)?)[X*](\d+(?:\.\d+)?)$', p)
    if m_chs:
        d, t = float(m_chs.group(1)), float(m_chs.group(2))
        return round(3.14159265 * (d - t) * t * 7.85e-3, 2)
    
    return 10.0

def normalize_grade(grade_str: str) -> str:
    if not grade_str or str(grade_str).strip().lower() in ['nan', 'none', '']:
        return "S355JR"
    g = str(grade_str).strip().upper().replace(" ", "")
    if "S355" in g: return "S355J2+N"
    if "S235" in g: return "S235JR"
    return g

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
                    if idx: curr_col = int(idx) - 1
                    data_elem = cell.find('ss:Data', ns)
                    val = data_elem.text if data_elem is not None else ""
                    row_cells[curr_col] = val
                    curr_col += 1
                if row_cells:
                    rows_data.append([row_cells.get(c, "") for c in range(max(row_cells.keys()) + 1)])

            raw_df = pd.DataFrame(rows_data)
            header_idx = None
            for idx, r in raw_df.iterrows():
                row_str = " ".join([str(v).lower() for v in r if v is not None])
                if "profil" in row_str and ("długość" in row_str or "length" in row_str or "materiał" in row_str or "pozycja" in row_str):
                    header_idx = idx
                    break

            if header_idx is not None:
                raw_df.columns = [str(c).strip() for c in raw_df.iloc[header_idx]]
                raw_df = raw_df.iloc[header_idx + 1:].reset_index(drop=True)
            return raw_df.dropna(how='all')
        except Exception:
            pass

    for engine in [None, 'openpyxl', 'xlrd']:
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

    raise ValueError("Nie rozpoznano formatu pliku.")

def clean_and_normalize_bom(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c_clean = str(col).lower().strip()
        if any(k in c_clean for k in ['pozycja', 'position', 'pos', 'mark', 'nr elementu']) and 'pos' not in col_map.values():
            col_map[col] = 'pos'
        elif any(k in c_clean for k in ['profil', 'profile', 'przekrój', 'section']) and 'profile' not in col_map.values():
            col_map[col] = 'profile'
        elif any(k in c_clean for k in ['materiał', 'material', 'gatunek', 'grade']) and 'grade' not in col_map.values():
            col_map[col] = 'grade'
        elif any(k in c_clean for k in ['ilość', 'ilosc', 'qty', 'szt']) and 'qty' not in col_map.values():
            col_map[col] = 'qty'
        elif any(k in c_clean for k in ['długość', 'length', 'l [mm]']) and 'suma' not in c_clean and 'length' not in col_map.values():
            col_map[col] = 'length'
        elif any(k in c_clean for k in ['szerokość', 'width', 'b [mm]']) and 'suma' not in c_clean and 'width' not in col_map.values():
            col_map[col] = 'width'
        elif any(k in c_clean for k in ['grubość', 'thick', 't [mm]']) and 'thick' not in col_map.values():
            col_map[col] = 'thick'

    df_renamed = df.rename(columns=col_map)
    valid_rows = []
    
    for _, row in df_renamed.iterrows():
        pos_val = str(row.get('pos', ''))
        prof_val = str(row.get('profile', '')).strip().upper().replace(" ", "")
        
        if any(w in pos_val.lower() for w in ['suma', 'total']) or not prof_val or prof_val == 'NAN':
            continue

        try:
            q = float(str(row.get('qty', '0')).replace(',', '.').strip())
            l = float(str(row.get('length', '0')).replace(',', '.').strip())
            w = float(str(row.get('width', '0')).replace(',', '.').strip()) if pd.notna(row.get('width')) else 0.0
            t = float(str(row.get('thick', '0')).replace(',', '.').strip()) if pd.notna(row.get('thick')) else 0.0
            grade = normalize_grade(row.get('grade', 'S355JR'))

            if q > 0 and l > 0:
                valid_rows.append({
                    'pos': pos_val if pos_val not in ['', 'None', 'nan'] else f"P_{len(valid_rows)+1}",
                    'profile': prof_val, 'grade': grade, 'qty': int(round(q)), 'length': l, 'width': w, 'thick': t
                })
        except ValueError:
            continue

    return pd.DataFrame(valid_rows)

# ==============================================================================
# 3. OPTYMALIZACJA 1D & 2D
# ==============================================================================

def optimize_1d(pieces: List[CutPiece1D], stock_lengths: List[float], kerf: float, trim_cut: float) -> List[StockBar1D]:
    groups = {}
    for p in pieces:
        groups.setdefault(ProfileGroupKey(p.grade, p.profile), []).append(p)

    all_bars = []
    bar_counter = 1

    for key, grp_pieces in groups.items():
        sorted_pieces = sorted(grp_pieces, key=lambda x: x.length, reverse=True)
        active_bars: List[StockBar1D] = []

        for piece in sorted_pieces:
            best_bar, best_rem = None, float('inf')
            for bar in active_bars:
                if bar.free_length >= (piece.length + kerf):
                    if (bar.free_length - piece.length - kerf) < best_rem:
                        best_rem = bar.free_length - piece.length - kerf
                        best_bar = bar
            
            if best_bar:
                best_bar.cuts.append(piece)
                best_bar.cuts_len += piece.length
            else:
                chosen_sl = min((sl for sl in sorted(stock_lengths) if sl - trim_cut >= piece.length + kerf), default=max(stock_lengths))
                new_bar = StockBar1D(
                    bar_id=bar_counter, stock_length=chosen_sl, cuts=[piece], cuts_len=piece.length,
                    kerf=kerf, trim_cut=trim_cut, profile=key.profile, grade=key.grade
                )
                bar_counter += 1
                active_bars.append(new_bar)
                
        all_bars.extend(active_bars)
    return all_bars

def pack_single_sheet(w: float, l: float, items: List[PlateItem], spacing: float, margin: float, sheet_id: int, format_name: str) -> Tuple[StockPlate, List[PlateItem]]:
    eff_w, eff_l = w - 2*margin, l - 2*margin
    plate = StockPlate(sheet_id=sheet_id, grade=items[0].grade, thickness=items[0].thickness, stock_w=w, stock_l=l, format_name=format_name)
    
    if eff_w <= 0 or eff_l <= 0: return plate, items
    
    unplaced, shelves = [], []
    for item in items:
        placed = False
        orientations = [(item.width, item.length), (item.length, item.width)] if item.width != item.length else [(item.width, item.length)]
        
        for o_w, o_l in orientations:
            if o_w > eff_w or o_l > eff_l: continue
            
            for shelf in shelves:
                if o_l <= shelf["h"] and (shelf["x"] + o_w) <= eff_w:
                    plate.packed_items.append(PackedRect(item.pos, margin + shelf["x"], margin + shelf["y"], o_w, o_l))
                    shelf["x"] += o_w + spacing
                    plate.used_area += (o_w * o_l)
                    placed = True
                    break
            if placed: break
            
            last_y = (shelves[-1]["y"] + shelves[-1]["h"] + spacing) if shelves else 0.0
            if (last_y + o_l) <= eff_l and o_w <= eff_w:
                shelves.append({"y": last_y, "h": o_l, "x": o_w + spacing})
                plate.packed_items.append(PackedRect(item.pos, margin, margin + last_y, o_w, o_l))
                plate.used_area += (o_w * o_l)
                placed = True
                break
                
        if not placed: unplaced.append(item)
        
    return plate, unplaced

def optimize_2d_multi_format(plates: List[PlateItem], formats: List[Tuple[float, float, str]], margin: float, spacing: float) -> List[StockPlate]:
    groups = {}
    for p in plates:
        groups.setdefault(PlateGroupKey(p.grade, p.thickness), []).append(p)

    all_sheets = []
    sheet_id = 1

    for key, grp_plates in groups.items():
        sorted_p = sorted(grp_plates, key=lambda x: max(x.length, x.width), reverse=True)
        unplaced = sorted_p
        
        while unplaced:
            best_plate, best_unplaced, min_scrap = None, None, float('inf')
            
            for w, l, fname in formats:
                plate, rem = pack_single_sheet(w, l, unplaced, spacing, margin, sheet_id, fname)
                if len(rem) < len(unplaced) or (len(rem) == 0 and plate.scrap_area < min_scrap):
                    if plate.scrap_area < min_scrap:
                        min_scrap = plate.scrap_area
                        best_plate = plate
                        best_unplaced = rem
            
            if not best_plate: 
                break # Oversized items
            
            all_sheets.append(best_plate)
            unplaced = best_unplaced
            sheet_id += 1

    return all_sheets

# ==============================================================================
# 4. MODUŁ STYKOWANIA PROFILI
# ==============================================================================

def get_passes(t: float) -> int:
    if t <= 5.6: return 2
    if t <= 10.0: return 3
    if t <= 15.0: return 4
    if t <= 17.0: return 5
    if t <= 20.0: return 6
    if t <= 25.0: return 8
    if t <= 30.0: return 12
    if t <= 35.0: return 18
    return 22

def guess_dims(prof: str) -> Tuple[float, float, float, float]:
    # Uproszczona estymata dla HEA/HEB/IPE do celów szybkiej kalkulacji (h, b, tw, tf) w mm
    p = prof.upper()
    if "HEB600" in p: return 600.0, 300.0, 15.5, 30.0
    if "HEB800" in p: return 800.0, 300.0, 17.5, 33.0
    if "HEA240" in p: return 230.0, 240.0, 7.5, 12.0
    if "IPE300" in p: return 300.0, 150.0, 7.1, 10.7
    
    m = re.search(r'(\d+)', p)
    if m:
        h = float(m.group(1))
        b = h / 2.0 if "IPE" in p else h
        return h, b, max(5.0, h*0.02), max(8.0, h*0.03)
    return 200.0, 100.0, 6.0, 10.0

def calc_splice_cost(prof: str, rate_h: float, rate_ut: float) -> dict:
    h, b, tw, tf = guess_dims(prof)
    
    L_weld_web = 2.0 * (h / 1000.0)
    L_weld_flange = 2.0 * (b / 1000.0)
    L_skladanie = L_weld_web + L_weld_flange
    L_ut = (h + 2*b) / 1000.0
    
    t_cut = 20.0
    t_assemble = 20.0 * L_skladanie
    t_weld_web = 20.0 * get_passes(tw) * L_weld_web
    t_weld_flange = 20.0 * get_passes(tf) * L_weld_flange
    
    t_total = t_cut + t_assemble + t_weld_web + t_weld_flange
    cost_labor = t_total * (rate_h / 60.0)
    cost_ut = L_ut * rate_ut
    total_cost = cost_labor + cost_ut
    
    return {
        "Czas [min]": round(t_total, 1),
        "Koszt Robocizny [PLN]": round(cost_labor, 2),
        "Koszt UT [PLN]": round(cost_ut, 2),
        "Koszt Styku Całkowity [PLN]": round(total_cost, 2)
    }

# ==============================================================================
# 5. GENERATOR EXCELA (EKSPORT)
# ==============================================================================

def build_excel_export(df_order, df_workshop_1d, df_prof_summary, df_kpi, df_splicing=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        if not df_order.empty: df_order.to_excel(writer, sheet_name="Do Zamówienia", index=False)
        if not df_workshop_1d.empty: df_workshop_1d.to_excel(writer, sheet_name="Rozkrój Warsztat 1D", index=False)
        if not df_prof_summary.empty: df_prof_summary.to_excel(writer, sheet_name="Odpad wg Profili 1D", index=False)
        if df_splicing is not None and not df_splicing.empty: df_splicing.to_excel(writer, sheet_name="Analiza Stykowania", index=False)
        if not df_kpi.empty: df_kpi.to_excel(writer, sheet_name="Statystyka", index=False)
    return buf.getvalue()

def plot_2d_plate_plan(sheet: StockPlate):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.add_patch(patches.Rectangle((0, 0), sheet.stock_l, sheet.stock_w, edgecolor='#0F172A', facecolor='#F1F5F9'))

    for item in sheet.packed_items:
        ax.add_patch(patches.Rectangle((item.y, item.x), item.h, item.w, linewidth=1, edgecolor='#1E3A8A', facecolor='#60A5FA', alpha=0.85))
        ax.text(item.y + item.h / 2, item.x + item.w / 2, f"{item.pos}\n{item.h:.0f}x{item.w:.0f}",
                ha='center', va='center', fontsize=7, color='black', weight='bold')

    ax.set_xlim(-100, sheet.stock_l + 100)
    ax.set_ylim(-100, sheet.stock_w + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    st.caption(f"**Arkusz #{sheet.sheet_id}**: {sheet.format_name} | #{sheet.thickness:.0f} mm | {sheet.grade} | Wykorzystanie: **{sheet.efficiency:.1f}%**")
    st.pyplot(fig)
    plt.close(fig)

# ==============================================================================
# 6. INTERFEJS STREAMLIT
# ==============================================================================

st.set_page_config(page_title="SteelOpt - Optymalizator Hutniczy", layout="wide", page_icon="🏗️")

st.markdown("""
<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
    <h2 style="color: #F8FAFC; margin: 0;">🏗️ SteelOpt: Optymalizator Rozkroju i Stykowania</h2>
</div>
""", unsafe_allow_html=True)

# Pasek boczny
st.sidebar.header("⚙️ Konfiguracja Cięcia 1D")
kerf = st.sidebar.number_input("Rzaz piły [mm]:", value=4.5, step=0.5)
trim_cut = st.sidebar.number_input("Naddatek (odcięcie) [mm]:", value=40.0, step=5.0)

s6 = st.sidebar.checkbox("6 000 mm", value=True)
s12 = st.sidebar.checkbox("12 100 mm", value=True)
s15 = st.sidebar.checkbox("15 100 mm", value=True)
chosen_stocks = [sl for b, sl in zip([s6, s12, s15], [6000.0, 12100.0, 15100.0]) if b] or [12100.0]

st.sidebar.header("⚙️ Formaty Blach 2D")
plate_formats = st.sidebar.multiselect(
    "Dostępne formaty:",
    ["1500x3000", "1500x6000", "2000x6000", "2000x12000", "2500x6000", "2500x12000"],
    default=["2000x6000", "1500x3000"]
)
parsed_formats = [(float(f.split('x')[0]), float(f.split('x')[1]), f) for f in plate_formats]
edge_margin = st.sidebar.number_input("Margines od krawędzi [mm]:", value=20.0)
spacing = st.sidebar.number_input("Odstęp termiczny [mm]:", value=12.0)

st.sidebar.header("⚙️ Stykowanie Profili")
use_splicing = st.sidebar.checkbox("Aktywuj moduł stykowania (BEP)", value=True)
rate_h = st.sidebar.number_input("Stawka RBH [PLN/h]:", value=130.0)
rate_ut = st.sidebar.number_input("Badania UT [PLN/mb]:", value=70.0)
price_steel = st.sidebar.number_input("Cena stali [PLN/kg]:", value=3.60)
price_scrap = st.sidebar.number_input("Cena złomu [PLN/kg]:", value=1.20)

# Import
uploaded_file = st.file_uploader("Wgraj zestawienie materiałowe (BOM):", type=["xlsx", "xls", "csv"])

if uploaded_file:
    raw_df = parse_bom_file(uploaded_file)
    clean_df = clean_and_normalize_bom(raw_df)

    pieces_1d, plates_2d = [], []
    for _, r in clean_df.iterrows():
        prof = str(r['profile']).upper()
        if r['width'] > 0 and r['thick'] > 0 and any(p in prof for p in ['BLACHA', 'PL', 'FORMATKA']):
            for _ in range(int(r['qty'])):
                plates_2d.append(PlateItem(r['pos'], r['grade'], r['thick'], r['width'], r['length']))
        else:
            for i in range(int(r['qty'])):
                pieces_1d.append(CutPiece1D(f"{r['pos']}_{i+1}", r['pos'], r['length'], prof, r['grade']))

    bars_1d = optimize_1d(pieces_1d, chosen_stocks, kerf, trim_cut) if pieces_1d else []
    sheets_2d = optimize_2d_multi_format(plates_2d, parsed_formats, edge_margin, spacing) if plates_2d else []

    # KPI 1D
    prof_summary_rows = []
    for (grade, prof), grp in pd.DataFrame([b.__dict__ for b in bars_1d]).groupby(['grade', 'profile']) if bars_1d else []:
        b_list = [b for b in bars_1d if b.grade == grade and b.profile == prof]
        uw = get_unit_weight(prof)
        tot_cut_m = sum(c.length for b in b_list for c in b.cuts) / 1000.0
        tot_stock_m = sum(b.stock_length for b in b_list) / 1000.0
        waste_m = tot_stock_m - tot_cut_m
        
        prof_summary_rows.append({
            "Profil": prof, "Gatunek": grade, "Masa 1mb [kg]": uw, "Liczba sztang": len(b_list),
            "Dł. netto [m]": round(tot_cut_m, 2), "Dł. brutto [m]": round(tot_stock_m, 2),
            "Odpad [m]": round(waste_m, 2),
            "Masa Netto [kg]": round(tot_cut_m * uw, 1),
            "Masa Brutto [kg]": round(tot_stock_m * uw, 1),
            "Odpad [kg]": round(waste_m * uw, 1),
            "Odpad [%]": f"{(waste_m/tot_stock_m*100) if tot_stock_m>0 else 0:.1f}%"
        })
    
    df_prof_summary = pd.DataFrame(prof_summary_rows)
    
    net_mass_1d = sum(p.get("Masa netto [kg]", p.get("Masa Netto [kg]", 0.0)) for p in prof_summary_rows)
    gross_mass_1d = sum(p.get("Masa brutto [kg]", p.get("Masa Brutto [kg]", 0.0)) for p in prof_summary_rows)

    net_mass_2d = sum(s.used_area / 1e6 * s.thickness * 7.85 for s in sheets_2d)
    gross_mass_2d = sum(s.total_area / 1e6 * s.thickness * 7.85 for s in sheets_2d)

    tot_net, tot_gross = net_mass_1d + net_mass_2d, gross_mass_1d + gross_mass_2d
    
    st.write(f"### Zapotrzebowanie: {tot_gross/1000:.2f} t brutto | Odpad całkowity: {((tot_gross-tot_net)/tot_gross*100) if tot_gross else 0:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛒 Lista Zakupowa", "📊 Warsztat 1D", "📋 Nesting 2D", "⚙️ Stykowanie (BEP)", "📥 Eksport Excel"])

    with tab1:
        st.dataframe(df_prof_summary, use_container_width=True)

    with tab2:
        for b in bars_1d[:30]:
            fig, ax = plt.subplots(figsize=(10, 0.8))
            cx = b.trim_cut
            ax.broken_barh([(0, b.trim_cut)], (0, 8), facecolors='#EF4444')
            for c in b.cuts:
                ax.broken_barh([(cx, c.length)], (0, 8), facecolors='#3B82F6', edgecolor='black')
                ax.text(cx + c.length/2, 4, f"{c.pos}", ha='center', va='center', color='white', fontsize=8)
                cx += c.length + b.kerf
            ax.broken_barh([(cx, b.stock_length - cx)], (0, 8), facecolors='#CBD5E1', hatch='//')
            ax.set_xlim(0, b.stock_length); ax.set_ylim(-1, 9); ax.axis('off')
            st.pyplot(fig); plt.close(fig)

    with tab3:
        for s in sheets_2d[:15]:
            plot_2d_plate_plan(s)

    with tab4:
        if use_splicing and bars_1d:
            st.subheader("Analiza Progu Rentowności (BEP) dla złączy spawanych")
            splicing_data = []
            unique_profs = set(b.profile for b in bars_1d)
            for prf in unique_profs:
                cost_data = calc_splice_cost(prf, rate_h, rate_ut)
                c_styk = cost_data["Koszt Styku Całkowity [PLN]"]
                uw = get_unit_weight(prf)
                # BEP (kg) = C_styk / (Cena_stali - Cena_zlomu)
                bep_kg = c_styk / (price_steel - price_scrap)
                bep_m = bep_kg / uw if uw > 0 else 0
                splicing_data.append({
                    "Profil": prf,
                    "Koszt Styku [PLN]": c_styk,
                    "Czas [RBH]": round(cost_data["Czas [min]"]/60, 2),
                    "Odpad krytyczny (BEP) [kg]": round(bep_kg, 1),
                    "Dł. krytyczna (BEP) [m]": round(bep_m, 2),
                    "Decyzja": f"Stykuj jeśli odpad > {round(bep_m, 2)} m"
                })
            df_splice = pd.DataFrame(splicing_data)
            st.dataframe(df_splice, use_container_width=True)
        else:
            st.info("Moduł stykowania jest wyłączony lub brak profili 1D.")

    with tab5:
        st.write("Generowanie raportu...")
        df_order = df_prof_summary.copy()
        df_kpi = pd.DataFrame([{"Masa Brutto [t]": tot_gross/1000, "Masa Netto [t]": tot_net/1000}])
        df_w1d = pd.DataFrame([{"Nr": b.bar_id, "Profil": b.profile, "Sztanga": b.stock_length, "Odpad": b.waste_length} for b in bars_1d])
        
        df_splice_export = df_splice if use_splicing and 'df_splice' in locals() else pd.DataFrame()

        xls_data = build_excel_export(df_order, df_w1d, df_prof_summary, df_kpi, df_splice_export)
        st.download_button("📥 Pobierz Pełny Raport (.xlsx)", data=xls_data, file_name="Raport_SteelOpt.xlsx", mime="application/vnd.ms-excel")
