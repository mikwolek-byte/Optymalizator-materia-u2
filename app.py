import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ==============================================================================
# 1. KATALOG MAS I KLASYFIKACJA ASORTYMENTOWA
# ==============================================================================

PROFILE_WEIGHTS: Dict[str, float] = {
    # IPE
    "IPE80": 6.0, "IPE100": 8.1, "IPE120": 10.4, "IPE140": 12.9, "IPE160": 15.8,
    "IPE180": 18.8, "IPE200": 22.4, "IPE220": 26.2, "IPE240": 30.7, "IPE270": 36.1,
    "IPE300": 42.2, "IPE330": 49.1, "IPE360": 57.1, "IPE400": 66.3, "IPE450": 77.6,
    "IPE500": 90.7, "IPE550": 106.0, "IPE600": 122.0,
    # HEA
    "HEA100": 16.7, "HEA120": 19.9, "HEA140": 24.7, "HEA160": 30.4, "HEA180": 35.5,
    "HEA200": 42.3, "HEA220": 50.5, "HEA240": 60.3, "HEA260": 68.2, "HEA280": 76.4,
    "HEA300": 88.3, "HEA320": 97.6, "HEA340": 105.0, "HEA360": 112.0, "HEA400": 125.0,
    "HEA450": 140.0, "HEA500": 155.0, "HEA550": 166.0, "HEA600": 178.0, "HEA650": 190.0,
    "HEA700": 204.0, "HEA800": 224.0, "HEA900": 252.0, "HEA1000": 272.0,
    # HEB
    "HEB100": 20.4, "HEB120": 26.7, "HEB140": 33.7, "HEB160": 42.6, "HEB180": 51.2,
    "HEB200": 61.3, "HEB220": 71.5, "HEB240": 83.2, "HEB260": 93.0, "HEB280": 103.0,
    "HEB300": 117.0, "HEB320": 127.0, "HEB340": 134.0, "HEB360": 142.0, "HEB400": 155.0,
    "HEB450": 171.0, "HEB500": 187.0, "HEB550": 199.0, "HEB600": 212.0, "HEB650": 225.0,
    "HEB700": 241.0, "HEB800": 262.0, "HEB900": 291.0, "HEB1000": 314.0,
    # HEM
    "HEM100": 41.8, "HEM120": 52.1, "HEM140": 63.2, "HEM160": 76.2, "HEM180": 88.9,
    "HEM200": 103.0, "HEM220": 117.0, "HEM240": 157.0, "HEM260": 172.0, "HEM280": 189.0,
    "HEM300": 238.0, "HEM320": 245.0, "HEM340": 248.0, "HEM360": 250.0, "HEM400": 256.0,
    "HEM450": 263.0, "HEM500": 270.0, "HEM550": 277.0, "HEM600": 285.0,
    # UNP / UPN
    "UNP80": 8.64, "UNP100": 10.6, "UNP120": 13.4, "UNP140": 16.0, "UNP160": 18.8,
    "UNP180": 22.0, "UNP200": 25.3, "UNP220": 29.4, "UNP240": 33.2, "UNP260": 37.9,
    "UNP280": 41.8, "UNP300": 46.2, "UNP320": 59.5, "UNP350": 60.6, "UNP380": 63.1,
    "UNP400": 71.8,
    # UPE
    "UPE80": 7.9, "UPE100": 9.82, "UPE120": 12.1, "UPE140": 14.5, "UPE160": 17.0,
    "UPE180": 19.7, "UPE200": 22.8, "UPE220": 26.6, "UPE240": 30.2, "UPE270": 35.2,
    "UPE300": 44.4, "UPE330": 53.2, "UPE360": 61.2, "UPE400": 72.2
}

def get_unit_weight(profile_str: str) -> float:
    p = str(profile_str).upper().replace(" ", "").replace(",", ".")
    
    if p in PROFILE_WEIGHTS:
        return PROFILE_WEIGHTS[p]

    m_he = re.match(r'^HE(\d+)([ABM])$', p)
    if m_he:
        num, letter = m_he.groups()
        key = f"HE{letter}{num}"
        if key in PROFILE_WEIGHTS:
            return PROFILE_WEIGHTS[key]

    # Kątowniki: L100x100x10, L100*10, L50*4
    m_angle = re.match(r'^L(\d+)(?:[X*](\d+))?[X*](\d+(?:\.\d+)?)$', p)
    if m_angle:
        w = float(m_angle.group(1))
        h = float(m_angle.group(2)) if m_angle.group(2) else w
        t = float(m_angle.group(3))
        area_mm2 = (w + h - t) * t
        return round(area_mm2 * 7.85e-3, 2)

    # Rury prostokątne i kwadratowe: CFRHS300*150*8, RHS200*100*6, RK100*100*5, RP120*60*4
    m_rhs = re.match(r'^(?:CF)?(?:RHS|SHS|RK|RP|RKR)(\d+)[X*](\d+)(?:[X*](\d+(?:\.\d+)?))?$', p)
    if m_rhs:
        h = float(m_rhs.group(1))
        w = float(m_rhs.group(2))
        t = float(m_rhs.group(3)) if m_rhs.group(3) else 0.0
        if t == 0.0 and m_rhs.group(3) is None:
            t = w
            w = h
        area_mm2 = 2.0 * t * (h + w - 2.0 * t)
        return round(area_mm2 * 7.85e-3, 2)

    # Rury okrągłe: ROHR114.3*5, CHS88.9*4, RO60.3*3.2
    m_chs = re.match(r'^(?:CHS|RO|ROHR)(\d+(?:\.\d+)?)[X*](\d+(?:\.\d+)?)$', p)
    if m_chs:
        d = float(m_chs.group(1))
        t = float(m_chs.group(2))
        return round(3.14159265 * (d - t) * t * 7.85e-3, 2)

    # Pręty okrągłe: D30, FI30, Ø30
    m_bar = re.match(r'^(?:D|FI|Ø)(\d+(?:\.\d+)?)$', p)
    if m_bar:
        d = float(m_bar.group(1))
        area = 3.14159265 * ((d / 2.0) ** 2)
        return round(area * 7.85e-3, 2)

    # Płaskowniki / pasy: BL20*80, PD40*5, FL80*10, PL20*80
    m_flat = re.match(r'^(?:BL|PD|FL|PL|P)(\d+(?:\.\d+)?)[X*](\d+(?:\.\d+)?)$', p)
    if m_flat:
        t = float(m_flat.group(1))
        w = float(m_flat.group(2))
        return round(t * w * 7.85e-3, 2)

    return 10.0


# ==============================================================================
# 2. PARSER PLIKÓW (XML 2003, EXCEL, CSV)
# ==============================================================================

def parse_bom_file(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    prefix = raw_bytes[:1500].strip()

    # 1. Obsługa formatu Tekla / Advance Steel XML Spreadsheet 2003 (rozszerzenie .xls)
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

            # Znalezienie wiersza nagłówkowego
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

    # 2. Standardowy plik Excel (.xlsx, .xls)
    for engine in [None, 'openpyxl', 'xlrd']:
        try:
            if engine:
                return pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
            return pd.read_excel(io.BytesIO(raw_bytes))
        except Exception:
            continue

    # 3. Pliki HTML podszywające się pod XLS
    try:
        dfs = pd.read_html(io.BytesIO(raw_bytes))
        if dfs:
            return dfs[0]
    except Exception:
        pass

    # 4. Pliki tekstowe CSV
    for enc in ['utf-8', 'windows-1250', 'iso-8859-2']:
        for sep in [';', ',', '\t']:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=enc)
            except Exception:
                continue

    raise ValueError("Nie udało się rozpoznać formatu pliku.")


def clean_and_normalize_bom(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c_clean = str(col).lower().strip()
        if any(k in c_clean for k in ['pozycja', 'position', 'pos', 'mark', 'nr elementu', 'nr części']) and 'pos' not in col_map.values():
            col_map[col] = 'pos'
        elif any(k in c_clean for k in ['profil', 'profile', 'przekrój', 'section', 'asortyment']) and 'profile' not in col_map.values():
            col_map[col] = 'profile'
        elif any(k in c_clean for k in ['materiał', 'material', 'gatunek', 'grade', 'stal']) and 'grade' not in col_map.values():
            col_map[col] = 'grade'
        elif any(k in c_clean for k in ['ilość', 'ilosc', 'quantity', 'qty', 'szt', 'liczba']) and 'qty' not in col_map.values():
            col_map[col] = 'qty'
        elif any(k in c_clean for k in ['długość', 'dlugosc', 'length', 'l [mm]']) and not any(k in c_clean for k in ['całk', 'total', 'suma']) and 'length' not in col_map.values():
            col_map[col] = 'length'
        elif any(k in c_clean for k in ['szerokość', 'szerokosc', 'width', 'b [mm]']) and not any(k in c_clean for k in ['całk', 'total', 'suma']) and 'width' not in col_map.values():
            col_map[col] = 'width'
        elif any(k in c_clean for k in ['grubość', 'grubosc', 'thick', 't [mm]']) and 'thick' not in col_map.values():
            col_map[col] = 'thick'

    df_renamed = df.rename(columns=col_map)

    valid_rows = []
    for _, row in df_renamed.iterrows():
        pos_val = str(row.get('pos', ''))
        prof_val = str(row.get('profile', '')).strip()
        qty_val = str(row.get('qty', ''))
        len_val = str(row.get('length', ''))

        # Pomijanie wierszy podsumowujących raporty Tekli
        if any(w in pos_val.lower() for w in ['suma', 'total']) or any(w in len_val.lower() for w in ['suma', 'total']):
            continue

        if prof_val in ['', 'None', 'nan']:
            continue

        try:
            q = float(str(qty_val).replace(',', '.').strip())
            l = float(str(len_val).replace(',', '.').strip())
            w = float(str(row.get('width', 0.0)).replace(',', '.').strip()) if pd.notna(row.get('width')) else 0.0
            t = float(str(row.get('thick', 0.0)).replace(',', '.').strip()) if pd.notna(row.get('thick')) else 0.0
            grade = str(row.get('grade', 'S355JR')).strip()
            if grade in ['', 'None', 'nan']:
                grade = 'S355JR'

            if q > 0 and l > 0:
                valid_rows.append({
                    'pos': pos_val if pos_val not in ['', 'None', 'nan'] else f"P_{len(valid_rows)+1}",
                    'profile': prof_val,
                    'grade': grade,
                    'qty': int(round(q)),
                    'length': l,
                    'width': w,
                    'thick': t
                })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(valid_rows)


# ==============================================================================
# 3. SILNIK OPTYMALIZACJI 1D (ROZKRÓJ PROFILI ZE SZTANG)
# ==============================================================================

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


def optimize_1d(pieces: List[CutPiece1D], stock_lengths: List[float], kerf: float, trim_cut: float) -> List[StockBar1D]:
    groups = {}
    for p in pieces:
        groups.setdefault((p.profile, p.grade), []).append(p)

    all_bars = []
    bar_counter = 1

    for (prof, grade), grp_pieces in groups.items():
        sorted_pieces = sorted(grp_pieces, key=lambda x: x.length, reverse=True)
        active_bars: List[StockBar1D] = []

        for piece in sorted_pieces:
            best_bar = None
            best_rem = float('inf')

            for bar in active_bars:
                needed = piece.length + kerf
                if bar.free_length >= needed:
                    rem = bar.free_length - needed
                    if rem < best_rem:
                        best_rem = rem
                        best_bar = bar

            if best_bar:
                best_bar.cuts.append(piece)
                best_bar.cuts_len += piece.length
            else:
                chosen_sl = None
                for sl in sorted(stock_lengths):
                    if (sl - trim_cut) >= (piece.length + kerf):
                        chosen_sl = sl
                        break
                if not chosen_sl:
                    chosen_sl = max(stock_lengths)

                new_bar = StockBar1D(
                    bar_id=bar_counter,
                    stock_length=chosen_sl,
                    cuts=[piece],
                    cuts_len=piece.length,
                    kerf=kerf,
                    trim_cut=trim_cut,
                    profile=prof,
                    grade=grade
                )
                bar_counter += 1
                active_bars.append(new_bar)

        all_bars.extend(active_bars)

    return all_bars


# ==============================================================================
# 4. SILNIK OPTYMALIZACJI 2D (NESTING FORMATER BLACH)
# ==============================================================================

@dataclass
class PlateItem:
    id: str
    pos: str
    length: float
    width: float
    thick: float
    grade: str

@dataclass
class StockPlateSheet:
    sheet_id: int
    length: float
    width: float
    thick: float
    grade: str
    placed: List[Tuple[PlateItem, float, float, float, float]]  # item, x, y, w, h
    used_area: float

    @property
    def total_area(self) -> float:
        return self.length * self.width

    @property
    def efficiency(self) -> float:
        return (self.used_area / self.total_area) * 100.0 if self.total_area > 0 else 0.0


def optimize_2d(plates: List[PlateItem], stock_formats: List[Tuple[float, float]], edge_margin: float, spacing: float) -> List[StockPlateSheet]:
    if not plates:
        return []

    groups = {}
    for p in plates:
        groups.setdefault((p.thick, p.grade), []).append(p)

    all_sheets = []
    sheet_id_counter = 1

    for (thick, grade), grp_plates in groups.items():
        sorted_p = sorted(grp_plates, key=lambda x: max(x.length, x.width), reverse=True)
        sw, sl = stock_formats[0]

        curr_sheet = StockPlateSheet(sheet_id=sheet_id_counter, length=sl, width=sw, thick=thick, grade=grade, placed=[], used_area=0.0)
        sheet_id_counter += 1

        shelves = []
        usable_w = sw - 2 * edge_margin
        usable_l = sl - 2 * edge_margin
        curr_y = edge_margin

        for p in sorted_p:
            pw, pl = min(p.width, p.length), max(p.width, p.length)
            placed = False

            for s in shelves:
                if s['height'] >= pw and (usable_l - s['curr_x']) >= pl:
                    curr_sheet.placed.append((p, s['curr_x'], s['y'], pl, pw))
                    s['curr_x'] += pl + spacing
                    curr_sheet.used_area += p.length * p.width
                    placed = True
                    break
                elif s['height'] >= pl and (usable_l - s['curr_x']) >= pw:
                    curr_sheet.placed.append((p, s['curr_x'], s['y'], pw, pl))
                    s['curr_x'] += pw + spacing
                    curr_sheet.used_area += p.length * p.width
                    placed = True
                    break

            if not placed:
                shelf_h = pw
                if (curr_y + shelf_h <= sw - edge_margin) and (pl <= usable_l):
                    shelves.append({'y': curr_y, 'height': shelf_h, 'curr_x': edge_margin + pl + spacing})
                    curr_sheet.placed.append((p, edge_margin, curr_y, pl, pw))
                    curr_sheet.used_area += p.length * p.width
                    curr_y += shelf_h + spacing
                    placed = True

            if not placed:
                all_sheets.append(curr_sheet)
                curr_sheet = StockPlateSheet(sheet_id=sheet_id_counter, length=sl, width=sw, thick=thick, grade=grade, placed=[], used_area=0.0)
                sheet_id_counter += 1
                shelves = []
                curr_y = edge_margin
                shelf_h = pw
                shelves.append({'y': curr_y, 'height': shelf_h, 'curr_x': edge_margin + pl + spacing})
                curr_sheet.placed.append((p, edge_margin, curr_y, pl, pw))
                curr_sheet.used_area += p.length * p.width
                curr_y += shelf_h + spacing

        all_sheets.append(curr_sheet)

    return all_sheets


# ==============================================================================
# 5. INTERFEJS UŻYTKOWNIKA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="SteelOpt - Optymalizator Hutniczy", layout="wide", page_icon="🏗️")

st.markdown("""
<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
    <h2 style="color: #F8FAFC; margin: 0;">🏗️ SteelOpt: Optymalizator Rozkroju Stali i Zamówień Hutniczych</h2>
    <p style="color: #94A3B8; margin: 5px 0 0 0;">Obsługa Tekla Structures, Advance Steel, Bocad | Profile IPE, HEA, HEB, HEM, UNP, UPE, Rury, Kątowniki | Blachy 2D</p>
</div>
""", unsafe_allow_html=True)

# Pasek boczny: Parametry technologiczne cięcia
st.sidebar.header("⚙️ Parametry Cięcia Hutniczego")
kerf = st.sidebar.number_input("Rzaz piły taśmowej [mm]:", min_value=1.0, max_value=15.0, value=4.5, step=0.5)
trim_cut = st.sidebar.number_input("Naddatek końców fabrycznych [mm]:", min_value=0.0, max_value=100.0, value=40.0, step=5.0)

st.sidebar.subheader("Długości handlowe sztang [mm]")
s12 = st.sidebar.checkbox("12 000 mm", value=True)
s121 = st.sidebar.checkbox("12 100 mm", value=True)
s14 = st.sidebar.checkbox("14 000 mm", value=True)
s15 = st.sidebar.checkbox("15 000 mm", value=True)

chosen_stocks = []
if s12: chosen_stocks.append(12000.0)
if s121: chosen_stocks.append(12100.0)
if s14: chosen_stocks.append(14000.0)
if s15: chosen_stocks.append(15000.0)
if not chosen_stocks:
    chosen_stocks = [12000.0]

st.sidebar.subheader("Formaty arkuszy blach [mm]")
plate_format_choice = st.sidebar.selectbox("Standardowy arkusz:", ["2000 x 6000", "1500 x 6000", "1500 x 3000", "2000 x 12000"])
pf_w, pf_l = [float(x.strip()) for x in plate_format_choice.split("x")]
edge_margin = st.sidebar.number_input("Margines od krawędzi blachy [mm]:", value=20.0, step=5.0)
spacing = st.sidebar.number_input("Odstęp cięcia termicznego [mm]:", value=12.0, step=2.0)

# Import danych
c1, c2 = st.columns([3, 1])
with c1:
    uploaded_file = st.file_uploader("Wgraj plik z zestawieniem materiałowym (BOM):", type=["xlsx", "xls", "csv"])
with c2:
    st.write("")
    st.write("")
    use_demo = st.button("🚀 Załaduj Przykładowy BOM")

raw_df = None
if uploaded_file is not None:
    try:
        raw_df = parse_bom_file(uploaded_file)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
elif use_demo:
    raw_df = pd.DataFrame([
        {"Pos": "B1", "Profile": "IPE300", "Grade": "S355J2+N", "Qty": 6, "Length_mm": 5420, "Width_mm": 0, "Thick_mm": 0},
        {"Pos": "B2", "Profile": "IPE300", "Grade": "S355J2+N", "Qty": 8, "Length_mm": 6250, "Width_mm": 0, "Thick_mm": 0},
        {"Pos": "C1", "Profile": "HEA240", "Grade": "S355J2+N", "Qty": 8, "Length_mm": 4150, "Width_mm": 0, "Thick_mm": 0},
        {"Pos": "R1", "Profile": "UNP160", "Grade": "S235JR", "Qty": 14, "Length_mm": 2950, "Width_mm": 0, "Thick_mm": 0},
        {"Pos": "K1", "Profile": "L100x100x10", "Grade": "S235JR", "Qty": 12, "Length_mm": 3800, "Width_mm": 0, "Thick_mm": 0},
        {"Pos": "PL1", "Profile": "BLACHA", "Grade": "S355J2+N", "Qty": 16, "Length_mm": 650, "Width_mm": 450, "Thick_mm": 20},
        {"Pos": "PL2", "Profile": "BLACHA", "Grade": "S355J2+N", "Qty": 32, "Length_mm": 350, "Width_mm": 250, "Thick_mm": 20},
    ])

if raw_df is not None:
    clean_df = clean_and_normalize_bom(raw_df)

    if clean_df.empty:
        st.warning("Nie znaleziono pozycji o prawidłowych wymiarach i ilościach.")
    else:
        st.success(f"Pomyślnie przetworzono {len(clean_df)} unikalnych pozycji zestawienia materiałowego.")

        # Rozdzielenie na elementy 1D i 2D
        pieces_1d: List[CutPiece1D] = []
        plates_2d: List[PlateItem] = []

        for _, r in clean_df.iterrows():
            prof = str(r['profile']).upper().strip()
            w = float(r['width'])
            t = float(r['thick'])
            l = float(r['length'])
            q = int(r['qty'])
            pos = str(r['pos'])
            grd = str(r['grade'])

            # Blachy formatowe (2D) to elementy o podanej szerokości i grubości oraz profilu typu BLACHA/PL
            if w > 0 and t > 0 and any(p in prof for p in ['BLACHA', 'PL', 'FORMATKA']):
                for i in range(q):
                    plates_2d.append(PlateItem(id=f"{pos}_{i+1}", pos=pos, length=l, width=w, thick=t, grade=grd))
            else:
                for i in range(q):
                    pieces_1d.append(CutPiece1D(id=f"{pos}_{i+1}", pos=pos, length=l, profile=prof, grade=grd))

        # Obliczenia optymalizacyjne
        bars_1d = optimize_1d(pieces_1d, chosen_stocks, kerf, trim_cut) if pieces_1d else []
        sheets_2d = optimize_2d(plates_2d, [(pf_w, pf_l)], edge_margin, spacing) if plates_2d else []

        # ==============================================================================
        # BILANS ODPADU I STATYSTYKI
        # ==============================================================================
        
        # 1D: Zestawienie odpadu per profil
        prof_summary_rows = []
        total_bars_gross_mass = 0.0
        total_bars_net_mass = 0.0

        if bars_1d:
            prof_grade_groups = {}
            for b in bars_1d:
                prof_grade_groups.setdefault((b.profile, b.grade), []).append(b)

            for (prof, grd), grp_bars in prof_grade_groups.items():
                uw = get_unit_weight(prof)
                n_bars = len(grp_bars)
                tot_cut_m = sum(c.length for b in grp_bars for c in b.cuts) / 1000.0
                tot_stock_m = sum(b.stock_length for b in grp_bars) / 1000.0
                waste_m = tot_stock_m - tot_cut_m
                waste_pct = (waste_m / tot_stock_m * 100.0) if tot_stock_m > 0 else 0.0

                net_m_kg = tot_cut_m * uw
                gross_m_kg = tot_stock_m * uw
                waste_m_kg = waste_m * uw

                total_bars_net_mass += net_m_kg
                total_bars_gross_mass += gross_m_kg

                prof_summary_rows.append({
                    "Profil": prof,
                    "Gatunek": grd,
                    "Masa 1mb [kg]": uw,
                    "Liczba sztang": n_bars,
                    "Dł. netto [m]": round(tot_cut_m, 2),
                    "Dł. brutto [m]": round(tot_stock_m, 2),
                    "Odpad [m]": round(waste_m, 2),
                    "Masa netto [kg]": round(net_m_kg, 1),
                    "Masa brutto [kg]": round(gross_m_kg, 1),
                    "Odpad [kg]": round(waste_m_kg, 1),
                    "Odpad [%]": f"{waste_pct:.1f}%"
                })

        df_prof_summary = pd.DataFrame(prof_summary_rows)

        # 2D: Masy blach
        total_plates_net_mass = 0.0
        total_plates_gross_mass = 0.0
        if sheets_2d:
            for s in sheets_2d:
                area_m2 = (s.length * s.width) / 1e6
                used_area_m2 = s.used_area / 1e6
                sheet_gross_kg = area_m2 * s.thick * 7.85
                sheet_net_kg = used_area_m2 * s.thick * 7.85
                total_plates_gross_mass += sheet_gross_kg
                total_plates_net_mass += sheet_net_kg

        tot_gross_all = total_bars_gross_mass + total_plates_gross_mass
        tot_net_all = total_bars_net_mass + total_plates_net_mass
        tot_waste_kg = tot_gross_all - tot_net_all
        tot_waste_pct = (tot_waste_kg / tot_gross_all * 100.0) if tot_gross_all > 0 else 0.0

        # Wskaźniki KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Masa Zamówienia (Brutto)", f"{tot_gross_all / 1000.0:.2f} t")
        kpi2.metric("Masa Elementów (Netto)", f"{tot_net_all / 1000.0:.2f} t")
        kpi3.metric("Masa Odpadu Hutniczego", f"{tot_waste_kg / 1000.0:.2f} t")
        kpi4.metric("Średni Odpad Całkowity", f"{tot_waste_pct:.1f}%")

        # Zakładki prezentacji wyników
        tab_prof, tab_plates, tab_order, tab_export = st.tabs([
            "📊 Rozkrój Profili (1D)",
            "📋 Rozkrój Blach (2D)",
            "🛒 Zamówienie Hutnicze",
            "📥 Eksport Wyników (Excel)"
        ])

        # TAB 1: Rozkrój Profili 1D
        with tab_prof:
            if df_prof_summary.empty:
                st.info("Brak profili 1D do wyświetlenia.")
            else:
                st.subheader("Bilans Materiału i Odpadu wg Poszczególnych Profili")
                st.dataframe(df_prof_summary, use_container_width=True)

                st.subheader("Karty Rozkroju Sztang (Wizualizacja Warsztatowa)")
                max_display_bars = min(len(bars_1d), 35)
                for i in range(max_display_bars):
                    b = bars_1d[i]
                    fig, ax = plt.subplots(figsize=(10, 0.9))
                    curr_x = 0.0

                    # Trim cut
                    if b.trim_cut > 0:
                        ax.broken_barh([(0, b.trim_cut)], (0, 8), facecolors='#EF4444', edgecolor='black')
                        curr_x += b.trim_cut

                    # Elementy gotowe
                    for cut in b.cuts:
                        ax.broken_barh([(curr_x, cut.length)], (0, 8), facecolors='#3B82F6', edgecolor='black')
                        ax.text(curr_x + cut.length / 2, 4, f"{cut.pos} ({cut.length:.0f})",
                                ha='center', va='center', color='white', fontsize=7, fontweight='bold')
                        curr_x += cut.length
                        if b.kerf > 0:
                            ax.broken_barh([(curr_x, b.kerf)], (0, 8), facecolors='#1E293B')
                            curr_x += b.kerf

                    # Odpad
                    waste_len = b.stock_length - curr_x
                    if waste_len > 0:
                        ax.broken_barh([(curr_x, waste_len)], (0, 8), facecolors='#CBD5E1', hatch='//')
                        ax.text(curr_x + waste_len / 2, 4, f"Odpad: {waste_len:.0f}mm",
                                ha='center', va='center', color='#334155', fontsize=7)

                    ax.set_xlim(0, b.stock_length)
                    ax.set_ylim(-1, 9)
                    ax.axis('off')
                    st.caption(f"**Sztanga #{b.bar_id}**: {b.profile} | {b.grade} | L handlowe: **{b.stock_length:.0f} mm** | Wykorzystanie: **{b.efficiency:.1f}%** (Odpad: {b.waste_length:.0f} mm)")
                    st.pyplot(fig)
                    plt.close(fig)

                if len(bars_1d) > max_display_bars:
                    st.info(f"Wyświetlono 35 z {len(bars_1d)} sztang. Pełne zestawienie znajduje się w generowanym pliku Excel.")

        # TAB 2: Rozkrój Blach 2D
        with tab_plates:
            if not sheets_2d:
                st.info("Brak pozycji zakwalifikowanych jako formatki blach (2D).")
            else:
                st.subheader(f"Arkusze Blach: {len(sheets_2d)} szt. formatu {plate_format_choice} mm")
                for s in sheets_2d:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.add_patch(patches.Rectangle((0, 0), s.length, s.width, edgecolor='#0F172A', facecolor='#F1F5F9'))

                    for item, x, y, w, h in s.placed:
                        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#1E3A8A', facecolor='#60A5FA', alpha=0.85)
                        ax.add_patch(rect)
                        ax.text(x + w / 2, y + h / 2, f"{item.pos}\n{item.length:.0f}x{item.width:.0f}",
                                ha='center', va='center', fontsize=6, color='black', weight='bold')

                    ax.set_xlim(-50, s.length + 50)
                    ax.set_ylim(-50, s.width + 50)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    st.caption(f"**Arkusz #{s.sheet_id}**: Grubość #**{s.thick:.0f} mm** | {s.grade} | {s.width:.0f} x {s.length:.0f} mm | Wykorzystanie: **{s.efficiency:.1f}%**")
                    st.pyplot(fig)
                    plt.close(fig)

        # TAB 3: Zamówienie Hutnicze
        with tab_order:
            st.subheader("Lista Zakupowa do Dystrybutora Stali")
            order_items = []

            if bars_1d:
                bar_orders = {}
                for b in bars_1d:
                    key = (b.profile, b.stock_length, b.grade)
                    bar_orders[key] = bar_orders.get(key, 0) + 1

                for (prof, sl, grd), count in bar_orders.items():
                    uw = get_unit_weight(prof)
                    tot_len_m = (count * sl) / 1000.0
                    tot_mass_kg = tot_len_m * uw
                    order_items.append({
                        "Typ": "Profil Hutniczy",
                        "Asortyment": prof,
                        "Wymiar Handlowy [mm]": f"L = {sl:.0f}",
                        "Gatunek": grd,
                        "Liczba Sztuk": count,
                        "Łączna Długość [m]": round(tot_len_m, 2),
                        "Masa Całkowita [kg]": round(tot_mass_kg, 1),
                        "Masa Całkowita [t]": round(tot_mass_kg / 1000.0, 3)
                    })

            if sheets_2d:
                sheet_orders = {}
                for s in sheets_2d:
                    key = (s.thick, s.width, s.length, s.grade)
                    sheet_orders[key] = sheet_orders.get(key, 0) + 1

                for (th, w, l, grd), count in sheet_orders.items():
                    sheet_area = (w * l) / 1e6
                    tot_mass_kg = count * sheet_area * th * 7.85
                    order_items.append({
                        "Typ": "Blacha Gruba",
                        "Asortyment": f"#{th:.0f} mm",
                        "Wymiar Handlowy [mm]": f"{w:.0f} x {l:.0f}",
                        "Gatunek": grd,
                        "Liczba Sztuk": count,
                        "Łączna Długość [m]": count * (l / 1000.0),
                        "Masa Całkowita [kg]": round(tot_mass_kg, 1),
                        "Masa Całkowita [t]": round(tot_mass_kg / 1000.0, 3)
                    })

            df_order_table = pd.DataFrame(order_items)
            st.dataframe(df_order_table, use_container_width=True)

        # TAB 4: Eksport do Excela
        with tab_export:
            st.subheader("Generowanie Raportu Handlowego i Warsztatowego (.xlsx)")

            # Przygotowanie danych warsztatowych
            workshop_cuts = []
            for b in bars_1d:
                for c in b.cuts:
                    workshop_cuts.append({
                        "Nr Sztangi": b.bar_id,
                        "Profil": b.profile,
                        "Gatunek": b.grade,
                        "Dł. Handlowa [mm]": b.stock_length,
                        "Pozycja": c.pos,
                        "Długość Detalu [mm]": c.length,
                        "Odpad na sztandze [mm]": round(b.waste_length, 1),
                        "Wykorzystanie [%]": round(b.efficiency, 1)
                    })
            df_workshop = pd.DataFrame(workshop_cuts)

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                df_order_table.to_excel(writer, sheet_name="Do Zamówienia", index=False)
                df_workshop.to_excel(writer, sheet_name="Rozkrój Warsztat", index=False)
                if not df_prof_summary.empty:
                    df_prof_summary.to_excel(writer, sheet_name="Odpad wg Profili 1D", index=False)
                
                # Podsumowanie ogólne
                df_kpi = pd.DataFrame([{
                    "Masa Brutto [t]": round(tot_gross_all / 1000.0, 3),
                    "Masa Netto [t]": round(tot_net_all / 1000.0, 3),
                    "Odpad [t]": round(tot_waste_kg / 1000.0, 3),
                    "Odpad Średni [%]": round(tot_waste_pct, 2)
                }])
                df_kpi.to_excel(writer, sheet_name="Statystyka Odpadu", index=False)

            st.download_button(
                label="📥 Pobierz Gotowy Arkusz Excel (.xlsx)",
                data=excel_buf.getvalue(),
                file_name="Zamowienie_i_Rozkroj_Stali.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
