import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SteelOpt - Optymalizator i Stykowanie",
    page_icon="🏗️",
    layout="wide",
)

STEEL_DENSITY_KG_M3 = 7850.0

# Baza mas jednostkowych najpopularniejszych profili wg norm europejskich (kg/m)
EURO_PROFILE_WEIGHTS = {
    "IPE100": 8.1, "IPE120": 10.4, "IPE160": 15.8, "IPE200": 22.4, "IPE300": 42.2, "IPE400": 66.3,
    "HEA100": 16.7, "HEA120": 19.9, "HEA160": 30.4, "HEA200": 42.3, "HEA240": 60.3, "HEA300": 88.3,
    "HEB100": 20.4, "HEB160": 42.6, "HEB200": 61.3, "HEB300": 117.0,
    "UNP100": 10.6, "UNP160": 18.8, "UNP200": 25.3,
}

@dataclass
class Item1D:
    mark: str
    length: float
    profile: str
    grade: str
    quantity: int = 1

def get_unit_weight_1d(profile_str: str) -> float:
    """Oblicza lub odnajduje masę 1 mb profilu hutniczego [kg/m]."""
    raw = str(profile_str).upper().replace(" ", "").replace("×", "X").replace(",", ".")
    m_he = re.match(r"^HE(\d+)([ABM])$", raw)
    if m_he:
        raw = f"HE{m_he.group(2)}{m_he.group(1)}"
    clean_prof = re.sub(r'[^A-Z0-9]', '', raw)
    
    for key, weight in EURO_PROFILE_WEIGHTS.items():
        if key == clean_prof or clean_prof.startswith(key):
            return weight

    # Wyliczanie dla kątowników
    m_angle = re.search(r'(?:L|KAT|KĄT)?\s*(\d+(?:\.\d+)?)[X](\d+(?:\.\d+)?)(?:[X](\d+(?:\.\d+)?))?', raw)
    if m_angle and any(prefix in raw for prefix in ["L", "KAT", "KĄT"]):
        a, b = float(m_angle.group(1)), float(m_angle.group(2))
        t = float(m_angle.group(3)) if m_angle.group(3) else a
        return round((a + b - t) * t * 1.012 * (STEEL_DENSITY_KG_M3 / 1_000_000.0), 2)
    return 20.0

def get_allowed_lengths(profile: str) -> List[float]:
    """
    Zwraca dozwolone długości handlowe w zależności od typu profilu.
    Reguła biznesowa: HEA, HEB, IPE, HEM -> 12.1m i 15.1m. Reszta -> 6.0m i 12.0m.
    """
    prof = str(profile).upper()
    if prof.startswith(("HEA", "HEB", "IPE", "HEM")):
        return [12100.0, 15100.0]
    return [6000.0, 12000.0]

def group_1d_items_by_material(items: List[Item1D]) -> Dict[Tuple[str, str], List[Item1D]]:
    """Grupuje elementy względem profilu oraz gatunku stali."""
    grouped = {}
    for item in items:
        prof = str(item.profile).strip().upper()
        # Standaryzacja nazw HE np. HE240A na HEA240
        m_he = re.match(r"^HE(\d+)([ABM])$", prof.replace(" ", ""))
        if m_he: prof = f"HE{m_he.group(2)}{m_he.group(1)}"
        
        grade = str(item.grade).strip().upper() if pd.notna(item.grade) and item.grade else "S355J2+N"
        
        key = (prof, grade)
        grouped.setdefault(key, []).append(item)
    return grouped

def optimize_no_splice(items: List[Item1D], allowed_lengths: List[float], kerf: float, trim: float) -> Tuple[Dict[float, int], float]:
    """Optymalizacja klasyczna (bez łączenia) - heurystyka First Fit Decreasing."""
    expanded_cuts = [float(it.length) for it in items for _ in range(it.quantity)]
    expanded_cuts.sort(reverse=True)
    
    sorted_stocks = sorted(allowed_lengths)
    bars_used = []
    
    for cut in expanded_cuts:
        placed = False
        for bar in bars_used:
            if bar['stock'] - bar['used'] - trim >= cut + kerf:
                bar['used'] += cut + kerf
                placed = True
                break
        if not placed:
            # Dobór najmniejszej wystarczającej sztangi, ew. największej, gdy element przekracza gabaryty
            chosen_stock = sorted_stocks[0] if (sorted_stocks[0] - trim >= cut) else sorted_stocks[-1]
            bars_used.append({'stock': chosen_stock, 'used': cut})
            
    stock_counts = {}
    total_purchased_len = 0.0
    for bar in bars_used:
        stock_counts[bar['stock']] = stock_counts.get(bar['stock'], 0) + 1
        total_purchased_len += bar['stock']
        
    return stock_counts, total_purchased_len

def optimize_with_splice(items: List[Item1D], allowed_lengths: List[float], kerf: float, trim: float) -> Tuple[Dict[float, int], float]:
    """
    Optymalizacja ze stykowaniem - zlicza całe zapotrzebowanie jako jeden element,
    a następnie optymalnie dobiera z dostępnych długości handlowych.
    """
    total_net_len = sum(it.length * it.quantity for it in items)
    total_cuts = sum(it.quantity for it in items)
    required_len = total_net_len + (total_cuts * kerf)
    
    sorted_stocks = sorted(allowed_lengths, reverse=True)
    l1 = sorted_stocks[0]
    l2 = sorted_stocks[1] if len(sorted_stocks) > 1 else sorted_stocks[0]
    
    min_waste = float('inf')
    best_combination = (0, 0)
    
    # Przeszukiwanie optymalnej kombinacji z dwóch dozwolonych długości
    max_l1 = int(math.ceil(required_len / (l1 - trim)))
    for cnt1 in range(max_l1 + 1):
        rem = required_len - cnt1 * (l1 - trim)
        cnt2 = 0 if rem <= 0 else int(math.ceil(rem / (l2 - trim)))
        waste = (cnt1 * l1 + cnt2 * l2) - (cnt1 * trim + cnt2 * trim + required_len)
        if 0 <= waste < min_waste:
            min_waste = waste
            best_combination = (cnt1, cnt2)
            
    stock_counts = {}
    if best_combination[0] > 0: stock_counts[l1] = best_combination[0]
    if best_combination[1] > 0: stock_counts[l2] = best_combination[1]
    
    total_purchased_len = (best_combination[0] * l1) + (best_combination[1] * l2)
    return stock_counts, total_purchased_len

def parse_bom_file(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
    except Exception:
        return pd.read_csv(io.BytesIO(uploaded_file.getvalue()), sep=';')

def map_imported_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizacja nazw kolumn z polskich/angielskich/CAD na standard aplikacji."""
    col_map = {}
    for col in df.columns:
        c_clean = str(col).lower().strip()
        if any(k in c_clean for k in ['pozycja', 'pos', 'mark']): col_map[col] = 'mark'
        elif any(k in c_clean for k in ['profil', 'profile']): col_map[col] = 'profile'
        elif any(k in c_clean for k in ['materiał', 'grade', 'gatunek']): col_map[col] = 'grade'
        elif any(k in c_clean for k in ['ilość', 'qty', 'szt']): col_map[col] = 'qty'
        elif any(k in c_clean for k in ['długość', 'length', 'l [mm]']) and 'całk' not in c_clean: col_map[col] = 'length'

    df_ren = df.rename(columns=col_map)
    clean_rows = []
    
    for _, row in df_ren.iterrows():
        try:
            q = int(float(str(row.get('qty', '1')).replace(',', '.')))
            l = float(str(row.get('length', '0')).replace(',', '.'))
            prof = str(row.get('profile', '')).strip()
            
            if q > 0 and l > 0 and prof and prof.lower() != 'nan':
                clean_rows.append({
                    "mark": str(row.get('mark', 'P')),
                    "profile": prof,
                    "grade": str(row.get('grade', 'S355J2+N')),
                    "length": l,
                    "qty": q,
                })
        except ValueError:
            continue
    return pd.DataFrame(clean_rows)

def build_excel_export(df_opt1: pd.DataFrame, df_opt2: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_opt1.to_excel(writer, sheet_name="Opcja 1 (Bez Styku)", index=False)
        df_opt2.to_excel(writer, sheet_name="Opcja 2 (Ze Stykiem)", index=False)
    return buffer.getvalue()

with st.sidebar:
    st.header("⚙️ Parametry")
    kerf_1d = st.number_input("Rzaz piły [mm]", min_value=1.0, value=4.5)
    trim_1d = st.number_input("Naddatek obcięcia [mm]", min_value=0.0, value=40.0)
    price_per_kg = st.number_input("Cena stali [PLN/kg]", min_value=1.0, value=4.60)
    
    st.info("ℹ️ Zgodnie z regułą biznesową aplikacja automatycznie dobierze długości:\n\n• HEA, HEB, IPE, HEM: 12.1 m oraz 15.1 m\n• Pozostałe: 6.0 m oraz 12.0 m")

st.title("🏗️ SteelOpt: Optymalizator i Stykowanie")

uploaded_file = st.file_uploader("Wczytaj Zestawienie BOM (Excel/CSV):")

if uploaded_file:
    df_raw = parse_bom_file(uploaded_file)
    df_clean = map_imported_columns(df_raw)
    
    # Oddzielenie blach (aplikacja wspiera logikę tylko dla profili wg założeń)
    is_plate = df_clean["profile"].str.contains(r"PL|BLACHA|#", case=False, regex=True)
    df_1d = df_clean[~is_plate].copy()
    
    raw_items_1d = [Item1D(**row) for _, row in df_1d.iterrows()]
    grouped_1d = group_1d_items_by_material(raw_items_1d)
    
    rows_opt1 = []
    rows_opt2 = []
    
    total_cost_opt1 = 0.0
    total_cost_opt2 = 0.0
    
    for (prof, grade), items in grouped_1d.items():
        allowed_lengths = get_allowed_lengths(prof)
        unit_wt = get_unit_weight_1d(prof)
        net_len = sum(it.length * it.quantity for it in items)
        
        # Opcja 1: Bez styku
        counts_opt1, gross_len_opt1 = optimize_no_splice(items, allowed_lengths, kerf_1d, trim_1d)
        mass_opt1 = (gross_len_opt1 / 1000.0) * unit_wt
        cost_1 = mass_opt1 * price_per_kg
        total_cost_opt1 += cost_1
        waste_1 = ((gross_len_opt1 - net_len) / gross_len_opt1 * 100) if gross_len_opt1 > 0 else 0
        
        for length, qty in counts_opt1.items():
            rows_opt1.append({"Profil": prof, "Gatunek": grade, "Sztanga [mm]": length, "Szt.": qty, "Masa brutto [kg]": round((length/1000)*unit_wt*qty, 1), "Odpad [%]": round(waste_1, 1), "Koszt [PLN]": round((length/1000)*unit_wt*qty*price_per_kg, 2)})
            
        # Opcja 2: Ze stykiem
        counts_opt2, gross_len_opt2 = optimize_with_splice(items, allowed_lengths, kerf_1d, trim_1d)
        mass_opt2 = (gross_len_opt2 / 1000.0) * unit_wt
        cost_2 = mass_opt2 * price_per_kg
        total_cost_opt2 += cost_2
        waste_2 = ((gross_len_opt2 - net_len) / gross_len_opt2 * 100) if gross_len_opt2 > 0 else 0
        
        for length, qty in counts_opt2.items():
            rows_opt2.append({"Profil": prof, "Gatunek": grade, "Sztanga [mm]": length, "Szt.": qty, "Masa brutto [kg]": round((length/1000)*unit_wt*qty, 1), "Odpad [%]": round(waste_2, 1), "Koszt [PLN]": round((length/1000)*unit_wt*qty*price_per_kg, 2)})

    df_opt1 = pd.DataFrame(rows_opt1)
    df_opt2 = pd.DataFrame(rows_opt2)

    tab1, tab2 = st.tabs(["🛒 Wyniki i Eksport", "✉️ Szablon Maila"])
    
    with tab1:
        st.success(f"Pomyślnie przeanalizowano pozycje dla profili. Odrzucono pozycje blachowe.")
        col1, col2 = st.columns(2)
        col1.metric("Opcja 1 (Bez Styku) - Koszt Całkowity", f"{total_cost_opt1:,.2f} PLN")
        col2.metric("Opcja 2 (Ze Stykiem) - Koszt Całkowity", f"{total_cost_opt2:,.2f} PLN")
        
        st.markdown("Plik Excel zawiera przygotowane rozbicie kosztów i odpadu dla obydwu scenariuszy.")
        excel_data = build_excel_export(df_opt1, df_opt2)
        st.download_button("📥 Pobierz Zestawienie w Excelu (2 Opcje)", data=excel_data, file_name="Zestawienie_SteelOpt.xlsx", use_container_width=True)

    with tab2:
        st.markdown("### Szablon zapytania ofertowego")
        st.info("Skopiuj poniższy tekst i wyślij do dostawcy wraz z wygenerowanym Excelem.")
        
        email_body = (
            "Dzień dobry,\n\n"
            "Proszę o przygotowanie oferty cenowej oraz podanie dostępności dla załączonego zestawienia wyrobów hutniczych.\n\n"
            "Wymagania dodatkowe:\n"
            "- Atest materiałowy 3.1 (PN-EN 10204) dla wszystkich pozycji.\n"
            "- Proszę o uwzględnienie kosztów transportu do naszego zakładu.\n\n"
            "Z góry dziękuję za odpowiedź.\n\n"
            "Pozdrawiam,\n[Twój Podpis]"
        )
        
        st.text_area("Gotowa treść:", value=email_body, height=250)
