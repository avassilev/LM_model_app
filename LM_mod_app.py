# LM_mod_app.py

import numpy as np
import pandas as pd
import panel as pn
from pathlib import Path
import sys, os

pn.extension('tabulator')

import matplotlib.pyplot as plt

sys.path.append('.')



# ---------- 1. Helper that works in BOTH environments -------------
def _read_csv(filename):
    """
    Reads a CSV either
      - directly from disk (server / local Python) or
      - via HTTP in the browser (Pyodide)
    """
    if "pyodide" in sys.modules:                       # we are in WASM
        from pyodide.http import open_url
        return pd.read_csv(open_url(f"processed_data/{filename}"))
    else:                                              # normal Python
        return pd.read_csv(Path(__file__).parent /
                            "processed_data" / filename)

# ---------- 2. Load once at start ---------------------------------
def load_data():
    global demproj, act, edu
    demproj = _read_csv("dem_proj_BG.csv")
    act      = _read_csv("activity_1.csv")
    edu      = _read_csv("education_1.csv")

load_data() 

def compute_projection(A_gr_bar, kappa, k_bar, nu, demographic_scenario=None, add_params=False):
    if demproj is None or act is None or edu is None:
        raise ValueError("Data not loaded. Run load_data() first.")
    
    # Initial constants
    unemp_rate = 0.04
    beta_L = 0.10
    beta_M = 0.55
    beta_H = 0.35
    alpha = 0.34

    init_yr = 2024
    frst_forec_yr = init_yr + 1
    end_yr = 2100

    eH_init = 1098300
    eL_init = 241600
    eM_init = 1585500

    A_gr_init = 0.025
    k_init = 2.33
    
    Ak_df = pd.DataFrame({'year': range(init_yr, end_yr+1)})
    Ak_df['A_gr'] = Ak_df['k'] = pd.NA
    Ak_df.loc[Ak_df.year == init_yr, 'A_gr'] = A_gr_init
    Ak_df.loc[Ak_df.year == init_yr, 'k'] = k_init

    for i in Ak_df.index[1:]:
        Ak_df.loc[i, 'A_gr'] = kappa * A_gr_bar + (1 - kappa) * Ak_df.loc[i-1, 'A_gr']
        Ak_df.loc[i, 'k'] = nu * k_bar + (1 - nu) * Ak_df.loc[i-1, 'k']

    Ak_df = Ak_df[['year', 'A_gr', 'k']]

    pop = (
        demproj.groupby(["projection", "year"])
        .agg({"pop": "sum"})
        .reset_index()
        .sort_values(["projection", "year"])
    )
    pop["pop_gr"] = pop.groupby("projection")["pop"].pct_change()
    pop.drop(columns=["pop"], inplace=True)

    lab = demproj.merge(edu, how="left").dropna()
    lab["L"] = lab["pop"] * lab["edu_L"]
    lab["M"] = lab["pop"] * lab["edu_M"]
    lab["H"] = lab["pop"] * lab["edu_H"]
    lab.drop(columns=["pop", "edu_H", "edu_L", "edu_M"], inplace=True)
    lab = pd.melt(
        lab,
        id_vars=["projection", "year", "age", "age_group", "sex"],
        value_vars=["L", "M", "H"],
        var_name="edu",
        value_name="pop",
    )
    lab = lab.merge(act, how="left")
    lab["pop_active"] = lab["pop"] * lab["l_active"]
    lab["employment"] = lab["pop_active"] * (1 - unemp_rate)
    lab = lab.groupby(['projection', 'year', 'edu']).agg({'employment': 'sum'}).reset_index()

    lab = lab.pivot(columns='edu', values='employment', index=['projection', 'year']) \
             .rename(columns={'H': 'eH', 'M': 'eM', 'L': 'eL'}) \
             .reset_index()

    lab_s_gr = lab.copy()
    lab_s_gr[["eH_gr", "eM_gr", "eL_gr"]] = lab_s_gr.groupby("projection")[["eH", "eM", "eL"]].pct_change()
    lab_s_gr = lab_s_gr[lab_s_gr.year >= init_yr]
    lab_s_gr["e_gr"] = (
        (1 + lab_s_gr.eH_gr) ** beta_H *
        (1 + lab_s_gr.eM_gr) ** beta_M *
        (1 + lab_s_gr.eL_gr) ** beta_L - 1
    )

    lab_s_abs = lab_s_gr.copy()
    lab_s_abs[["eHg", "eMg", "eLg"]] = lab_s_abs[["eH_gr", "eM_gr", "eL_gr"]] + 1
    lab_s_abs.loc[lab_s_abs.year == init_yr, ["eHg", "eMg", "eLg"]] = 1
    lab_s_abs[["eHgcum", "eMgcum", "eLgcum"]] = lab_s_abs.groupby("projection")[["eHg", "eMg", "eLg"]].cumprod()
    lab_s_abs["eH"] = lab_s_abs["eHgcum"] * eH_init
    lab_s_abs["eM"] = lab_s_abs["eMgcum"] * eM_init
    lab_s_abs["eL"] = lab_s_abs["eLgcum"] * eL_init

    pr = lab_s_gr.merge(Ak_df)
    proj = pd.DataFrame()

    for sc in pr.projection.unique():
        p = pr[pr.projection == sc].copy()
        p["Y_gr"] = (
            (1 + p.A_gr) ** (1 / (1 - alpha)) *
            (nu * k_bar / p["k"].shift() + (1 - nu)) ** (alpha / (1 - alpha)) *
            (1 + p.e_gr) - 1
        )
        proj = pd.concat([proj, p])

    proj = proj.dropna()
    proj = proj.merge(pop, how="left")
    proj["Y_pc_gr"] = (1 + proj["Y_gr"]) / (1 + proj["pop_gr"]) - 1
    proj["A_contrib"] = proj.A_gr
    proj["L_contrib"] = (1 - alpha) * proj.e_gr
    proj["K_contrib"] = proj.Y_gr - proj.A_contrib - proj.L_contrib

    if add_params:
        proj["k_bar"] = k_bar
        proj["A_gr_bar"] = A_gr_bar
        proj["kappa"] = kappa
        proj["nu"] = nu
        proj = proj[
            ["projection", "k_bar", "A_gr_bar", "kappa", "nu", "year", "eH_gr", "eM_gr", "eL_gr", "e_gr", "A_gr", "k",
             "Y_gr", "pop_gr", "Y_pc_gr", "A_contrib", "L_contrib", "K_contrib"]
        ]

    if demographic_scenario:
        proj = proj[proj.projection == demographic_scenario]

    return proj



# --- Widgets ---

A_gr_bar_slider = pn.widgets.FloatSlider(name="Равновесен растеж на ОФП (%)", start=0.5, end=2.0, step=0.5, value=1.0)
kappa_slider = pn.widgets.FloatSlider(name="Скорост на конвергенция на ОФП", start=0.5, end=0.95, step=0.05, value=0.9)
k_bar_slider = pn.widgets.FloatSlider(name="Равновесно съотношение капитал-БВП", start=2.7, end=3.1, step=0.1, value=3.0)
nu_slider = pn.widgets.FloatSlider(name="Скорост на конвергенция на съотношението капитал-БВП", start=0.05, end=0.4, step=0.05, value=0.1)

# Scenario selector
scenario_select = pn.widgets.Select(
    name="Демографски сценарий", 
    options=['Базисен (BSL)', 'Висока миграция (HMIGR)', 'Ниска раждаемост (LFRT)', 'Ниска миграция (LMIGR)', 'Ниска смъртност (LMRT)', 'Отсъствие на миграция (NMIGR)'], 
    value="Базисен (BSL)"
)

calculate_button = pn.widgets.Button(name="Изчисли", button_type="primary")

# Output placeholders

table_output = pn.widgets.Tabulator(pd.DataFrame(), height=250, show_index=False)
plot_output = pn.pane.Matplotlib(height=400)

# --- Callback ---
def run_projection(event=None):
    A_gr_bar = A_gr_bar_slider.value/100.0
    kappa = kappa_slider.value
    k_bar = k_bar_slider.value
    nu = nu_slider.value

    scenario_coding = {"Базисен (BSL)":"BSL", 
                       "Висока миграция (HMIGR)":"HMIGR", 
                       "Ниска раждаемост (LFRT)":"LFRT", 
                       "Ниска миграция (LMIGR)":"LMIGR", 
                       "Ниска смъртност (LMRT)":"LMRT", 
                       "Отсъствие на миграция (NMIGR)":"NMIGR"}
    scenario = scenario_coding[scenario_select.value]

    df = compute_projection(A_gr_bar, kappa, k_bar, nu, demographic_scenario=scenario)

    p = df.copy()
    p = p.loc[p.year.isin([2025, 2030, 2035, 2040, 2045, 2050]),['year','e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']]
    p[['e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']] *= 100
    p['year'] = p['year'].map("{:d}".format)
    for v in ['e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']:
        p[v] = p[v].map("{:.2f}".format)
    p.rename(columns = {'year':'Година','e_gr':'Синтетична заетост (годишно изменение, %)','Y_gr':'Реален БВП (годишно изменение, %)', 'pop_gr':'Население (годишно изменение, %)', 'Y_pc_gr':'Реален БВП на глава от населението (годишно изменение, %)'}, inplace=True)

    gr = df.copy()
    gr = gr.loc[gr.year.isin([2025, 2030, 2035, 2040, 2045, 2050]),['year','Y_gr', 'A_contrib', 'L_contrib', 'K_contrib']]
    gr[['Y_gr', 'A_contrib', 'L_contrib', 'K_contrib']] *= 100

    table_output.value = p

    # Plot: Y_gr as line, contributions as stacked bars
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar width and offset
    width = 1
    x = gr["year"]
    A = gr["A_contrib"]
    L = gr["L_contrib"]
    K = gr["K_contrib"]

    base_pos = np.zeros_like(A)
    base_neg = np.zeros_like(A)



    ax.bar(x, A, width, label="ОФП", color="tab:blue")
    base_pos += np.where(A > 0, A, 0)
    base_neg += np.where(A < 0, A, 0)
    
    ax.bar(x, L, width, bottom=np.where(L >= 0, base_pos, base_neg), label="Синтетичен труд", color="tab:orange")
    base_pos += np.where(L > 0, L, 0)
    base_neg += np.where(L < 0, L, 0)
    
    ax.bar(x, K, width, bottom=np.where(K >= 0, base_pos, base_neg), label="Капитал", color="tab:green")

    # Overlay line plot for Y_gr
    ax.plot(x, gr["Y_gr"], marker='o', color='black', linewidth=2, label="Реален БВП (годишно изменение, %)")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title("Приноси за растежа на реалния БВП")
    # ax.set_xlabel("Ð“Ð¾Ð´Ð¸Ð½Ð°")
    ax.set_ylabel("%")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(x.astype(str), rotation=45)

    plot_output.object = fig

# Attach callback
calculate_button.on_click(run_projection)

# Initial placeholder render (optional)
run_projection()



# --- Layout ---
controls = pn.Column(
    pn.pane.Markdown("### Основни параметри на модела"),
    A_gr_bar_slider,
    kappa_slider,
    k_bar_slider,
    nu_slider,
    scenario_select,
    calculate_button,
)

layout = pn.Column(
    controls,
    pn.pane.Markdown("### Резултати"),
    table_output,
    pn.pane.Markdown("### Фактори за икономическия растеж"),
    plot_output,
    pn.pane.HTML(
        '<a href="documentation/index.html" target="_blank">Документация (HTML)</a>',
        sizing_mode="stretch_width"
    ),
    pn.pane.HTML(
        '<a href="documentation/LM_app_doc.pdf" target="_blank">Документация (PDF)</a>',
        sizing_mode="stretch_width"
    )
)

# layout.servable()
layout.servable(target='app')
