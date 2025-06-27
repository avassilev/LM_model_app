importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/wheels/bokeh-3.6.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.6.0/dist/wheels/panel-1.6.0-py3-none-any.whl', 'pyodide-http==0.2.1', 'matplotlib', 'numpy', 'pandas']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  \nimport asyncio\n\nfrom panel.io.pyodide import init_doc, write_doc\n\ninit_doc()\n\n# LM_mod_app.py\n\nimport numpy as np\nimport pandas as pd\nimport panel as pn\nfrom pathlib import Path\nimport sys, os\n\npn.extension('tabulator')\n\nimport matplotlib.pyplot as plt\n\nsys.path.append('.')\n\n\n\n# ---------- 1. Helper that works in BOTH environments -------------\ndef _read_csv(filename):\n    """\n    Reads a CSV either\n      - directly from disk (server / local Python) or\n      - via HTTP in the browser (Pyodide)\n    """\n    if "pyodide" in sys.modules:                       # we are in WASM\n        from pyodide.http import open_url\n        return pd.read_csv(open_url(f"processed_data/{filename}"))\n    else:                                              # normal Python\n        return pd.read_csv(Path(__file__).parent /\n                            "processed_data" / filename)\n\n# ---------- 2. Load once at start ---------------------------------\ndef load_data():\n    global demproj, act, edu\n    demproj = _read_csv("dem_proj_BG.csv")\n    act      = _read_csv("activity_1.csv")\n    edu      = _read_csv("education_1.csv")\n\nload_data() \n\ndef compute_projection(A_gr_bar, kappa, k_bar, nu, demographic_scenario=None, add_params=False):\n    if demproj is None or act is None or edu is None:\n        raise ValueError("Data not loaded. Run load_data() first.")\n    \n    # Initial constants\n    unemp_rate = 0.04\n    beta_L = 0.10\n    beta_M = 0.55\n    beta_H = 0.35\n    alpha = 0.34\n\n    init_yr = 2024\n    frst_forec_yr = init_yr + 1\n    end_yr = 2100\n\n    eH_init = 1098300\n    eL_init = 241600\n    eM_init = 1585500\n\n    A_gr_init = 0.025\n    k_init = 2.33\n    \n    Ak_df = pd.DataFrame({'year': range(init_yr, end_yr+1)})\n    Ak_df['A_gr'] = Ak_df['k'] = pd.NA\n    Ak_df.loc[Ak_df.year == init_yr, 'A_gr'] = A_gr_init\n    Ak_df.loc[Ak_df.year == init_yr, 'k'] = k_init\n\n    for i in Ak_df.index[1:]:\n        Ak_df.loc[i, 'A_gr'] = kappa * A_gr_bar + (1 - kappa) * Ak_df.loc[i-1, 'A_gr']\n        Ak_df.loc[i, 'k'] = nu * k_bar + (1 - nu) * Ak_df.loc[i-1, 'k']\n\n    Ak_df = Ak_df[['year', 'A_gr', 'k']]\n\n    pop = (\n        demproj.groupby(["projection", "year"])\n        .agg({"pop": "sum"})\n        .reset_index()\n        .sort_values(["projection", "year"])\n    )\n    pop["pop_gr"] = pop.groupby("projection")["pop"].pct_change()\n    pop.drop(columns=["pop"], inplace=True)\n\n    lab = demproj.merge(edu, how="left").dropna()\n    lab["L"] = lab["pop"] * lab["edu_L"]\n    lab["M"] = lab["pop"] * lab["edu_M"]\n    lab["H"] = lab["pop"] * lab["edu_H"]\n    lab.drop(columns=["pop", "edu_H", "edu_L", "edu_M"], inplace=True)\n    lab = pd.melt(\n        lab,\n        id_vars=["projection", "year", "age", "age_group", "sex"],\n        value_vars=["L", "M", "H"],\n        var_name="edu",\n        value_name="pop",\n    )\n    lab = lab.merge(act, how="left")\n    lab["pop_active"] = lab["pop"] * lab["l_active"]\n    lab["employment"] = lab["pop_active"] * (1 - unemp_rate)\n    lab = lab.groupby(['projection', 'year', 'edu']).agg({'employment': 'sum'}).reset_index()\n\n    lab = lab.pivot(columns='edu', values='employment', index=['projection', 'year']) \\\n             .rename(columns={'H': 'eH', 'M': 'eM', 'L': 'eL'}) \\\n             .reset_index()\n\n    lab_s_gr = lab.copy()\n    lab_s_gr[["eH_gr", "eM_gr", "eL_gr"]] = lab_s_gr.groupby("projection")[["eH", "eM", "eL"]].pct_change()\n    lab_s_gr = lab_s_gr[lab_s_gr.year >= init_yr]\n    lab_s_gr["e_gr"] = (\n        (1 + lab_s_gr.eH_gr) ** beta_H *\n        (1 + lab_s_gr.eM_gr) ** beta_M *\n        (1 + lab_s_gr.eL_gr) ** beta_L - 1\n    )\n\n    lab_s_abs = lab_s_gr.copy()\n    lab_s_abs[["eHg", "eMg", "eLg"]] = lab_s_abs[["eH_gr", "eM_gr", "eL_gr"]] + 1\n    lab_s_abs.loc[lab_s_abs.year == init_yr, ["eHg", "eMg", "eLg"]] = 1\n    lab_s_abs[["eHgcum", "eMgcum", "eLgcum"]] = lab_s_abs.groupby("projection")[["eHg", "eMg", "eLg"]].cumprod()\n    lab_s_abs["eH"] = lab_s_abs["eHgcum"] * eH_init\n    lab_s_abs["eM"] = lab_s_abs["eMgcum"] * eM_init\n    lab_s_abs["eL"] = lab_s_abs["eLgcum"] * eL_init\n\n    pr = lab_s_gr.merge(Ak_df)\n    proj = pd.DataFrame()\n\n    for sc in pr.projection.unique():\n        p = pr[pr.projection == sc].copy()\n        p["Y_gr"] = (\n            (1 + p.A_gr) ** (1 / (1 - alpha)) *\n            (nu * k_bar / p["k"].shift() + (1 - nu)) ** (alpha / (1 - alpha)) *\n            (1 + p.e_gr) - 1\n        )\n        proj = pd.concat([proj, p])\n\n    proj = proj.dropna()\n    proj = proj.merge(pop, how="left")\n    proj["Y_pc_gr"] = (1 + proj["Y_gr"]) / (1 + proj["pop_gr"]) - 1\n    proj["A_contrib"] = proj.A_gr\n    proj["L_contrib"] = (1 - alpha) * proj.e_gr\n    proj["K_contrib"] = proj.Y_gr - proj.A_contrib - proj.L_contrib\n\n    if add_params:\n        proj["k_bar"] = k_bar\n        proj["A_gr_bar"] = A_gr_bar\n        proj["kappa"] = kappa\n        proj["nu"] = nu\n        proj = proj[\n            ["projection", "k_bar", "A_gr_bar", "kappa", "nu", "year", "eH_gr", "eM_gr", "eL_gr", "e_gr", "A_gr", "k",\n             "Y_gr", "pop_gr", "Y_pc_gr", "A_contrib", "L_contrib", "K_contrib"]\n        ]\n\n    if demographic_scenario:\n        proj = proj[proj.projection == demographic_scenario]\n\n    return proj\n\n\n\n# --- Widgets ---\n\nA_gr_bar_slider = pn.widgets.FloatSlider(name="\u0420\u0430\u0432\u043d\u043e\u0432\u0435\u0441\u0435\u043d \u0440\u0430\u0441\u0442\u0435\u0436 \u043d\u0430 \u041e\u0424\u041f (%)", start=0.5, end=2.0, step=0.5, value=1.0)\nkappa_slider = pn.widgets.FloatSlider(name="\u0421\u043a\u043e\u0440\u043e\u0441\u0442 \u043d\u0430 \u043a\u043e\u043d\u0432\u0435\u0440\u0433\u0435\u043d\u0446\u0438\u044f \u043d\u0430 \u041e\u0424\u041f", start=0.5, end=0.95, step=0.05, value=0.9)\nk_bar_slider = pn.widgets.FloatSlider(name="\u0420\u0430\u0432\u043d\u043e\u0432\u0435\u0441\u043d\u043e \u0441\u044a\u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0435 \u043a\u0430\u043f\u0438\u0442\u0430\u043b-\u0411\u0412\u041f", start=2.7, end=3.1, step=0.1, value=3.0)\nnu_slider = pn.widgets.FloatSlider(name="\u0421\u043a\u043e\u0440\u043e\u0441\u0442 \u043d\u0430 \u043a\u043e\u043d\u0432\u0435\u0440\u0433\u0435\u043d\u0446\u0438\u044f \u043d\u0430 \u0441\u044a\u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0435\u0442\u043e \u043a\u0430\u043f\u0438\u0442\u0430\u043b-\u0411\u0412\u041f", start=0.05, end=0.4, step=0.05, value=0.1)\n\n# Scenario selector\nscenario_select = pn.widgets.Select(\n    name="\u0414\u0435\u043c\u043e\u0433\u0440\u0430\u0444\u0441\u043a\u0438 \u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439", \n    options=['\u0411\u0430\u0437\u0438\u0441\u0435\u043d (BSL)', '\u0412\u0438\u0441\u043e\u043a\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (HMIGR)', '\u041d\u0438\u0441\u043a\u0430 \u0440\u0430\u0436\u0434\u0430\u0435\u043c\u043e\u0441\u0442 (LFRT)', '\u041d\u0438\u0441\u043a\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (LMIGR)', '\u041d\u0438\u0441\u043a\u0430 \u0441\u043c\u044a\u0440\u0442\u043d\u043e\u0441\u0442 (LMRT)', '\u041e\u0442\u0441\u044a\u0441\u0442\u0432\u0438\u0435 \u043d\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (NMIGR)'], \n    value="\u0411\u0430\u0437\u0438\u0441\u0435\u043d (BSL)"\n)\n\ncalculate_button = pn.widgets.Button(name="\u0418\u0437\u0447\u0438\u0441\u043b\u0438", button_type="primary")\n\n# Output placeholders\n\ntable_output = pn.widgets.Tabulator(pd.DataFrame(), height=250, show_index=False)\nplot_output = pn.pane.Matplotlib(height=400)\n\n# --- Callback ---\ndef run_projection(event=None):\n    A_gr_bar = A_gr_bar_slider.value/100.0\n    kappa = kappa_slider.value\n    k_bar = k_bar_slider.value\n    nu = nu_slider.value\n\n    scenario_coding = {"\u0411\u0430\u0437\u0438\u0441\u0435\u043d (BSL)":"BSL", \n                       "\u0412\u0438\u0441\u043e\u043a\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (HMIGR)":"HMIGR", \n                       "\u041d\u0438\u0441\u043a\u0430 \u0440\u0430\u0436\u0434\u0430\u0435\u043c\u043e\u0441\u0442 (LFRT)":"LFRT", \n                       "\u041d\u0438\u0441\u043a\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (LMIGR)":"LMIGR", \n                       "\u041d\u0438\u0441\u043a\u0430 \u0441\u043c\u044a\u0440\u0442\u043d\u043e\u0441\u0442 (LMRT)":"LMRT", \n                       "\u041e\u0442\u0441\u044a\u0441\u0442\u0432\u0438\u0435 \u043d\u0430 \u043c\u0438\u0433\u0440\u0430\u0446\u0438\u044f (NMIGR)":"NMIGR"}\n    scenario = scenario_coding[scenario_select.value]\n\n    df = compute_projection(A_gr_bar, kappa, k_bar, nu, demographic_scenario=scenario)\n\n    p = df.copy()\n    p = p.loc[p.year.isin([2025, 2030, 2035, 2040, 2045, 2050]),['year','e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']]\n    p[['e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']] *= 100\n    p['year'] = p['year'].map("{:d}".format)\n    for v in ['e_gr','Y_gr', 'pop_gr', 'Y_pc_gr']:\n        p[v] = p[v].map("{:.2f}".format)\n    p.rename(columns = {'year':'\u0413\u043e\u0434\u0438\u043d\u0430','e_gr':'\u0421\u0438\u043d\u0442\u0435\u0442\u0438\u0447\u043d\u0430 \u0437\u0430\u0435\u0442\u043e\u0441\u0442 (\u0433\u043e\u0434\u0438\u0448\u043d\u043e \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435, %)','Y_gr':'\u0420\u0435\u0430\u043b\u0435\u043d \u0411\u0412\u041f (\u0433\u043e\u0434\u0438\u0448\u043d\u043e \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435, %)', 'pop_gr':'\u041d\u0430\u0441\u0435\u043b\u0435\u043d\u0438\u0435 (\u0433\u043e\u0434\u0438\u0448\u043d\u043e \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435, %)', 'Y_pc_gr':'\u0420\u0435\u0430\u043b\u0435\u043d \u0411\u0412\u041f \u043d\u0430 \u0433\u043b\u0430\u0432\u0430 \u043e\u0442 \u043d\u0430\u0441\u0435\u043b\u0435\u043d\u0438\u0435\u0442\u043e (\u0433\u043e\u0434\u0438\u0448\u043d\u043e \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435, %)'}, inplace=True)\n\n    gr = df.copy()\n    gr = gr.loc[gr.year.isin([2025, 2030, 2035, 2040, 2045, 2050]),['year','Y_gr', 'A_contrib', 'L_contrib', 'K_contrib']]\n    gr[['Y_gr', 'A_contrib', 'L_contrib', 'K_contrib']] *= 100\n\n    table_output.value = p\n\n    # Plot: Y_gr as line, contributions as stacked bars\n    fig, ax = plt.subplots(figsize=(8, 5))\n\n    # Bar width and offset\n    width = 1\n    x = gr["year"]\n    A = gr["A_contrib"]\n    L = gr["L_contrib"]\n    K = gr["K_contrib"]\n\n    base_pos = np.zeros_like(A)\n    base_neg = np.zeros_like(A)\n\n\n\n    ax.bar(x, A, width, label="\u041e\u0424\u041f", color="tab:blue")\n    base_pos += np.where(A > 0, A, 0)\n    base_neg += np.where(A < 0, A, 0)\n    \n    ax.bar(x, L, width, bottom=np.where(L >= 0, base_pos, base_neg), label="\u0421\u0438\u043d\u0442\u0435\u0442\u0438\u0447\u0435\u043d \u0442\u0440\u0443\u0434", color="tab:orange")\n    base_pos += np.where(L > 0, L, 0)\n    base_neg += np.where(L < 0, L, 0)\n    \n    ax.bar(x, K, width, bottom=np.where(K >= 0, base_pos, base_neg), label="\u041a\u0430\u043f\u0438\u0442\u0430\u043b", color="tab:green")\n\n    # Overlay line plot for Y_gr\n    ax.plot(x, gr["Y_gr"], marker='o', color='black', linewidth=2, label="\u0420\u0435\u0430\u043b\u0435\u043d \u0411\u0412\u041f (\u0433\u043e\u0434\u0438\u0448\u043d\u043e \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435, %)")\n    ax.axhline(0, color="black", linewidth=0.8)\n\n    ax.set_title("\u041f\u0440\u0438\u043d\u043e\u0441\u0438 \u0437\u0430 \u0440\u0430\u0441\u0442\u0435\u0436\u0430 \u043d\u0430 \u0440\u0435\u0430\u043b\u043d\u0438\u044f \u0411\u0412\u041f")\n    # ax.set_xlabel("\xd0\u201c\xd0\xbe\xd0\xb4\xd0\xb8\xd0\xbd\xd0\xb0")\n    ax.set_ylabel("%")\n    ax.legend()\n    ax.set_xticks(x)\n    ax.set_xticklabels(x.astype(str), rotation=45)\n\n    plot_output.object = fig\n\n# Attach callback\ncalculate_button.on_click(run_projection)\n\n# Initial placeholder render (optional)\nrun_projection()\n\n\n\n# --- Layout ---\ncontrols = pn.Column(\n    pn.pane.Markdown("### \u041e\u0441\u043d\u043e\u0432\u043d\u0438 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u0438 \u043d\u0430 \u043c\u043e\u0434\u0435\u043b\u0430"),\n    A_gr_bar_slider,\n    kappa_slider,\n    k_bar_slider,\n    nu_slider,\n    scenario_select,\n    calculate_button,\n)\n\nlayout = pn.Column(pn.pane.Markdown("# \u041f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0437\u0430 \u043e\u0446\u0435\u043d\u043a\u0430 \u043d\u0430 \u043c\u0430\u043a\u0440\u043e\u0438\u043a\u043e\u043d\u043e\u043c\u0438\u0447\u0435\u0441\u043a\u0438\u0442\u0435 \u0435\u0444\u0435\u043a\u0442\u0438 \u043e\u0442 \u0434\u0435\u043c\u043e\u0433\u0440\u0430\u0444\u0441\u043a\u0430\u0442\u0430 \u0434\u0438\u043d\u0430\u043c\u0438\u043a\u0430"),\n    controls,\n    pn.pane.Markdown("### \u0420\u0435\u0437\u0443\u043b\u0442\u0430\u0442\u0438"),\n    table_output,\n    pn.pane.Markdown("### \u0424\u0430\u043a\u0442\u043e\u0440\u0438 \u0437\u0430 \u0438\u043a\u043e\u043d\u043e\u043c\u0438\u0447\u0435\u0441\u043a\u0438\u044f \u0440\u0430\u0441\u0442\u0435\u0436"),\n    plot_output,\n    pn.pane.HTML(\n        '<a href="documentation/index.html" target="_blank">\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u044f (HTML)</a>',\n        sizing_mode="stretch_width"\n    ),\n    pn.pane.HTML(\n        '<a href="documentation/LM_app_doc.pdf" target="_blank">\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u044f (PDF)</a>',\n        sizing_mode="stretch_width"\n    )\n)\n\n# layout.servable()\nlayout.servable(target='app')\n\n\nawait write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    from panel.io.pyodide import _convert_json_patch
    state.curdoc.apply_json_patch(_convert_json_patch(patch), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()