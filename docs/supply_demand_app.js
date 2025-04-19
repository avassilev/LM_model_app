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
  \nimport asyncio\n\nfrom panel.io.pyodide import init_doc, write_doc\n\ninit_doc()\n\nimport panel as pn\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\npn.extension('tabulator', sizing_mode="stretch_width")\n\n# Distinct intercepts\ndemand_intercepts = {\n    "Low": 100,\n    "Normal": 80,\n    "High": 60\n}\n\nsupply_intercepts = {\n    "Low": 10,\n    "Normal": 30,\n    "High": 50\n}\n\n# Widgets\nsupply_selector = pn.widgets.Select(name="Supply Level", options=["Low", "Normal", "High"], value="Normal")\ndemand_selector = pn.widgets.Select(name="Demand Level", options=["Low", "Normal", "High"], value="Normal")\n\ndemand_slope_slider = pn.widgets.FloatSlider(name="Demand Slope", start=-5, end=-0.1, step=0.1, value=-1)\nsupply_slope_slider = pn.widgets.FloatSlider(name="Supply Slope", start=0.1, end=5, step=0.1, value=1)\n\n# Output widgets\nresult_table = pn.widgets.Tabulator(pd.DataFrame({"Price": [], "Quantity": []}), height=150)\nplot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width")\n\ndef update(*events):\n    # Get current values\n    intercept_supply = supply_intercepts[supply_selector.value]\n    intercept_demand = demand_intercepts[demand_selector.value]\n    slope_demand = demand_slope_slider.value\n    slope_supply = supply_slope_slider.value\n\n    # Calculate equilibrium\n    try:\n        eq_quantity = (intercept_demand - intercept_supply) / (slope_supply - slope_demand)\n        eq_price = intercept_demand + slope_demand * eq_quantity\n    except ZeroDivisionError:\n        eq_quantity, eq_price = np.nan, np.nan\n\n    # Update table\n    df = pd.DataFrame({"Price": [round(eq_price, 2)], "Quantity": [round(eq_quantity, 2)]})\n    result_table.value = df\n\n    # Generate curves\n    q = np.linspace(0, 100, 100)\n    demand = intercept_demand + slope_demand * q\n    supply = intercept_supply + slope_supply * q\n\n    fig, ax = plt.subplots()\n    ax.plot(q, demand, label="Demand", color="blue")\n    ax.plot(q, supply, label="Supply", color="orange")\n    \n    # Plot equilibrium if it's within view\n    if 0 <= eq_quantity <= 100 and 0 <= eq_price <= 120:\n        ax.plot(eq_quantity, eq_price, 'ro', label="Equilibrium")\n\n    ax.set_xlabel("Quantity")\n    ax.set_ylabel("Price")\n    ax.set_xlim(0, 100)\n    ax.set_ylim(0, 120)\n    ax.legend()\n    ax.set_title("Supply and Demand")\n\n    plot_pane.object = fig\n\n# Watch all widgets\nfor w in [supply_selector, demand_selector, supply_slope_slider, demand_slope_slider]:\n    w.param.watch(update, 'value')\n\n# Initial plot\nupdate()\n\n# Layout\ncontrols = pn.Column(\n    pn.Row(supply_selector, demand_selector),\n    pn.Row(supply_slope_slider, demand_slope_slider),\n)\n\napp = pn.Column(\n    controls,\n    plot_pane,\n    pn.pane.Markdown("### Equilibrium"),\n    result_table\n)\n\napp.servable()\n\n# Reminder:\n# panel convert supply_demand_app.py --to pyodide-worker --out docs --index\n\nawait write_doc()
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