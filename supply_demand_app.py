import panel as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pn.extension('tabulator', sizing_mode="stretch_width")

# Distinct intercepts
demand_intercepts = {
    "Low": 100,
    "Normal": 80,
    "High": 60
}

supply_intercepts = {
    "Low": 10,
    "Normal": 30,
    "High": 50
}

# Widgets
supply_selector = pn.widgets.Select(name="Supply Level", options=["Low", "Normal", "High"], value="Normal")
demand_selector = pn.widgets.Select(name="Demand Level", options=["Low", "Normal", "High"], value="Normal")

demand_slope_slider = pn.widgets.FloatSlider(name="Demand Slope", start=-5, end=-0.1, step=0.1, value=-1)
supply_slope_slider = pn.widgets.FloatSlider(name="Supply Slope", start=0.1, end=5, step=0.1, value=1)

# Output widgets
result_table = pn.widgets.Tabulator(pd.DataFrame({"Price": [], "Quantity": []}), height=150)
plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width")

def update(*events):
    # Get current values
    intercept_supply = supply_intercepts[supply_selector.value]
    intercept_demand = demand_intercepts[demand_selector.value]
    slope_demand = demand_slope_slider.value
    slope_supply = supply_slope_slider.value

    # Calculate equilibrium
    try:
        eq_quantity = (intercept_demand - intercept_supply) / (slope_supply - slope_demand)
        eq_price = intercept_demand + slope_demand * eq_quantity
    except ZeroDivisionError:
        eq_quantity, eq_price = np.nan, np.nan

    # Update table
    df = pd.DataFrame({"Price": [round(eq_price, 2)], "Quantity": [round(eq_quantity, 2)]})
    result_table.value = df

    # Generate curves
    q = np.linspace(0, 100, 100)
    demand = intercept_demand + slope_demand * q
    supply = intercept_supply + slope_supply * q

    fig, ax = plt.subplots()
    ax.plot(q, demand, label="Demand", color="blue")
    ax.plot(q, supply, label="Supply", color="orange")
    
    # Plot equilibrium if it's within view
    if 0 <= eq_quantity <= 100 and 0 <= eq_price <= 120:
        ax.plot(eq_quantity, eq_price, 'ro', label="Equilibrium")

    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.legend()
    ax.set_title("Supply and Demand")

    plot_pane.object = fig

# Watch all widgets
for w in [supply_selector, demand_selector, supply_slope_slider, demand_slope_slider]:
    w.param.watch(update, 'value')

# Initial plot
update()

# Layout
controls = pn.Column(
    pn.Row(supply_selector, demand_selector),
    pn.Row(supply_slope_slider, demand_slope_slider),
)

app = pn.Column(
    controls,
    plot_pane,
    pn.pane.Markdown("### Equilibrium"),
    result_table
)

app.servable()

# Reminder:
# panel convert supply_demand_app.py --to pyodide-worker --out docs
# make a copy of convert supply_demand_app.html and rename it to index.html
# to test: python -m http.server 8000
# then open in browser: http://localhost:8000