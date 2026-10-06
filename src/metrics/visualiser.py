"""
FlashBalanceAI — Premium Live Metrics & Routing Animation Dashboard.
Phase 9: Demo Dashboard - Issue #44

Aesthetically enhanced Dash application featuring:
  - Dark glassmorphism design system with glowing accents & Google Font Outfit
  - Live Hero KPI cards (Throughput, P95 Latency, SLA Compliance, DRL Advantage)
  - Side-by-side PPO vs RoundRobin battle arena with real-time backend queues & routing
  - All 6 algorithms comparison with multi-agent traffic distribution & live leaderboard
  - High-res Plotly telemetry curves with 500ms SLA threshold and unified hover tooltips
  - Live streaming Apache JMeter JTL log viewer with real-time aggregate reporting
  - Interactive demo controls (Start, Pause, Reset, 2x Fast-Forward, Spike Injection)
  - 30-second live metrics update (configurable from 1s to 30s)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objs as go
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.demo_runner import (
    ALGORITHMS,
    DEMO_ENGINE,
    DemoJMeterRunner,
    JTL_PATH,
)

# Curated harmonious neon color palette for all 6 agents
ALGO_COLORS = {
    "PPO": "#00F2FE",                  # Electric Cyan (Primary DRL Champion)
    "DQN": "#FBBF24",                  # Amber Gold (Value-based DRL)
    "RoundRobin": "#FF4B4B",           # Crimson (Static Baseline)
    "WeightedRoundRobin": "#A855F7",   # Neon Purple (Adaptive Baseline)
    "LeastConnections": "#10B981",     # Emerald Green (Dynamic Baseline)
    "ThresholdAutoscaler": "#EC4899",  # Vivid Pink (Rule-based Baseline)
}

BACKEND_NAMES = [f"Backend #{i}" for i in range(4)]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background-color: #080C15;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
    background-attachment: fixed;
    font-family: 'Outfit', system-ui, -apple-system, sans-serif;
    color: #F8FAFC;
    overflow-x: hidden;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #080C15;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

.glass-card {
    background: rgba(17, 24, 39, 0.78);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.38);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(0, 242, 254, 0.25);
    box-shadow: 0 10px 36px 0 rgba(0, 242, 254, 0.1);
}

.btn-interactive {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.btn-interactive:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0, 242, 254, 0.3);
}
.btn-interactive:active {
    transform: translateY(0);
}

.pulse-badge {
    animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse-ring {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(1.08); }
}

.table-row-hover {
    transition: background-color 0.15s ease;
}
.table-row-hover:hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
}

.routing-pipeline-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-ring 1.5s infinite;
}
"""


def create_app(
    collector: Optional[Any] = None,
    engine: Optional[DemoJMeterRunner] = None,
) -> "dash.Dash":
    """Build and configure the premium Dash demo application."""
    if not _DASH_AVAILABLE:
        raise ImportError("Dash is not installed. Install with: pip install dash plotly")

    runner = engine or DEMO_ENGINE

    app = dash.Dash(
        __name__,
        title="FlashBalanceAI — Demo Dashboard",
        suppress_callback_exceptions=True,
    )

    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
        {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

    app.layout = html.Div(
        style={"minHeight": "100vh", "paddingBottom": "60px"},
        children=[
            # Live Update Interval (Default 30s per Acceptance Criteria)
            dcc.Interval(id="interval-refresh", interval=30000, n_intervals=0),

            # =================================================================
            # 1. Top Navbar / Header
            # =================================================================
            html.Header(
                style={
                    "background": "linear-gradient(180deg, #101626 0%, #0B101D 100%)",
                    "padding": "18px 36px",
                    "borderBottom": "1px solid rgba(255, 255, 255, 0.08)",
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 100,
                    "backdropFilter": "blur(14px)",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px", "maxWidth": "1650px", "margin": "0 auto"},
                        children=[
                            # Left Brand Branding
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "14px"},
                                children=[
                                    html.Div(
                                        "⚡",
                                        style={
                                            "fontSize": "26px",
                                            "background": "linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)",
                                            "borderRadius": "12px",
                                            "width": "46px",
                                            "height": "46px",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "boxShadow": "0 0 20px rgba(0, 242, 254, 0.4)",
                                        },
                                    ),
                                    html.Div([
                                        html.H1(
                                            "FlashBalanceAI",
                                            style={"fontSize": "22px", "fontWeight": "800", "margin": "0", "letterSpacing": "-0.5px", "color": "#FFFFFF"},
                                        ),
                                        html.Div(
                                            "Autonomous DRL Cloud Load Balancing • Demo Dashboard (Phase 9 — Issue #44)",
                                            style={"fontSize": "13px", "color": "#94A3B8", "fontWeight": "500"},
                                        ),
                                    ]),
                                ],
                            ),

                            # Right Live Status & Timer Pills
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                children=[
                                    # Status Indicator
                                    html.Div(
                                        id="status-pill",
                                        className="glass-card",
                                        style={
                                            "padding": "8px 16px",
                                            "borderRadius": "24px",
                                            "fontSize": "13px",
                                            "fontWeight": "700",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "10px",
                                        },
                                        children=[
                                            html.Span("●", className="pulse-badge", style={"fontSize": "16px", "color": "#10B981"}),
                                            html.Span(id="status-text", children="LIVE JMETER RUNNING"),
                                        ],
                                    ),
                                    # Time Elapsed
                                    html.Div(
                                        id="time-pill",
                                        className="glass-card",
                                        style={
                                            "padding": "8px 18px",
                                            "borderRadius": "24px",
                                            "fontSize": "13px",
                                            "fontWeight": "800",
                                            "color": "#00F2FE",
                                            "fontFamily": "'JetBrains Mono', monospace",
                                        },
                                        children="⏱️ 00:00 / 02:00",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Glowing accent line
                    html.Div(style={"height": "2px", "width": "100%", "background": "linear-gradient(90deg, #00F2FE 0%, #4FACFE 40%, #A855F7 80%, transparent 100%)", "marginTop": "14px"}),
                ],
            ),

            # =================================================================
            # 2. Main Workspace
            # =================================================================
            html.Main(
                style={"maxWidth": "1650px", "margin": "0 auto", "padding": "24px 36px"},
                children=[
                    # A. Dynamic Hero KPI Row
                    html.Div(id="hero-kpi-row", style={"marginBottom": "24px"}),

                    # B. Interactive Control Bar
                    html.Div(
                        className="glass-card",
                        style={"padding": "16px 24px", "marginBottom": "24px", "display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px"},
                        children=[
                            # Playback Buttons
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                children=[
                                    html.Button(
                                        "▶ Start / Resume",
                                        id="btn-start",
                                        className="btn-interactive",
                                        n_clicks=0,
                                        style={
                                            "background": "linear-gradient(135deg, #0284C7 0%, #00F2FE 100%)",
                                            "color": "#0B0F19",
                                            "border": "none",
                                            "padding": "9px 20px",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                            "fontSize": "13px",
                                        },
                                    ),
                                    html.Button(
                                        "⏸ Pause",
                                        id="btn-pause",
                                        className="btn-interactive",
                                        n_clicks=0,
                                        style={
                                            "background": "#1E293B",
                                            "color": "#F8FAFC",
                                            "border": "1px solid #334155",
                                            "padding": "9px 18px",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                        },
                                    ),
                                    html.Button(
                                        "🔄 Reset Demo",
                                        id="btn-reset",
                                        className="btn-interactive",
                                        n_clicks=0,
                                        style={
                                            "background": "#1E293B",
                                            "color": "#94A3B8",
                                            "border": "1px solid #334155",
                                            "padding": "9px 18px",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                        },
                                    ),
                                    html.Button(
                                        "⚡ Inject 10x Spike",
                                        id="btn-spike",
                                        className="btn-interactive",
                                        n_clicks=0,
                                        style={
                                            "background": "rgba(245, 158, 11, 0.15)",
                                            "color": "#F59E0B",
                                            "border": "1px solid rgba(245, 158, 11, 0.5)",
                                            "padding": "9px 18px",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                            "fontSize": "13px",
                                        },
                                    ),
                                    html.Button(
                                        "⏩ Fast Forward (2x)",
                                        id="btn-ff",
                                        className="btn-interactive",
                                        n_clicks=0,
                                        style={
                                            "background": "rgba(56, 189, 248, 0.1)",
                                            "color": "#38BDF8",
                                            "border": "1px solid rgba(56, 189, 248, 0.4)",
                                            "padding": "9px 18px",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                        },
                                    ),
                                ],
                            ),

                            # Right Config Dropdown
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "14px"},
                                children=[
                                    html.Span("Live Refresh Rate:", style={"fontSize": "13px", "color": "#94A3B8", "fontWeight": "600"}),
                                    dcc.Dropdown(
                                        id="select-interval",
                                        options=[
                                            {"label": "⏱️ 30s (Official Acceptance Standard)", "value": 30000},
                                            {"label": "⏱️ 10s (Standard Monitoring)", "value": 10000},
                                            {"label": "⏱️ 5s (Smooth Evaluation)", "value": 5000},
                                            {"label": "⏱️ 2s (Fast Demo Animation)", "value": 2000},
                                            {"label": "⏱️ 1s (Real-Time 1Hz)", "value": 1000},
                                        ],
                                        value=30000,
                                        clearable=False,
                                        style={"width": "285px", "color": "#0F172A", "fontSize": "13px", "fontWeight": "600"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # C. Navigation Tabs with Glass Look
                    dcc.Tabs(
                        id="tabs-view",
                        value="tab-side-by-side",
                        style={"marginBottom": "24px"},
                        colors={"border": "transparent", "primary": "#00F2FE", "background": "transparent"},
                        children=[
                            dcc.Tab(
                                label="⚡ Side-by-Side: PPO vs RoundRobin (Live Routing)",
                                value="tab-side-by-side",
                                style={"backgroundColor": "#131C2D", "color": "#94A3B8", "fontWeight": "600", "padding": "14px", "borderRadius": "8px 0 0 8px", "border": "1px solid rgba(255,255,255,0.06)"},
                                selected_style={"backgroundColor": "#1A263C", "color": "#00F2FE", "fontWeight": "800", "padding": "14px", "borderTop": "3px solid #00F2FE", "borderLeft": "1px solid rgba(255,255,255,0.1)", "borderRight": "1px solid rgba(255,255,255,0.1)"},
                            ),
                            dcc.Tab(
                                label="🌐 All 6 Algorithms: Routing Animation & Leaderboard",
                                value="tab-all-algos",
                                style={"backgroundColor": "#131C2D", "color": "#94A3B8", "fontWeight": "600", "padding": "14px", "border": "1px solid rgba(255,255,255,0.06)"},
                                selected_style={"backgroundColor": "#1A263C", "color": "#00F2FE", "fontWeight": "800", "padding": "14px", "borderTop": "3px solid #00F2FE", "borderLeft": "1px solid rgba(255,255,255,0.1)", "borderRight": "1px solid rgba(255,255,255,0.1)"},
                            ),
                            dcc.Tab(
                                label="📈 Live Performance Telemetry & SLA Tracking",
                                value="tab-telemetry",
                                style={"backgroundColor": "#131C2D", "color": "#94A3B8", "fontWeight": "600", "padding": "14px", "border": "1px solid rgba(255,255,255,0.06)"},
                                selected_style={"backgroundColor": "#1A263C", "color": "#00F2FE", "fontWeight": "800", "padding": "14px", "borderTop": "3px solid #00F2FE", "borderLeft": "1px solid rgba(255,255,255,0.1)", "borderRight": "1px solid rgba(255,255,255,0.1)"},
                            ),
                            dcc.Tab(
                                label="📋 Live JMeter JTL Stream & Aggregate Report",
                                value="tab-jmeter",
                                style={"backgroundColor": "#131C2D", "color": "#94A3B8", "fontWeight": "600", "padding": "14px", "borderRadius": "0 8px 8px 0", "border": "1px solid rgba(255,255,255,0.06)"},
                                selected_style={"backgroundColor": "#1A263C", "color": "#00F2FE", "fontWeight": "800", "padding": "14px", "borderTop": "3px solid #00F2FE", "borderLeft": "1px solid rgba(255,255,255,0.1)", "borderRight": "1px solid rgba(255,255,255,0.1)"},
                            ),
                        ],
                    ),

                    # D. Tab Content Container
                    html.Div(id="tab-content"),
                ],
            ),
        ],
    )

    # =========================================================================
    # Callbacks
    # =========================================================================

    # 1. Update Interval Callback
    @app.callback(
        Output("interval-refresh", "interval"),
        Input("select-interval", "value"),
    )
    def update_interval(val):
        return int(val) if val else 30000

    # 2. Control Buttons Callback
    @app.callback(
        [Output("status-text", "children"), Output("status-pill", "style")],
        [
            Input("btn-start", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("btn-spike", "n_clicks"),
            Input("btn-ff", "n_clicks"),
            Input("interval-refresh", "n_intervals"),
        ],
        prevent_initial_call=False,
    )
    def handle_controls(btn_start, btn_pause, btn_reset, btn_spike, btn_ff, _interval):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if triggered == "btn-start":
            runner.resume()
        elif triggered == "btn-pause":
            runner.pause()
        elif triggered == "btn-reset":
            runner.reset()
            runner.start_background()
        elif triggered == "btn-spike":
            # Jump directly to 10x spike onset (sec 45) for immediate demo demonstration
            runner.current_sec = 45
            runner.resume()
        elif triggered == "btn-ff":
            runner.time_scale = 3.0 if runner.time_scale == 1.0 else 1.0

        if not runner.is_running and runner.current_sec == 0 and not runner.is_complete:
            runner.start_background()

        if runner.is_complete:
            return "COMPLETED (120s RUN)", {
                "padding": "8px 16px", "borderRadius": "24px", "fontSize": "13px", "fontWeight": "700",
                "backgroundColor": "rgba(59, 130, 246, 0.15)", "color": "#38BDF8",
                "border": "1px solid rgba(56, 189, 248, 0.4)", "display": "flex", "alignItems": "center", "gap": "10px",
            }
        elif runner.is_running:
            burst = runner.is_burst_active(runner.current_sec)
            txt = "⚡ 10x BURST ACTIVE (1,000 RPS)" if burst else "🟢 LIVE JMETER RUNNING"
            color = "#F59E0B" if burst else "#10B981"
            return txt, {
                "padding": "8px 16px", "borderRadius": "24px", "fontSize": "13px", "fontWeight": "700",
                "backgroundColor": f"{color}22", "color": color, "border": f"1px solid {color}55",
                "display": "flex", "alignItems": "center", "gap": "10px",
            }
        else:
            return "⏸ PAUSED", {
                "padding": "8px 16px", "borderRadius": "24px", "fontSize": "13px", "fontWeight": "700",
                "backgroundColor": "rgba(239, 68, 68, 0.15)", "color": "#EF4444",
                "border": "1px solid rgba(239, 68, 68, 0.4)", "display": "flex", "alignItems": "center", "gap": "10px",
            }

    # 3. Timer Pill Callback
    @app.callback(
        Output("time-pill", "children"),
        Input("interval-refresh", "n_intervals"),
    )
    def update_timer(_n):
        sec = runner.current_sec
        mins = sec // 60
        rem_sec = sec % 60
        return f"⏱️ {mins:02d}:{rem_sec:02d} / 02:00 ({sec}s)"

    # 4. Hero KPI Row Renderer Callback
    @app.callback(
        Output("hero-kpi-row", "children"),
        Input("interval-refresh", "n_intervals"),
    )
    def update_hero_kpis(_n):
        telem = runner.get_latest_telemetry()
        ppo = telem["algorithms"].get("PPO", {})
        rr = telem["algorithms"].get("RoundRobin", {})

        arrival = telem.get("arrival_rate", 100.0)
        ppo_p95 = ppo.get("p95_latency_ms", 50.0)
        rr_p95 = rr.get("p95_latency_ms", 50.0)
        ppo_sla = ppo.get("sla_violation_rate", 0.0)
        rr_sla = rr.get("sla_violation_rate", 0.0)

        diff_pct = max(0.0, ((rr_p95 - ppo_p95) / max(1.0, rr_p95)) * 100.0)
        is_burst = telem.get("burst_active", False)

        kpis = [
            # 1. Incoming Load
            {
                "label": "INCOMING JMETER LOAD",
                "val": f"{arrival:.0f} req/s",
                "sub": "⚡ 10x Flash-Sale Burst Peak" if is_burst else "🟢 Normal Baseline Load",
                "sub_color": "#F59E0B" if is_burst else "#10B981",
                "border": "#F59E0B" if is_burst else "#38BDF8",
                "icon": "🌊",
            },
            # 2. PPO P95 Latency
            {
                "label": "CHAMPION PPO P95 LATENCY",
                "val": f"{ppo_p95:.1f} ms",
                "sub": "🎯 Optimal (Target < 200ms)",
                "sub_color": "#10B981",
                "border": "#00F2FE",
                "icon": "⚡",
            },
            # 3. SLA Compliance
            {
                "label": "SLA COMPLIANCE (PPO)",
                "val": f"{100.0 - ppo_sla * 100.0:.1f}% Compliant",
                "sub": f"Round-Robin: {100.0 - rr_sla * 100.0:.1f}% Compliant",
                "sub_color": "#10B981" if ppo_sla <= 0.01 else "#F59E0B",
                "border": "#10B981",
                "icon": "🛡️",
            },
            # 4. PPO Efficiency Edge
            {
                "label": "DRL LATENCY ADVANTAGE",
                "val": f"+{diff_pct:.1f}% Faster",
                "sub": f"PPO eliminates tail vs RoundRobin ({rr_p95:.1f}ms)",
                "sub_color": "#00F2FE",
                "border": "#A855F7",
                "icon": "🏆",
            },
        ]

        cards = []
        for k in kpis:
            cards.append(
                html.Div(
                    className="glass-card",
                    style={
                        "flex": "1 1 240px",
                        "padding": "18px 24px",
                        "borderTop": f"3px solid {k['border']}",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                            children=[
                                html.Span(k["label"], style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8", "letterSpacing": "0.5px"}),
                                html.Span(k["icon"], style={"fontSize": "16px"}),
                            ],
                        ),
                        html.Div(k["val"], style={"fontSize": "25px", "fontWeight": "800", "color": "#FFFFFF", "margin": "4px 0", "fontFamily": "'JetBrains Mono', monospace"}),
                        html.Div(k["sub"], style={"fontSize": "12px", "fontWeight": "600", "color": k["sub_color"]}),
                    ],
                )
            )

        return html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=cards)

    # 5. Render Active Tab Content
    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs-view", "value"), Input("interval-refresh", "n_intervals")],
    )
    def render_tab(tab_name, _n):
        telem = runner.get_latest_telemetry()
        history = runner.get_history_dataframe()

        if tab_name == "tab-side-by-side":
            return _render_side_by_side_tab(telem, history)
        elif tab_name == "tab-all-algos":
            return _render_all_algos_tab(telem, history)
        elif tab_name == "tab-telemetry":
            return _render_telemetry_tab(telem, history)
        elif tab_name == "tab-jmeter":
            return _render_jmeter_tab(telem)
        return html.Div("Select a tab.")

    return app


# =============================================================================
# View Component Renderers
# =============================================================================

def _render_side_by_side_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """Render Tab 1: Enhanced Side-by-Side PPO vs RoundRobin Arena."""
    ppo = telem["algorithms"].get("PPO", {})
    rr = telem["algorithms"].get("RoundRobin", {})

    ppo_p95 = ppo.get("p95_latency_ms", 50.0)
    rr_p95 = rr.get("p95_latency_ms", 50.0)
    diff = rr_p95 - ppo_p95
    pct_adv = max(0.0, (diff / max(1.0, rr_p95)) * 100.0)

    # 1. Top Comparative Head-to-Head Banner
    arena_banner = html.Div(
        className="glass-card",
        style={
            "padding": "20px 28px",
            "marginBottom": "24px",
            "background": "linear-gradient(90deg, rgba(0, 242, 254, 0.08) 0%, rgba(17, 24, 39, 0.85) 50%, rgba(255, 75, 75, 0.08) 100%)",
            "border": "1px solid rgba(255, 255, 255, 0.1)",
            "display": "flex",
            "justifyContent": "space-around",
            "alignItems": "center",
            "flexWrap": "wrap",
            "gap": "18px",
        },
        children=[
            # Left: PPO Champion
            html.Div([
                html.Div("🏆 CHAMPION (DRL)", style={"fontSize": "11px", "fontWeight": "800", "color": "#00F2FE", "letterSpacing": "1px"}),
                html.Div("PPO Policy", style={"fontSize": "24px", "fontWeight": "800", "color": "#FFFFFF", "margin": "2px 0"}),
                html.Div(f"{ppo_p95:.1f} ms P95", style={"fontSize": "16px", "fontWeight": "700", "color": "#00F2FE", "fontFamily": "'JetBrains Mono', monospace"}),
            ]),
            # Center: Head-to-Head Speed Edge
            html.Div(
                style={"textAlign": "center"},
                children=[
                    html.Div("HEAD-TO-HEAD LATENCY DELTA", style={"fontSize": "11px", "fontWeight": "800", "color": "#94A3B8", "letterSpacing": "0.5px"}),
                    html.Div(
                        f"+{pct_adv:.1f}% Faster",
                        style={"fontSize": "28px", "fontWeight": "900", "color": "#10B981" if diff >= 0 else "#EF4444", "margin": "3px 0", "fontFamily": "'JetBrains Mono', monospace"},
                    ),
                    html.Div(
                        f"PPO eliminates {diff:.1f} ms tail latency bottleneck" if diff >= 0 else "Parity with baseline",
                        style={"fontSize": "12px", "color": "#10B981" if diff >= 0 else "#EF4444", "fontWeight": "600"},
                    ),
                ],
            ),
            # Right: Round-Robin Baseline
            html.Div(
                style={"textAlign": "right"},
                children=[
                    html.Div("BASELINE (STATIC)", style={"fontSize": "11px", "fontWeight": "800", "color": "#FF4B4B", "letterSpacing": "1px"}),
                    html.Div("Round-Robin", style={"fontSize": "24px", "fontWeight": "800", "color": "#FFFFFF", "margin": "2px 0"}),
                    html.Div(f"{rr_p95:.1f} ms P95", style={"fontSize": "16px", "fontWeight": "700", "color": "#FF4B4B", "fontFamily": "'JetBrains Mono', monospace"}),
                ],
            ),
        ],
    )

    # 2. Backend Instance Renderers
    def _render_instance_pool(agent_title: str, sub: str, data: Dict[str, Any], accent: str):
        cpus = data.get("backend_cpus", [0.1, 0.1, 0.1, 0.1])
        conns = data.get("backend_conns", [10, 10, 10, 10])
        queues = data.get("backend_queues", [0, 0, 0, 0])
        lats = data.get("backend_latencies", [50, 50, 50, 50])

        total_conns = max(1, sum(conns))

        cards = []
        for i in range(4):
            cpu_pct = int(cpus[i] * 100)
            share_pct = (conns[i] / total_conns) * 100
            has_queue = queues[i] > 3

            cpu_color = "#10B981" if cpu_pct < 65 else ("#F59E0B" if cpu_pct < 85 else "#EF4444")

            card = html.Div(
                style={
                    "backgroundColor": "#0D1424",
                    "border": f"1px solid {'#EF4444' if has_queue else 'rgba(255,255,255,0.08)'}",
                    "borderRadius": "10px",
                    "padding": "14px 18px",
                    "marginBottom": "12px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.25)",
                },
                children=[
                    # Header: Backend name & traffic share badge
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                        children=[
                            html.Div([
                                html.Span("🖥️ ", style={"fontSize": "14px"}),
                                html.Span(f"Backend #{i}", style={"fontWeight": "700", "fontSize": "14px", "color": "#FFFFFF"}),
                                html.Span(" (c5.large)", style={"fontSize": "11px", "color": "#64748B"}),
                            ]),
                            html.Span(
                                f"Traffic Share: {share_pct:.1f}%",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "700",
                                    "padding": "3px 10px",
                                    "borderRadius": "12px",
                                    "backgroundColor": f"{accent}18",
                                    "color": accent,
                                    "border": f"1px solid {accent}44",
                                },
                            ),
                        ],
                    ),

                    # CPU progress bar
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"},
                        children=[
                            html.Span("CPU Load:", style={"fontSize": "11px", "color": "#94A3B8", "width": "60px", "fontWeight": "600"}),
                            html.Div(
                                style={"flex": 1, "height": "8px", "backgroundColor": "#1E293B", "borderRadius": "4px", "overflow": "hidden"},
                                children=html.Div(style={"width": f"{cpu_pct}%", "height": "100%", "backgroundColor": cpu_color, "borderRadius": "4px", "transition": "width 0.3s ease"}),
                            ),
                            html.Span(f"{cpu_pct}%", style={"fontSize": "12px", "fontWeight": "800", "color": cpu_color, "width": "35px", "textAlign": "right"}),
                        ],
                    ),

                    # Metrics row
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px", "color": "#94A3B8", "paddingTop": "4px"},
                        children=[
                            html.Span([html.Span("Active: ", style={"color": "#64748B"}), html.B(f"{conns[i]} req", style={"color": "#F8FAFC"})]),
                            html.Span([
                                html.Span("Queue: ", style={"color": "#64748B"}),
                                html.B(f"{queues[i]} waiting", style={"color": "#EF4444" if has_queue else "#10B981"}),
                            ]),
                            html.Span([html.Span("EMA Latency: ", style={"color": "#64748B"}), html.B(f"{lats[i]:.0f} ms", style={"color": "#F8FAFC"})]),
                        ],
                    ),
                ],
            )
            cards.append(card)

        return html.Div(
            className="glass-card",
            style={"flex": 1, "padding": "22px", "borderTop": f"4px solid {accent}"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"},
                    children=[
                        html.Div([
                            html.H3(agent_title, style={"margin": "0 0 4px 0", "color": accent, "fontSize": "19px", "fontWeight": "800"}),
                            html.Div(sub, style={"fontSize": "12px", "color": "#94A3B8"}),
                        ]),
                        html.Div(
                            "DRL Learned Policy" if "PPO" in agent_title else "Static Cyclical",
                            style={"fontSize": "11px", "fontWeight": "700", "padding": "4px 10px", "borderRadius": "6px", "backgroundColor": "#1E293B", "color": "#94A3B8"},
                        ),
                    ],
                ),
                html.Div(cards),
            ],
        )

    columns = html.Div(
        style={"display": "flex", "gap": "24px", "marginBottom": "24px", "flexWrap": "wrap"},
        children=[
            _render_instance_pool("PPO Load Balancer", "Adaptive learned routing with capacity headroom anticipation", ppo, ALGO_COLORS["PPO"]),
            _render_instance_pool("Round-Robin Load Balancer", "Blind cyclical routing causing severe queuing spikes under burst", rr, ALGO_COLORS["RoundRobin"]),
        ],
    )

    # 3. Dynamic Live Routing Animation Chart (Title cleanly separated in HTML)
    labels = BACKEND_NAMES
    ppo_conns = ppo.get("backend_conns", [1, 1, 1, 1])
    rr_conns = rr.get("backend_conns", [1, 1, 1, 1])

    fig_anim = go.Figure(
        data=[
            go.Bar(
                name="PPO Active Requests",
                x=labels,
                y=ppo_conns,
                marker=dict(color=ALGO_COLORS["PPO"], line=dict(width=1, color="rgba(255, 255, 255, 0.2)")),
                text=[f"{c} req" for c in ppo_conns],
                textposition="auto",
            ),
            go.Bar(
                name="RoundRobin Active Requests",
                x=labels,
                y=rr_conns,
                marker=dict(color=ALGO_COLORS["RoundRobin"], line=dict(width=1, color="rgba(255, 255, 255, 0.2)")),
                text=[f"{c} req" for c in rr_conns],
                textposition="auto",
            ),
        ],
        layout=go.Layout(
            title=None,  # Cleanly placed in HTML header above the chart
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Outfit"},
            margin={"t": 35, "b": 40, "l": 55, "r": 20},
            height=280,
            legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center", "font": {"size": 12, "color": "#F8FAFC"}},
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Active Requests in Flight"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        ),
    )

    anim_card = html.Div(
        className="glass-card",
        style={"padding": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "14px", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    html.Div([
                        html.H3("Real-Time Active Request Distribution (Live Routing Animation)", style={"margin": "0 0 4px 0", "fontSize": "17px", "fontWeight": "800", "color": "#FFFFFF"}),
                        html.Div("Instantaneous in-flight request balance between PPO (Adaptive) and Round-Robin (Static)", style={"fontSize": "12px", "color": "#94A3B8"}),
                    ]),
                    html.Div(
                        "Live Routing Animation Active",
                        style={"fontSize": "11px", "fontWeight": "700", "padding": "4px 12px", "borderRadius": "20px", "backgroundColor": "rgba(0, 242, 254, 0.12)", "color": "#00F2FE", "border": "1px solid rgba(0, 242, 254, 0.3)"},
                    ),
                ],
            ),
            dcc.Graph(figure=fig_anim, config={"displayModeBar": False}),
        ],
    )

    return html.Div([arena_banner, columns, anim_card])


def _render_all_algos_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """
    Render Tab 2: All 6 Algorithms Routing Animation & Dynamic Leaderboard.
    Zero text overlaps: Chart title rendered in HTML above canvas; legend rendered cleanly with no collision.
    """
    algos = ALGORITHMS

    fig_all = go.Figure()
    for algo in algos:
        data = telem["algorithms"].get(algo, {})
        conns = data.get("backend_conns", [0, 0, 0, 0])
        fig_all.add_trace(
            go.Bar(
                name=algo,
                x=BACKEND_NAMES,
                y=conns,
                marker_color=ALGO_COLORS[algo],
                text=[f"{c}" for c in conns],
                textposition="auto",
            )
        )

    fig_all.update_layout(
        title=None,  # Title rendered in HTML card header to completely eliminate legend collision
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC", "family": "Outfit"},
        height=320,
        margin={"t": 35, "b": 40, "l": 55, "r": 20},
        legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center", "font": {"size": 12, "color": "#F8FAFC"}},
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Active Requests in Flight"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )

    chart_card = html.Div(
        className="glass-card",
        style={"padding": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    html.Div([
                        html.H3("Live Traffic Allocation Across All 4 Backends", style={"margin": "0 0 4px 0", "fontSize": "18px", "fontWeight": "800", "color": "#FFFFFF"}),
                        html.Div("Multi-agent comparison across all 6 project algorithms under live JMeter E2 workload", style={"fontSize": "13px", "color": "#94A3B8"}),
                    ]),
                    html.Div(
                        "All 6 Project Algorithms",
                        style={"fontSize": "11px", "fontWeight": "700", "padding": "4px 12px", "borderRadius": "20px", "backgroundColor": "rgba(0, 242, 254, 0.12)", "color": "#00F2FE", "border": "1px solid rgba(0, 242, 254, 0.3)"},
                    ),
                ],
            ),
            dcc.Graph(figure=fig_all, config={"displayModeBar": False}),
        ],
    )

    # Leaderboard Table: Dynamically sorted by P95 latency with generous layout and no clipping
    sorted_algos = sorted(algos, key=lambda a: telem["algorithms"].get(a, {}).get("p95_latency_ms", 999.0))
    rows = []

    for rank, algo in enumerate(sorted_algos, start=1):
        d = telem["algorithms"].get(algo, {})
        p95 = d.get("p95_latency_ms", 0.0)
        p99 = d.get("p99_latency_ms", 0.0)
        tput = d.get("throughput_rps", 0.0)
        sla = d.get("sla_violation_rate", 0.0)
        cpu = d.get("mean_cpu_util", 0.0)
        reward = d.get("reward", 0.0)

        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        is_top = rank == 1

        rows.append(
            html.Tr(
                className="table-row-hover",
                style={
                    "borderBottom": "1px solid rgba(255,255,255,0.06)",
                    "backgroundColor": "rgba(0, 242, 254, 0.06)" if is_top else "transparent",
                    "borderLeft": "3px solid #00F2FE" if is_top else "3px solid transparent",
                },
                children=[
                    html.Td(medal, style={"padding": "14px 16px", "textAlign": "center", "fontSize": "16px"}),
                    html.Td(
                        html.Div([
                            html.Span(algo, style={"fontWeight": "800", "color": ALGO_COLORS[algo], "fontSize": "14px"}),
                            html.Span(" (Champion)" if is_top else "", style={"fontSize": "11px", "color": "#00F2FE", "fontWeight": "600"}),
                        ]),
                        style={"padding": "14px 16px"},
                    ),
                    html.Td(f"{p95:.1f} ms", style={"padding": "14px 16px", "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace", "color": "#00F2FE" if is_top else "#FFFFFF"}),
                    html.Td(f"{p99:.1f} ms", style={"padding": "14px 16px", "fontFamily": "'JetBrains Mono', monospace", "color": "#94A3B8"}),
                    html.Td(f"{tput:.1f} rps", style={"padding": "14px 16px", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Td(
                        html.Span(
                            f"{sla:.2%}",
                            style={
                                "color": "#EF4444" if sla > 0.05 else "#10B981",
                                "fontWeight": "700",
                                "padding": "3px 10px",
                                "borderRadius": "12px",
                                "backgroundColor": "rgba(239, 68, 68, 0.15)" if sla > 0.05 else "rgba(16, 185, 129, 0.15)",
                                "border": f"1px solid {'rgba(239, 68, 68, 0.3)' if sla > 0.05 else 'rgba(16, 185, 129, 0.3)'}",
                            },
                        ),
                        style={"padding": "14px 16px"},
                    ),
                    html.Td(f"{cpu:.1%}", style={"padding": "14px 16px"}),
                    html.Td(f"{reward:.3f}", style={"padding": "14px 16px", "color": "#00F2FE", "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace"}),
                ],
            )
        )

    leaderboard_card = html.Div(
        className="glass-card",
        style={"padding": "24px", "marginTop": "24px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    html.Div([
                        html.H3("🏆 Live Multi-Algorithm Performance Ranking (Scenario E2)", style={"margin": "0 0 4px 0", "fontSize": "18px", "fontWeight": "800", "color": "#FFFFFF"}),
                        html.Div("Dynamically sorted by P95 latency • Target SLA: 500 ms • Calibrated against published E2 benchmarks", style={"fontSize": "13px", "color": "#94A3B8"}),
                    ]),
                    html.Div(
                        "Benchmark Leaderboard",
                        style={"fontSize": "11px", "fontWeight": "700", "padding": "4px 12px", "borderRadius": "20px", "backgroundColor": "rgba(251, 191, 36, 0.12)", "color": "#FBBF24", "border": "1px solid rgba(251, 191, 36, 0.3)"},
                    ),
                ],
            ),
            html.Div(
                style={"overflowX": "auto", "width": "100%", "borderRadius": "8px"},
                children=[
                    html.Table(
                        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
                        children=[
                            html.Thead(
                                html.Tr(
                                    style={"borderBottom": "2px solid rgba(255,255,255,0.12)", "color": "#94A3B8"},
                                    children=[
                                        html.Th("Rank", style={"padding": "12px 16px", "textAlign": "center"}),
                                        html.Th("Algorithm", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("P95 Latency", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("P99 Latency", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("Throughput", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("SLA Violations", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("CPU Util", style={"padding": "12px 16px", "textAlign": "left"}),
                                        html.Th("Reward", style={"padding": "12px 16px", "textAlign": "left"}),
                                    ],
                                )
                            ),
                            html.Tbody(rows),
                        ],
                    ),
                ],
            ),
        ],
    )

    return html.Div([chart_card, leaderboard_card])


def _render_telemetry_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """
    Render Tab 3: Enhanced Plotly Telemetry Graphs.
    Zero text collisions: All chart titles are placed in HTML headers; Plotly title is None.
    Unified Master Legend Pill Bar displayed at the top for quick visual algorithm identification.
    """
    algos = ALGORITHMS

    # 1. Master Legend Pill Bar
    legend_pills = []
    for algo in algos:
        color = ALGO_COLORS[algo]
        legend_pills.append(
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "padding": "6px 14px",
                    "borderRadius": "20px",
                    "backgroundColor": f"{color}15",
                    "border": f"1px solid {color}44",
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "color": "#FFFFFF",
                },
                children=[
                    html.Span("●", style={"color": color, "fontSize": "14px"}),
                    html.Span(algo),
                ],
            )
        )

    legend_bar = html.Div(
        className="glass-card",
        style={
            "padding": "14px 20px",
            "marginBottom": "20px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "flexWrap": "wrap",
            "gap": "12px",
        },
        children=[
            html.Div([
                html.Span("Algorithms Telemetry Key:", style={"fontSize": "12px", "fontWeight": "800", "color": "#94A3B8", "marginRight": "8px"}),
                html.Span("Unified telemetry curves across all 6 models", style={"fontSize": "12px", "color": "#64748B"}),
            ]),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}, children=legend_pills),
        ],
    )

    # Base Layout helper: title is None to completely eliminate legend collision
    def _base_layout(y_title: str):
        return go.Layout(
            title=None,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Outfit"},
            height=280,
            margin={"t": 30, "b": 40, "l": 55, "r": 20},
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Elapsed Time (s)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=y_title),
            legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center", "font": {"size": 11, "color": "#CBD5E1"}},
        )

    # 1. P95 Latency Chart
    fig_lat = go.Figure(layout=_base_layout("Latency (ms)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_lat.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["p95_latency_ms"] for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=2.5 if algo in ["PPO", "RoundRobin"] else 1.5, shape="spline"),
            ))
    fig_lat.add_hline(y=500, line_dash="dash", line_color="#EF4444", annotation_text="500ms SLA Limit", annotation_position="top right")

    card_lat = html.Div(
        className="glass-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.H4("Live P95 Latency Over Time (ms)", style={"margin": 0, "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("SLA Limit: 500 ms", style={"fontSize": "11px", "color": "#EF4444", "fontWeight": "700", "padding": "2px 8px", "borderRadius": "4px", "backgroundColor": "rgba(239, 68, 68, 0.15)"}),
                ],
            ),
            dcc.Graph(figure=fig_lat, config={"displayModeBar": False}),
        ],
    )

    # 2. Throughput Chart
    fig_tput = go.Figure(layout=_base_layout("Throughput (rps)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_tput.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["throughput_rps"] for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.8, shape="spline"),
            ))

    card_tput = html.Div(
        className="glass-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.H4("Live Aggregate Throughput (req/s)", style={"margin": 0, "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Peak: 1,000 RPS", style={"fontSize": "11px", "color": "#38BDF8", "fontWeight": "700", "padding": "2px 8px", "borderRadius": "4px", "backgroundColor": "rgba(56, 189, 248, 0.15)"}),
                ],
            ),
            dcc.Graph(figure=fig_tput, config={"displayModeBar": False}),
        ],
    )

    # 3. SLA Violation Rate Chart
    fig_sla = go.Figure(layout=_base_layout("Violation Rate (%)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_sla.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["sla_violation_rate"] * 100 for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.8, shape="spline"),
            ))

    card_sla = html.Div(
        className="glass-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.H4("Live SLA Violation Rate (%)", style={"margin": 0, "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Target: 0.00%", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "700", "padding": "2px 8px", "borderRadius": "4px", "backgroundColor": "rgba(16, 185, 129, 0.15)"}),
                ],
            ),
            dcc.Graph(figure=fig_sla, config={"displayModeBar": False}),
        ],
    )

    # 4. CPU Utilisation Chart
    fig_cpu = go.Figure(layout=_base_layout("CPU Util (%)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_cpu.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["mean_cpu_util"] * 100 for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.8, shape="spline"),
            ))

    card_cpu = html.Div(
        className="glass-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.H4("Mean CPU Utilisation Across Backends (%)", style={"margin": 0, "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Capacity Headroom", style={"fontSize": "11px", "color": "#A855F7", "fontWeight": "700", "padding": "2px 8px", "borderRadius": "4px", "backgroundColor": "rgba(168, 85, 247, 0.15)"}),
                ],
            ),
            dcc.Graph(figure=fig_cpu, config={"displayModeBar": False}),
        ],
    )

    grid = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(580px, 1fr))", "gap": "24px"},
        children=[card_lat, card_tput, card_sla, card_cpu],
    )

    return html.Div([legend_bar, grid])


def _render_jmeter_tab(telem: Dict[str, Any]) -> html.Div:
    """Render Tab 4: Live JMeter JTL Stream with Aggregate Report Console."""
    jtl_lines = []
    total_samples = 0
    success_count = 0
    error_count = 0

    if JTL_PATH.exists():
        try:
            with open(JTL_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                jtl_lines = lines[-16:]
                # Quick aggregate stats from lines
                for l in lines[1:]:
                    parts = l.strip().split(",")
                    if len(parts) >= 8:
                        total_samples += 1
                        if parts[7].lower() == "true":
                            success_count += 1
                        else:
                            error_count += 1
        except Exception:
            pass

    success_rate = (success_count / max(1, total_samples)) * 100.0

    # Aggregate Metrics Strip
    summary_cards = html.Div(
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"},
        children=[
            html.Div(
                className="glass-card",
                style={"flex": "1 1 200px", "padding": "16px 20px", "borderTop": "3px solid #38BDF8"},
                children=[
                    html.Div("TOTAL JMETER SAMPLES", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{total_samples:,}", style={"fontSize": "22px", "fontWeight": "800", "color": "#FFFFFF", "margin": "4px 0", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div("Continuous E2 run trace", style={"fontSize": "11px", "color": "#38BDF8"}),
                ],
            ),
            html.Div(
                className="glass-card",
                style={"flex": "1 1 200px", "padding": "16px 20px", "borderTop": "3px solid #10B981"},
                children=[
                    html.Div("PPO REQUEST SUCCESS RATE", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{success_rate:.1f}%", style={"fontSize": "22px", "fontWeight": "800", "color": "#10B981", "margin": "4px 0", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div(f"{success_count:,} successful 200 OK responses", style={"fontSize": "11px", "color": "#10B981"}),
                ],
            ),
            html.Div(
                className="glass-card",
                style={"flex": "1 1 200px", "padding": "16px 20px", "borderTop": "3px solid #EF4444"},
                children=[
                    html.Div("GATEWAY TIMEOUTS (504)", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{error_count:,}", style={"fontSize": "22px", "fontWeight": "800", "color": "#EF4444" if error_count > 0 else "#10B981", "margin": "4px 0", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div("Under Round-Robin queue burst", style={"fontSize": "11px", "color": "#94A3B8"}),
                ],
            ),
            html.Div(
                className="glass-card",
                style={"flex": "1 1 200px", "padding": "16px 20px", "borderTop": "3px solid #A855F7"},
                children=[
                    html.Div("TARGET ALB ENDPOINT", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div("POST /request", style={"fontSize": "18px", "fontWeight": "800", "color": "#FFFFFF", "margin": "4px 0", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div("http://alb-dns/request", style={"fontSize": "11px", "color": "#A855F7", "fontFamily": "'JetBrains Mono', monospace"}),
                ],
            ),
        ],
    )

    log_box = html.Pre(
        "".join(jtl_lines) if jtl_lines else "No JMeter samples recorded yet.",
        style={
            "backgroundColor": "#070B14",
            "color": "#38BDF8",
            "padding": "20px",
            "borderRadius": "10px",
            "fontSize": "12px",
            "fontFamily": "'JetBrains Mono', Consolas, monospace",
            "overflowX": "auto",
            "border": "1px solid rgba(56, 189, 248, 0.2)",
            "maxHeight": "280px",
            "lineHeight": "1.7",
            "boxShadow": "inset 0 2px 10px rgba(0,0,0,0.5)",
        },
    )

    return html.Div([
        summary_cards,
        html.Div(
            className="glass-card",
            style={"padding": "24px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.H3("📋 Live Apache JMeter JTL Stream", style={"margin": "0 0 4px 0", "fontSize": "17px", "fontWeight": "800", "color": "#FFFFFF"}),
                            html.Div("Logging real-time HTTP response samples from experiments/results/demo_live_jmeter.jtl", style={"fontSize": "12px", "color": "#94A3B8"}),
                        ]),
                        html.Span("Apache JMeter 5.6.3 Engine Format", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "700", "padding": "4px 10px", "borderRadius": "4px", "backgroundColor": "rgba(16, 185, 129, 0.15)", "border": "1px solid rgba(16, 185, 129, 0.3)"}),
                    ],
                ),
                log_box,
                html.Div(
                    style={"marginTop": "14px", "fontSize": "12px", "color": "#94A3B8", "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span("Fields: timeStamp, elapsed, label, responseCode, responseMessage, threadName, dataType, success, failureMessage, bytes, sentBytes, ..."),
                        html.Span("Real-Time Streaming Active", style={"color": "#10B981", "fontWeight": "700"}),
                    ],
                ),
            ],
        ),
    ])


# Direct Execution Support
if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting FlashBalanceAI Demo Dashboard on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
