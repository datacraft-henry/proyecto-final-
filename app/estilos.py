"""Paletas (modo claro y oscuro) y CSS de la app de riesgo (marca DataCraft).

Va separado de la app para que `streamlit_app.py` se lea como composición de
pantalla y no como una hoja de estilos. El color nunca va solo: siempre con
ícono y texto, así se entiende también en escala de grises.

Cada paleta trae, además de los tokens del CSS propio, la clave `streamlit`
con las opciones `theme.*` que se aplican en caliente para que la grilla, los
inputs y los botones nativos acompañen el modo elegido.
"""

LIGHT = {
    "name": "light",
    "bg": "#FBF8F2",
    "surface": "#FFFFFF",
    "surface_alt": "#F1EADC",
    "line": "#E6DECF",
    "text": "#13294B",
    "muted": "#4A5B78",
    "accent": "#E8833A",
    "accent_text": "#B4541A",
    "header_bg": "#13294B",
    "header_line": "#13294B",
    "header_text": "#FBF8F2",
    "header_sub": "#D9DEE8",
    "active_bg": "#13294B",
    "active_text": "#FBF8F2",
    "hint_bg": "#F1EADC",
    "hint_text": "#4A5B78",
    "picked_bg": "#E8EEF9",
    "picked_text": "#13294B",
    "picked_line": "#C9D5EC",
    "level": {"alto": "#C0392B", "medio": "#D69E2E", "bajo": "#2F855A"},
    "level_soft": {"alto": "#FDECEA", "medio": "#FDF4DC", "bajo": "#E6F4EC"},
    "level_text": {"alto": "#8E2A1F", "medio": "#7A5A0F", "bajo": "#22633F"},
    "on_bar": "#FFFFFF",
    "shadow_soft": "0 1px 2px rgba(19,41,75,.05)",
    "shadow_card": "0 1px 2px rgba(19,41,75,.04), 0 8px 24px rgba(19,41,75,.07)",
    "shadow_hover": "0 4px 8px rgba(19,41,75,.06), 0 14px 32px rgba(19,41,75,.12)",
    "streamlit": {
        "base": "light",
        "primaryColor": "#B4541A",
        "backgroundColor": "#FBF8F2",
        "secondaryBackgroundColor": "#F1EADC",
        "textColor": "#13294B",
        "borderColor": "#E6DECF",
        "dataframeBorderColor": "#E6DECF",
        "dataframeHeaderBackgroundColor": "#F1EADC",
        "codeBackgroundColor": "#F1EADC",
        "linkColor": "#B4541A",
    },
}

DARK = {
    "name": "dark",
    "bg": "#0F172A",
    "surface": "#1E293B",
    "surface_alt": "#1E293B",
    "line": "#334155",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "accent": "#E8833A",
    "accent_text": "#F5B07A",
    "header_bg": "#0B1220",
    "header_line": "#334155",
    "header_text": "#F8FAFC",
    "header_sub": "#94A3B8",
    "active_bg": "#F8FAFC",
    "active_text": "#0F172A",
    "hint_bg": "#273449",
    "hint_text": "#94A3B8",
    "picked_bg": "rgba(232,131,58,.18)",
    "picked_text": "#F5B07A",
    "picked_line": "rgba(232,131,58,.45)",
    "level": {"alto": "#F87171", "medio": "#FBBF24", "bajo": "#4ADE80"},
    "level_soft": {"alto": "rgba(239,68,68,.15)", "medio": "rgba(245,158,11,.15)", "bajo": "rgba(34,197,94,.15)"},
    "level_text": {"alto": "#FCA5A5", "medio": "#FCD34D", "bajo": "#86EFAC"},
    "on_bar": "#0F172A",
    "shadow_soft": "0 1px 2px rgba(0,0,0,.35)",
    "shadow_card": "0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35)",
    "shadow_hover": "0 4px 8px rgba(0,0,0,.3), 0 14px 32px rgba(0,0,0,.45)",
    "streamlit": {
        "base": "dark",
        "primaryColor": "#E8833A",
        "backgroundColor": "#0F172A",
        "secondaryBackgroundColor": "#1E293B",
        "textColor": "#F8FAFC",
        "borderColor": "#334155",
        "dataframeBorderColor": "#334155",
        "dataframeHeaderBackgroundColor": "#1E293B",
        "codeBackgroundColor": "#1E293B",
        "linkColor": "#F5B07A",
    },
}

PALETTES = {"light": LIGHT, "dark": DARK}


def build_css(p: dict) -> str:
    """Hoja de estilos de la pantalla para una paleta."""
    lvl, soft, ink = p["level"], p["level_soft"], p["level_text"]
    return f"""
<style>
h1, h2, h3, .dc-serif {{ font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}

/* Cabecera con el selector de modo claro / oscuro */
.st-key-dc-header {{ background: {p["header_bg"]}; border: 1px solid {p["header_line"]}; border-radius: 14px;
  padding: 10px 24px; margin-bottom: 18px; }}
/* La marca mide lo mismo que el botón (40 px) y ambas columnas se centran en la barra. El
   contenedor de markdown no sigue el alto del texto grande, por eso se fija también. */
.st-key-dc-header [data-testid="stMarkdown"]:has(.dc-brand),
.st-key-dc-header [data-testid="stMarkdownContainer"]:has(> .dc-brand) {{ height: 40px; }}
.dc-brand {{ display: flex; align-items: center; gap: 18px; height: 40px; white-space: nowrap; overflow: hidden; }}
.dc-logo {{ font-family: Georgia, serif; font-size: 1.7rem; font-weight: 600; line-height: 1; color: {p["header_text"]}; }}
.dc-logo span {{ color: {p["accent"]}; }}
.dc-sub {{ color: {p["header_sub"]}; font-size: 1rem; line-height: 1; overflow: hidden; text-overflow: ellipsis; }}
.st-key-dc-header button {{ background: transparent !important; border: 1px solid {p["header_sub"]} !important;
  transition: background .15s ease, border-color .15s ease; }}
.st-key-dc-header button:hover {{ background: rgba(255,255,255,.1) !important; border-color: {p["header_text"]} !important; }}
.st-key-dc-header button * {{ color: {p["header_text"]} !important; }}
.dc-hero {{ font-family: Georgia, serif; font-size: 1.45rem; line-height: 1.35; color: {p["text"]}; margin: 4px 0 2px; }}
.dc-hero b {{ color: {p["accent_text"]}; }}
.dc-muted {{ color: {p["muted"]}; font-size: 0.9rem; }}
.dc-lead {{ font-size: 1.1rem; line-height: 1.5; color: {p["text"]}; margin: 0 0 4px; }}
.dc-lead b {{ color: {p["accent_text"]}; }}

/* Pestañas: pedidos a revisar y vendedores recomendados */
.st-key-dc-tabs [data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; color: {p["muted"]}; }}
.st-key-dc-tabs [data-baseweb="tab"][aria-selected="true"] p {{ color: {p["text"]}; }}

/* Barras de control: filtros globales arriba, segmento + descarga sobre la tabla */
.st-key-dc-toolbar, .st-key-dc-tablebar {{ background: {p["surface_alt"]}; border: 1px solid {p["line"]};
  border-radius: 14px; padding: 12px 18px 14px; box-shadow: {p["shadow_soft"]}; }}
.st-key-dc-toolbar {{ margin-bottom: 14px; }}
.st-key-dc-tablebar {{ padding: 10px 14px; margin-bottom: 10px; }}
.st-key-dc-toolbar [data-testid="stWidgetLabel"] p {{ font-weight: 600; color: {p["text"]}; }}
.dc-toolbar-title {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: {p["muted"]}; margin-bottom: 2px; }}

/* Selector de nivel (radios del segmented control): el activo se ve de lejos */
.st-key-dc-tablebar [role="radiogroup"] button[role="radio"] {{
  border-color: {p["line"]}; background: {p["surface"]}; transition: background .15s ease, transform .15s ease; }}
.st-key-dc-tablebar [role="radiogroup"] button[role="radio"]:hover {{ background: {p["hint_bg"]}; }}
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"],
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"]:hover,
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"]:active {{
  background: {p["active_bg"]} !important; border-color: {p["active_bg"]} !important; box-shadow: inset 0 -3px 0 {p["accent"]}; }}
.st-key-dc-tablebar [role="radiogroup"] button[aria-checked="true"] * {{ color: {p["active_text"]} !important; font-weight: 600; }}
.st-key-dc-tablebar [role="radiogroup"] button:active {{ transform: scale(0.98); }}

/* Tarjetas de nivel */
.dc-card {{ background: {p["surface"]}; border: 1px solid {p["line"]}; border-left: 6px solid var(--lvl);
  border-radius: 12px; padding: 14px 16px; height: 100%; box-shadow: {p["shadow_soft"]};
  transition: transform .18s ease, box-shadow .18s ease; }}
.dc-card:hover {{ transform: translateY(-2px); box-shadow: {p["shadow_hover"]}; }}
.dc-card .t {{ font-weight: 600; color: {p["text"]}; font-size: 0.95rem; }}
.dc-card .n {{ font-family: Georgia, serif; font-size: 2rem; color: {p["text"]}; line-height: 1.2; }}
.dc-card .p {{ color: {p["muted"]}; font-size: 0.9rem; }}
.dc-card .a {{ color: {p["text"]}; font-size: 0.88rem; margin-top: 8px; }}

/* Lift: cuánto concentra retrasos el nivel alto */
.dc-lift {{ background: {p["surface"]}; border: 1px solid {p["line"]}; border-top: 5px solid {lvl["alto"]};
  border-radius: 12px; padding: 14px 16px; box-shadow: {p["shadow_soft"]}; }}
.dc-lift .n {{ font-family: Georgia, serif; font-size: 2.2rem; line-height: 1.1; color: {lvl["alto"]}; }}
.dc-lift .t {{ font-weight: 600; color: {p["text"]}; margin-top: 2px; }}
.dc-lift .s {{ color: {p["muted"]}; font-size: 0.85rem; margin-top: 6px; }}

/* Inspector de pedido */
.st-key-dc-inspector {{ background: {p["surface"]}; border: 1px solid {p["line"]}; border-radius: 14px;
  padding: 18px 18px 14px; box-shadow: {p["shadow_card"]}; gap: 0.6rem; }}
.dc-eyebrow {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {p["muted"]}; }}
.dc-hint {{ display: inline-block; margin-top: 6px; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
  background: {p["hint_bg"]}; color: {p["hint_text"]}; border: 1px solid {p["line"]}; line-height: 1.3; }}
.dc-hint.picked {{ background: {p["picked_bg"]}; color: {p["picked_text"]}; border-color: {p["picked_line"]}; }}
.dc-oid {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.82rem; color: {p["text"]};
  word-break: break-all; background: {p["bg"]}; border: 1px solid {p["line"]}; border-radius: 8px; padding: 6px 10px; margin-top: 10px; }}
.dc-level {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 10px; }}
.dc-badge {{ display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.88rem; font-weight: 700;
  color: var(--ink); background: var(--soft); border: 1.5px solid var(--lvl); }}
.dc-score {{ color: {p["muted"]}; font-size: 0.85rem; }}
.dc-score b {{ font-family: Georgia, serif; font-size: 1.1rem; color: {p["text"]}; }}
.dc-facts {{ margin: 12px 0 2px; padding: 2px 12px; background: {p["bg"]}; border: 1px solid {p["line"]}; border-radius: 10px; }}
.dc-facts > div {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  padding: 7px 0; border-bottom: 1px solid {p["line"]}; }}
.dc-facts > div:last-child {{ border-bottom: 0; }}
.dc-facts small {{ color: {p["muted"]}; font-size: 0.8rem; }}
.dc-facts b {{ font-family: Georgia, serif; font-size: 1rem; color: {p["text"]}; white-space: nowrap; }}

/* Factores: chip + mini barra de impacto */
.dc-sec {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  margin: 14px 0 6px; padding-bottom: 4px; border-bottom: 1px solid {p["line"]}; }}
.dc-sec-t {{ font-weight: 700; color: {p["text"]}; font-size: 0.92rem; }}
.dc-sec-m {{ font-size: 0.75rem; font-weight: 600; }}
.dc-sec-up {{ color: {lvl["alto"]}; }}
.dc-sec-down {{ color: {lvl["bajo"]}; }}
.dc-fx {{ padding: 6px 0 4px; }}
.dc-fx-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
.dc-chip {{ display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; white-space: nowrap; }}
.dc-chip i {{ font-style: normal; font-weight: 800; }}
.dc-fx-up .dc-chip {{ background: {soft["alto"]}; color: {ink["alto"]}; }}
.dc-fx-down .dc-chip {{ background: {soft["bajo"]}; color: {ink["bajo"]}; }}
.dc-fx-val {{ color: {p["text"]}; font-size: 0.88rem; font-weight: 600; text-align: right; }}
.dc-fx-bar {{ height: 6px; margin-top: 6px; border-radius: 999px; background: {p["hint_bg"]}; overflow: hidden; }}
.dc-fx-bar i {{ display: block; height: 100%; border-radius: 999px; transition: width .3s ease; }}
.dc-fx-up .dc-fx-bar i {{ background: {lvl["alto"]}; }}
.dc-fx-down .dc-fx-bar i {{ background: {lvl["bajo"]}; }}
.dc-fx-note {{ color: {p["muted"]}; font-size: 0.78rem; margin: 8px 0 0; }}

/* Acción sugerida */
.dc-action {{ margin-top: 14px; border-radius: 12px; padding: 12px 14px; background: var(--soft);
  border: 1px solid var(--lvl); border-left-width: 6px; }}
.dc-action .k {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink); }}
.dc-action .t {{ color: {p["text"]}; font-weight: 600; margin-top: 4px; line-height: 1.35; }}

/* Estado vacío */
.dc-empty {{ text-align: center; padding: 36px 24px 22px; border: 1px dashed {p["line"]}; border-radius: 14px; background: {p["surface"]}; }}
.dc-empty .i {{ font-size: 2rem; }}
.dc-empty .t {{ font-family: Georgia, serif; font-size: 1.25rem; color: {p["text"]}; margin-top: 6px; }}
.dc-empty .s {{ color: {p["muted"]}; margin-top: 4px; }}
</style>
"""
