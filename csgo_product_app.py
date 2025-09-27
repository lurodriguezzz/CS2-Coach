
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import gradio as gr
import requests
import os, json, warnings
warnings.filterwarnings('ignore')

PATH_MODEL_WIN = "./flwinner_csgo_product_api.keras"
PATH_MODEL_CLUST = "./cluster_csgo_product_api.keras"
PATH_SHAP_BG = "./X_shap_background_csgo_product_api.npy"
PATH_CLUSTER_T = "./cluster_t_product_2_best_api.csv"

CACHE_PATH = "./steam_accum_cache.json"

TOPK_SHAP = 8
APPID_CS  = 730

model_ = tf.keras.models.load_model(PATH_MODEL_WIN)
try:
    model_c = tf.keras.models.load_model(PATH_MODEL_CLUST)
    HAS_MODEL_C = True
except Exception:
    model_c = None
    HAS_MODEL_C = False

# =========================
# ORDEM DE FEATURES
# =========================
STEAM_BASE17 = [
    'qtKill','qtDeath','qtHs','qtBombeDefuse','qtBombePlant',
    'vlDamage','qtHits','qtShots','qtRoundsPlayed',
    'descMapName_de_ancient','descMapName_de_dust2','descMapName_de_inferno',
    'descMapName_de_mirage','descMapName_de_nuke','descMapName_de_overpass',
    'descMapName_de_train','descMapName_de_vertigo'
]

feature_order_base = STEAM_BASE17[:]
win_in = int(model_.input_shape[-1])
if len(feature_order_base) != win_in:
    if len(feature_order_base) > win_in:
        feature_order_base = feature_order_base[:win_in]
    else:
        feature_order_base = feature_order_base + [f"__PAD__{i}" for i in range(win_in - len(feature_order_base))]

DERIVED_COLS = ["kd_ratio","hs_pct","acc_shot","dmg_per_round","surv_per_round"]
REQUIRED_COLUMNS_WIN = feature_order_base[:]
required_columns_clust = feature_order_base[:] + DERIVED_COLS
if HAS_MODEL_C:
    cl_in = int(model_c.input_shape[-1])
    if len(required_columns_clust) != cl_in:
        if len(required_columns_clust) > cl_in:
            required_columns_clust = required_columns_clust[:cl_in]
        else:
            required_columns_clust = required_columns_clust + [f"__PAD_C__{i}" for i in range(cl_in - len(required_columns_clust))]
REQUIRED_COLUMNS_CLUST = required_columns_clust

# =========================
# LOAD DEMAIS ARQUIVOS
# =========================
xbg = np.load(PATH_SHAP_BG).astype("float32")

cluster_t_raw = pd.read_csv(PATH_CLUSTER_T)
cluster_t = cluster_t_raw.set_index(cluster_t_raw.columns[0], drop=True).copy()
cluster_t.index = cluster_t.index.astype(str).str.strip()
cluster_t.index.name = None
cluster_t = cluster_t.drop(index=["flWinner"], errors="ignore")
cluster_t.columns = [str(c).strip() for c in cluster_t.columns]

coach_labels = {
    "0": {"perfil": "Entry fragger bruto",  "sugestao": "Melhorar trade discipline e reduzir mortes rápidas"},
    "1": {"perfil": "Flex equilibrado",     "sugestao": "Buscar mais impacto (FK/R) mantendo consistência"},
    "2": {"perfil": "Entry agressivo",      "sugestao": "Converter impacto em rounds ganhos"},
    "3": {"perfil": "Âncora CT",            "sugestao": "Fortalecer micro-posicionamento e utilidade"},
    "4": {"perfil": "Lurker/estrela",       "sugestao": "Trabalhar timings e precisão"},
    "5": {"perfil": "Suporte",              "sugestao": "Aumentar impacto individual com utilidade"},
    "6": {"perfil": "Entry sacrificial",    "sugestao": "Sobreviver mais em execuções"},
}

# =========================
# FUNÇÕES AUXILIARES
# =========================
def add_derived_to_row(row_like: pd.Series) -> pd.Series:
    r = row_like.copy(); eps = 1e-6
    r["kd_ratio"]       = float(r.get("qtKill",0)) / (float(r.get("qtDeath",0)) + eps)
    r["hs_pct"]         = float(r.get("qtHs",0))   / (float(r.get("qtKill",0))  + eps)
    r["acc_shot"]       = float(r.get("qtHits",0)) / (float(r.get("qtShots",0)) + eps)
    r["dmg_per_round"]  = float(r.get("vlDamage",0)) / (float(r.get("qtRoundsPlayed",0)) + eps)
    r["surv_per_round"] = float(r.get("qtSurvived",0)) / (float(r.get("qtRoundsPlayed",0)) + eps)
    return r

def make_input_for_model_win(d: dict):
    row = pd.Series(d, dtype="float32").reindex(REQUIRED_COLUMNS_WIN).fillna(0.0).astype("float32")
    return row.values.reshape(1, -1), row

def make_input_for_model_c(d: dict):
    base = pd.Series(d, dtype="float32").reindex(feature_order_base).fillna(0.0)
    row  = add_derived_to_row(base).reindex(REQUIRED_COLUMNS_CLUST).fillna(0.0).astype("float32")
    return row.values.reshape(1, -1), row

def f_pred(X: np.ndarray):
    y = model_.predict(X, verbose=0)
    if y.ndim == 2 and y.shape[1] == 1:
        p1 = y.ravel()
        return np.column_stack([1 - p1, p1])
    return y

if xbg.shape[1] != len(REQUIRED_COLUMNS_WIN):
    if xbg.shape[1] > len(REQUIRED_COLUMNS_WIN):
        xbg = xbg[:, :len(REQUIRED_COLUMNS_WIN)]
    else:
        pad = np.zeros((xbg.shape[0], len(REQUIRED_COLUMNS_WIN) - xbg.shape[1]), dtype="float32")
        xbg = np.hstack([xbg, pad]).astype("float32")

explainer = shap.KernelExplainer(lambda X: f_pred(X.astype("float32")), xbg.astype("float32"))

def _extract_positive_shap(sv_list, expected_values):
    if isinstance(sv_list, (list, tuple)):
        sv = np.asarray(sv_list[-1]).ravel()
    else:
        sv = np.asarray(sv_list).ravel()
    if isinstance(expected_values, (list, tuple, np.ndarray)):
        ev = np.asarray(expected_values).ravel()
        base_value = float(ev[-1])
    else:
        base_value = float(expected_values)
    return sv, base_value

def make_shap_explanation_scaled(Xw: np.ndarray, roww: pd.Series, topk: int | None = None):
    sv_list = explainer.shap_values(Xw, check_additivity=False)
    sv_full, _ = _extract_positive_shap(sv_list, explainer.expected_value)
    m = min(len(sv_full), len(REQUIRED_COLUMNS_WIN), roww.values.shape[0])
    sv_full  = np.asarray(sv_full[:m], dtype=float)
    row_vals = np.asarray(roww.values[:m], dtype=float)
    feat_all = REQUIRED_COLUMNS_WIN[:m]
    contrib = sv_full * row_vals
    order = np.argsort(np.abs(contrib))[::-1]
    if isinstance(topk, int) and topk > 0:
        order = order[:min(topk, m)]
    contrib_sel = contrib[order]
    data_sel    = row_vals[order]
    feat_names  = [f"{feat_all[i]} (φ×x)" for i in order]
    return shap.Explanation(values=contrib_sel, base_values=0.0,
                            data=data_sel, feature_names=feat_names)

def kpis_from_row_base(row: pd.Series):
    R = max(float(row.get("qtRoundsPlayed", 0) or 0), 1.0)
    return {
        "KPR": float(row.get("qtKill",0))/R,
        "DPR": float(row.get("qtDeath",0))/R,
        "ADR": float(row.get("vlDamage",0))/R,
        "Acc": float(row.get("qtHits",0))/max(float(row.get("qtShots",0) or 0), 1.0),
        "HSR": float(row.get("qtHs",0))/max(float(row.get("qtKill",0)  or 0), 1.0),
    }

def ref_from_cluster_t(cid: int):
    c = str(cid)
    if c not in cluster_t.columns or "qtRoundsPlayed" not in cluster_t.index:
        return {k:0.0 for k in ["KPR","DPR","ADR","Acc","HSR"]}
    R = max(float(cluster_t.loc["qtRoundsPlayed", c]), 1.0)
    return {
        "KPR": float(cluster_t.loc["qtKill", c]) / R,
        "DPR": float(cluster_t.loc["qtDeath", c]) / R,
        "ADR": float(cluster_t.loc["vlDamage", c]) / R,
        "Acc": float(cluster_t.loc["qtHits", c]) / max(float(cluster_t.loc["qtShots", c]), 1.0),
        "HSR": float(cluster_t.loc["qtHs", c]) / max(float(cluster_t.loc["qtKill", c]), 1.0),
    }

def predict_cluster_from_dict(d: dict) -> int:
    if not HAS_MODEL_C:
        return -1
    Xc, _ = make_input_for_model_c(d)
    y = model_c.predict(Xc, verbose=0)
    y = np.asarray(y)
    return int(np.argmax(y, axis=1)[0]) if (y.ndim == 2 and y.shape[1] > 1) else int(np.rint(y).astype(int).ravel()[0])

def predict_win_and_shap_from_dict(d: dict, topk: int | None = None):
    Xw, roww = make_input_for_model_win(d)
    y = model_.predict(Xw, verbose=0)
    pwin = float(y.ravel()[0]) if (y.ndim == 2 and y.shape[1] == 1) else float(y[0, 1])
    expl = make_shap_explanation_scaled(Xw, roww, topk=topk)
    return pwin, expl, roww

def build_card_from_stats(stats: dict, topk: int | None = None):
    cid = predict_cluster_from_dict(stats)
    pwin, expl, row_base = predict_win_and_shap_from_dict(stats, topk=topk)
    kpis = kpis_from_row_base(row_base)
    ref  = ref_from_cluster_t(cid) if cid >= 0 else {k:0.0 for k in ["KPR","DPR","ADR","Acc","HSR"]}
    delta = {k: kpis[k] - ref.get(k, 0.0) for k in kpis}
    label = coach_labels.get(str(cid), {}) if cid >= 0 else {}
    perfil   = (label.get("perfil") or (f"Cluster {cid}" if cid >= 0 else "—")).strip()
    sugestao = (label.get("sugestao") or "Ajuste seu jogo nos pontos fracos.").strip()
    card = {
        "cluster_id": cid,
        "perfil": perfil,
        "sugestao": sugestao,
        "prob_vitoria": round(pwin, 4),
        "kpis": {
            "KPR": round(kpis["KPR"],3),
            "DPR": round(kpis["DPR"],3),
            "ADR": round(kpis["ADR"],1),
            "Acc%": round(kpis["Acc"]*100,2),
            "HS%": round(kpis["HSR"]*100,2),
        },
        "vs_cluster": {
            "KPR":   round(delta.get("KPR",0.0),3),
            "DPR":   round(delta.get("DPR",0.0),3),
            "ADR":   round(delta.get("ADR",0.0),1),
            "Acc%":  round(delta.get("Acc",0.0)*100,1),
            "HS%":   round(delta.get("HSR",0.0)*100,1),
        }
    }
    return card, expl

#CACHE
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
def _save_cache(cache: dict):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

#Steam API + adapters
def _steam_get_user_stats(api_key: str, steam_id64: str, appid: int = APPID_CS) -> dict:
    url = "https://api.steampowered.com/ISteamUserStats/GetUserStatsForGame/v2/"
    params = {"key": api_key, "steamid": steam_id64, "appid": appid}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _stat(stats_list: list, name: str, default: float = 0.0) -> float:
    try:
        for it in stats_list:
            if it.get("name") == name:
                return float(it.get("value", default))
    except Exception:
        pass
    return float(default)

def _steam_totals(api_payload: dict) -> dict:
    stats = api_payload.get("playerstats", {}).get("stats", []) or []
    def g(name: str, default: float = 0.0) -> float:
        return _stat(stats, name, default)
    de_maps = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train","de_vertigo"]
    return {
        "total_kills":          g("total_kills", 0),
        "total_deaths":         g("total_deaths", 0),
        "total_shots_fired":    g("total_shots_fired", 0),
        "total_shots_hit":      g("total_shots_hit", 0),
        "total_damage_done":    g("total_damage_done", 0),
        "total_kills_headshot": g("total_kills_headshot", 0),
        "total_rounds_played":  g("total_rounds_played", 0),
        "total_planted_bombs":  g("total_planted_bombs", 0),
        "total_defused_bombs":  g("total_defused_bombs", 0),
        "total_assists":        g("total_assists", 0),
        "rounds_per_map": {m: g(f"total_rounds_map_{m}", 0) for m in de_maps},
    }

def _delta_from_totals(curr: dict, prev: dict | None) -> dict | None:
    if not prev:
        return None
    keys_to_check = [
        "total_rounds_played","total_kills","total_deaths",
        "total_shots_fired","total_damage_done"
    ]
    if any(float(curr.get(k,0)) < float(prev.get(k,0)) for k in keys_to_check):
        prev = {k: 0.0 for k in curr.keys()}
        if isinstance(curr.get("rounds_per_map"), dict):
            prev["rounds_per_map"] = {m: 0.0 for m in curr["rounds_per_map"].keys()}

    def d(key): return max(float(curr.get(key, 0)) - float(prev.get(key, 0)), 0.0)

    total_kills   = d("total_kills")
    total_deaths  = d("total_deaths")
    shots_fired   = d("total_shots_fired")
    shots_hit     = d("total_shots_hit")
    damage        = d("total_damage_done")
    hs_kills      = d("total_kills_headshot")
    rounds        = d("total_rounds_played")
    planted       = d("total_planted_bombs")
    defused       = d("total_defused_bombs")
    assists       = d("total_assists")
    survived      = max(rounds - total_deaths, 0.0)

    de_maps = list(curr.get("rounds_per_map", {}).keys())
    rounds_per_map_delta = {}
    for m in de_maps:
        pm = float(prev.get("rounds_per_map", {}).get(m, 0))
        cm = float(curr.get("rounds_per_map", {}).get(m, 0))
        rounds_per_map_delta[m] = max(cm - pm, 0.0)
    best_map = max(rounds_per_map_delta, key=rounds_per_map_delta.get) if any(v>0 for v in rounds_per_map_delta.values()) else None

    return {
        "qtKill": float(total_kills),
        "qtAssist": float(assists),
        "qtDeath": float(total_deaths),
        "qtHs": float(hs_kills),
        "qtBombeDefuse": float(defused),
        "qtBombePlant": float(planted),
        "vlDamage": float(damage),
        "qtHits": float(shots_hit),
        "qtShots": float(shots_fired),
        "qtRoundsPlayed": float(rounds),
        "qtSurvived": float(survived),
        "descMapName_de_ancient": 1.0 if best_map == "de_ancient" else 0.0,
        "descMapName_de_dust2":   1.0 if best_map == "de_dust2"   else 0.0,
        "descMapName_de_inferno": 1.0 if best_map == "de_inferno" else 0.0,
        "descMapName_de_mirage":  1.0 if best_map == "de_mirage"  else 0.0,
        "descMapName_de_nuke":    1.0 if best_map == "de_nuke"    else 0.0,
        "descMapName_de_overpass":1.0 if best_map == "de_overpass"else 0.0,
        "descMapName_de_train":   1.0 if best_map == "de_train"   else 0.0,
        "descMapName_de_vertigo": 1.0 if best_map == "de_vertigo" else 0.0,
    }

def steam_to_model_input(api_payload: dict) -> dict:
    stats = api_payload.get("playerstats", {}).get("stats", []) or []
    def g(name: str, default: float = 0.0) -> float:
        return _stat(stats, name, default)
    total_kills        = g("total_kills", 0)
    total_deaths       = g("total_deaths", 0)
    total_shots_fired  = g("total_shots_fired", 0)
    total_shots_hit    = g("total_shots_hit", 0)
    total_damage_done  = g("total_damage_done", 0)
    total_kills_hs     = g("total_kills_headshot", 0)
    total_rounds       = g("total_rounds_played", 0)
    total_planted      = g("total_planted_bombs", 0)
    total_defused      = g("total_defused_bombs", 0)
    total_assists      = g("total_assists", 0)
    qt_survived = max(total_rounds - total_deaths, 0)
    de_maps = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train","de_vertigo"]
    rounds_per_map = {m: g(f"total_rounds_map_{m}", 0) for m in de_maps}
    best_map = max(rounds_per_map, key=rounds_per_map.get) if any(v > 0 for v in rounds_per_map.values()) else None
    return {
        "qtKill": float(total_kills),
        "qtAssist": float(total_assists),
        "qtDeath": float(total_deaths),
        "qtHs": float(total_kills_hs),
        "qtBombeDefuse": float(total_defused),
        "qtBombePlant": float(total_planted),
        "vlDamage": float(total_damage_done),
        "qtHits": float(total_shots_hit),
        "qtShots": float(total_shots_fired),
        "qtRoundsPlayed": float(total_rounds),
        "qtSurvived": float(qt_survived),
        "descMapName_de_ancient": 1.0 if best_map == "de_ancient" else 0.0,
        "descMapName_de_dust2":   1.0 if best_map == "de_dust2"   else 0.0,
        "descMapName_de_inferno": 1.0 if best_map == "de_inferno" else 0.0,
        "descMapName_de_mirage":  1.0 if best_map == "de_mirage"  else 0.0,
        "descMapName_de_nuke":    1.0 if best_map == "de_nuke"    else 0.0,
        "descMapName_de_overpass":1.0 if best_map == "de_overpass"else 0.0,
        "descMapName_de_train":   1.0 if best_map == "de_train"   else 0.0,
        "descMapName_de_vertigo": 1.0 if best_map == "de_vertigo" else 0.0,
    }

# =========================
# FUNÇÃO DA UI (API + Δ)
# =========================
def run_infer(api_key: str, steam_id64: str, max_display: int, use_incremental: bool):
    if not api_key or not steam_id64:
        msg = "**Erro:** Informe Steam API Key e SteamID64."
        fig = plt.figure(figsize=(6,2)); plt.axis("off"); plt.text(0.5,0.5,msg,ha="center",va="center")
        return "—", msg, "", fig

    try:
        raw = _steam_get_user_stats(api_key.strip(), steam_id64.strip(), appid=APPID_CS)
    except Exception as e:
        msg = f"**Erro na Steam API:** {type(e).__name__}: {e}"
        fig = plt.figure(figsize=(6,2)); plt.axis("off"); plt.text(0.5,0.5,"Falha na API",ha="center",va="center")
        return "—", msg, "", fig

    curr_tot = _steam_totals(raw)
    cache = _load_cache()
    cache_key = steam_id64.strip()
    prev_tot = cache.get(cache_key)

    msg_extra = ""
    if bool(use_incremental):
        delta_stats = _delta_from_totals(curr_tot, prev_tot)
        if delta_stats is None:
            msg_extra = ("**Primeira execução para este SteamID64.** Snapshot salvo. "
                         "Jogue novas partidas e rode novamente para ver os deltas (≈ sessão).")
            cache[cache_key] = curr_tot; _save_cache(cache)
            fig = plt.figure(figsize=(6,2)); plt.axis("off"); plt.text(0.5,0.5,msg_extra,ha="center",va="center")
            return "—", msg_extra, "", fig
        stats = delta_stats
        if all((float(v)==0.0) for v in stats.values()):
            msg_extra = ("**Δ vazio:** sem mudanças desde o último snapshot (ou impacto zero nas métricas).")
    else:
        stats = steam_to_model_input(raw)
        if all((float(v)==0.0) for v in stats.values()):
            msg_extra = ("**Aviso:** API é acumulativa; ative o modo incremental (Δ) para aproximar por-partida.")

    cache[cache_key] = curr_tot
    _save_cache(cache)

    topk = int(max_display) if isinstance(max_display, (int, float)) else TOPK_SHAP
    card, expl = build_card_from_stats(stats, topk=topk)

    perfil   = card.get("perfil", "—")
    sugestao = card.get("sugestao", "—")
    p        = card.get("prob_vitoria", None)
    header = f"**{perfil}**"
    sub    = f"**Sugestão:** {sugestao}"
    if isinstance(p, (int, float)):
        sub += f"  \n**Prob. de vitória (model_)**: {float(p):.3f}"
    if msg_extra:
        sub += f"\n\n{msg_extra}"

    k  = card["kpis"]; vs = card["vs_cluster"]
    BIGGER_BETTER = {"KPR": True, "DPR": False, "ADR": True, "Acc%": True, "HS%": True}
    PCT_KEYS      = {"Acc%", "HS%", "Surv%"}
    def fmt_val(val, key, prec=3):
        if val is None: return "—"
        try: v = float(val)
        except: return "—"
        if key in PCT_KEYS: return f"{v:.{2 if key!='Surv%' else 1}f}%"
        return f"{v:.{prec}f}"
    def fmt_delta_arrow(val, key, prec_num=3, prec_pct=1):
        if val is None: return "—"
        try: v = float(val)
        except: return "—"
        arrow = "↑" if v > 0 else ("↓" if v < 0 else "→")
        good  = (v > 0 and BIGGER_BETTER.get(key, True)) or (v < 0 and not BIGGER_BETTER.get(key, True))
        color = "#1a7f37" if good else ("#b42318" if v != 0 else "#6b7280")
        disp  = f"{v:.{prec_pct}f} pp" if key in PCT_KEYS else f"{v:.{prec_num}f}"
        return f"<span style='color:{color};font-weight:700'>{arrow} {disp}</span>"

    table = f"""
    <style>
      .kpi-wrap {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }}
      .kpi-table {{ border-collapse: collapse; width: 100%; table-layout: fixed; border-radius: 14px; overflow: hidden; }}
      .kpi-table thead th {{ background: #f3f4f6; font-weight: 800; font-size: 18px; padding: 14px 16px; border-bottom: 2px solid #e5e7eb; }}
      .kpi-table td {{ font-size: 16px; padding: 12px 16px; border-bottom: 1px solid #e5e7eb; vertical-align: middle; }}
      .kpi-table tbody tr:nth-child(odd) {{ background: #fafafa; }}
      .kpi-kpi {{ width: 18%; font-weight: 700; color: #111827; }}
      .kpi-value {{ width: 32%; font-variant-numeric: tabular-nums; }}
      .kpi-delta {{ width: 50%; font-variant-numeric: tabular-nums; }}
      .kpi-legend {{ margin-top: 10px; color: #6b7280; font-size: 13px; }}
    </style>
    <div class="kpi-wrap">
      <table class="kpi-table">
        <thead><tr><th class="kpi-kpi">KPI</th><th class="kpi-value">Valor</th><th class="kpi-delta">Δ vs classe</th></tr></thead>
        <tbody>
          <tr><td class="kpi-kpi">KPR</td><td class="kpi-value">{fmt_val(k.get('KPR'),'KPR')}</td><td class="kpi-delta">{fmt_delta_arrow(vs.get('KPR'),'KPR')}</td></tr>
          <tr><td class="kpi-kpi">DPR</td><td class="kpi-value">{fmt_val(k.get('DPR'),'DPR')}</td><td class="kpi-delta">{fmt_delta_arrow(vs.get('DPR'),'DPR')}</td></tr>
          <tr><td class="kpi-kpi">ADR</td><td class="kpi-value">{fmt_val(k.get('ADR'),'ADR',prec=1)}</td><td class="kpi-delta">{fmt_delta_arrow(vs.get('ADR'),'ADR',prec_num=1)}</td></tr>
          <tr><td class="kpi-kpi">Acc%</td><td class="kpi-value">{fmt_val(k.get('Acc%'),'Acc%')}</td><td class="kpi-delta">{fmt_delta_arrow(vs.get('Acc%'),'Acc%')}</td></tr>
          <tr><td class="kpi-kpi">HS%</td><td class="kpi-value">{fmt_val(k.get('HS%'),'HS%')}</td><td class="kpi-delta">{fmt_delta_arrow(vs.get('HS%'),'HS%')}</td></tr>
        </tbody>
      </table>
      <div class="kpi-legend">Δ positivo em verde = melhor (exceto DPR). Percentuais em pp.</div>
    </div>
    """

    try:
        plt.close("all")
        fig = plt.figure(figsize=(8.4, 4.4))
        shap.plots.bar(expl, show=False, max_display=topk)
        ax = plt.gca()
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(False); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="y", pad=26)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_width_px = 0
        for t in ax.get_yticklabels():
            label_width_px = max(label_width_px, t.get_window_extent(renderer=renderer).width)
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        extra_left = (label_width_px + 24) / fig_w_px
        left = min(0.12 + extra_left, 0.70)
        fig.subplots_adjust(left=left)
        ax.set_xmargin(0.22)
        fig.canvas.draw_idle()
    except Exception:
        plt.clf()
        fig = plt.figure(figsize=(7,3))
        plt.text(0.5,0.6,"Não foi possível renderizar o SHAP (φ×x)", ha="center", va="center")
        plt.axis("off")

    return header, sub, table, fig

# =========================
# UI GRADIO
# =========================
with gr.Blocks(title="CSGO Coach — SOMENTE Steam API (Δ incremental)") as demo:
    gr.HTML("""
  <style>
  #wf_plot { max-width: 900px; margin: 0 auto; }
  #wf_plot > div { display:flex; justify-content:center; }
  #wf_plot canvas, #wf_plot svg, #wf_plot img, #wf_plot figure { margin:0 auto; }
  </style>
  """)
    gr.Markdown("## 🧠 CSGO Coach — Classe (model_c), KPIs e SHAP (φ×x no model_)")
    gr.Markdown("> **Steam Web API acumulativa** + **modo incremental (Δ)** com snapshot por SteamID64.")

    with gr.Row():
        api_key  = gr.Textbox(value=os.getenv("STEAM_API_KEY",""), label="Steam API Key", type="password", placeholder="cole sua chave aqui")
        steam_id = gr.Textbox(value=os.getenv("STEAM_ID64",""), label="SteamID64", placeholder="ex.: 7656119xxxxxxxxxx")

    with gr.Row():
        maxk    = gr.Slider(3, 20, value=TOPK_SHAP, step=1, label="Top-K (|φ×x|)")
        use_incremental = gr.Checkbox(value=True, label="Incremental (Δ desde o último snapshot)")
        run_btn = gr.Button("Rodar inferência", variant="primary")

    header_md = gr.Markdown(); sub_md = gr.Markdown(); kpi_html = gr.HTML(); wf_plot = gr.Plot()
    run_btn.click(run_infer, inputs=[api_key, steam_id, maxk, use_incremental],
                  outputs=[header_md, sub_md, kpi_html, wf_plot])

demo.launch()