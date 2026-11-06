"""Koshi Landslide Explorer: Streamlit dashboard for the MSc study on landslide susceptibility in the Koshi Basin.

Run from the notebook's folder (the one that contains `outputs/`, and `app/` after Section 10b):
    pip install -r requirements.txt
    streamlit run koshi_dashboard.py
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title='Koshi Landslide Explorer', page_icon='⛰️', layout='wide', initial_sidebar_state='collapsed')
_v = tuple(int(x) for x in st.__version__.split('.')[:2])
WIDE = {'width': 'stretch'} if _v >= (1, 46) else {'use_container_width': True}

BASE = Path(__file__).resolve().parent
SEARCH = [BASE, BASE.parent]
COLORS = {'Low': [26, 152, 80], 'Medium': [240, 190, 60], 'High': [215, 48, 39]}
HEX = {'Low': '#1a9850', 'Medium': '#d9a21b', 'High': '#d73027'}
TINT = {'Low': '#e3f2e8', 'Medium': '#fbf0d2', 'High': '#fbe3e1'}
EXAMPLES = {'Middle hills': (27.6, 86.0), 'Steep slope': (27.6, 87.0), 'High valley': (27.6, 87.1), 'Kathmandu Valley': (27.68, 85.32)}
STUDY = {'model': 'Gradient Boosting', 'test_auc': 0.861, 'test_recall': 0.849, 'test_blocks': 78, 'block_km': 10,
         'n_landslides': 9005, 'max_elev': 4500, 'dem': 'Copernicus GLO-30', 'steep_regime_auc': 0.772,
         'inventory': 'ICIMOD Landslide Inventory of Koshi Basin 2024',
         'share_oof': {'Low': 12.3, 'Moderate': 29.2, 'High': 58.6},
         'medians': {'elevation': {'landslide': 1475, 'background': 1502}, 'slope': {'landslide': 35.8, 'background': 26.7},
                     'tri': {'landslide': 54.0, 'background': 37.8}, 'relief': {'landslide': 216, 'background': 172}}}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Public Sans', system-ui, sans-serif; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }
.band { background: #18252b; color: #eef3f1; border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: .9rem;
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: .6rem 2rem; }
.band h1 { color: #ffffff; font-size: 1.55rem; margin: 0; padding: 0; font-weight: 700; letter-spacing: -0.01em; }
.band p { margin: .15rem 0 0; color: #b8c7c4; font-size: .92rem; max-width: 62ch; }
.band .facts { display: flex; gap: 1.6rem; }
.band .facts div { text-align: right; } .band .facts b { display: block; font-size: 1.15rem; color: #fff; }
.band .facts span { font-size: .78rem; color: #9fb3b0; }
.panel { background: #ffffff; border: 1px solid #d6dfdc; border-radius: 14px; padding: 1rem 1.1rem; }
.verdict { border-radius: 14px; padding: 1rem 1.15rem 1.05rem; border: 1px solid #d6dfdc; border-left-width: 8px; background: #fff; }
.verdict .cls { font-size: 2.1rem; font-weight: 700; line-height: 1.05; margin: 0; }
.verdict .what { font-size: .82rem; color: #5b6b70; margin: 0 0 .15rem; }
.verdict .big { font-size: 1.05rem; margin: .35rem 0 0; color: #1e2b30; }
.ladder { position: relative; height: 12px; border-radius: 6px; margin: .9rem 0 .2rem;
  background: linear-gradient(90deg, #1a9850 0 33.33%, #f0be3c 33.33% 66.66%, #d73027 66.66% 100%); }
.ladder span { position: absolute; top: -6px; width: 6px; height: 24px; border-radius: 3px; background: #18252b; border: 2px solid #fff; }
.ladder-lab { display: flex; justify-content: space-between; font-size: .75rem; color: #5b6b70; }
.small { font-size: .84rem; color: #5b6b70; line-height: 1.5; }
.tile { background: #fff; border: 1px solid #d6dfdc; border-radius: 12px; padding: .75rem .9rem; height: 100%; }
.tile .k { font-size: .78rem; color: #5b6b70; margin: 0; }
.tile .v { font-size: 1.45rem; font-weight: 700; margin: .1rem 0 0; color: #1e2b30; }
.tile .c { font-size: .78rem; margin: .1rem 0 0; }
.why { display: grid; grid-template-columns: 1.6rem 1fr; gap: .4rem; padding: .6rem 0; border-top: 1px solid #e3e9e7; font-size: .93rem; }
.why:first-of-type { border-top: 0; }
.why .i { font-weight: 700; text-align: center; border-radius: 6px; height: 1.6rem; line-height: 1.6rem; font-size: .8rem; }
.i.up { background: #fbe3e1; color: #b3261e; } .i.down { background: #e3f2e8; color: #17703f; } .i.flat { background: #eceff0; color: #5b6b70; }
.why small { display: block; color: #5b6b70; font-size: .8rem; margin-top: .15rem; }
.sec { font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 .5rem; color: #1e2b30; }
.lead { border-radius: 12px; padding: .85rem 1rem; background: #fff; border: 1px solid #d6dfdc; height: 100%; }
.lead b { display: block; font-size: 1.35rem; } .lead span { font-size: .84rem; color: #5b6b70; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
def find(*names):
    for root in SEARCH:
        for n in names:
            p = root / n
            if p.exists():
                return p
    return None


def to_latlon(x, y):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True).transform(np.asarray(x), np.asarray(y))
    return np.asarray(lat), np.asarray(lon)


FEATURES = ['elevation', 'slope', 'aspect_sin', 'aspect_cos', 'tri', 'relief', 'x_km', 'y_km']


def _estimator(name):
    """The same estimators the notebook tuned (Section 6)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    sc = lambda m: Pipeline([('scale', StandardScaler()), ('model', m)])
    return {'Baseline (majority)': DummyClassifier(strategy='prior'),
            'Logistic Regression': sc(LogisticRegression(max_iter=3000)),
            'LDA': sc(LinearDiscriminantAnalysis()), 'Gaussian Naive Bayes': sc(GaussianNB()),
            'KNN': sc(KNeighborsClassifier()), 'SVM (RBF)': sc(SVC(kernel='rbf', random_state=42)),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42),
            'Gradient Boosting': HistGradientBoostingClassifier(max_iter=300, random_state=42)}.get(name)


def build_all_models(grid_df):
    """Refit the ten tuned models on all samples (as for the final map) and score every 1 km cell.
    Needs outputs/model_dataset.csv and outputs/table_spatial_results.csv; results are cached to outputs/."""
    import ast
    ds, sp = find('outputs/model_dataset.csv', 'model_dataset.csv'), find('outputs/table_spatial_results.csv', 'table_spatial_results.csv')
    if not ds or not sp or any(f not in grid_df for f in FEATURES):
        return None
    cache = sp.parent / 'dashboard_model_scores.npz'
    res = pd.read_csv(sp)
    names = list(res['Model'])
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z['names']) == names and z['scores'].shape[1] == len(grid_df):
            scores, kinds = z['scores'], list(z['kinds'])
        else:
            cache.unlink(); return build_all_models(grid_df)
    else:
        data = pd.read_csv(ds)
        X, y = data[FEATURES].astype('float64'), data['label'].values
        G = grid_df[FEATURES].astype('float64')
        scores, kinds = [], []
        bar = st.progress(0.0, text='Training the ten models once on your saved data (about a minute)…')
        for k, n in enumerate(names):
            est = _estimator(n)
            params = ast.literal_eval(str(res.loc[k, 'Best params'])) if 'Best params' in res and pd.notna(res.loc[k, 'Best params']) else {}
            est.set_params(**params).fit(X, y)
            if hasattr(est, 'predict_proba'):
                scores.append(est.predict_proba(G)[:, 1]); kinds.append('probability')
            else:
                scores.append(est.decision_function(G)); kinds.append('margin')
            bar.progress((k + 1) / len(names), text=f'Trained {n} ({k + 1} of {len(names)})')
        bar.empty()
        scores = np.vstack(scores)
        np.savez_compressed(cache, names=np.array(names, dtype=object), scores=scores, kinds=np.array(kinds, dtype=object))
    ci = {}
    bc = find('outputs/table_bootstrap_ci.csv', 'table_bootstrap_ci.csv')
    if bc:
        b_ = pd.read_csv(bc); b_ = b_.rename(columns={b_.columns[0]: 'Model'}).set_index('Model')
        ci = {n: [float(b_.loc[n, 'lo']), float(b_.loc[n, 'hi'])] for n in names if n in b_.index and 'Baseline' not in n}
    best = res.loc[res['ROC-AUC'].idxmax(), 'Model']
    out = []
    for k, n in enumerate(names):
        sc_ = scores[k]; q1, q2 = np.quantile(sc_, [1 / 3, 2 / 3])
        r = res.iloc[k]
        out.append({'name': n, 'best': n == best, 'kind': kinds[k], 'constant': bool(np.ptp(sc_) < 1e-9),
                    'auc': round(float(r['ROC-AUC']), 3), 'pr_auc': round(float(r['PR-AUC']), 3), 'recall': round(float(r['Recall']), 3),
                    'precision': round(float(r['Precision']), 3), 'mcc': round(float(r['MCC']), 3), 'cv_auc': round(float(r['CV AUC']), 3),
                    'train_auc': round(float(r['Train AUC']), 3), 'ci': ci.get(n), 'q1': float(q1), 'q2': float(q2), 'scores': sc_})
    return out


@st.cache_data(show_spinner='Loading model results…')
def load():
    meta = dict(STUDY); models = []; source = ''
    js = find('app/koshi_app_data.json', 'koshi_app_data.json', 'outputs/koshi_app_data.json')
    if js:
        D = json.load(open(js))
        C = D['cells']
        cells = pd.DataFrame({'lat': C['lat'], 'lon': C['lon'], 'elevation': C['e'], 'slope': C['sl'], 'aspect': C['a'],
                              'tri': C['t'], 'relief': C['rl'], 's_best': C['s'],
                              'p_oof': [np.nan if v is None else v for v in C['so']]})
        m = D['meta']
        meta.update({k: m[k] for k in ['test_auc', 'test_recall', 'test_blocks', 'block_km', 'n_landslides', 'max_elev',
                                        'medians', 'oof_q1', 'oof_q2', 'share_oof', 'steep_regime_auc'] if k in m and m[k] is not None})
        meta['model'] = m.get('model', meta['model']); meta['dem'] = m.get('dem', meta['dem'])
        for mm in D.get('models', []):
            models.append({k: v for k, v in mm.items() if k != 'scores'} | {'scores': np.asarray(mm['scores'], float)})
        if not models:
            s = cells['s_best'].values
            models = [{'name': meta['model'], 'best': True, 'kind': 'probability', 'constant': False, 'auc': meta['test_auc'],
                       'recall': meta['test_recall'], 'ci': None, 'scores': s, 'q1': m['q1'], 'q2': m['q2']}]
        L = D.get('landslides', {})
        ls = pd.DataFrame({k: L[k] for k in L})
        source = f'{js.relative_to(js.parents[1]) if len(js.parents) > 1 else js.name}'
    else:
        rg = find('outputs/risk_grid.csv', 'risk_grid.csv')
        if rg is None:
            return None
        g = pd.read_csv(rg)
        lat, lon = to_latlon(g['x'], g['y'])
        cells = pd.DataFrame({'lat': lat, 'lon': lon, 'elevation': g['elevation'], 'slope': g['slope'], 'aspect': g['aspect'],
                              'tri': g['tri'], 'relief': g['relief'], 's_best': g['p'],
                              'p_oof': g['p_oof'] if 'p_oof' in g else np.nan})
        res = find('outputs/results.json', 'results.json')
        if res:
            R = json.load(open(res)); st_ = R.get('spatial_test', {})
            meta.update({'test_auc': st_.get('ROC-AUC', meta['test_auc']), 'test_recall': st_.get('Recall', meta['test_recall']),
                         'model': R.get('best_model', meta['model']), 'dem': str(R.get('dem_source', meta['dem'])).split(' (')[0]})
        q1, q2 = np.quantile(g['p'], [1 / 3, 2 / 3])
        models = [{'name': meta['model'], 'best': True, 'kind': 'probability', 'constant': False, 'auc': meta['test_auc'],
                   'recall': meta['test_recall'], 'ci': None, 'scores': g['p'].values, 'q1': q1, 'q2': q2}]
        if 'aspect_sin' not in g:
            g['aspect_sin'], g['aspect_cos'] = np.sin(np.radians(g['aspect'])), np.cos(np.radians(g['aspect']))
        if 'x_km' not in g:
            g['x_km'], g['y_km'] = g['x'] / 1000, g['y'] / 1000
        allm = build_all_models(g)
        if allm:
            models = allm
            meta['model'] = next(m['name'] for m in models if m['best'])
        ls = pd.DataFrame()
        inv = find('Koshi_Landslide_Inventory_2024.xlsx', 'data/Koshi_Landslide_Inventory_2024.xlsx')
        if inv:
            I = pd.read_excel(inv, sheet_name='Landslide_Inventory').drop_duplicates('ID').dropna(subset=['Centroid_X', 'Centroid_Y'])
            la, lo = to_latlon(I['Centroid_X'], I['Centroid_Y'])
            ls = pd.DataFrame({'lat': la, 'lon': lo, 'area': I['Area_'].round().astype(int), 'elev': I['Elevation'],
                               'slope': I['Slope'], 'aspect': I['Aspect'], 'id': I['ID'],
                               'date': I['Image_ID'].str.extract(r'_(\d{8})T')[0].fillna('')})
        source = str(rg.name)
    for mm in models:
        mm['sorted'] = np.sort(mm['scores'])
    return {'cells': cells, 'models': models, 'meta': meta, 'ls': ls, 'source': source}




# ---------------------------------------------------------------- helpers
def km_between(la1, lo1, la2, lo2):
    dy = (la1 - la2) * 110.574
    dx = (lo1 - lo2) * 111.320 * np.cos(np.radians((la1 + la2) / 2))
    return np.hypot(dx, dy)


def classify(s, q1, q2):
    return 'High' if s > q2 else 'Medium' if s > q1 else 'Low'


COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west']
compass = lambda a: COMPASS[int(((a + 22.5) % 360) // 45)]


def factors(c, md):
    t, sl, a, e, rl = c['tri'], c['slope'], c['aspect'], c['elevation'], c['relief']
    cos_a = np.cos(np.radians(a)); out = []
    note_tri = f"Typical landslide site {md['tri']['landslide']:.1f}; typical stable ground {md['tri']['background']:.1f}."
    if t >= 60: out.append(('up', 'Ruggedness', f'Very rugged ground (TRI {t:.1f}); the model’s most important factor reaches its plateau above TRI ≈ 60.', note_tri))
    elif t >= 39: out.append(('up', 'Ruggedness', f'Rugged ground (TRI {t:.1f}), above TRI 39, the first threshold the decision tree uses.', note_tri))
    elif t >= 25: out.append(('flat', 'Ruggedness', f'Moderately rugged ground (TRI {t:.1f}), below the TRI 39 threshold.', note_tri))
    else: out.append(('down', 'Ruggedness', f'Smooth ground (TRI {t:.1f}); risk is lowest below TRI ≈ 20–25.', note_tri))
    if sl >= md['slope']['landslide']: out.append(('up', 'Slope', f'Steep slope of {sl:.1f}°, at or above the landslide median ({md["slope"]["landslide"]:.1f}°).', 'Slope and ruggedness carry almost the same information (correlation 0.96).'))
    elif sl >= md['slope']['background']: out.append(('flat', 'Slope', f'Slope of {sl:.1f}°, between stable ground ({md["slope"]["background"]:.1f}°) and landslide sites ({md["slope"]["landslide"]:.1f}°).', ''))
    else: out.append(('down', 'Slope', f'Gentle slope of {sl:.1f}°, below typical stable ground ({md["slope"]["background"]:.1f}°).', ''))
    d = compass(a)
    if cos_a <= -0.5: out.append(('up', 'Slope direction', f'Faces {d} ({a:.0f}°). South-facing slopes hold 55% of mapped landslides but 27% of stable ground.', 'South-facing Himalayan slopes are drier and less densely forested.'))
    elif cos_a >= 0.5: out.append(('down', 'Slope direction', f'Faces {d} ({a:.0f}°). Only 2.8% of mapped landslides face north, against 23% of stable ground.', 'Caution: 95.5% of landslides were mapped from one winter image, when north faces are dimly lit.'))
    else: out.append(('flat', 'Slope direction', f'Faces {d} ({a:.0f}°), between high-risk south faces and low-risk north faces.', ''))
    if e > 2400: out.append(('down', 'Elevation', f'{e:,.0f} m. The model’s risk drops sharply near 2,400 m.', 'Part of this drop may reflect winter snow hiding landslides in the mapping image.'))
    elif e > 2000: out.append(('flat', 'Elevation', f'{e:,.0f} m, close to the band where risk begins to fall.', ''))
    else: out.append(('flat', 'Elevation', f'{e:,.0f} m; below 2,000 m elevation has little effect.', ''))
    out.append(('up' if rl >= md['relief']['landslide'] else 'down' if rl <= md['relief']['background'] else 'flat', 'Local relief',
                f'{rl:,.0f} m height range within about 330 m; adds little once ruggedness is known.', ''))
    return out


ACTION = {'High': 'Priority for field inspection before building, road cutting or the monsoon: look for cracks, seepage, bulging ground, tilted trees or fresh scars, and check the slopes above for debris sources.',
          'Medium': 'Field-check before major cuts or construction, avoid undercutting the toe of the slope and keep drainage clear.',
          'Low': 'Least susceptible third of the basin, but riverbank erosion, debris from slopes above and fresh road cuts can still cause failures.'}


DATA = load()
if DATA is None:
    st.markdown('<div class="band"><div><h1>Koshi Landslide Explorer</h1><p>No model results found.</p></div></div>', unsafe_allow_html=True)
    st.warning('Put this file in the notebook’s folder, next to `outputs/` (and `app/` after running Section 10b of the notebook), then reload.')
    st.stop()

cells, models, M, LS = DATA['cells'], DATA['models'], DATA['meta'], DATA['ls']
best_idx = next((k for k, m in enumerate(models) if m.get('best')), 0)
md = M['medians']

st.markdown(f"""<div class="band">
  <div><h1>⛰️ Koshi Landslide Explorer</h1>
  <p>Where are slopes in Nepal’s Koshi Basin most likely to fail? A research demonstration of ten machine learning models, ranked on ground they never saw.</p></div>
  <div class="facts">
    <div><b>{len(cells):,}</b><span>map cells of 1 km</span></div>
    <div><b>{M['n_landslides']:,}</b><span>mapped landslides</span></div>
    <div><b>{M['test_auc']:.3f}</b><span>best test ROC-AUC</span></div>
  </div></div>""", unsafe_allow_html=True)

PAGES = ['Explore a site', 'Model performance', 'About the study']
if hasattr(st, 'segmented_control'):
    page = st.segmented_control('Page', PAGES, default=PAGES[0], label_visibility='collapsed') or PAGES[0]
else:
    page = st.radio('Page', PAGES, horizontal=True, label_visibility='collapsed')


def site_rank(mm, i):
    s = float(mm['scores'][i])
    if mm.get('constant'):
        return s, None, None
    return s, classify(s, mm['q1'], mm['q2']), 100 * np.searchsorted(mm['sorted'], s, 'left') / len(mm['sorted'])


# ================================================================ explore
if page == 'Explore a site':
    if 'lat' not in st.session_state:
        st.session_state.lat, st.session_state.lon = EXAMPLES['Middle hills']

    def pick():
        k = st.session_state.get('ex')
        if k in EXAMPLES:
            st.session_state.lat, st.session_state.lon = EXAMPLES[k]

    map_col, site_col = st.columns([1.55, 1], gap='large')
    with site_col:
        st.markdown('<p class="sec" style="margin-top:0">Choose a site</p>', unsafe_allow_html=True)
        if hasattr(st, 'pills'):
            st.pills('Examples', list(EXAMPLES), key='ex', on_change=pick, label_visibility='collapsed')
        else:
            st.radio('Examples', list(EXAMPLES), key='ex', on_change=pick, horizontal=True, label_visibility='collapsed')
        a, b = st.columns(2)
        a.number_input('Latitude (° N)', 26.0, 29.0, step=0.01, format='%.4f', key='lat')
        b.number_input('Longitude (° E)', 84.0, 89.0, step=0.01, format='%.4f', key='lon')
        mi = st.selectbox('Model', range(len(models)), index=best_idx,
                          format_func=lambda k: f"{models[k]['name']}{' (best)' if models[k].get('best') else ''} · AUC {models[k]['auc']:.3f}")
    lat, lon, m = float(st.session_state.lat), float(st.session_state.lon), models[mi]
    d = km_between(lat, lon, cells['lat'].values, cells['lon'].values)
    i = int(np.argmin(d)); dist = float(d[i]); inside = dist <= 0.75
    c = cells.iloc[i]
    s, cls, pct = site_rank(m, i)

    with site_col:
        if not inside:
            st.error(f'No prediction here. The nearest map cell is {dist:.1f} km away: the point is outside the study area or above {M["max_elev"]:,} m.')
        elif cls is None:
            st.info(f"{m['name']} gives every cell the same score (AUC {m['auc']:.3f}), so it can’t rank sites. Choose another model.")
        else:
            share = M.get('share_oof', {}).get({'Medium': 'Moderate'}.get(cls, cls)) if m.get('best') else None
            st.markdown(f"""<div class="verdict" style="border-left-color:{HEX[cls]};background:{TINT[cls]}">
              <p class="what">Landslide susceptibility · {m['name']}</p>
              <p class="cls" style="color:{HEX[cls]}">{cls}</p>
              <p class="big">More susceptible than <b>{pct:.0f}%</b> of the basin</p>
              <div class="ladder"><span style="left:calc({min(99.0, max(1.0, pct)):.1f}% - 3px)"></span></div>
              <div class="ladder-lab"><span>Low third</span><span>Medium</span><span>High third</span></div>
              <p class="small" style="margin:.7rem 0 0">Score {s:.3f}{' (SVM margin)' if m.get('kind') == 'margin' else ''} · nearest cell {dist * 1000:,.0f} m away{f' · this third holds {share:.1f}% of landslides on unseen ground' if share is not None else ''}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<p class="small" style="margin-top:.6rem"><b>What to do.</b> {ACTION[cls]}</p>', unsafe_allow_html=True)

    with map_col:
        show = cells[['lat', 'lon']].copy()
        if m.get('constant'):
            show['color'] = [[154, 167, 170, 200]] * len(show); show['label'] = 'no ranking'
        else:
            cl = np.where(m['scores'] > m['q2'], 'High', np.where(m['scores'] > m['q1'], 'Medium', 'Low'))
            show['color'] = [COLORS[k] + [185] for k in cl]; show['label'] = cl
        show['score'] = np.round(m['scores'], 3)
        layers = [pdk.Layer('ScatterplotLayer', show, get_position='[lon, lat]', get_fill_color='color', get_radius=520, pickable=True),
                  pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                            get_fill_color=[24, 37, 43, 255], get_line_color=[255, 255, 255], stroked=True,
                            line_width_min_pixels=3, get_radius=2200)]
        if len(LS) and st.toggle('Show mapped landslides', value=False):
            layers.insert(1, pdk.Layer('ScatterplotLayer', LS[['lat', 'lon']], get_position='[lon, lat]',
                                       get_fill_color=[24, 37, 43, 140], get_radius=180))
        st.pydeck_chart(pdk.Deck(layers=layers, map_style=pdk.map_styles.CARTO_LIGHT,
                                 initial_view_state=pdk.ViewState(latitude=lat if inside else float(cells['lat'].mean()),
                                                                  longitude=lon if inside else float(cells['lon'].mean()), zoom=7.2 if inside else 6.6),
                                 tooltip={'text': '{label} susceptibility\nscore {score}'}), height=470, **WIDE)
        st.markdown('<p class="small">Each dot is a 1 km cell coloured by the chosen model: green low, amber medium, red high third of the basin. Hover for the score; the dark ring marks your site.</p>', unsafe_allow_html=True)

    if not inside:
        st.stop()

    # ---- terrain strip
    st.markdown('<p class="sec">Terrain at this site</p>', unsafe_allow_html=True)
    tiles = [('Ruggedness (TRI)', f"{c['tri']:.1f}", c['tri'] - md['tri']['background'], 'vs typical stable ground'),
             ('Slope', f"{c['slope']:.1f}°", c['slope'] - md['slope']['background'], '° vs typical stable ground'),
             ('Faces', compass(c['aspect']).capitalize(), None, f"{c['aspect']:.0f}° from north"),
             ('Elevation', f"{c['elevation']:,.0f} m", None, 'risk falls above 2,400 m'),
             ('Local relief', f"{c['relief']:,.0f} m", c['relief'] - md['relief']['background'], 'm vs typical stable ground')]
    for col, (k, v, delta, note) in zip(st.columns(5), tiles):
        if delta is None:
            sub = f'<p class="c" style="color:#5b6b70">{note}</p>'
        else:
            sub = f'<p class="c" style="color:{"#b3261e" if delta > 0 else "#17703f"}">{delta:+.1f} {note}</p>'
        col.markdown(f'<div class="tile"><p class="k">{k}</p><p class="v">{v}</p>{sub}</div>', unsafe_allow_html=True)

    # ---- why + consensus
    left, right = st.columns([1, 1.05], gap='large')
    with left:
        st.markdown('<p class="sec">Why this result</p>', unsafe_allow_html=True)
        if not m.get('best'):
            st.markdown(f'<p class="small">{m["name"]} uses the same eight measures; these notes come from the study’s analysis of the best model.</p>', unsafe_allow_html=True)
        sym = {'up': '▲', 'down': '▼', 'flat': '●'}
        html = ''.join(f'<div class="why"><span class="i {dr}" title="{ {"up": "raises risk", "down": "lowers risk", "flat": "little effect"}[dr] }">{sym[dr]}</span>'
                       f'<div><b>{name}.</b> {text}{f"<small>{note}</small>" if note else ""}</div></div>'
                       for dr, name, text, note in factors(c, md))
        st.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)
        notes = []
        if c['slope'] >= 35 and M.get('steep_regime_auc'):
            notes.append(f"Steep terrain: the model ranks sites less accurately here (AUC {M['steep_regime_auc']:.3f} against {M['test_auc']:.3f} overall).")
        if m.get('best') and np.isfinite(c['p_oof']) and M.get('oof_q1') is not None and cls:
            oc = classify(c['p_oof'], M['oof_q1'], M['oof_q2'])
            notes.append(f'A model that never saw this area gives {oc}' + (' too, so the class is stable.' if oc == cls else f' instead of {cls}: treat the class as uncertain.'))
        for n_ in notes:
            st.markdown(f'<p class="small" style="margin-top:.5rem">⚠︎ {n_}</p>', unsafe_allow_html=True)

    with right:
        st.markdown('<p class="sec">Do the ten models agree?</p>', unsafe_allow_html=True)
        rows = []
        for k, mm in enumerate(models):
            sc, ccls, p_ = site_rank(mm, i)
            rows.append({'Model': mm['name'], 'AUC': mm['auc'], 'Percentile': p_, 'Class': ccls or 'cannot rank',
                         'Score': None if ccls is None else round(sc, 3), 'Selected': k == mi})
        cmp = pd.DataFrame(rows).sort_values('AUC', ascending=False)
        ranked = cmp[cmp['Class'] != 'cannot rank']
        if len(ranked) > 1:
            cnt = ranked['Class'].value_counts(); top3 = ranked.head(3)['Class']
            st.markdown(f'<p class="small"><b>{cnt.max()} of {len(ranked)}</b> models put this site in the <b>{cnt.idxmax().lower()}</b> third. '
                        + ('The three most accurate models agree.' if top3.nunique() == 1 else 'The three most accurate models disagree, so treat the class as uncertain.')
                        + '</p>', unsafe_allow_html=True)
            order = list(ranked['Model'])
            bands = pd.DataFrame({'x': [0, 33.33, 66.66], 'x2': [33.33, 66.66, 100], 'Class': ['Low', 'Medium', 'High']})
            bg = alt.Chart(bands).mark_rect(opacity=.16).encode(x=alt.X('x:Q', scale=alt.Scale(domain=[0, 100]), title='Percentile of this site in the basin'),
                                                                x2='x2:Q', color=alt.Color('Class:N', scale=alt.Scale(domain=list(HEX), range=list(HEX.values())), legend=None))
            rule = alt.Chart(ranked).mark_rule(color='#c9d3d0').encode(x=alt.datum(0), x2='Percentile:Q', y=alt.Y('Model:N', sort=order, title=None))
            dots = alt.Chart(ranked).mark_circle(size=170, opacity=1, stroke='#18252b').encode(
                x='Percentile:Q', y=alt.Y('Model:N', sort=order, title=None, axis=alt.Axis(labelLimit=240)),
                color=alt.Color('Class:N', scale=alt.Scale(domain=list(HEX), range=list(HEX.values())), legend=None),
                strokeWidth=alt.condition('datum.Selected', alt.value(2.5), alt.value(0)),
                tooltip=['Model', alt.Tooltip('AUC:Q', format='.3f'), alt.Tooltip('Percentile:Q', format='.0f'), 'Class', 'Score'])
            st.altair_chart((bg + rule + dots).properties(height=max(160, 34 * len(ranked) + 40)), **WIDE)
            st.markdown('<p class="small">Models are ordered by test ROC-AUC, best at the top; the outlined dot is the model you chose. Percentiles compare this site with every basin cell under the same model, so they are comparable across models.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel small">Only one model is loaded. Run Section 10b of the notebook to export all ten models and compare them here.</div>', unsafe_allow_html=True)

    # ---- evidence + downloads
    left, right = st.columns([1, 1.05], gap='large')
    with left:
        if len(LS):
            st.markdown('<p class="sec">Mapped landslides nearby</p>', unsafe_allow_html=True)
            dl = km_between(lat, lon, LS['lat'].values, LS['lon'].values)
            n1, n3 = int((dl <= 1).sum()), int((dl <= 3).sum())
            basin = len(LS) / float(len(cells))
            a, b, c3 = st.columns(3)
            a.markdown(f'<div class="tile"><p class="k">Within 1 km</p><p class="v">{n1}</p></div>', unsafe_allow_html=True)
            b.markdown(f'<div class="tile"><p class="k">Within 3 km</p><p class="v">{n3}</p><p class="c" style="color:#5b6b70">{n3 / (np.pi * 9) / basin:.1f}× basin density</p></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="tile"><p class="k">Nearest</p><p class="v">{dl.min():.2f} km</p></div>', unsafe_allow_html=True)
            if 'area' in LS and n3:
                near = LS.assign(dist_km=dl).query('dist_km <= 3').sort_values('dist_km').head(6)
                near = near.assign(Faces=near['aspect'].map(compass), Date=near['date'].map(lambda x: f'{x[6:8]}/{x[4:6]}/{x[:4]}' if x else ''))
                st.dataframe(near.rename(columns={'id': 'ID', 'dist_km': 'Distance (km)', 'area': 'Area (m²)', 'slope': 'Slope (°)', 'elev': 'Elevation (m)'})
                             [['ID', 'Distance (km)', 'Area (m²)', 'Slope (°)', 'Faces', 'Elevation (m)', 'Date']].round({'Distance (km)': 2}),
                             hide_index=True, **WIDE)
    with right:
        st.markdown('<p class="sec">Take the results with you</p>', unsafe_allow_html=True)
        site = pd.DataFrame([{'latitude': lat, 'longitude': lon, 'model': m['name'], 'score': s, 'class': cls, 'basin_percentile': pct,
                              'elevation_m': c['elevation'], 'slope_deg': c['slope'], 'aspect_deg': c['aspect'], 'tri': c['tri'], 'relief_m': c['relief']}])
        st.download_button('Download this site’s result (CSV)', site.to_csv(index=False), 'koshi_site_result.csv', 'text/csv', **WIDE)
        if len(models) > 1:
            st.download_button('Download the ten-model comparison (CSV)', cmp.drop(columns='Selected').to_csv(index=False),
                               'koshi_model_comparison.csv', 'text/csv', **WIDE)
        if not m.get('constant'):
            full = cells[['lat', 'lon', 'elevation', 'slope', 'aspect', 'tri', 'relief']].copy()
            full['score'] = m['scores']; full['class'] = np.where(full['score'] > m['q2'], 'High', np.where(full['score'] > m['q1'], 'Medium', 'Low'))
            st.download_button(f"Download the full {m['name']} map, {len(full):,} cells (CSV)", full.to_csv(index=False),
                               f"koshi_{m['name'].replace(' ', '_').lower()}_map.csv", 'text/csv', **WIDE)
        st.markdown('<p class="small">Scores rank sites within the basin; they are not the probability that a landslide will happen, and they say nothing about when.</p>', unsafe_allow_html=True)

# ================================================================ performance
elif page == 'Model performance':
    perf = pd.DataFrame([{'Model': mm['name'], 'Test ROC-AUC': mm['auc'], 'CI low': mm['ci'][0] if mm.get('ci') else None,
                          'CI high': mm['ci'][1] if mm.get('ci') else None, 'PR-AUC': mm.get('pr_auc'), 'Recall': mm.get('recall'),
                          'Precision': mm.get('precision'), 'MCC': mm.get('mcc'), 'Block-CV AUC': mm.get('cv_auc'),
                          'Training AUC': mm.get('train_auc')} for mm in models])
    # the leaderboard needs only test metrics, which the notebook saves for every model
    sp = find('outputs/table_spatial_results.csv', 'table_spatial_results.csv')
    if sp and (len(perf) < 2 or perf['Recall'].isna().any()):
        t = pd.read_csv(sp)
        perf = pd.DataFrame({'Model': t['Model'], 'Test ROC-AUC': t['ROC-AUC'], 'CI low': np.nan, 'CI high': np.nan,
                             'PR-AUC': t.get('PR-AUC'), 'Recall': t.get('Recall'), 'Precision': t.get('Precision'), 'MCC': t.get('MCC'),
                             'Block-CV AUC': t.get('CV AUC'), 'Training AUC': t.get('Train AUC')})
    bc = find('outputs/table_bootstrap_ci.csv', 'table_bootstrap_ci.csv')
    if bc and perf['CI low'].isna().all():
        b_ = pd.read_csv(bc); b_ = b_.rename(columns={b_.columns[0]: 'Model'}).set_index('Model')
        perf['CI low'] = perf['Model'].map(b_['lo']) if 'lo' in b_ else np.nan
        perf['CI high'] = perf['Model'].map(b_['hi']) if 'hi' in b_ else np.nan
        perf.loc[perf['Model'].str.contains('Baseline'), ['CI low', 'CI high']] = np.nan
    perf = perf.sort_values('Test ROC-AUC', ascending=False).reset_index(drop=True)
    b = perf.iloc[0]; learned = perf[~perf['Model'].str.contains('Baseline')]
    linear = learned[learned['Model'].isin(['Logistic Regression', 'LDA'])]['Test ROC-AUC']
    cards = [(b['Model'], 'best model on held-out blocks'), (f"{b['Test ROC-AUC']:.3f}", 'its test ROC-AUC'),
             (f"{b['Test ROC-AUC'] - linear.max():+.3f}" if len(linear) else f"{b['Recall']:.3f}", 'gain over the best linear model' if len(linear) else 'recall'),
             (f"{M['share_oof'].get('High', 58.6):.1f}%", 'of landslides in the top third, unseen ground')]
    for col, (v, k) in zip(st.columns(4), cards):
        col.markdown(f'<div class="lead"><b>{v}</b><span>{k}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec">Leaderboard on {M["test_blocks"]} held-out 10 km blocks</p>', unsafe_allow_html=True)
    order = list(perf['Model'])
    pf = perf.copy()
    pf['label_x'] = pf[['Test ROC-AUC', 'CI high']].max(axis=1)
    pf['label'] = pf['Test ROC-AUC'].map(lambda v: f'{v:.3f}') + np.where(pf['Model'].str.contains('Baseline'), ' (chance)', '')
    pf['is_best'] = pf['Model'] == b['Model']
    X = alt.Scale(domain=[0.5, 0.9], clamp=True)
    Y = alt.Y('Model:N', sort=order, title=None,
              axis=alt.Axis(labelLimit=260, labelFontSize=13, labelColor='#1e2b30', labelPadding=8, ticks=False, domain=False))
    base_ = alt.Chart(pf)
    bars = base_.mark_bar(height=20, cornerRadiusEnd=4, clip=True).encode(
        x=alt.X('Test ROC-AUC:Q', scale=X, title='Test ROC-AUC (bars start at 0.5, chance level)',
                axis=alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9], format='.1f', grid=True, titlePadding=10)),
        x2=alt.datum(0.5), y=Y,
        color=alt.condition('datum.is_best', alt.value('#18252b'), alt.value('#a9bbb7')),
        tooltip=['Model', alt.Tooltip('Test ROC-AUC:Q', format='.3f'), alt.Tooltip('CI low:Q', format='.3f'),
                 alt.Tooltip('CI high:Q', format='.3f'), alt.Tooltip('Recall:Q', format='.3f')])
    chart = bars
    if pf['CI low'].notna().any():
        chart = chart + alt.Chart(pf.dropna(subset=['CI low'])).mark_rule(color='#d73027', strokeWidth=2.5, clip=True).encode(
            x=alt.X('CI low:Q', scale=X), x2='CI high:Q', y=Y)
    labels = base_.mark_text(align='left', dx=8, fontSize=13, color='#1e2b30').encode(
        x=alt.X('label_x:Q', scale=X), y=Y, text='label:N')
    st.altair_chart((chart + labels).properties(height=max(180, 42 * len(pf)), padding={'left': 5, 'right': 70, 'top': 5, 'bottom': 10})
                    .configure_view(strokeWidth=0), **WIDE)
    st.markdown('<p class="small">' + ('Red lines are 95% block-bootstrap intervals; the dark bar is the best model.' if perf['CI low'].notna().any()
                else 'The dark bar is the best model. Confidence intervals appear once the notebook has saved table_bootstrap_ci.csv.') + '</p>', unsafe_allow_html=True)
    with st.expander('All metrics', expanded=True):
        st.dataframe(perf.round(3), hide_index=True, **WIDE)
    tables = [('Random versus block validation', 'table_cv_random_vs_spatial.csv'), ('Repeated random and block splits', 'table_repeated_splits.csv'),
              ('Cross-validated AUC against block size (0 = random)', 'table_block_size.csv'),
              ('Permutation importance of the best model', 'table_feature_importance.csv'),
              ('Landslides per susceptibility third, unseen ground', 'table_density_per_class_oof.csv'),
              ('Landslides per susceptibility third, in-sample', 'table_density_per_class.csv')]
    found = [(t, find(f'outputs/{f}', f)) for t, f in tables]
    if any(p for _, p in found):
        st.markdown('<p class="sec">Validation detail</p>', unsafe_allow_html=True)
        for title, p in found:
            if p:
                with st.expander(title):
                    st.dataframe(pd.read_csv(p), hide_index=True, **WIDE)
    figs = [('ROC and precision–recall curves', 'fig7_roc_pr'), ('Partial dependence of the best model', 'fig11b_partial_dependence'),
            ('Basin susceptibility map', 'fig12_risk_map')]
    found_f = [(t, find(f'figures/{f}.png', f'figures/{f}.jpg')) for t, f in figs]
    if any(p for _, p in found_f):
        st.markdown('<p class="sec">Figures</p>', unsafe_allow_html=True)
        for title, p in found_f:
            if p:
                with st.expander(title):
                    st.image(str(p), **WIDE)

# ================================================================ about
else:
    l, r = st.columns([1.4, 1], gap='large')
    with l:
        st.markdown(f"""
<p class="sec" style="margin-top:0">A comparative analysis of ten machine learning models for predicting landslide susceptibility in the Koshi Basin, Nepal</p>
<p class="small">Kaushal Khanal · MSc Data Science and Computational Intelligence · Softwarica College of IT and E-Commerce (Coventry University) · STW7072CEM Machine Learning</p>

**What it predicts.** How likely each 1 km cell is to host the start of a landslide, compared with the rest of the basin. It ranks places; it does not forecast when a slope will fail or where debris will travel.

**Data.** {M['n_landslides']:,} landslides from the {M['inventory']} (CC BY 4.0) below {M['max_elev']:,} m, and the same number of stable points at least 50 m beyond each landslide footprint. Eight terrain measures come from the {M['dem']} elevation model: elevation, slope, two aspect components, terrain ruggedness, 330 m local relief, easting and northing.

**Method.** Ten models were tuned by five-fold block cross-validation and tested once on {M['test_blocks']} held-out 10 km blocks: a baseline, logistic regression, LDA, Gaussian naive Bayes, KNN, an RBF support vector machine, a decision tree, random forest, extra trees and gradient boosting.

**Findings.** Gradient Boosting ranked best (ROC-AUC {M['test_auc']:.3f}), significantly ahead of Random Forest. Ruggedness and slope direction drive the predictions. A random split inflated cross-validated AUC by up to 0.024, so block validation matters most when choosing a model.
""", unsafe_allow_html=True)
    with r:
        st.markdown("""<div class="panel">
<p class="sec" style="margin-top:0">Read the results with care</p>
<p class="small">• Terrain only: no rainfall, rock type, land cover or earthquake shaking.</p>
<p class="small">• 95.5% of landslides were mapped from one winter image, so north-facing and high slopes may be under-mapped and under-rated.</p>
<p class="small">• Accuracy drops in the steepest terrain (AUC 0.772), where stable ground is steep too.</p>
<p class="small">• Results are for 1 km cells; a site visit decides.</p>
<p class="small">• Land-use decisions in Nepal require review under the Disaster Risk Reduction and Management Act 2017.</p>
</div>""", unsafe_allow_html=True)