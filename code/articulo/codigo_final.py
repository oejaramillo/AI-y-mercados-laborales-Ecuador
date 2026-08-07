# ============================================================
# ANÁLISIS DE EXPOSICIÓN DE LLMs - ECUADOR 2025
#
# Qué hace, en orden:
#   1. Carga ENEMDU 2025, crosswalks y catálogo CIUO
#   2. Calcula scores de exposición por ocupación (tarea -> SOC -> CIUO)
#   3. Une con ENEMDU por p41 (NO p40) sobre población ocupada (condact 1-6)
#   4. Imputa por proximidad los CIUO sin match (4->3->2->1 dígitos + semántica)
#   5. Calcula y aplica el factor de ajuste F
#   6. Produce resultados, tablas .tex y figuras
#   7. Compara crosswalk BLS vs manual
# ============================================================

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# RUTA DEL REPOSITORIO
# ============================================================

RUTA = Path(r'C:\Users\jairo\OneDrive\Documentos\Artículos académicos y opinión\market labour and Ai\GitHub\AI-y-mercados-laborales-Ecuador')

if not RUTA.exists():
    raise SystemExit(f'ERROR: no existe la ruta\n  {RUTA}\nEdita la variable RUTA arriba.')

# Carpetas de salida (se crean si no existen)
(RUTA / 'figures').mkdir(exist_ok=True)
(RUTA / 'paper' / 'tablas').mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('EXPOSICIÓN A LLMs - ECUADOR 2025')
print('=' * 70)
print(f'Repositorio: {RUTA}')


# ============================================================
# PARÁMETROS
# ============================================================

# Rúbrica de Eloundou et al. (2023): etiqueta -> (alpha, beta, gamma)
RUBRICA = {'E1': (1.0, 1.0, 1.0),
           'E2': (0.0, 0.5, 1.0),
           'E0': (0.0, 0.0, 0.0)}

# Al agregar tareas dentro de una ocupación, las Core pesan el doble
PESO_CORE, PESO_SUPP = 2.0, 1.0

# Población ocupada: 1=adecuado, 2=subempleo tiempo, 3=subempleo ingresos,
# 4=otro no pleno, 5=no remunerado, 6=no clasificado
CONDACT_OCUPADOS = [1, 2, 3, 4, 5, 6]

# Similitud mínima para aceptar una imputación semántica
UMBRAL_SEMANTICO = 0.35

COLS = ['alpha_gpt', 'beta_gpt', 'gamma_gpt', 'alpha_h', 'beta_h', 'gamma_h']

# ------------------------------------------------------------
# IMPUTACIÓN MANUAL 
#
# Códigos CIUO-8 ecuatorianos sin equivalente directo en el crosswalk BLS,
# asignados uno por uno a la familia de ocupaciones SOC más cercana.
# Proviene de las cuatro rondas de `merges p41.ipynb`.
#
# El valor es un prefijo de código SOC: '29-10' toma el promedio de todas
# las ocupaciones que empiezan así; '15-1211' identifica una sola.
#
# Tiene prioridad sobre la imputación jerárquica automática, porque una
# decisión razonada vence a truncar dígitos.
# ------------------------------------------------------------

IMPUTACION_MANUAL = {
    # --- Salud y tecnologías de la información ---
    '2211': '29-10',    # Médicos generales -> ocupaciones médicas
    '2212': '29-10',    # Médicos especialistas
    '2511': '15-1211',  # Analistas de sistemas -> Computer Systems Analysts
    '2512': '15-1252',  # Desarrolladores de software -> Software Developers
    '2513': '15-12',    # Desarrolladores web -> TI general
    '2514': '15-12',    # Programadores de aplicaciones
    '2521': '15-12',    # Diseñadores de bases de datos
    '2522': '15-1244',  # Administradores de sistemas -> Network and Systems Admin
    '2523': '15-1241',  # Arquitectos de redes -> Computer Network Architects
    '2529': '15-12',    # Especialistas en bases de datos y redes n.e.p.
    '3511': '15-12',    # Técnicos en operaciones de TI
    '3512': '15-1232',  # Técnicos de soporte -> Computer User Support Specialists
    '3513': '15-1231',  # Técnicos de redes -> Computer Network Support Specialists
    '3514': '15-1241',  # Técnicos en tecnología web
    '2519': '15-12',    # Analistas y desarrolladores de software n.e.p.

    # --- Cuidados, agro, educación ---
    '5321': '31-1',     # Cuidados personales en instituciones -> Home Health Aides
    '5322': '31-1',     # Cuidados personales a domicilio
    '3258': '29-20',    # Ayudantes de ambulancia -> EMT y paramédicos
    '3252': '29-20',    # Documentación sanitaria
    '3142': '19-40',    # Técnicos agropecuarios -> Agricultural/Food Scientists
    '3143': '19-40',    # Técnicos forestales
    '2622': '25-40',    # Bibliotecarios -> ocupaciones de biblioteca
    '5312': '25-90',    # Auxiliares de maestros -> Teacher Assistants

    # --- Fuerzas Armadas ---
    '0110': '11-',      # Oficiales -> directivos y gerentes
    '0210': '33-',      # Suboficiales -> supervisores de seguridad
    '0310': '33-',      # Otros miembros

    # --- Varios ---
    '1115': '11-1031',  # Miembros del Poder Ejecutivo -> Legislators
    '2230': '29-12',    # Medicina tradicional -> médicos especialistas
    '2659': '27-10',    # Artistas creativos
    '3413': '21-20',    # Auxiliares religiosos -> Community Workers
    '4414': '43-60',    # Escribientes públicos -> secretarias
    '5161': '39-90',    # Astrólogos y adivinadores -> Personal Care
    '5162': '39-90',    # Acompañantes y ayudantes de cámara
    '6340': '45-3031',  # Pescadores y cazadores -> Fishing and Hunting Workers
    '7132': '47-2141',  # Barnizadores -> Painters, Construction
    '7542': '47-5032',  # Dinamiteros -> Explosives Workers
    '8155': '51-6',     # Operadores de máquinas de peletería -> textil
    '8159': '51-6',     # Operadores de máquinas textiles n.e.p.
    '9332': '45-2',     # Conductores de animales de tracción -> agrícola
    '9334': '43-5071',  # Reponedores -> Shipping and Inventory Clerks
    '9510': '41-',      # Trabajadores ambulantes -> Sales Workers
    '9613': '37-20',    # Barrenderos -> Janitors and Cleaners
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def norm_ciuo(valor, digitos=4):
    """Convierte cualquier código CIUO/ISCO a texto de 4 dígitos.
    Maneja floats ('3341.0'), comas decimales y espacios."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    s = str(valor).strip().replace(',', '.').split('.')[0]
    s = ''.join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(digitos) if len(s) <= digitos else s[:digitos]


def prom_pond(df, col, peso='fexp'):
    """Media ponderada por factor de expansión, ignorando NaN."""
    m = df[col].notna() & df[peso].notna()
    if not m.any():
        return np.nan
    return float(np.average(df.loc[m, col], weights=df.loc[m, peso]))


def leer_csv(ruta, **kw):
    """Lee CSV probando las codificaciones que aparecen en el repo."""
    for enc in ('utf-8', 'latin-1', 'ISO-8859-1'):
        try:
            return pd.read_csv(ruta, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'No se pudo leer {ruta}')


def limpiar_texto(s):
    """Minúsculas, sin tildes, sin puntuación. Para comparar descripciones."""
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', s)).strip()


def scores_de_etiqueta(serie):
    """Aplica la rúbrica a una columna de etiquetas E0/E1/E2."""
    et = serie.astype(str).str.strip().str.upper()
    vals = et.map(lambda e: RUBRICA.get(e, (np.nan, np.nan, np.nan)))
    return pd.DataFrame(vals.tolist(), index=serie.index,
                        columns=['alpha', 'beta', 'gamma'])


# ============================================================
# PASO 1: CARGAR DATOS
# ============================================================

print('\n' + '=' * 70)
print('PASO 1: CARGAR DATOS')
print('=' * 70)

enemdu, meta = pyreadstat.read_sav(str(RUTA / 'data' / 'BDDenemdu_personas_2025_anual.sav'))
print(f'  ENEMDU: {len(enemdu):,} registros')
print(f'  Población expandida: {enemdu["fexp"].sum():,.0f}')

cw_bls = leer_csv(RUTA / 'crosswalk' / 'cw-bls.csv')
cw_own = leer_csv(RUTA / 'crosswalk' / 'cw-own.csv')
print(f'  Crosswalk BLS:    {len(cw_bls):,} filas de tarea')
print(f'  Crosswalk manual: {len(cw_own):,} filas de tarea')

ciuo = leer_csv(RUTA / 'data' / 'Clasificación nacional de ocupaciones.csv')
ciuo['ciuo'] = ciuo['CODIGO'].map(norm_ciuo)
catalogo = ciuo.dropna(subset=['ciuo']).drop_duplicates('ciuo')[['ciuo', 'DESCRIPCION']]
print(f'  Catálogo CIUO: {len(catalogo):,} códigos')


# ============================================================
# PASO 2: SCORES DE EXPOSICIÓN POR OCUPACIÓN
# Agrega tarea -> SOC (ponderado Core/Supplemental) -> CIUO (promedio simple)
# ============================================================

print('\n' + '=' * 70)
print('PASO 2: SCORES POR OCUPACIÓN')
print('=' * 70)


def construir_scores(cw, nombre):
    cw = cw.copy()

    # GPT-4 y humano, cada uno con su rúbrica
    cw[['alpha_gpt', 'beta_gpt', 'gamma_gpt']] = scores_de_etiqueta(cw['gpt4_exposure'])
    cw[['alpha_h', 'beta_h', 'gamma_h']] = scores_de_etiqueta(cw['human_exposure_agg'])

    cw['peso'] = np.where(cw['Task Type'].astype(str).str.strip() == 'Core',
                          PESO_CORE, PESO_SUPP)

    # tarea -> SOC
    filas = []
    for soc, g in cw.groupby('2010 SOC Code'):
        fila = {'soc': soc}
        for c in COLS:
            m = g[c].notna()
            fila[c] = np.average(g.loc[m, c], weights=g.loc[m, 'peso']) if m.any() else np.nan
        fila['n_tareas'] = g['Task ID'].nunique()
        filas.append(fila)
    soc_df = pd.DataFrame(filas)

    # SOC -> CIUO
    puente = cw[['2010 SOC Code', 'ISCO-08 Code']].dropna().drop_duplicates()
    puente.columns = ['soc', 'isco']
    soc_df = soc_df.merge(puente, on='soc', how='left')
    soc_df['ciuo'] = soc_df['isco'].map(norm_ciuo)

    out = (soc_df.dropna(subset=['ciuo'])
           .groupby('ciuo')
           .agg(**{c: (c, 'mean') for c in COLS},
                n_soc=('soc', 'nunique'))
           .reset_index())

    print(f'  {nombre}: {len(soc_df):,} SOC -> {len(out):,} códigos CIUO')
    # Se devuelve también el nivel SOC: hace falta para la imputación manual
    return out, soc_df


scores_bls, soc_bls = construir_scores(cw_bls, 'BLS   ')
scores_own, soc_own = construir_scores(cw_own, 'Manual')

# ------------------------------------------------------------
# Universo SOC completo, para la imputación manual.
#
# cw-bls.csv solo trae las ocupaciones SOC que tienen correspondencia
# ISCO-08. Las reglas de IMPUTACION_MANUAL apuntan justamente a las que
# NO la tienen, así que hay que leer el labelset O*NET completo.
# ------------------------------------------------------------

onet = pd.read_csv(RUTA / 'data' / 'full_labelset.tsv', sep='\t', index_col=0)


def norm_soc(x):
    """'11-1011.00' -> '11-1011'"""
    if pd.isna(x):
        return None
    m = re.search(r'(\d{2})[-.]?(\d{4})', str(x))
    return f'{m.group(1)}-{m.group(2)}' if m else None


onet['soc'] = onet['O*NET-SOC Code'].map(norm_soc)
onet[['alpha_gpt', 'beta_gpt', 'gamma_gpt']] = scores_de_etiqueta(onet['gpt4_exposure'])
onet[['alpha_h', 'beta_h', 'gamma_h']] = scores_de_etiqueta(onet['human_exposure_agg'])
onet['peso'] = np.where(onet['Task Type'].astype(str).str.strip() == 'Core',
                        PESO_CORE, PESO_SUPP)

filas_onet = []
for soc, g in onet.dropna(subset=['soc']).groupby('soc'):
    fila = {'soc': soc}
    for c in COLS:
        m = g[c].notna()
        fila[c] = np.average(g.loc[m, c], weights=g.loc[m, 'peso']) if m.any() else np.nan
    filas_onet.append(fila)
soc_onet = pd.DataFrame(filas_onet)

print(f'  Universo O*NET completo: {len(soc_onet):,} ocupaciones SOC')

# La especificación principal es BLS (estandarizada y trazable)
scores = scores_bls


# ============================================================
# PASO 3: PREPARAR p41 Y FILTRAR POBLACIÓN OCUPADA
# OJO: p40 = rama de actividad (CIIU). p41 = grupo de ocupación (CIUO).
# ============================================================

print('\n' + '=' * 70)
print('PASO 3: PREPARAR ENEMDU')
print('=' * 70)

enemdu['ciuo'] = enemdu['p41'].map(norm_ciuo)

ocupados = enemdu[enemdu['condact'].isin(CONDACT_OCUPADOS)].copy()
sin_codigo = ocupados['ciuo'].isna()

print(f'  Población total:          {enemdu["fexp"].sum():>15,.0f}')
print(f'  Ocupados (condact 1-6):   {ocupados["fexp"].sum():>15,.0f}  ({len(ocupados):,} registros)')
print(f'  Ocupados sin código p41:  {ocupados.loc[sin_codigo, "fexp"].sum():>15,.0f}  ({sin_codigo.mean()*100:.2f}%)')

base = ocupados[~sin_codigo].copy()
pop_base = base['fexp'].sum()


# ============================================================
# PASO 4: IMPUTACIÓN POR PROXIMIDAD
# Los CIUO sin match directo heredan el promedio de su subgrupo de
# 3 dígitos; si no existe, del grupo de 2; luego del gran grupo de 1.
# Lo que aún quede se resuelve por similitud de descripciones.
# ============================================================

print('\n' + '=' * 70)
print('PASO 4: IMPUTACIÓN POR PROXIMIDAD')
print('=' * 70)

# Tablas de referencia por nivel jerárquico
tablas = {4: scores.set_index('ciuo')[COLS]}
for n in (3, 2, 1):
    tmp = scores.copy()
    tmp['padre'] = tmp['ciuo'].str[:n]
    tablas[n] = tmp.groupby('padre')[COLS].mean()

# Tabla de imputación manual: promedio de las SOC bajo cada prefijo asignado
manual = {}
sin_regla = []
for ciuo_cod, prefijo_soc in IMPUTACION_MANUAL.items():
    sel = soc_onet[soc_onet['soc'].astype(str).str.startswith(prefijo_soc, na=False)]
    if len(sel):
        promedio = sel[COLS].mean()
        if promedio.notna().any():
            manual[ciuo_cod] = promedio
            continue
    sin_regla.append(f'{ciuo_cod} ({prefijo_soc})')

print(f'  Reglas de imputación manual disponibles: {len(manual)} de {len(IMPUTACION_MANUAL)}')
if sin_regla:
    print(f'  Sin SOC coincidente: {", ".join(sin_regla)}')

codigos = sorted(base['ciuo'].unique())
asignado = []
n_manual = 0

for cod in codigos:
    fila = {'ciuo': cod}
    niveles = []
    uso_manual = False

    # Orden de preferencia, columna por columna:
    #   1. match directo a 4 dígitos con valor válido
    #   2. imputación manual (criterio experto)
    #   3. jerarquía CIUO 3 -> 2 -> 1 dígitos
    # No basta con que el código exista en la tabla: el valor tiene que ser
    # válido. Hay ocupaciones en el crosswalk cuyas tareas no traen etiqueta
    # de exposición y cuyo promedio es NaN.
    for c in COLS:
        valor, nivel = np.nan, np.nan

        if cod in tablas[4].index and pd.notna(tablas[4].at[cod, c]):
            valor, nivel = tablas[4].at[cod, c], 4
        elif cod in manual and pd.notna(manual[cod][c]):
            valor, nivel = manual[cod][c], 5
            uso_manual = True
        else:
            for n in (3, 2, 1):
                clave = cod[:n]
                if clave in tablas[n].index:
                    v = tablas[n].at[clave, c]
                    if pd.notna(v):
                        valor, nivel = v, n
                        break

        fila[c] = valor
        if pd.notna(nivel):
            niveles.append(nivel)

    # Se reporta el nivel más grueso que hizo falta. La imputación manual (5)
    # no se considera "más gruesa" que el match directo, así que se marca aparte.
    jerarquicos = [n for n in niveles if n != 5]
    if uso_manual:
        fila['nivel_imputacion'] = 5
        n_manual += 1
    else:
        fila['nivel_imputacion'] = min(jerarquicos) if jerarquicos else np.nan
    asignado.append(fila)

asignado = pd.DataFrame(asignado)

if n_manual:
    print(f'  {n_manual} códigos resueltos por imputación manual')

# --- fallback semántico para lo que quedó sin cubrir
pendientes = asignado.loc[asignado['nivel_imputacion'].isna(), 'ciuo'].tolist()
print(f'  Sin match jerárquico: {len(pendientes)} códigos')

if pendientes:
    desc = dict(zip(catalogo['ciuo'], catalogo['DESCRIPCION']))
    donantes = [c for c in scores['ciuo'] if c in desc]
    consultas = [c for c in pendientes if c in desc]

    if donantes and consultas:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
            M = vec.fit_transform([limpiar_texto(desc[c]) for c in donantes] +
                                  [limpiar_texto(desc[c]) for c in consultas])
            sim = cosine_similarity(M[len(donantes):], M[:len(donantes)])
        except ImportError:
            from difflib import SequenceMatcher
            print('    (sklearn no instalado, usando difflib)')
            sim = np.array([[SequenceMatcher(None, limpiar_texto(desc[c]),
                                             limpiar_texto(desc[d])).ratio()
                             for d in donantes] for c in consultas])

        ref = scores.set_index('ciuo')
        asignado = asignado.set_index('ciuo')
        n_ok = 0
        for i, cod in enumerate(consultas):
            j = sim[i].argmax()
            if sim[i, j] >= UMBRAL_SEMANTICO:
                asignado.loc[cod, COLS] = ref.loc[donantes[j], COLS].values
                asignado.loc[cod, 'nivel_imputacion'] = 0   # 0 = semántico
                n_ok += 1
        asignado = asignado.reset_index()
        print(f'  Resueltos por similitud semántica: {n_ok}')

# --- reporte por nivel
pob_cod = base.groupby('ciuo')['fexp'].sum()
etiquetas_nivel = {4: 'directo (4 dígitos)', 5: 'imputación manual',
                   3: 'subgrupo (3 dígitos)', 2: 'grupo (2 dígitos)',
                   1: 'gran grupo (1 dígito)', 0: 'similitud semántica'}

print()
for nivel in [4, 5, 3, 2, 1, 0]:
    cods = asignado.loc[asignado['nivel_imputacion'] == nivel, 'ciuo']
    if len(cods) == 0:
        continue
    pob = pob_cod.reindex(cods).sum()
    print(f'  {etiquetas_nivel[nivel]:24} {len(cods):>4} códigos | {pob:>12,.0f} personas ({pob/pop_base*100:5.2f}%)')

sin_cob = asignado.loc[asignado['nivel_imputacion'].isna(), 'ciuo']
pob_sin = pob_cod.reindex(sin_cob).sum() if len(sin_cob) else 0.0
print(f'  {"sin cobertura":24} {len(sin_cob):>4} códigos | {pob_sin:>12,.0f} personas ({pob_sin/pop_base*100:5.2f}%)')


# ============================================================
# PASO 5: MERGE FINAL Y COBERTURA
# ============================================================

print('\n' + '=' * 70)
print('PASO 5: MERGE Y COBERTURA')
print('=' * 70)

df = base.merge(asignado, on='ciuo', how='left', validate='many_to_one')
assert len(df) == len(base), 'ERROR: el merge duplicó filas'

pop_con_score = df.loc[df['beta_gpt'].notna(), 'fexp'].sum()
pop_directo = df.loc[df['nivel_imputacion'] == 4, 'fexp'].sum()

print(f'  Cobertura sobre ocupados con p41:  {pop_con_score/pop_base*100:.2f}%')
print(f'  Cobertura sobre todos los ocupados: {pop_con_score/ocupados["fexp"].sum()*100:.2f}%')
print(f'  De ella, sin imputar:               {pop_directo/pop_base*100:.2f}%')


# ============================================================
# PASO 6: FACTOR DE AJUSTE F
# F = 0.5*media(ratios educación) + 0.5*media(ratios acceso)
# Se recalcula desde el CSV, no se escribe a mano.
# ============================================================

print('\n' + '=' * 70)
print('PASO 6: FACTOR DE AJUSTE')
print('=' * 70)

fact = leer_csv(RUTA / 'code' / 'adjustment_factors_table8_v3.csv')
ind = fact[fact['group'].isin(['Education', 'Access_Skills'])].copy()


def calcular_F(pais, cambios=None):
    cambios = cambios or {}
    edu, acc = [], []
    for _, f in ind.iterrows():
        valor = float(cambios.get(f['indicator'], f[pais]))
        ratio = valor / float(f['US'])
        (edu if f['group'] == 'Education' else acc).append(ratio)
    return 0.5 * np.mean(edu) + 0.5 * np.mean(acc), np.mean(edu), np.mean(acc)


NOMBRES_IND = {
    'literacy': 'Alfabetización (\\% 15+)',
    'schooling': 'Años de escolaridad (25+)',
    'edu_spending': 'Gasto público en educación (\\% PIB)',
    'pisa': 'PISA Matemáticas',
    'internet': 'Usuarios de internet (\\%)',
    'broadband': 'Banda ancha fija (por 100 hab.)',
    'firm_tech': 'Absorción tecnológica de firmas (1--7)',
    'digital_skills': 'Habilidades digitales (1--7)',
}

PAISES = [p for p in ['Chile', 'Mexico', 'Peru', 'Ecuador'] if p in fact.columns]

factores, subindices = {}, {}
for pais in PAISES:
    F_p, e, a = calcular_F(pais)
    factores[pais], subindices[pais] = F_p, (e, a)
    print(f'  {pais:8} F = {F_p:.4f}   (educación {e:.4f}, acceso {a:.4f})')

# Matriz de ratios país/US por indicador
ratios_df = pd.DataFrame({
    'grupo': ind['group'].values,
    'indicador': ind['indicator'].values,
    'US': ind['US'].astype(float).values,
})
for pais in PAISES:
    ratios_df[pais] = ind[pais].astype(float).values / ind['US'].astype(float).values
    ratios_df[pais + '_valor'] = ind[pais].astype(float).values

# Escenarios de sensibilidad para Ecuador
escenarios = {
    'Base (PISA-D 377, ENEMDU 10.4)': {},
    'PISA CAF composite (432.9)': {'pisa': 432.9},
    'Escolaridad UNDP (8.97)': {'schooling': 8.97},
    'PISA CAF + UNDP': {'pisa': 432.9, 'schooling': 8.97},
    'firm\\_tech GCR (4.2)': {'firm_tech': 4.2},
}

print('\n  Sensibilidad (Ecuador):')
filas_sens = []
for nombre, cambios in escenarios.items():
    F_e, e_e, a_e = calcular_F('Ecuador', cambios)
    filas_sens.append({'escenario': nombre, 'F': F_e, 'edu': e_e, 'acc': a_e})
    print(f'    {nombre:34} F = {F_e:.4f}')
sens = pd.DataFrame(filas_sens)
print(f'    Rango: [{sens["F"].min():.4f}, {sens["F"].max():.4f}]')

# Se usa el escenario base
F = calcular_F('Ecuador')[0]
print(f'\n  Factor aplicado: F = {F:.4f}')

for c in COLS:
    df[c + '_adj'] = df[c] * F
df['factor_ajuste'] = F


# ============================================================
# PASO 7: RESULTADOS
# ============================================================

print('\n' + '=' * 70)
print('PASO 7: RESULTADOS AGREGADOS')
print('=' * 70)

for c in ['alpha_h', 'beta_h', 'gamma_h', 'alpha_gpt', 'beta_gpt', 'gamma_gpt']:
    tec = prom_pond(df, c) * 100
    aju = prom_pond(df, c + '_adj') * 100
    print(f'  {c:10}  técnico {tec:6.2f}%   ajustado {aju:6.2f}%')

# Robustez: solo match directo, sin imputados
print('\n  Solo match directo (sin imputación):')
solo_dir = df[df['nivel_imputacion'] == 4]
for c in ['beta_h', 'beta_gpt']:
    print(f'    {c:10}  {prom_pond(solo_dir, c)*100:6.2f}%')


ETIQUETAS = {
    'secemp':  {1: 'Formal', 2: 'Informal', 3: 'Doméstico', 4: 'No clasificado'},
    'nnivins': {1: 'Ninguno', 2: 'Alfabetización', 3: 'Básica',
                4: 'Media/Bachillerato', 5: 'Superior'},
    'area':    {1: 'Urbana', 2: 'Rural'},
    'p02':     {1: 'Hombre', 2: 'Mujer'},
    'grupo1':  {1: 'Directivos', 2: 'Profesionales', 3: 'Técnicos nivel medio',
                4: 'Empleados de oficina', 5: 'Servicios y comercio',
                6: 'Agropecuarios y pesqueros', 7: 'Oficiales y artesanos',
                8: 'Operadores de maquinaria', 9: 'No calificados',
                10: 'Fuerzas Armadas'},
}


def corte(var):
    """Exposición media por categoría de una variable."""
    filas = []
    for cod in sorted(df[var].dropna().unique()):
        sub = df[df[var] == cod]
        if sub['fexp'].sum() <= 0:
            continue
        ing = sub[(sub['ingrl'] > 0) & (sub['ingrl'] < 999999)]
        filas.append({
            'categoria': ETIQUETAS.get(var, {}).get(int(cod), f'{var}={int(cod)}'),
            'pct': sub['fexp'].sum() / df['fexp'].sum() * 100,
            'beta_h': prom_pond(sub, 'beta_h') * 100,
            'beta_gpt': prom_pond(sub, 'beta_gpt') * 100,
            'beta_h_adj': prom_pond(sub, 'beta_h_adj') * 100,
            'beta_gpt_adj': prom_pond(sub, 'beta_gpt_adj') * 100,
            'ingreso': prom_pond(ing, 'ingrl'),
        })
    return pd.DataFrame(filas)


cortes = {}
for var in ['secemp', 'nnivins', 'grupo1', 'area', 'p02']:
    if var not in df.columns:
        print(f'  aviso: la variable {var} no está en la ENEMDU')
        continue
    t = corte(var)
    cortes[var] = t
    print(f'\n  --- Por {var} ---')
    for _, r in t.iterrows():
        print(f'    {r["categoria"]:26} {r["pct"]:5.1f}% empleo | '
              f'beta_gpt {r["beta_gpt"]:5.2f}% | ajustado {r["beta_gpt_adj"]:5.2f}%')

# Brecha formal vs informal
if 'secemp' in df.columns:
    print('\n  --- Brecha formal vs informal ---')
    fo, inf = df[df['secemp'] == 1], df[df['secemp'] == 2]
    for c in ['beta_h', 'beta_gpt', 'beta_h_adj', 'beta_gpt_adj']:
        a, b = prom_pond(fo, c) * 100, prom_pond(inf, c) * 100
        print(f'    {c:14} formal {a:6.2f}%  informal {b:6.2f}%  '
              f'brecha {a-b:+6.2f} pp  (ratio {a/b:.2f})')

# Exposición por ocupación
por_ocup = (df.groupby('ciuo')
            .apply(lambda g: pd.Series({
                'beta_gpt': prom_pond(g, 'beta_gpt') * 100,
                'beta_h': prom_pond(g, 'beta_h') * 100,
                'poblacion': g['fexp'].sum()}), include_groups=False)
            .reset_index()
            .merge(catalogo, on='ciuo', how='left')
            .sort_values('beta_gpt', ascending=False))

# Separar las ocupaciones con score de las que quedaron sin cobertura.
# Sin esto, los NaN se van al final del sort y contaminan el "menos expuestas".
con_score = por_ocup[por_ocup['beta_gpt'].notna()]
sin_score = por_ocup[por_ocup['beta_gpt'].isna()].sort_values('poblacion', ascending=False)

print(f'\n  Ocupaciones con score: {len(con_score)} | sin score: {len(sin_score)}')

print('\n  --- Top 15 ocupaciones más expuestas ---')
for _, r in con_score.head(15).iterrows():
    print(f'    {r["ciuo"]} | {str(r["DESCRIPCION"])[:50]:50} | '
          f'{r["beta_gpt"]:5.1f}% | {r["poblacion"]:>10,.0f}')

print('\n  --- Top 15 menos expuestas ---')
for _, r in con_score.tail(15).iterrows():
    print(f'    {r["ciuo"]} | {str(r["DESCRIPCION"])[:50]:50} | '
          f'{r["beta_gpt"]:5.1f}% | {r["poblacion"]:>10,.0f}')

if len(sin_score):
    pob_sin_score = sin_score['poblacion'].sum()
    print(f'\n  --- SIN COBERTURA: {len(sin_score)} ocupaciones, '
          f'{pob_sin_score:,.0f} personas ({pob_sin_score/df["fexp"].sum()*100:.2f}%) ---')
    print('      (no tienen score; NO son ocupaciones poco expuestas)')
    for _, r in sin_score.head(20).iterrows():
        print(f'    {r["ciuo"]} | {str(r["DESCRIPCION"])[:50]:50} | '
              f'{r["poblacion"]:>10,.0f}')
    sin_score.to_csv(RUTA / 'data' / 'ocupaciones_sin_cobertura.csv',
                     index=False, encoding='utf-8')


# ============================================================
# PASO 8: TABLAS LaTeX PARA EL PAPER
# ============================================================

print('\n' + '=' * 70)
print('PASO 8: TABLAS LaTeX')
print('=' * 70)


def escribir_tabla(tab, caption, label, col, archivo):
    L = [r'\begin{table}[htbp]', r'\centering',
         rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\small',
         r'\begin{tabular}{lrrrrr}', r'\toprule',
         rf'{col} & \% empleo & $\beta_h$ & $\beta_{{gpt}}$ & '
         r'$\beta_h^{adj}$ & $\beta_{gpt}^{adj}$ \\', r'\midrule']
    for _, r in tab.iterrows():
        L.append(f'{r["categoria"]} & {r["pct"]:.1f} & {r["beta_h"]:.2f} & '
                 f'{r["beta_gpt"]:.2f} & {r["beta_h_adj"]:.2f} & '
                 f'{r["beta_gpt_adj"]:.2f} \\\\')
    L += [r'\bottomrule', r'\end{tabular}',
          r'\begin{flushleft}\footnotesize',
          r'\textit{Nota:} exposición en \%, media ponderada por factor de '
          r'expansión. $\beta = \alpha + 0{,}5 \times E2$. Las columnas '
          r'ajustadas aplican $F^{EC}$.',
          r'\\ \textit{Fuente:} ENEMDU 2025 (INEC) y Eloundou et al. (2023). '
          r'Elaboración propia.',
          r'\end{flushleft}', r'\end{table}']
    (RUTA / 'paper' / 'tablas' / archivo).write_text('\n'.join(L), encoding='utf-8')
    print(f'  {archivo}')


titulos = {
    'secemp': ('Exposición a LLMs por sector de empleo, Ecuador 2025',
               'tab:exp_formalidad', 'Sector', 'tabla_exp_formalidad.tex'),
    'nnivins': ('Exposición a LLMs por nivel de instrucción, Ecuador 2025',
                'tab:exp_educacion', 'Nivel de instrucción', 'tabla_exp_educacion.tex'),
    'grupo1': ('Exposición a LLMs por grupo ocupacional, Ecuador 2025',
               'tab:exp_grupo', 'Grupo ocupacional', 'tabla_exp_grupo.tex'),
    'area': ('Exposición a LLMs por área, Ecuador 2025',
             'tab:exp_area', 'Área', 'tabla_exp_area.tex'),
    'p02': ('Exposición a LLMs por sexo, Ecuador 2025',
            'tab:exp_sexo', 'Sexo', 'tabla_exp_sexo.tex'),
}
for var, (cap, lab, col, arch) in titulos.items():
    if var in cortes:
        escribir_tabla(cortes[var], cap, lab, col, arch)


# ---- 8.2 Tabla del factor de ajuste (indicadores y ratios por país) ----

L = [r'\begin{table}[htbp]', r'\centering',
     r'\caption{Factor de ajuste: indicadores y ratios respecto a Estados Unidos}',
     r'\label{tab:adjustment_factors}', r'\small',
     r'\begin{tabular}{ll' + 'c' * (len(PAISES) + 1) + '}', r'\toprule',
     'Grupo & Indicador & US & ' + ' & '.join(PAISES) + r' \\', r'\midrule']

for grupo, etiqueta in [('Education', 'Educación'), ('Access_Skills', 'Acceso y habilidades')]:
    sub = ratios_df[ratios_df['grupo'] == grupo]
    for i, (_, r) in enumerate(sub.iterrows()):
        celda_grupo = rf'\textbf{{{etiqueta}}}' if i == 0 else ''
        vals = ' & '.join(f'{r[p]:.3f}' for p in PAISES)
        L.append(f'{celda_grupo} & {NOMBRES_IND.get(r["indicador"], r["indicador"])} & '
                 f'{r["US"]:.1f} & {vals} \\\\')
    L.append(r'\midrule')

L.append(' & \\textit{Sub-índice educación} & & ' +
         ' & '.join(f'{subindices[p][0]:.3f}' for p in PAISES) + r' \\')
L.append(' & \\textit{Sub-índice acceso} & & ' +
         ' & '.join(f'{subindices[p][1]:.3f}' for p in PAISES) + r' \\')
L.append(r'\midrule')
L.append(r' & \textbf{Factor de ajuste $F^c$} & & ' +
         ' & '.join(rf'\textbf{{{factores[p]:.3f}}}' for p in PAISES) + r' \\')
L += [r'\bottomrule', r'\end{tabular}',
      r'\begin{flushleft}\footnotesize',
      r'\textit{Nota:} cada celda es el ratio $E_i^c / E_i^{US}$ (o $A_i^c / A_i^{US}$). '
      r'$F^c = 0{,}5 \times \overline{E^c} + 0{,}5 \times \overline{A^c}$, siguiendo a '
      r'Azuara, Ripani \& Torres (2024). Chile, México y Perú replican su Tabla 8; '
      r'Ecuador es original.',
      r'\\ \textit{Fuentes:} WDI (Banco Mundial); OCDE PISA 2018; PISA-D 2017; '
      r'ENEMDU 2023 (INEC); GITR 2016 (WEF); CAF IDED 2023.',
      r'\end{flushleft}', r'\end{table}']
(RUTA / 'paper' / 'tablas' / 'tabla_factor_ajuste.tex').write_text(
    '\n'.join(L), encoding='utf-8')
print('  tabla_factor_ajuste.tex')


# ---- 8.3 Tabla de sensibilidad ----

L = [r'\begin{table}[htbp]', r'\centering',
     r'\caption{Análisis de sensibilidad del factor de ajuste de Ecuador}',
     r'\label{tab:sensitivity}', r'\small',
     r'\begin{tabular}{lcccc}', r'\toprule',
     r' & & & \multicolumn{2}{c}{$\beta^*$ ajustado} \\',
     r'\cmidrule(lr){4-5}',
     r'Escenario & $F^{EC}$ & $\overline{E}$ & Humano & GPT-4 \\', r'\midrule']

b_h_tec = prom_pond(df, 'beta_h')
b_g_tec = prom_pond(df, 'beta_gpt')
for _, r in sens.iterrows():
    L.append(f'{r["escenario"]} & {r["F"]:.4f} & {r["edu"]:.4f} & '
             f'{b_h_tec * r["F"]:.4f} & {b_g_tec * r["F"]:.4f} \\\\')
L += [r'\midrule',
      f'Rango & [{sens["F"].min():.4f}, {sens["F"].max():.4f}] & & '
      f'[{b_h_tec * sens["F"].min():.4f}, {b_h_tec * sens["F"].max():.4f}] & '
      f'[{b_g_tec * sens["F"].min():.4f}, {b_g_tec * sens["F"].max():.4f}] \\\\',
      r'\bottomrule', r'\end{tabular}',
      r'\begin{flushleft}\footnotesize',
      rf'\textit{{Nota:}} $\beta^*_h = {b_h_tec:.4f} \times F^{{EC}}$, '
      rf'$\beta^*_g = {b_g_tec:.4f} \times F^{{EC}}$. El escenario base es la '
      r'especificación principal. Los escenarios alternativos varían la fuente de los '
      r'indicadores ecuatorianos y son aporte propio, no parte de Azuara et al. (2024).',
      r'\end{flushleft}', r'\end{table}']
(RUTA / 'paper' / 'tablas' / 'tabla_sensibilidad.tex').write_text(
    '\n'.join(L), encoding='utf-8')
print('  tabla_sensibilidad.tex')


# ============================================================
# PASO 9: FIGURAS
# ============================================================

print('\n' + '=' * 70)
print('PASO 9: FIGURAS')
print('=' * 70)

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 11, 'legend.frameon': False})
AZUL, NARANJA = '#2c6fbb', '#e07b39'


def guardar(fig, nombre):
    for ext in ('pdf', 'png'):
        fig.savefig(RUTA / 'figures' / f'{nombre}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  {nombre}.pdf / .png')


# 9.1 Barras por categoría (formalidad, educación, grupo ocupacional)
for var, titulo, nombre in [
        ('secemp', 'Exposición a LLMs por sector de empleo', 'fig_exp_formalidad'),
        ('nnivins', 'Exposición a LLMs por nivel de instrucción', 'fig_exp_educacion'),
        ('grupo1', 'Exposición a LLMs por grupo ocupacional', 'fig_exp_grupo')]:
    if var not in cortes:
        continue
    t = cortes[var].sort_values('beta_gpt')
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.45 * len(t) + 1.5)))
    ax.barh(y - 0.2, t['beta_gpt'], 0.4, color=AZUL, label='Técnica')
    ax.barh(y + 0.2, t['beta_gpt_adj'], 0.4, color=NARANJA, label='Ajustada')
    ax.set_yticks(y)
    ax.set_yticklabels(t['categoria'])
    ax.set_xlabel(r'Exposición media $\beta_{gpt}$ (%)')
    ax.set_title(f'{titulo}, Ecuador 2025')
    ax.legend(loc='lower right')
    guardar(fig, nombre)

# 9.2 Distribución de beta
sub = df[df['beta_gpt'].notna()]
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 1, 41)
ax.hist(sub['beta_gpt'], bins=bins, weights=sub['fexp'], color=AZUL, alpha=0.65,
        density=True, label='Técnica')
ax.hist(sub['beta_gpt_adj'], bins=bins, weights=sub['fexp'], color=NARANJA, alpha=0.65,
        density=True, label='Ajustada')
ax.set_xlabel(r'$\beta$ (fracción de tareas expuestas)')
ax.set_ylabel('Densidad (ponderada por población)')
ax.set_title('Distribución de la exposición ocupacional, Ecuador 2025')
ax.legend()
guardar(fig, 'fig_distribucion_beta')

# 9.3 Técnico vs ajustado
medidas = ['alpha_h', 'beta_h', 'alpha_gpt', 'beta_gpt']
tec = [prom_pond(df, m) * 100 for m in medidas]
aju = [prom_pond(df, m + '_adj') * 100 for m in medidas]
x = np.arange(len(medidas))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(x - 0.2, tec, 0.4, color=AZUL, label='Técnica')
ax.bar(x + 0.2, aju, 0.4, color=NARANJA, label='Ajustada')
for i, (a, b) in enumerate(zip(tec, aju)):
    ax.text(i - 0.2, a + 0.3, f'{a:.1f}', ha='center', fontsize=9)
    ax.text(i + 0.2, b + 0.3, f'{b:.1f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([r'$\alpha_h$', r'$\beta_h$', r'$\alpha_{gpt}$', r'$\beta_{gpt}$'])
ax.set_ylabel('Exposición media (%)')
ax.set_title(f'Efecto del factor de ajuste (F = {F:.4f})')
ax.legend()
guardar(fig, 'fig_tecnico_vs_ajustado')

VERDE, GRIS, MORADO = '#4c9a52', '#8a8a8a', '#7b5aa6'

# ---- 9.4 Estructura del mercado laboral ----
pet = enemdu[enemdu['p03'] >= 15]['fexp'].sum()
pea = enemdu[enemdu['condact'].isin([1, 2, 3, 4, 5, 6, 7, 8])]['fexp'].sum()
pei = enemdu[enemdu['condact'] == 9]['fexp'].sum()
ocup = ocupados['fexp'].sum()
adec = df[df['condact'] == 1]['fexp'].sum()

fig, ax = plt.subplots(figsize=(8, 4.5))
nombres = ['PET\n(15+ años)', 'PEA', 'PEI', 'Ocupados\n(condact 1-6)', 'Empleo\nadecuado']
vals = [pet, pea, pei, ocup, adec]
barras = ax.bar(nombres, [v / 1e6 for v in vals],
                color=[AZUL, VERDE, GRIS, NARANJA, MORADO], width=0.6)
for b, v in zip(barras, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
            f'{v/1e6:.2f}M', ha='center', fontsize=9)
ax.set_ylabel('Millones de personas')
ax.set_title('Estructura del mercado laboral ecuatoriano, 2025')
guardar(fig, 'fig_estructura_mercado')

# ---- 9.5 Condición de actividad ----
etiq_cond = {1: 'Empleo\nadecuado', 2: 'Subempleo\ntiempo', 3: 'Subempleo\ningresos',
             4: 'Otro empleo\nno pleno', 5: 'No\nremunerado', 6: 'No\nclasificado'}
vals_cond = [df[df['condact'] == c]['fexp'].sum() for c in etiq_cond]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
barras = ax.bar(list(etiq_cond.values()), [v / 1e6 for v in vals_cond],
                color=AZUL, width=0.6)
total_cond = sum(vals_cond)
for b, v in zip(barras, vals_cond):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
            f'{v/total_cond*100:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('Millones de personas')
ax.set_title('Población ocupada por condición de actividad, Ecuador 2025')
guardar(fig, 'fig_condicion_actividad')

# ---- 9.6 Curva de Lorenz del ingreso laboral ----
ing = df[(df['ingrl'] > 0) & (df['ingrl'] < 999999)].sort_values('ingrl')
cum_p = np.concatenate([[0], (ing['fexp'].cumsum() / ing['fexp'].sum()).values])
cum_y = np.concatenate([[0], ((ing['ingrl'] * ing['fexp']).cumsum() /
                              (ing['ingrl'] * ing['fexp']).sum()).values])
gini = 1 - 2 * np.trapezoid(cum_y, cum_p) if hasattr(np, 'trapezoid') \
    else 1 - 2 * np.trapz(cum_y, cum_p)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], ls='--', color=GRIS, lw=1, label='Igualdad perfecta')
ax.plot(cum_p, cum_y, color=AZUL, lw=2, label=f'Lorenz (Gini {gini:.3f})')
ax.fill_between(cum_p, cum_y, cum_p, color=AZUL, alpha=0.15)
ax.set_xlabel('Proporción acumulada de ocupados')
ax.set_ylabel('Proporción acumulada del ingreso')
ax.set_title('Distribución del ingreso laboral, Ecuador 2025')
ax.set_aspect('equal')
ax.legend(loc='upper left')
guardar(fig, 'fig_lorenz')
print(f'    Gini del ingreso laboral: {gini:.4f}')

# ---- 9.7 Top y bottom ocupaciones por exposición ----
top = pd.concat([con_score.head(12), con_score.tail(12)])
fig, ax = plt.subplots(figsize=(9, 9))
colores = [NARANJA] * 12 + [AZUL] * 12
y = np.arange(len(top))
ax.barh(y, top['beta_gpt'], color=colores)
ax.set_yticks(y)
ax.set_yticklabels([str(d)[:42] for d in top['DESCRIPCION']], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(r'$\beta_{gpt}$ (%)')
ax.set_title('Ocupaciones más y menos expuestas a LLMs, Ecuador 2025')
ax.axhline(11.5, color=GRIS, ls='--', lw=1)
guardar(fig, 'fig_top_ocupaciones')

# ---- 9.8 Exposición contra ingreso, por ocupación ----
ing_ocup = (df[(df['ingrl'] > 0) & (df['ingrl'] < 999999)]
            .groupby('ciuo')
            .apply(lambda g: prom_pond(g, 'ingrl'), include_groups=False)
            .rename('ingreso').reset_index())
m = con_score.merge(ing_ocup, on='ciuo').dropna(subset=['ingreso', 'beta_gpt'])
if len(m) > 5:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(m['ingreso'], m['beta_gpt'], s=np.sqrt(m['poblacion']) / 6,
               alpha=0.5, color=AZUL, edgecolor='none')
    pend, inter = np.polyfit(np.log(m['ingreso']), m['beta_gpt'], 1)
    xs = np.linspace(m['ingreso'].min(), m['ingreso'].max(), 100)
    ax.plot(xs, inter + pend * np.log(xs), color=NARANJA, lw=2)
    ax.set_xscale('log')
    ax.set_xlabel('Ingreso laboral mensual medio (USD, escala log)')
    ax.set_ylabel(r'$\beta_{gpt}$ (%)')
    corr = m['beta_gpt'].corr(np.log(m['ingreso']))
    ax.set_title(f'Exposición e ingreso por ocupación (corr. {corr:.3f})')
    guardar(fig, 'fig_exposicion_ingreso')
    print(f'    Correlación beta vs log(ingreso): {corr:.3f}')

# ---- 9.9 Factor de ajuste por país ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
orden = sorted(PAISES, key=lambda p: factores[p])
barras = ax.bar(orden, [factores[p] for p in orden],
                color=[NARANJA if p == 'Ecuador' else AZUL for p in orden], width=0.55)
for b, p in zip(barras, orden):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f'{factores[p]:.3f}', ha='center', fontsize=10)
ax.axhline(1.0, color=GRIS, ls='--', lw=1)
ax.text(len(orden) - 0.4, 1.01, 'Estados Unidos = 1', fontsize=9, color=GRIS, ha='right')
ax.set_ylim(0, 1.12)
ax.set_ylabel('Factor de ajuste $F^c$')
ax.set_title('Factor de ajuste por país')
guardar(fig, 'fig_factor_por_pais')

# ---- 9.10 Descomposición: educación contra acceso ----
x = np.arange(len(PAISES))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - 0.2, [subindices[p][0] for p in PAISES], 0.4, color=AZUL, label='Educación')
ax.bar(x + 0.2, [subindices[p][1] for p in PAISES], 0.4, color=NARANJA, label='Acceso digital')
for i, p in enumerate(PAISES):
    ax.text(i - 0.2, subindices[p][0] + 0.01, f'{subindices[p][0]:.3f}',
            ha='center', fontsize=8)
    ax.text(i + 0.2, subindices[p][1] + 0.01, f'{subindices[p][1]:.3f}',
            ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(PAISES)
ax.set_ylabel('Sub-índice (ratio respecto a EE.UU.)')
ax.set_title('Descomposición del factor de ajuste')
ax.legend()
guardar(fig, 'fig_descomposicion_factor')

# ---- 9.11 Heatmap de ratios ----
mat = ratios_df[PAISES].values
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mat, cmap='RdYlGn', vmin=0.2, vmax=1.05, aspect='auto')
ax.set_xticks(range(len(PAISES)))
ax.set_xticklabels(PAISES)
ax.set_yticks(range(len(ratios_df)))
ax.set_yticklabels([NOMBRES_IND.get(i, i).replace('\\%', '%').replace('--', '-')
                    for i in ratios_df['indicador']], fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('Ratios por indicador respecto a Estados Unidos')
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.7, label='Ratio país / EE.UU.')
guardar(fig, 'fig_heatmap_ratios')

# ---- 9.12 Sensibilidad del factor ----
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(sens))
etiquetas_sens = [e.replace('\\_', '_') for e in sens['escenario']]
ax.scatter(sens['F'], y, s=90, color=AZUL, zorder=3)
ax.axvline(sens['F'].iloc[0], color=NARANJA, ls='--', lw=1.5,
           label=f'Base ({sens["F"].iloc[0]:.4f})')
ax.axvspan(sens['F'].min(), sens['F'].max(), color=AZUL, alpha=0.08)
for i, v in enumerate(sens['F']):
    ax.text(v + 0.0015, i, f'{v:.4f}', va='center', fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(etiquetas_sens, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Factor de ajuste $F^{EC}$')
ax.set_title('Sensibilidad del factor a la fuente de los indicadores')
ax.legend(loc='lower right')
guardar(fig, 'fig_sensibilidad_factor')

# ---- 9.13 Contribución de cada indicador al déficit (1 - ratio) ----
ec = ratios_df[['indicador', 'grupo', 'Ecuador']].copy()
ec['deficit'] = 1 - ec['Ecuador']
ec = ec.sort_values('deficit')
fig, ax = plt.subplots(figsize=(8.5, 5))
colores = [AZUL if g == 'Education' else NARANJA for g in ec['grupo']]
y = np.arange(len(ec))
ax.barh(y, ec['deficit'], color=colores)
ax.set_yticks(y)
ax.set_yticklabels([NOMBRES_IND.get(i, i).replace('\\%', '%').replace('--', '-')
                    for i in ec['indicador']], fontsize=9)
for i, v in enumerate(ec['deficit']):
    ax.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('Déficit respecto a EE.UU. (1 - ratio)')
ax.set_title('Qué indicadores explican la brecha de Ecuador')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AZUL),
                   plt.Rectangle((0, 0), 1, 1, color=NARANJA)],
          labels=['Educación', 'Acceso digital'], loc='lower right')
guardar(fig, 'fig_contribucion_deficit')


# ============================================================
# PASO 10: GUARDAR
# Va ANTES de la comparación: los resultados principales se guardan
# siempre, aunque algo falle en los extras.
# ============================================================

print('\n' + '=' * 70)
print('PASO 10: GUARDAR RESULTADOS')
print('=' * 70)

por_ocup.to_csv(RUTA / 'data' / 'exposicion_por_ocupacion.csv', index=False, encoding='utf-8')
print('  data/exposicion_por_ocupacion.csv')

# Base completa: pesada, está en el .gitignore
salida = RUTA / 'data' / 'enemdu_scores_ajustados.csv'
df.to_csv(salida, index=False, encoding='utf-8')
print(f'  data/enemdu_scores_ajustados.csv  ({os.path.getsize(salida)/1024**2:.0f} MB)')


# ============================================================
# PASO 11: COMPARACIÓN CROSSWALK BLS vs MANUAL  (robustez, opcional)
#
# Compara los scores que salen del crosswalk oficial del BLS contra los
# del crosswalk manual. No afecta ningún resultado: todo lo anterior usa
# el BLS. Sirve solo para mostrar cuánto cambia el ORDEN de ocupaciones
# según la correspondencia que uses.
#
# Envuelto en try/except: si falla, avisa pero no tumba el script.
# ============================================================

print('\n' + '=' * 70)
print('PASO 11: COMPARACIÓN DE CROSSWALKS (robustez)')
print('=' * 70)

try:
    comp = scores_bls.merge(scores_own, on='ciuo', suffixes=('_bls', '_own'))
    print(f'  Ocupaciones en ambos crosswalks: {len(comp):,}')

    ok = comp['beta_gpt_bls'].notna() & comp['beta_gpt_own'].notna()
    c = comp[ok]
    print(f'  Con score válido en ambos: {len(c):,}')

    if len(c) < 10:
        raise ValueError(
            f'solo {len(c)} ocupaciones comparables; no alcanza para correlacionar. '
            'Revisa que los códigos CIUO de ambos crosswalks tengan el mismo formato.')

    pear = c['beta_gpt_bls'].corr(c['beta_gpt_own'])
    # Spearman = Pearson sobre los rangos. Se calcula así para no depender
    # de scipy, que no tiene versión para Python 3.14.
    spear = c['beta_gpt_bls'].rank().corr(c['beta_gpt_own'].rank())
    print(f'  Pearson:  {pear:.3f}')
    print(f'  Spearman: {spear:.3f}')
    print(f'  Media BLS {c["beta_gpt_bls"].mean():.3f} vs manual {c["beta_gpt_own"].mean():.3f}')
    print(f'  Diferencia mediana: {(c["beta_gpt_bls"] - c["beta_gpt_own"]).median():+.3f}')

    # Concordancia en el decil superior de exposición
    u_b = c['beta_gpt_bls'] >= c['beta_gpt_bls'].quantile(0.9)
    u_o = c['beta_gpt_own'] >= c['beta_gpt_own'].quantile(0.9)
    jac = (u_b & u_o).sum() / max(1, (u_b | u_o).sum())
    print(f'  En el decil superior bajo ambos: {(u_b & u_o).sum()} | Jaccard {jac:.3f}')

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(c['beta_gpt_bls'], c['beta_gpt_own'], s=18, alpha=0.45, color=AZUL,
               edgecolor='none')
    ax.plot([0, 1], [0, 1], color='#8a8a8a', ls='--', lw=1, label='45°')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\beta_{gpt}$ — crosswalk BLS')
    ax.set_ylabel(r'$\beta_{gpt}$ — crosswalk manual')
    ax.set_title('Exposición bajo dos correspondencias\n'
                 rf'$\rho = {pear:.3f}$, $n = {len(c):,}$')
    ax.legend(loc='upper left')
    guardar(fig, 'fig_comparacion_crosswalk')

    c[['ciuo', 'beta_gpt_bls', 'beta_gpt_own', 'beta_h_bls', 'beta_h_own']].to_csv(
        RUTA / 'data' / 'comparacion_crosswalk.csv', index=False, encoding='utf-8')
    print('  data/comparacion_crosswalk.csv')

except Exception as e:
    import traceback
    print(f'\n  !! La comparación falló: {type(e).__name__}: {e}')
    print('  !! Los resultados principales YA se guardaron en el PASO 10.')
    print('  !! Detalle del error:')
    traceback.print_exc()


print('\n' + '=' * 70)
print('PROCESO COMPLETADO')
print('=' * 70)