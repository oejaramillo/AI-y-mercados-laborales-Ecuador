import os
import time
import json
import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

# Absolute path to the .env next to this file (api.py)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Debug (safe): shows if it loaded, without printing the secret
key = os.getenv("DEEPSEEK_API_KEY")
print("Loaded .env from:", ENV_PATH)
print("DEEPSEEK_API_KEY present?:", key is not None and len(key) > 0)

import pandas as pd
import requests

# Opcional (solo para barra de progreso)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================
# 1) CONFIGURACIÓN
# =========================

# (A) Credenciales
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("Falta DEEPSEEK_API_KEY en variables de entorno.")

# (B) Endpoint y modelo
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_COMPLETIONS_PATH = os.getenv("DEEPSEEK_CHAT_PATH", "/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# (C) Parámetros de corrida
N_RUNS_PER_TASK = int(os.getenv("N_RUNS_PER_TASK", "3"))  # 3–5 recomendado
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))      # bajo para estabilidad
TOP_P = float(os.getenv("TOP_P", "1.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# (D) Control de tasa / reintentos
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.05"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
BACKOFF_BASE_SEC = float(os.getenv("BACKOFF_BASE_SEC", "1.5"))
BACKOFF_JITTER = float(os.getenv("BACKOFF_JITTER", "0.2"))

# (E) Archivos de salida
INPUT_FILE = os.getenv("INPUT_FILE", "/home/edu/Dropbox/Edu/repositorios/AI y mercados laborales Ecuador/data/full_labelset.tsv")
OUT_RUNS_JSONL = os.getenv("OUT_RUNS_JSONL", "deepseek_runs.jsonl")
OUT_AGG_CSV = os.getenv("OUT_AGG_CSV", "deepseek_agg.csv")

# (F) Columnas esperadas
COL_ONET = "O*NET-SOC Code"
COL_TASK_ID = "Task ID"
COL_TASK = "Task"
COL_TASK_TYPE = "Task Type"
COL_TITLE = "Title"

REQUIRED_COLS = [COL_ONET, COL_TASK_ID, COL_TASK, COL_TASK_TYPE, COL_TITLE]


# =========================
# 2) PROMPT + ESQUEMA JSON
# =========================

RUBRIC_VERSION = "2026-04_prob_nested_v1"

SYSTEM_PROMPT = (
    "You are an expert in labor economics and task-level exposure measurement to AI.\n"
    "You must follow the rubric strictly and output ONLY valid JSON with the specified keys.\n"
    "No extra text, no markdown, no code fences."
)

USER_PROMPT_TEMPLATE = """\
Classify exposure of the following work task to current LLM-based AI systems.

RUBRIC (based on Eloundou et al. taxonomy E0/E1/E2; updated wording for clarity):
- E0 (No exposure): The task cannot be meaningfully improved or accelerated by current LLM-based AI, even with typical digital tools.
- E1 (Direct exposure): A general-purpose LLM can directly assist to perform the task (e.g., drafting, summarizing, reasoning over text, generating standard code) such that the task time could plausibly be reduced by >= 50% while keeping comparable quality, WITHOUT requiring custom integrations.
- E2 (Extended exposure): The task could plausibly be reduced by >= 50% with comparable quality ONLY when the LLM is combined with additional tools/integrations (e.g., internal databases, APIs, automation workflows, specialized software, OCR pipelines, enterprise systems). Not fully direct via chat alone.

IMPORTANT:
- Focus on current capabilities (no future speculation).
- Evaluate technical feasibility, not legal/regulatory permission.
- Be conservative on borderline cases.
- Output probabilities using the nested decision structure below.

NESTED PROBABILITIES TO OUTPUT:
1) p_direct = P(E1)
2) p_extended_given_not_direct = P(E2 | NOT E1)
Then implied:
p_E2 = (1 - p_direct) * p_extended_given_not_direct
p_E0 = 1 - p_direct - p_E2

Return JSON with EXACT keys:
{{
  "p_direct": number,                         // 0..1
  "p_extended_given_not_direct": number,       // 0..1
  "confidence": number,                       // 0..1 (your confidence in these probabilities)
  "hard_label": "E0"|"E1"|"E2",               // based on highest implied probability among E0/E1/E2
  "flags": [string, ...],                     // optional: e.g., "borderline_E1_E2", "physical_task", "high_context_needed"
  "short_justification": string               // <= 35 words
}}

TASK CONTEXT:
- Occupation title: {title}
- O*NET-SOC: {onet_soc}
- Task type: {task_type}
- Task description: {task_text}
""".strip()


def build_messages(row: pd.Series) -> List[Dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=str(row[COL_TITLE]),
        onet_soc=str(row[COL_ONET]),
        task_type=str(row[COL_TASK_TYPE]),
        task_text=str(row[COL_TASK]),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# =========================
# 3) LLAMADA API (ROBUSTA)
# =========================

def _post_chat_completions(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = BASE_URL.rstrip("/") + CHAT_COMPLETIONS_PATH
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        # incluir texto ayuda a debug, pero cuidado con tokens.
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def call_model(messages: List[Dict[str, str]], temperature: float, top_p: float) -> str:
    """
    Devuelve el contenido crudo del assistant (string).
    Ajusta aquí si tu API requiere nombres distintos.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    data = _post_chat_completions(payload)
    # OpenAI-compatible: choices[0].message.content
    return data["choices"][0]["message"]["content"]


def robust_call(messages: List[Dict[str, str]], temperature: float, top_p: float) -> Tuple[str, int]:
    """
    Reintenta con backoff exponencial. Retorna (texto, intentos_usados).
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = call_model(messages, temperature=temperature, top_p=top_p)
            return text, attempt
        except Exception as e:
            last_err = e
            sleep_sec = (BACKOFF_BASE_SEC ** (attempt - 1)) + random.random() * BACKOFF_JITTER
            time.sleep(sleep_sec)
    raise RuntimeError(f"Falló tras {MAX_RETRIES} intentos. Último error: {last_err}")


# =========================
# 4) PARSING + CÁLCULOS
# =========================

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def clamp01(x: float) -> float:
    if math.isnan(x):
        return x
    return max(0.0, min(1.0, x))


def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Espera JSON puro. Si el modelo se equivoca y añade texto extra,
    intenta rescatar el primer objeto JSON.
    """
    text = text.strip()

    # Intento directo
    try:
        return json.loads(text)
    except Exception:
        pass

    # Rescate: busca primer '{' y último '}' y parsea
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidate = text[i : j + 1]
        return json.loads(candidate)

    raise ValueError("No se pudo parsear JSON.")


def implied_probs(p_direct: float, p_ext_cond: float) -> Tuple[float, float, float]:
    """
    Devuelve (pE0, pE1, pE2) consistentes con estructura anidada.
    """
    p_direct = clamp01(p_direct)
    p_ext_cond = clamp01(p_ext_cond)
    pE1 = p_direct
    pE2 = (1.0 - p_direct) * p_ext_cond
    pE0 = 1.0 - pE1 - pE2
    # pequeña corrección numérica
    pE0 = max(0.0, pE0)
    return pE0, pE1, pE2


def hard_label_from_probs(pE0: float, pE1: float, pE2: float) -> str:
    m = max(pE0, pE1, pE2)
    # desempate conservador: si empata, elige menor exposición (E0 < E1 < E2)
    candidates = []
    if abs(pE0 - m) < 1e-12: candidates.append("E0")
    if abs(pE1 - m) < 1e-12: candidates.append("E1")
    if abs(pE2 - m) < 1e-12: candidates.append("E2")
    return sorted(candidates, key=lambda z: {"E0": 0, "E1": 1, "E2": 2}[z])[0]


def alpha_beta_zeta_soft(pE1: float, pE2: float) -> Tuple[float, float, float]:
    """
    Replicando lógica de Eloundou:
      alpha = E1
      beta  = E1 + 0.5*E2
      zeta  = E1 + E2
    en versión suave (probabilística).
    """
    alpha = pE1
    beta = pE1 + 0.5 * pE2
    zeta = pE1 + pE2
    return alpha, beta, zeta


# =========================
# 5) I/O INCREMENTAL (JSONL) + RESUMEN
# =========================

def load_existing_runs(jsonl_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Lee el JSONL existente y devuelve un dict:
    task_id -> lista de corridas ya guardadas para esa tarea.
    Deduplica por run_id si hiciera falta.
    """
    runs_by_task = defaultdict(list)

    if not os.path.exists(jsonl_path):
        return {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"Warning: línea JSON inválida en {jsonl_path}, línea {line_num}. Se ignora.")
                continue

            task_id = str(obj.get("task_id", "")).strip()
            if not task_id:
                continue

            runs_by_task[task_id].append(obj)

    # Deduplicar por run_id y ordenar
    cleaned = {}
    for task_id, runs in runs_by_task.items():
        by_run_id = {}
        for r in runs:
            run_id = r.get("run_id", None)
            if run_id is None:
                continue
            try:
                run_id = int(run_id)
            except Exception:
                continue

            # si hubiera duplicados, conservamos el primero
            if run_id not in by_run_id:
                by_run_id[run_id] = r

        cleaned[task_id] = [by_run_id[k] for k in sorted(by_run_id.keys())]

    return cleaned


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# 6) PIPELINE PRINCIPAL
# =========================

def main():
    # 6.1 Cargar datos
    df = pd.read_csv(INPUT_FILE, sep="\t", index_col=0)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en input: {missing}. Columnas disponibles: {list(df.columns)}")

    # Asegurar tipos
    df[COL_TASK_ID] = df[COL_TASK_ID].astype(str)
    df[COL_ONET] = df[COL_ONET].astype(str)

    # 6.2 Leer progreso existente desde JSONL
    existing_runs_by_task = load_existing_runs(OUT_RUNS_JSONL)
    summarize_progress(existing_runs_by_task, N_RUNS_PER_TASK)

    # número de corridas ya hechas por task_id
    run_counts = {task_id: len(runs) for task_id, runs in existing_runs_by_task.items()}

    # solo procesar tareas incompletas
    to_process = df[df[COL_TASK_ID].map(lambda x: run_counts.get(str(x), 0) < N_RUNS_PER_TASK)].copy()

    print(f"Tareas pendientes o incompletas: {len(to_process)}")

    rows_list = list(to_process.iterrows())
    if tqdm is not None:
        rows_list = tqdm(rows_list, total=len(to_process), desc="Tasks")

    # 6.3 Loop tareas
    for _, row in rows_list:
        task_id = str(row[COL_TASK_ID])

        already_done_runs = existing_runs_by_task.get(task_id, [])
        already_done_n = len(already_done_runs)

        if already_done_n >= N_RUNS_PER_TASK:
            continue

        print(f"Procesando task_id={task_id} | corridas existentes={already_done_n} | faltan={N_RUNS_PER_TASK - already_done_n}")

        messages = build_messages(row)

        # conservamos las corridas previas
        run_records = already_done_runs.copy()

        # continuar desde la corrida faltante
        for run_id in range(already_done_n + 1, N_RUNS_PER_TASK + 1):
            raw_text, attempts = robust_call(messages, temperature=TEMPERATURE, top_p=TOP_P)

            parsed = parse_json_strict(raw_text)

            p_direct = safe_float(parsed.get("p_direct"))
            p_ext_cond = safe_float(parsed.get("p_extended_given_not_direct"))
            confidence = safe_float(parsed.get("confidence"))
            flags = parsed.get("flags") if isinstance(parsed.get("flags"), list) else []
            short_just = str(parsed.get("short_justification", "")).strip()

            p_direct = clamp01(p_direct)
            p_ext_cond = clamp01(p_ext_cond)
            confidence = clamp01(confidence)

            pE0, pE1, pE2 = implied_probs(p_direct, p_ext_cond)
            hard = hard_label_from_probs(pE0, pE1, pE2)

            alpha, beta, zeta = alpha_beta_zeta_soft(pE1, pE2)

            out_obj = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "rubric_version": RUBRIC_VERSION,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "attempts": attempts,
                "run_id": run_id,

                "onet_soc": str(row[COL_ONET]),
                "task_id": task_id,
                "title": str(row[COL_TITLE]),
                "task_type": str(row[COL_TASK_TYPE]),
                "task_text": str(row[COL_TASK]),

                "p_direct": p_direct,
                "p_extended_given_not_direct": p_ext_cond,
                "pE0": pE0,
                "pE1": pE1,
                "pE2": pE2,
                "hard_label": hard,
                "confidence": confidence,
                "flags": flags,
                "short_justification": short_just,

                "alpha_soft": alpha,
                "beta_soft": beta,
                "zeta_soft": zeta,

                "raw_model_output": raw_text,
            }

            append_jsonl(OUT_RUNS_JSONL, out_obj)
            run_records.append(out_obj)

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        # actualizar memoria local para que si más abajo se consulta el mismo task_id quede completo
        existing_runs_by_task[task_id] = run_records

    # 6.4 reconstruir agregado completo desde JSONL
    rebuild_agg_from_jsonl(OUT_RUNS_JSONL, OUT_AGG_CSV, N_RUNS_PER_TASK)

    print("Listo.")
    print(f"Corridas guardadas en: {OUT_RUNS_JSONL}")
    print(f"Agregado reconstruido en: {OUT_AGG_CSV}")

def summarize_progress(existing_runs_by_task: Dict[str, List[Dict[str, Any]]], n_runs_per_task: int) -> None:
    total_tasks_seen = len(existing_runs_by_task)
    complete = sum(1 for runs in existing_runs_by_task.values() if len(runs) >= n_runs_per_task)
    partial = sum(1 for runs in existing_runs_by_task.values() if 0 < len(runs) < n_runs_per_task)
    total_runs = sum(len(runs) for runs in existing_runs_by_task.values())

    print(f"Corridas existentes en JSONL: {total_runs}")
    print(f"Tareas con al menos una corrida: {total_tasks_seen}")
    print(f"Tareas completas ({n_runs_per_task}/{n_runs_per_task}): {complete}")
    print(f"Tareas parciales: {partial}")

def sd(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def aggregate_task_runs(run_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrega corridas de una sola tarea.
    Asume que run_records pertenecen al mismo task_id.
    """
    first = run_records[0]

    pE0_mean = sum(r["pE0"] for r in run_records) / len(run_records)
    pE1_mean = sum(r["pE1"] for r in run_records) / len(run_records)
    pE2_mean = sum(r["pE2"] for r in run_records) / len(run_records)

    alpha_mean = sum(r["alpha_soft"] for r in run_records) / len(run_records)
    beta_mean = sum(r["beta_soft"] for r in run_records) / len(run_records)
    zeta_mean = sum(r["zeta_soft"] for r in run_records) / len(run_records)

    hard_agg = hard_label_from_probs(pE0_mean, pE1_mean, pE2_mean)

    return {
        "onet_soc": first["onet_soc"],
        "task_id": first["task_id"],
        "title": first["title"],
        "task_type": first["task_type"],

        "pE0_mean": pE0_mean,
        "pE1_mean": pE1_mean,
        "pE2_mean": pE2_mean,
        "pE0_sd": sd([r["pE0"] for r in run_records]),
        "pE1_sd": sd([r["pE1"] for r in run_records]),
        "pE2_sd": sd([r["pE2"] for r in run_records]),

        "alpha_soft_mean": alpha_mean,
        "beta_soft_mean": beta_mean,
        "zeta_soft_mean": zeta_mean,

        "hard_label_mean_probs": hard_agg,
        "mean_confidence": sum(r["confidence"] for r in run_records) / len(run_records),
        "n_runs": len(run_records),
    }

def rebuild_agg_from_jsonl(jsonl_path: str, out_csv: str, n_runs_per_task: int) -> None:
    existing_runs_by_task = load_existing_runs(jsonl_path)

    agg_rows = []
    for task_id, run_records in existing_runs_by_task.items():
        # solo agregamos tareas completas
        if len(run_records) >= n_runs_per_task:
            run_records = sorted(run_records, key=lambda x: int(x["run_id"]))[:n_runs_per_task]
            agg_rows.append(aggregate_task_runs(run_records))

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_csv, index=False)
    print(f"Agregado reconstruido desde JSONL: {out_csv} ({len(agg_df)} tareas)")

if __name__ == "__main__":
    main()