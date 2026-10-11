import logging
import multiprocessing
import os

# 1 thread de BLAS/OpenMP por processo (definido antes de importar numpy):
# o paralelismo vem da Pool; threads extras sobrecarregam a CPU.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from pathlib import Path
from hapi import SLIT_GAUSSIAN, LOCAL_TABLE_CACHE
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('spectra-generator')


ROOT_DATABASE_PATH = Path(__file__).resolve().parents[2] / 'database'

# pressao em atm
PRESSURE_MIN = 0.015
PRESSURE_MAX = 1

# temperatura em K
TEMPERATURE_MIN = 273
TEMPERATURE_MAX =  373

#caminho optico em cm
OPTICAL_PATH_MIN = 1
OPTICAL_PATH_MAX = 10

# Total de espectros, dividido igualmente entre as janelas e depois entre os
# splits (SPLIT_FRACTIONS deve somar 1).
QTE_TOTAL = 10_000
SPLIT_FRACTIONS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

#Configs de simulacao
# Passo < RESOLUTION_MIN (exigência do convolveSpectrum do HAPI); 0.001 dá ≥ 5 pontos
# por largura da fenda e é da ordem do passo dos espectros reais (0.00094 e 0.00188).
STEP_MIN = 0.001
STEP_MAX = 0.001
# Janelas: nome -> (tabela em database/hitran_database, centro em cm⁻¹).
# Largura fixa WINDOW_WIDTH (cm⁻¹) => todos os espectros têm o mesmo nº de pontos.
# Centros = meio de cada banda a 400 K.
WINDOW_WIDTH = 900
WINDOWS = {
    # 'hcl_10': ('hcl', 2818),
    'hcl_20': ('hcl', 5551),
    # 'co_10':  ('co',  2126),
    # 'co_20':  ('co',  4211),
    # 'hf_10':  ('hf',  3896),
    # 'hf_20':  ('hf',  7602),
}
# Aviso de banda cortada (ver check_windows): linha fora da janela, até
# BAND_CUT_MARGIN cm⁻¹ da borda, com intensidade > BAND_CUT_THRESHOLD x a mais forte.
BAND_CUT_THRESHOLD = 1e-3
BAND_CUT_MARGIN = 100
DILUENT = {'self': 1}

# Configs para simular função aparelho (resolução = FWHM da gaussiana, em cm⁻¹).
# Ajustadas nos espectros reais: a gaussiana (espectro apodizado) ajusta melhor que o
# Michelson; resolução ~0.025–0.03 nos reais de 40/42 °C e ≲ 0.01 nos de 20 °C.
# A faixa cobre os dois, para a rede não depender da resolução.
CONVOLVE = True
RESOLUTION_MIN = 0.005
RESOLUTION_MAX = 0.04
SLIT_FUNCTIONS = [SLIT_GAUSSIAN]

# Tipo de experimento
TYPE_EXPERIMENT = 'absorbance'  # 'transmittance' ou 'absorbance'

# Paralelismo
N_PROCESSES = max(1, os.cpu_count() - 1)


generator = SpectraGenerator()


def log_uniform(low: float, high: float, n: int) -> np.ndarray:
    """Amostra `n` valores log-uniformes em [low, high]."""
    return np.exp(np.random.uniform(np.log(low), np.log(high), n))


def sample_parameters(n: int) -> dict:
    """Sorteia `n` conjuntos de parâmetros, todos uniformes nas suas faixas."""
    return {
        'pressure':    np.random.uniform(PRESSURE_MIN, PRESSURE_MAX, n),
        'temperature': np.random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX, n),
        'optical_path':  np.random.uniform(OPTICAL_PATH_MIN, OPTICAL_PATH_MAX, n),
        'step':        np.random.uniform(STEP_MIN, STEP_MAX, n),
        'resolution':   np.random.uniform(RESOLUTION_MIN, RESOLUTION_MAX, n),
        'slit_function': np.random.choice(SLIT_FUNCTIONS, n),
    }


def split_counts(total: int, keys: list) -> dict:
    """Divide `total` igualmente entre `keys`; o resto vai para as primeiras."""
    base, rest = divmod(total, len(keys))
    return {k: base + (1 if i < rest else 0) for i, k in enumerate(keys)}


def plan_sizes() -> dict:
    """Nº de espectros por (janela, split)."""
    if not np.isclose(sum(SPLIT_FRACTIONS.values()), 1):
        raise ValueError(f"SPLIT_FRACTIONS deve somar 1, soma {sum(SPLIT_FRACTIONS.values())}.")
    sizes = {}
    for window, n_window in split_counts(QTE_TOTAL, list(WINDOWS)).items():
        n_split = {split: int(n_window * frac) for split, frac in SPLIT_FRACTIONS.items()}
        n_split['train'] += n_window - sum(n_split.values())  # sobra do arredondamento
        for split, n in n_split.items():
            sizes[(window, split)] = n
    return sizes


def window_range(window: str) -> tuple:
    """(início, fim) da janela em cm⁻¹."""
    _, center = WINDOWS[window]
    return center - WINDOW_WIDTH / 2, center + WINDOW_WIDTH / 2


def check_windows(database_path: Path) -> None:
    """Valida as janelas: erro se começarem em <= 0 cm⁻¹ ou se a tabela não
    foi baixada; aviso se a janela cortar a banda em TEMPERATURE_MAX."""
    bad = [w for w in WINDOWS if window_range(w)[0] <= 0]
    if bad:
        raise ValueError(f"Janelas começando em número de onda <= 0: {bad}.")
    missing = sorted({table for table, _ in WINDOWS.values()} - {p.stem for p in database_path.glob('*.data')})
    if missing:
        raise FileNotFoundError(f"Moléculas sem tabela em {database_path}: {missing}.")

    c2 = 1.4387769  # cm·K, 2ª constante de radiação (hc/k)
    for window, (table, _) in WINDOWS.items():
        data = LOCAL_TABLE_CACHE[table]['data']
        nu, sw, elower = (np.asarray(data[k]) for k in ('nu', 'sw', 'elower'))
        # Intensidade relativa em TEMPERATURE_MAX (a função de partição cancela na razão)
        strength = sw * np.exp(-c2 * elower * (1 / TEMPERATURE_MAX - 1 / 296))
        start, end = window_range(window)
        inside = (nu >= start) & (nu <= end)
        if not inside.any():
            logger.warning(f"[{window}] nenhuma linha em {start:.0f}–{end:.0f} cm⁻¹.")
            continue
        strong_out = ~inside & (strength > strength[inside].max() * BAND_CUT_THRESHOLD)
        # Só perto das bordas; mais longe já são outras bandas.
        below = nu[strong_out & (nu < start) & (nu > start - BAND_CUT_MARGIN)]
        above = nu[strong_out & (nu > end) & (nu < end + BAND_CUT_MARGIN)]
        cut = []
        if below.size:
            cut.append(f"{below.min():.0f}–{start:.0f}")
        if above.size:
            cut.append(f"{end:.0f}–{above.max():.0f}")
        if cut:
            logger.warning(f"[{window}] janela {start:.0f}–{end:.0f} cm⁻¹ corta a banda a "
                           f"{TEMPERATURE_MAX} K: linhas fortes em {', '.join(cut)} cm⁻¹.")


def ensure_split_dirs(root: Path, splits) -> None:
    """Cria `root/<split>/` para cada split, se não existir."""
    for split in splits:
        split_dir = root / split
        already_existed = split_dir.exists()
        split_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{split}] {'já existia' if already_existed else 'criada agora'}: {split_dir}")


def build_tasks(sizes: dict) -> list:
    """Uma tarefa (dict de kwargs de `simulate_and_save`) por espectro.
    `index` é contínuo dentro de cada split."""
    tasks = []
    next_index = {split: 0 for split in SPLIT_FRACTIONS}
    for (window, split), n in sizes.items():
        params = sample_parameters(n)
        for i in range(n):
            tasks.append({
                'window': window,
                'split': split,
                'index': next_index[split],
                'pressure': params['pressure'][i],
                'temperature': params['temperature'][i],
                'optical_path': params['optical_path'][i],
                'step': params['step'][i],
                'resolution': params['resolution'][i],
                'slit_function': params['slit_function'][i],
            })
            next_index[split] += 1
    return tasks


def simulate_and_save(window: str, split: str, index: int, pressure: float, temperature: float, optical_path: float,
                       step: float, resolution: float, slit_function) -> Path | None:
    """Simula um espectro e salva espectro + metadados em
    `ROOT_DATABASE_PATH/<split>/*.npz`. Retorna o caminho, ou `None` se falhar."""
    molecule, _ = WINDOWS[window]
    window_min, window_max = window_range(window)
    logger.info(
        f"[{split}] {index} {window} (pid={os.getpid()}) — T={temperature:.2f}K P={pressure:.4f}atm "
        f"L={optical_path:.2f}cm step={step:.4f} resolution={resolution:.4f} slit={slit_function.__name__}"
    )

    generator.simulateSpectra(
        molecule, DILUENT,
        {'p': pressure, 'T': temperature, 'l': optical_path},
        step=step, quantity=TYPE_EXPERIMENT, convolve=CONVOLVE, resolution=resolution,
        slit_function=slit_function, wavenumber_range=(window_min, window_max),
    )

    wavenumbers = generator.spectra['wavenumbers']
    absorption = generator.spectra['absorption']
    if wavenumbers is None or absorption is None or len(wavenumbers) == 0:
        logger.warning(
            f"[{split}] amostra {index} ({window}): simulação falhou ou devolveu espectro vazio "
            f"(resolution={resolution:.4f} pode estar grande demais pra essa janela de wavenumber), pulando."
        )
        return None

    filename = (
        f"{index:05d}_{window}_"
        f"T{temperature:.2f}K_P{pressure:.4f}atm_L{optical_path:.2f}cm.npz"
    )
    file_path = ROOT_DATABASE_PATH / split / filename

    np.savez_compressed(
        file_path,
        wavenumbers=wavenumbers.astype(np.float32),
        absorption=absorption.astype(np.float32),
        molecule=molecule,
        window=window,
        window_min=window_min,
        window_max=window_max,
        pressure_atm=pressure,
        temperature_k=temperature,
        optical_path_cm=optical_path,
        step=step,
        resolution=resolution,
        slit_function=slit_function.__name__,
        convolve=CONVOLVE,
        quantity=TYPE_EXPERIMENT,
        diluent_gases=np.array(list(DILUENT.keys())),
        diluent_fractions=np.array(list(DILUENT.values()), dtype=float),
        wavenumber_min=float(np.min(wavenumbers)),
        wavenumber_max=float(np.max(wavenumbers)),
        split=split,
        index=index,
    )
    return file_path


def _process_task(task: dict) -> tuple:
    """Worker da Pool. Retorna (split, sucesso)."""
    file_path = simulate_and_save(**task)
    return task['split'], file_path is not None


def main() -> None:
    check_windows(Path(generator.database_path))
    ensure_split_dirs(ROOT_DATABASE_PATH, SPLIT_FRACTIONS)

    sizes = plan_sizes()
    tasks = build_tasks(sizes)
    totals = {split: sum(n for (_, s), n in sizes.items() if s == split) for split in SPLIT_FRACTIONS}
    counts = {split: 0 for split in SPLIT_FRACTIONS}

    def summary() -> str:
        return ', '.join(f"{split} {counts[split]}/{totals[split]}" for split in SPLIT_FRACTIONS)

    logger.info(
        f"=== Iniciando geração de {len(tasks)} espectros das janelas {WINDOWS} "
        f"({', '.join(f'{k}={v}' for k, v in totals.items())}) com {N_PROCESSES} processos ==="
    )

    # spawn evita travamentos por estado de threads herdado do processo principal.
    with multiprocessing.get_context('spawn').Pool(processes=N_PROCESSES) as pool:
        for i, (split, ok) in enumerate(pool.imap_unordered(_process_task, tasks), start=1):
            if ok:
                counts[split] += 1
            if i % 100 == 0 or i == len(tasks):
                logger.info(f"progresso: {i}/{len(tasks)} processados — {summary()}")

    logger.info(f"=== Geração concluída: {summary()} ===")


if __name__ == "__main__":
    main()
