import logging
import multiprocessing
import os

# Limita threads internas de BLAS/OpenMP por processo ANTES de importar numpy —
# com N_PROCESSES processos rodando ao mesmo tempo, cada um tentando abrir seu
# próprio pool de threads pra álgebra linear, a CPU fica sobre-inscrita (N
# processos x M threads cada) e cada simulação fica ordens de magnitude mais
# lenta do que rodando sozinha. Cada processo aqui já É a unidade de paralelismo.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from pathlib import Path
from hapi import SLIT_MICHELSON, SLIT_TRIANGULAR
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('spectra-generator')


ROOT_DATABASE_PATH = Path(__file__).resolve().parents[2] / 'database' / 'spectra'

# pressao em atm
PRESSURE_MIN = 0.1
PRESSURE_MAX = 1

# temperatura em K
TEMPERATURE_MIN = 273
TEMPERATURE_MAX =  400

#caminho optico em cm
OPTICAL_PATH_MIN = 0.5
OPTICAL_PATH_MAX = 10

# Sizes 
QTE_TRAIN = 10_000
QTE_VALID = 1000

#Configs de simulacao
STEP_MIN = 0.01
STEP_MAX = 0.01
MOLECULE_DATABASE_NAME_REFERENCE = 'hcl'
DILUENT = {'self': 1}

# Configs para simular função aparelho
CONVOLVE = True
RESOLUTION_MIN = 0.02
RESOLUTION_MAX = 5
SLIT_FUNCTIONS = [SLIT_MICHELSON, SLIT_TRIANGULAR]

# Paralelismo
N_PROCESSES = max(1, os.cpu_count() - 1)


generator = SpectraGenerator()


def sample_parameters(n: int) -> dict:
    """Amostra uniformemente `n` valores de cada parâmetro que tem MIN/MAX definido,
    e sorteia (com reposição, `np.random.choice`) uma função de fenda de
    `SLIT_FUNCTIONS` pra cada amostra."""
    return {
        'pressure':    np.random.uniform(PRESSURE_MIN, PRESSURE_MAX, n),
        'temperature': np.random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX, n),
        'optical_path': np.random.uniform(OPTICAL_PATH_MIN, OPTICAL_PATH_MAX, n),
        'step':        np.random.uniform(STEP_MIN, STEP_MAX, n),
        'resolution':  np.random.uniform(RESOLUTION_MIN, RESOLUTION_MAX, n),
        'slit_function': np.random.choice(SLIT_FUNCTIONS, n),
    }


def ensure_split_dirs(root: Path, molecule: str, splits: tuple) -> None:
    """Garante que a pasta de saída de cada split (train/valid/...) existe dentro
    de `root/<molecule>/`, criando quando faltar — assim cada split fica separado
    em sua própria subpasta, e cada molécula na sua, em vez de tudo junto (rodar
    o script pra moléculas diferentes não mistura nem sobrescreve nada)."""
    for split in splits:
        split_dir = root / molecule / split
        already_existed = split_dir.exists()
        split_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{split}] {'já existia' if already_existed else 'criada agora'}: {split_dir}")


def build_tasks() -> list:
    """Monta a lista de tarefas — uma por espectro, treino e validação juntos.
    Cada tarefa é um dict com os parâmetros da simulação e o `split`, que é
    quem diz pro processo onde salvar (`ROOT_DATABASE_PATH/<split>/...`)."""
    tasks = []
    for split, n in (('train', QTE_TRAIN), ('valid', QTE_VALID)):
        params = sample_parameters(n)
        for i in range(n):
            tasks.append({
                'split': split,
                'index': i,
                'pressure': params['pressure'][i],
                'temperature': params['temperature'][i],
                'optical_path': params['optical_path'][i],
                'step': params['step'][i],
                'resolution': params['resolution'][i],
                'slit_function': params['slit_function'][i],
            })
    return tasks


def simulate_and_save(split: str, index: int, pressure: float, temperature: float, optical_path: float,
                       step: float, resolution: float, slit_function) -> Path | None:
    """Simula um espectro pra uma combinação de parâmetros e salva o espectro +
    metadados num `.npz` em `ROOT_DATABASE_PATH/<split>/`. Retorna o caminho do
    arquivo salvo, ou `None` se a simulação falhar (amostra é pulada, não
    derruba o resto da geração)."""
    logger.info(
        f"[{split}] {index} (pid={os.getpid()}) — T={temperature:.2f}K P={pressure:.4f}atm "
        f"L={optical_path:.2f}cm step={step:.4f} resolution={resolution:.4f} slit={slit_function.__name__}"
    )

    generator.simulateSpectra(
        MOLECULE_DATABASE_NAME_REFERENCE, DILUENT,
        {'p': pressure, 'T': temperature, 'l': optical_path},
        step=step, quantity='absorbance', convolve=CONVOLVE, resolution=resolution,
        slit_function=slit_function,
    )

    wavenumbers = generator.spectra['wavenumbers']
    absorption = generator.spectra['absorption']
    if wavenumbers is None or absorption is None or len(wavenumbers) == 0:
        logger.warning(
            f"[{split}] amostra {index}: simulação falhou ou devolveu espectro vazio "
            f"(resolution={resolution:.4f} pode estar grande demais pra essa janela de wavenumber), pulando."
        )
        return None

    filename = (
        f"{index:05d}_{MOLECULE_DATABASE_NAME_REFERENCE}_"
        f"T{temperature:.2f}K_P{pressure:.4f}atm_L{optical_path:.2f}cm.npz"
    )
    file_path = ROOT_DATABASE_PATH / MOLECULE_DATABASE_NAME_REFERENCE / split / filename

    np.savez_compressed(
        file_path,
        wavenumbers=wavenumbers,
        absorption=absorption,
        molecule=MOLECULE_DATABASE_NAME_REFERENCE,
        pressure_atm=pressure,
        temperature_k=temperature,
        optical_path_cm=optical_path,
        step=step,
        resolution=resolution,
        slit_function=slit_function.__name__,
        convolve=CONVOLVE,
        diluent_gases=np.array(list(DILUENT.keys())),
        diluent_fractions=np.array(list(DILUENT.values()), dtype=float),
        wavenumber_min=float(np.min(wavenumbers)),
        wavenumber_max=float(np.max(wavenumbers)),
        split=split,
        index=index,
    )
    return file_path


def _process_task(task: dict) -> tuple:
    """Roda numa das tarefas de `build_tasks()` — é o que cada processo da
    Pool executa. Retorna (split, sucesso) pra dar pra contabilizar o
    progresso por split lá em `main()`."""
    file_path = simulate_and_save(**task)
    return task['split'], file_path is not None


def main() -> None:
    ensure_split_dirs(ROOT_DATABASE_PATH, MOLECULE_DATABASE_NAME_REFERENCE, ('train', 'valid'))

    tasks = build_tasks()
    totals = {'train': QTE_TRAIN, 'valid': QTE_VALID}
    counts = {'train': 0, 'valid': 0}

    logger.info(
        f"=== Iniciando geração de {len(tasks)} espectros "
        f"(train={QTE_TRAIN}, valid={QTE_VALID}) com {N_PROCESSES} processos ==="
    )

    # spawn (não fork): evita travar quando o worker herda estado de threads do
    # HAPI/numpy já em uso no processo principal antes do Pool ser criado.
    with multiprocessing.get_context('spawn').Pool(processes=N_PROCESSES) as pool:
        for i, (split, ok) in enumerate(pool.imap_unordered(_process_task, tasks), start=1):
            if ok:
                counts[split] += 1
            if i % 100 == 0 or i == len(tasks):
                logger.info(
                    f"progresso: {i}/{len(tasks)} processados — "
                    f"train {counts['train']}/{totals['train']}, valid {counts['valid']}/{totals['valid']}"
                )

    logger.info(
        f"=== Geração concluída: train {counts['train']}/{totals['train']}, "
        f"valid {counts['valid']}/{totals['valid']} ==="
    )


if __name__ == "__main__":
    main()
