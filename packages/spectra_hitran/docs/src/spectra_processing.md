# `SpectraHitran.SpectraProcessing`

`src/SpectraHitran/SpectraProcessing/spectraprocessing.py` — detecção de
linhas espectrais e utilitário de visualização. Genérico para qualquer
molécula — sem classificação de ramo (P/Q/R) nem lógica específica de
espécie (removida deliberadamente; ver `docs/README.md`).

`Spectrum = Dict[str, np.ndarray]` — mesmo formato de `SpectraGenerator.spectra`:
`{'wavenumbers': ..., 'absorption': ...}`.

## `detectLines(spectra: Spectrum, findPeaksParams: Dict[str, float]) -> pd.DataFrame`

Detecta picos de absorção (`scipy.signal.find_peaks`) e retorna um
`DataFrame` com colunas `wavenumber`/`absorption` por pico, ordenado por
`wavenumber` crescente.

- `findPeaksParams`: repassado direto ao `find_peaks` — precisa das chaves
  `height`, `width`, `distance` (ver docs do `scipy.signal.find_peaks`).
- Sem picos que atendam aos critérios: retorna `DataFrame` vazio (mesmas colunas).

## `plotSpectrum(real: Spectrum, lines: pd.DataFrame) -> None`

Plota o espectro (matplotlib) com os centros de `lines` marcados por cima
(`plt.scatter`, vermelho). Só para uso exploratório em script/notebook.

## Exemplo de uso

```python
from SpectraHitran.SpectraProcessing.spectraprocessing import detectLines, plotSpectrum

lines = detectLines(gen.spectra, {'height': 0.01, 'width': 0, 'distance': 1})
plotSpectrum(gen.spectra, lines)
```
