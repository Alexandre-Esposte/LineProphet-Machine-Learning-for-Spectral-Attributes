# `SpectraHitran.SpectraGenerator`

`src/SpectraHitran/SpectraGenerator/spectragenerator.py` — baixa dados de
linhas do HITRAN e simula espectros de absorção via
[HAPI](https://hitran.org/hapi/).

## `class SpectraGenerator`

### `__init__(database_path=None)`

Inicializa o backend de banco de dados local do HAPI (`db_begin`).

- `database_path`: pasta de cache dos `.data`/`.header`. Padrão:
  `hitran_database/` na raiz do projeto (caminho absoluto, independe do
  diretório corrente de onde o script roda).
- `db_begin` é sempre chamado, mesmo que a pasta já exista — pular essa
  chamada faz o HAPI gravar tudo no diretório corrente do processo em vez
  da pasta configurada.
- Atributos: `self.spectra` (`{'wavenumbers', 'absorption'}`, populado por
  `simulateSpectra`) e `self._fetched_diluents` (diluentes já baixados por
  `TableName`, usado para validar `simulateSpectra`).

### `downloadMolecule(moleculeName, isotopologues, wavenumberRange, diluents=('air', 'self'))`

Baixa a lista de linhas de uma molécula/isotopólogo(s) do HITRAN pro cache
local (`fetch_by_ids`).

- `moleculeName`: chave local e arbitrária da tabela no cache — não precisa
  bater com o nome oficial do HITRAN.
- `isotopologues`: IDs de isotopólogo **globais** do HITRAN, da tabela
  `hapi.ISO` — **não** são índices locais (1, 2, 3...). Ex.: `(52, 53)`
  para H35Cl/H37Cl.
- `wavenumberRange`: `(min, max)` em cm⁻¹.
- `diluents`: gases de alargamento a baixar, subconjunto de `air`, `self`,
  `h2`, `co2`, `he`, `h2o` — precisam cobrir os gases usados depois em
  `simulateSpectra`. Baixa um grupo `voigt_<gas>` por diluente pedido, que
  é o mesmo dado reaproveitado pelos perfis `'voigt'`/`'lorentz'`/
  `'doppler'`/`'priority'` (não existe `profile` aqui porque só oferecemos
  perfis de simulação que não precisam de mais nada além disso).
  **Cuidado com a tupla unária:** `('self')` sem vírgula não é uma tupla,
  é só a string `'self'` — o código itera sobre ela caractere por
  caractere e você vê um erro de diluentes desconhecidos com letras soltas
  (`'e'`, `'f'`, `'l'`, `'s'`). Use `('self',)`. Passar uma `str` direto
  agora levanta um `ValueError` explícito em vez disso.
- `ValueError` (capturado e logado, não propaga) se `diluents` for uma
  string em vez de tupla/lista, ou se algum diluente não for suportado.
  Registra o que foi baixado em `self._fetched_diluents`.

**Por que só Voigt/Lorentz/Doppler/Priority?** O HITRAN publica parâmetro
de alargamento pra perfis mais específicos (Hartmann-Tran, Voigt
speed-dependent, Galatry) só em `air`/`self`, nunca nos demais diluentes,
em nenhuma molécula — não é falta de download, é o HITRAN mesmo que não
tem esse dado (Galatry também não tem função de cálculo nesta versão do
HAPI). Pra não expor um "perfil preso" a um subconjunto fixo de gases, só
oferecemos os que funcionam com qualquer diluente suportado.

### `simulateSpectra(moleculeName, diluent, enviroment, step=0.01, profile='voigt', quantity='absorbance', convolve=True, resolution=0.07, af_wing=0.1, slit_function=SLIT_MICHELSON, wavenumber_wing=50)`

Calcula o coeficiente de absorção via HAPI e deriva a grandeza pedida,
opcionalmente convoluindo com a fenda do instrumento. Resultado em
`self.spectra['wavenumbers']`/`['absorption']` (ou `(None, None)` em caso
de falha).

- `diluent`: frações de cada gás (ex.: `{'air': 0.3, 'self': 0.7}`),
  precisam somar `1` (tolerância `1e-6`) — checado manualmente porque o
  HAPI só valida que cada fração individual está em `[0, 1]`, não a soma;
  sem essa checagem uma soma `< 1` sub-alargaria as linhas silenciosamente,
  como se parte da mistura fosse vácuo. Também valida que todo gás em
  `diluent` já foi baixado para `moleculeName` (via `_fetched_diluents`) —
  essa validação só roda se `downloadMolecule` foi chamado nesta mesma
  instância; com cache de sessão anterior, é pulada.
- `enviroment`: `{'l': <cm>, 'p': <atm>, 'T': <K>}`.
- `profile`: `'voigt'` (padrão), `'lorentz'`, `'doppler'` ou `'priority'`
  (HAPI escolhe o melhor perfil disponível). O nome da função HAPI é
  resolvido via `globals()[...]` a cada chamada (não a referência direta),
  de propósito, para que testes possam substituí-la via monkeypatch.
- `quantity`: grandeza final em `self.spectra['absorption']`. O HAPI sempre
  calcula primeiro o coeficiente de absorção α(ν) via
  `absorptionCoefficient_<profile>`; todas as demais grandezas são
  **derivadas dele** (nunca da seção de choque). `HITRAN_units` só vira
  `True` (α em cm²/molécula em vez de cm⁻¹) quando `quantity='cross_section'`
  — em qualquer outro caso fica `False`.

  | `quantity` | grandeza | como é obtida |
  |---|---|---|
  | `'coefficient'` | α(ν), cm⁻¹ | retorno direto do HAPI (`HITRAN_units=False`) |
  | `'cross_section'` | σ(ν), cm²/molécula | mesmo cálculo, com `HITRAN_units=True`; não depende de `enviroment['l']` |
  | `'transmittance'` | T(ν) | `transmittanceSpectrum(α, l)` do HAPI |
  | `'absorptance'` | `1 - T(ν)` | `absorptionSpectrum(α, l)` do HAPI (nome do HAPI é enganoso — não é o coeficiente) |
  | `'absorbance'` (padrão) | profundidade ótica, `-ln(T) = α·l` | `transmitance = exp(-α·l)`, `absorption = -log(transmitance)`. **Fórmula escrita manualmente** — não há equivalente pronto no HAPI |
  | `'radiance'` | radiância emitida | `radianceSpectrum(α, l)` do HAPI, usa `T` (temperatura) de `enviroment` |
- `convolve`: se `True` (padrão), convolui com `convolveSpectrum`
  (`SlitFunction=slit_function`, `Resolution=resolution`,
  `AF_wing=af_wing`). Se `False`, guarda o espectro cru — útil para
  `'cross_section'` teórica.
- `resolution`, `af_wing`, `slit_function`: só usados se `convolve=True`.
  `af_wing` padrão `None` → `10 * resolution` (escala automaticamente com a
  resolução pedida). **`AF_wing` não escala com `Resolution` por conta
  própria no HAPI** — é um raio absoluto do kernel (o próprio HAPI usa
  `Resolution=0.1`/`AF_wing=10.0` por padrão, proporção ~100x); se você
  aumentar `resolution` sem também aumentar `af_wing`, o kernel da fenda
  (principalmente de cauda longa, como `SLIT_MICHELSON`, o padrão) fica
  truncado e renormalizado, e a degradação de resolução não aparece de
  fato no espectro — foi exatamente esse o bug reportado ao usar
  `resolution` alto com `af_wing` fixo em um valor pequeno.
- `wavenumber_wing`: alcance (cm⁻¹) considerado ao redor de cada linha no
  cálculo do perfil. Padrão `50`.
- Qualquer exceção reseta `self.spectra` para `{None, None}` e imprime o
  traceback — nunca deixa estado parcial de uma tentativa anterior.
- `self.diluent_warnings`: populado (não-fatal) a cada chamada. Um gás
  *baixado* mas sem alargamento medido para essa molécula/isotopólogo
  (coluna `gamma_<gas>` toda NaN) não gera exceção — o HAPI trata a
  contribuição como zero silenciosamente. `_diluents_without_data`
  detecta isso e preenche uma mensagem por gás (lista vazia se todos têm
  dado real). Ex.: para HCl, `h2o` não tem alargamento publicado.

### `plot(interactive=False, ax=None, show=True)`

Plota `self.spectra`. `interactive=False` (padrão) usa matplotlib
(estático); `interactive=True` usa Plotly (zoom/pan/hover com o valor
exato de cada ponto).

- Rótulo do eixo Y e título são derivados do contexto da última
  `simulateSpectra` bem-sucedida (`self._last_simulation`: `moleculeName`,
  `quantity`, `enviroment`) — `_QUANTITY_LABELS` mapeia cada `quantity`
  pro rótulo certo (ex.: `'coefficient'` → `'Coeficiente de absorção (cm⁻¹)'`).
- Se `self.spectra` estiver vazio (nenhuma simulação ainda, ou a última
  falhou), imprime um aviso e retorna `None` em vez de deixar o
  matplotlib/Plotly estourar um erro genérico sobre `None`.
- `ax`: só usado no modo estático — passe um `plt.Axes` existente para
  compor com outros plots; se `None`, cria uma figura nova.
- `show`: se `False`, não exibe a figura — útil pra salvar
  (`fig.savefig(...)` no estático) ou customizar antes de mostrar.
- Retorno: `(fig, ax)` do matplotlib no modo estático, ou um
  `plotly.graph_objects.Figure` no modo interativo.

## Exemplo de uso

```python
from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator

gen = SpectraGenerator()  # cache em hitran_database/ na raiz do projeto
gen.downloadMolecule('hcl_52-53', (52, 53), (2900, 2910), diluents=('air', 'self'))
gen.simulateSpectra('hcl_52-53', {'air': 0.2, 'self': 0.8}, {'l': 10, 'p': 1, 'T': 296})
gen.plot()                    # estático (matplotlib)
gen.plot(interactive=True)    # interativo (Plotly) — zoom/pan/hover

# seção de choque teórica, sem convolução instrumental, perfil Lorentz:
gen.simulateSpectra('hcl_52-53', {'air': 0.2, 'self': 0.8}, {'l': 10, 'p': 1, 'T': 296},
                    profile='lorentz', quantity='cross_section', convolve=False)
```
