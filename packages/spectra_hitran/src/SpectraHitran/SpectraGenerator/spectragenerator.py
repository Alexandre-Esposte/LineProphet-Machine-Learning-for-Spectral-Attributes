from hapi import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import traceback
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Any, Literal, Union

# Pasta padrão do cache local do HITRAN, ancorada na raiz do projeto
# (independe do diretório corrente de onde o script é executado).
_DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[5] / "database" / "hitran_database"

# Diluentes de alargamento de linha suportados pelo HITRAN/HAPI (além de air/self).
_SUPPORTED_DILUENTS = ('air', 'self', 'h2', 'co2', 'he', 'h2o')

# Perfis de linha disponíveis no HAPI para o CÁLCULO do coeficiente de absorção
# (usados em `simulateSpectra`). Guardamos o NOME da função (resolvido em
# `globals()` a cada chamada) em vez da função em si, para que testes possam
# substituir a função do HAPI via monkeypatch.
#
# Só ficam aqui os perfis que funcionam pra QUALQUER molécula/diluente que o
# Voigt padrão já suporta (`_SUPPORTED_DILUENTS` — air/self/h2/co2/he/h2o):
# 'voigt'/'lorentz'/'doppler' reaproveitam os mesmos parâmetros `voigt_<gas>`
# baixados por `downloadMolecule` (Lorentz = Voigt com largura Doppler zerada;
# Doppler = Voigt com largura de pressão zerada), e 'priority' degrada de
# forma graciosa pro que estiver disponível. Deixamos de fora de propósito
# 'ht'/'sdvoigt'/'galatry': o HITRAN só publica parâmetro real pra esses em
# air/self (nunca h2/co2/he/h2o, em nenhuma molécula — é uma limitação
# estrutural do HITRAN/HAPI, não coisa de download incompleto), e Galatry nem
# tem função de cálculo nesta versão do HAPI. Preferimos não expor um "perfil
# preso" que só funciona pra um subconjunto fixo de diluentes.
_PROFILE_FUNCTION_NAMES = {
    'voigt': 'absorptionCoefficient_Voigt',
    'lorentz': 'absorptionCoefficient_Lorentz',
    'doppler': 'absorptionCoefficient_Doppler',
    'priority': 'absorptionCoefficient_Priority',
}

Profile = Literal['voigt', 'lorentz', 'doppler', 'priority']

# Grandezas de saída suportadas em simulateSpectra. 'cross_section' precisa de
# HITRAN_units=True (cm²/molécula); as demais usam HITRAN_units=False (cm⁻¹).
_QUANTITIES = ('coefficient', 'cross_section', 'transmittance', 'absorptance', 'absorbance', 'radiance')

# Rótulo do eixo Y em plot(), por quantity — usado pra não rotular todo
# espectro como "Absorption (A.U)" independente da grandeza pedida.
_QUANTITY_LABELS = {
    'coefficient': 'Coeficiente de absorção (cm⁻¹)',
    'cross_section': 'Seção de choque (cm²/molécula)',
    'transmittance': 'Transmitância',
    'absorptance': 'Absorptância (1 - T)',
    'absorbance': 'Profundidade ótica (α·l)',
    'radiance': 'Radiância',
}


class SpectraGenerator:
    """Baixa dados de linhas do HITRAN e simula espectros de absorção via HAPI."""

    def __init__(self, database_path: Optional[str] = None) -> None:
        """Inicializa o backend de banco de dados local do HAPI.

        Args:
            database_path: pasta de cache do HITRAN. Padrão: `database/hitran_database`
                na raiz do projeto.

        Exemplo:
            >>> gen = SpectraGenerator()
        """
        self.database_path = str(database_path or _DEFAULT_DATABASE_PATH)

        db_begin(self.database_path)

        self.spectra: Dict[str, Optional[np.ndarray]] = {'wavenumbers': None, 'absorption': None}
        self._fetched_diluents: Dict[str, set] = {}
        # Contexto da última simulação bem-sucedida (molécula/quantity/condições),
        # usado só para rotular plot() — não afeta o cálculo.
        self._last_simulation: Optional[Dict[str, Any]] = None
        # Avisos (não-fatais) da última simulateSpectra: diluentes pedidos que o
        # HITRAN não tem dado real de alargamento publicado para essa molécula/
        # isotopólogo (o HAPI trata a contribuição desse gás como zero, em vez
        # de erro, então isso não aparece em nenhuma exceção).
        self.diluent_warnings: List[str] = []

        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['figure.figsize'] = (12, 6)

    def downloadMolecule(self, moleculeName: str, isotopologues: Tuple[int, ...], wavenumberRange: Tuple[float, float],
                         diluents: Tuple[str, ...] = ('air', 'self')) -> None:
        """Baixa a lista de linhas de uma molécula do HITRAN para o cache local.

        Args:
            moleculeName: nome usado pra identificar a tabela local (ex.: `'HCl'`). Apenas identificação local, sem fins praticos para HAPI
            isotopologues: IDs globais de isotopólogo do HITRAN (tabela
                `hapi.ISO`), não confundir com índices locais (1, 2, 3...). Ex.: `(52, 53)`
                para H35Cl/H37Cl.
            wavenumberRange: `(min, max)` em cm⁻¹.
            diluents: gases de alargamento a baixar, subconjunto de
                `('air', 'self', 'h2', 'co2', 'he', 'h2o')`. Precisam cobrir os
                gases usados depois em `simulateSpectra` (mesmo parâmetro
                `voigt_<gas>` para os perfis `'voigt'`/`'lorentz'`/`'doppler'`/
                `'priority'`). Padrão `('air', 'self')`. Para um único gás,
                não esqueça a vírgula da tupla unária: `('self',)` — `('self')`
                sem vírgula é só a string `'self'` e é iterada caractere por
                caractere.

        Exemplo:
            >>> gen.downloadMolecule('HCl', (52, 53), (2500, 3200), diluents=('air', 'self'))
            >>> gen.downloadMolecule('HCl', (52,), (2500, 3200), diluents=('self',))
        """
        try:
            if isinstance(diluents, str):
                raise ValueError(f"diluents deve ser uma tupla/lista de gases, não uma string: {diluents!r} "
                                 f"(lembre da vírgula da tupla unária, ex.: ('{diluents}',))")

            diluents = tuple(g.lower() for g in diluents)
            unknown = set(diluents) - set(_SUPPORTED_DILUENTS)
            if unknown:
                raise ValueError(f"Diluentes não suportados pelo HITRAN: {sorted(unknown)}")

            parameter_groups = ['160-char'] + [f'voigt_{gas}' for gas in diluents]
            fetch_by_ids(moleculeName, isotopologues, wavenumberRange[0], wavenumberRange[1],
                         ParameterGroups=parameter_groups)

            self._fetched_diluents[moleculeName] = set(diluents)
        except Exception:
            traceback.print_exc()

    def _diluents_without_data(self, moleculeName: str, gases: Dict[str, float]) -> List[str]:
        """Gases (de `gases`) cuja coluna `gamma_<gas>` está toda vazia (`#`/NaN)
        na tabela local — ou seja, o HITRAN não tem alargamento medido para essa
        molécula/isotopólogo com esse gás, mesmo que o parâmetro tenha sido baixado."""
        table = LOCAL_TABLE_CACHE.get(moleculeName, {}).get('data', {})
        without_data = []
        for gas in gases:
            column = table.get(f'gamma_{gas.lower()}')
            if column is None or np.all(np.isnan(np.asarray(column, dtype=float))):
                without_data.append(gas)
        return without_data

    def simulateSpectra(self, moleculeName: str, diluent: Dict[str, float], enviroment: Dict[str, float],
                       step: float = 0.01, profile: Profile = 'voigt', quantity: str = 'absorbance',
                       convolve: bool = True, resolution: float = 0.07, af_wing: Optional[float] = None,
                       slit_function: Callable[..., Any] = SLIT_MICHELSON, wavenumber_wing: float = 50) -> None:
        """Simula um espectro e, opcionalmente, convolui com a fenda do instrumento.

        Resultado em `self.spectra['wavenumbers']` / `self.spectra['absorption']`.
        Todos os parâmetros têm padrão razoável — informe só o que quiser mudar.

        Args:
            moleculeName: nome de identificação usado em `downloadMolecule` para essa molécula.
            diluent: fração de cada gás de alargamento (ex.: `{'air': 0.8, 'self': 0.2}`);
                as frações devem somar 1. Chaves devem estar cobertas pelos
                `diluents` já baixados em `downloadMolecule`.
            enviroment: condições físicas do HAPI, ex.: `{'p': 1, 'T': 296, 'l': 10}`
                (pressão em atm, temperatura em K, caminho ótico `l` em cm).
            step: passo em cm⁻¹ do cálculo (`WavenumberStep` do HAPI). Padrão `0.01`.
            profile: perfil de linha — `'voigt'` (padrão), `'lorentz'`, `'doppler'`
                ou `'priority'` (HAPI escolhe automaticamente). Todos reaproveitam
                o mesmo dado `voigt_<gas>` baixado em `downloadMolecule`; perfis
                mais específicos do HITRAN (Hartmann-Tran, Voigt speed-dependent,
                Galatry) não são oferecidos por só terem parâmetro real em
                `air`/`self`, nunca nos demais diluentes.
            quantity: grandeza guardada em `self.spectra['absorption']` —
                `'coefficient'` (cm⁻¹), `'cross_section'` (cm²/molécula),
                `'transmittance'`, `'absorptance'` (1-transmitância),
                `'absorbance'` (profundidade ótica de Beer-Lambert, `-ln(T) = α·l`, padrão) ou `'radiance'`.
            convolve: se `False`, pula a convolução e guarda o espectro cru
                (útil para seção de choque teórica). Padrão `True`.
            resolution: resolução (cm⁻¹) da fenda instrumental (`Resolution` do
                HAPI). Só usado se `convolve=True`. Padrão `0.07`.
            af_wing: alcance (cm⁻¹) do kernel da fenda considerado na convolução
                (`AF_wing` do HAPI). Só usado se `convolve=True`. Padrão `None`
                → `10 * resolution`, escalando automaticamente com a resolução
                pedida (o próprio HAPI usa proporção parecida: `Resolution=0.1`
                com `AF_wing=10.0` por padrão). Isso importa porque `AF_wing`
                **não** escala com `resolution` por conta própria — é um raio
                absoluto do kernel; se você aumentar `resolution` sem também
                aumentar `af_wing`, o kernel da fenda (principalmente de cauda
                longa, como `SLIT_MICHELSON`) fica truncado e a "degradação"
                de resolução não aparece de fato no espectro. Informe um valor
                explícito só se quiser um raio de kernel diferente do padrão.
            slit_function: função de fenda do HAPI (`SlitFunction`). Só usado se
                `convolve=True`. Padrão `SLIT_MICHELSON`; outras opções:
                `SLIT_RECTANGULAR`, `SLIT_TRIANGULAR`, `SLIT_GAUSSIAN`,
                `SLIT_DIFFRACTION`, `SLIT_COSINUS`, `SLIT_DISPERSION`. Todas
                precisam ser importadas explicitamente (`from hapi import
                SLIT_GAUSSIAN`, etc.) — são funções do HAPI, não strings.
            wavenumber_wing: alcance (cm⁻¹) considerado ao redor de cada linha
                no cálculo do perfil (`WavenumberWing` do HAPI). Padrão `50`.

        Exemplos:
            >>> gen.simulateSpectra('HCl', {'air': 0.8, 'self': 0.2}, {'p': 1, 'T': 296, 'l': 10})
            >>> gen.simulateSpectra('HCl', {'self': 1.0}, {'p': 1, 'T': 296, 'l': 10},
            ...                     quantity='cross_section', convolve=False)
        """
        self.diluent_warnings = []
        try:
            if profile not in _PROFILE_FUNCTION_NAMES:
                raise ValueError(f"Perfil desconhecido: {profile!r}. Opções: {sorted(_PROFILE_FUNCTION_NAMES)}")
            if quantity not in _QUANTITIES:
                raise ValueError(f"Grandeza desconhecida: {quantity!r}. Opções: {_QUANTITIES}")

            total_fraction = sum(diluent.values())
            if not np.isclose(total_fraction, 1.0, atol=1e-6):
                raise ValueError(f"As frações de diluent devem somar 1 (soma atual: {total_fraction}). "
                                f"diluent={diluent!r}")

            fetched = self._fetched_diluents.get(moleculeName)
            if fetched is not None:
                missing = set(g.lower() for g in diluent) - fetched
                if missing:
                    raise ValueError(f"Diluentes {sorted(missing)} não foram baixados para '{moleculeName}' "
                                    f"(baixados: {sorted(fetched)}). Chame downloadMolecule com diluents apropriado.")

            without_data = self._diluents_without_data(moleculeName, diluent)
            if without_data:
                self.diluent_warnings = [
                    f"'{gas}' não tem alargamento medido pelo HITRAN para esta molécula/isotopólogo "
                    f"— a contribuição desse gás foi tratada como zero." for gas in without_data
                ]

            absorptionCoefficient = globals()[_PROFILE_FUNCTION_NAMES[profile]]
            wavenumbers, absorptioncoef = absorptionCoefficient(SourceTables=moleculeName,
                                                                Diluent=diluent,
                                                                Environment=enviroment,
                                                                HITRAN_units=(quantity == 'cross_section'),
                                                                WavenumberStep=step,
                                                                WavenumberWing=wavenumber_wing)

            if quantity == 'coefficient' or quantity == 'cross_section':
                absorption = absorptioncoef
            elif quantity == 'transmittance':
                _, absorption = transmittanceSpectrum(wavenumbers, absorptioncoef, Environment=enviroment)
            elif quantity == 'absorptance':
                _, absorption = absorptionSpectrum(wavenumbers, absorptioncoef, Environment=enviroment)
            elif quantity == 'radiance':
                _, absorption = radianceSpectrum(wavenumbers, absorptioncoef, Environment=enviroment)
            else:  # 'absorbance' (default) — profundidade ótica (-ln(T) = α·l), fórmula escrita manualmente.
                transmitance = np.exp(-absorptioncoef * enviroment['l'])
                if np.any(transmitance <= 0):
                    self.diluent_warnings.append(
                        "absorbância saturada: a transmitância chegou a 0 em pelo menos um ponto "
                        "(coeficiente de absorção muito alto e/ou caminho ótico 'l' muito longo) "
                        "— o espectro resultante contém +inf nesses pontos."
                    )
                with np.errstate(divide='ignore'):
                    absorption = -np.log(transmitance)

            if convolve:
                effective_af_wing = af_wing if af_wing is not None else 10 * resolution
                wavenumbers, absorption, _, _, _ = convolveSpectrum(wavenumbers,
                                                                    absorption,
                                                                    SlitFunction=slit_function,
                                                                    Resolution=resolution,
                                                                    AF_wing=effective_af_wing)

            self.spectra['wavenumbers'], self.spectra['absorption'] = wavenumbers, absorption
            self._last_simulation = {'moleculeName': moleculeName, 'quantity': quantity, 'enviroment': enviroment}

        except Exception:
            self.spectra['wavenumbers'], self.spectra['absorption'] = None, None
            self._last_simulation = None
            traceback.print_exc()

    def _plot_ylabel(self) -> str:
        quantity = self._last_simulation['quantity'] if self._last_simulation else None
        return _QUANTITY_LABELS.get(quantity, 'Absorção (A.U)')

    def _plot_title(self) -> Optional[str]:
        if not self._last_simulation:
            return None
        molecule = self._last_simulation['moleculeName']
        condicoes = ', '.join(f"{chave}={valor}" for chave, valor in self._last_simulation['enviroment'].items())
        return f"{molecule} — {condicoes}"

    def plot(self, interactive: bool = False, ax: Optional[Any] = None,
             show: bool = True) -> Optional[Union[Tuple[Any, Any], go.Figure]]:
        """Plota o último espectro simulado — estático (matplotlib) ou interativo (Plotly).

        Args:
            interactive: se `False` (padrão), gera um gráfico estático com
                matplotlib. Se `True`, gera um gráfico interativo com Plotly
                (zoom, pan e hover mostrando o valor exato de cada ponto).
            ax: eixo matplotlib (`plt.Axes`) para plotar; só usado quando
                `interactive=False`. Se `None` (padrão), cria uma figura nova.
            show: se `True` (padrão), exibe a figura (`plt.show()` no modo
                estático, `fig.show()` no interativo). Use `False` para
                compor/salvar a figura você mesmo.

        Returns:
            Modo estático: `(fig, ax)` do matplotlib. Modo interativo:
            `plotly.graph_objects.Figure`. `None` se ainda não há espectro
            simulado (ou a última simulação falhou).

        Exemplo:
            >>> gen.plot()
            >>> fig, ax = gen.plot(show=False)  # pra customizar/salvar antes de exibir
            >>> fig.savefig('espectro.png', dpi=150)
            >>> gen.plot(interactive=True)  # zoom/pan/hover no navegador ou notebook
        """
        try:
            wavenumbers, absorption = self.spectra['wavenumbers'], self.spectra['absorption']
            if wavenumbers is None or absorption is None:
                print("Nenhum espectro simulado ainda (ou a última simulação falhou) — chame simulateSpectra antes de plot().")
                return None

            if interactive:
                fig = go.Figure(go.Scatter(x=wavenumbers, y=absorption, mode='lines',
                                           line=dict(color='#1f6fb2', width=1.4),
                                           fill='tozeroy', fillcolor='rgba(31,111,178,0.08)'))
                fig.update_layout(title=self._plot_title(), xaxis_title='Número de onda (cm⁻¹)',
                                  yaxis_title=self._plot_ylabel(), template='plotly_white')
                if show:
                    fig.show()
                return fig

            fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

            ax.plot(wavenumbers, absorption, color='#1f6fb2', linewidth=1.4)
            ax.fill_between(wavenumbers, absorption, color='#1f6fb2', alpha=0.08)
            ax.set_xlabel('Número de onda (cm⁻¹)')
            ax.set_ylabel(self._plot_ylabel())

            title = self._plot_title()
            if title:
                ax.set_title(title, fontsize=11, color='#333333')

            ax.margins(x=0)
            ax.grid(alpha=0.25, linewidth=0.6)
            fig.tight_layout()

            if show:
                plt.show()

            return fig, ax

        except Exception:
            traceback.print_exc()
            return None
