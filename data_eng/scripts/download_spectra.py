"""Baixa dados de linha do HITRAN pra uma molécula, salvando no cache local
(`database/hitran_database/`, compartilhado com `spectra-generator.py`).

Uso:
    uv run --package data-eng python data_eng/scripts/download_spectra.py \\
        --name hcl --isotopes 52,53 --range 5200,5900 --diluents air,self
"""

import argparse
import logging
import os

from SpectraHitran.SpectraGenerator.spectragenerator import SpectraGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('download-spectra')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                         help='Nome de identificação da molécula (ex.: hcl). Vira o nome da tabela local.')
    parser.add_argument('--isotopes', type=lambda s: tuple(int(x) for x in s.split(',')), required=True,
                         help='IDs globais de isotopólogo do HITRAN, separados por vírgula (ex.: 52,53).')
    parser.add_argument('--range', type=lambda s: tuple(float(x) for x in s.split(',')), required=True,
                         metavar='MIN,MAX', help='Faixa de número de onda em cm⁻¹ (ex.: 5200,5900).')
    parser.add_argument('--diluents', type=lambda s: tuple(g.strip().lower() for g in s.split(',')),
                         default=('air', 'self'),
                         help='Gases diluentes a baixar, separados por vírgula (padrão: air,self).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        f"Baixando '{args.name}' — isótopos={args.isotopes}, faixa={args.range} cm⁻¹, "
        f"diluentes={args.diluents}"
    )

    generator = SpectraGenerator()
    data_file = generator.database_path + f'/{args.name}.data'

    generator.downloadMolecule(args.name, args.isotopes, args.range, diluents=args.diluents)

    if os.path.exists(data_file):
        logger.info(f"Concluído: '{args.name}' salvo em {data_file}")
    else:
        logger.error(
            f"Download de '{args.name}' falhou — {data_file} não foi criado "
            f"(ver traceback impresso acima pelo HAPI)."
        )


if __name__ == "__main__":
    main()
