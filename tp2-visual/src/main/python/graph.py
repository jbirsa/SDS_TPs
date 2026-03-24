import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

# --- CONSTANTES ---
CANTIDAD_CORRIDAS = 50
USAR_DESVIO_ESTANDAR = True

def truncar_y_formatear(media, desvio):
    if desvio == 0 or math.isnan(desvio):
        return f"{media:.3f} ± 0.000"

    orden_magnitud = math.floor(math.log10(abs(desvio)))
    primer_digito = int(abs(desvio) / (10**orden_magnitud))

    if primer_digito == 1:
        orden_magnitud -= 1

    decimales = -orden_magnitud if orden_magnitud < 0 else 0
    err_redondeado = round(desvio, decimales)
    media_redondeada = round(media, decimales)

    if decimales > 0:
        formato = f"{{:.{decimales}f}} ± {{:.{decimales}f}}"
    else:
        formato = f"{{:.0f}} ± {{:.0f}}"

    return formato.format(media_redondeada, err_redondeado)


def procesar_y_graficar(archivo_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(archivo_csv, sep=';')

    nombres_base = ['Sin líder', 'Líder dirección fija', 'Líder circular']
    cortes = [0] + df.index[df['eta'].diff() < 0].tolist() + [len(df)]

    cantidad_bloques = len(cortes) - 1
    if cantidad_bloques <= 0:
        raise ValueError("No se detectaron bloques de escenarios")

    if cantidad_bloques <= len(nombres_base):
        nombres_escenarios = nombres_base[:cantidad_bloques]
    else:
        nombres_escenarios = nombres_base + [
            f"Escenario {i + 1}" for i in range(len(nombres_base), cantidad_bloques)
        ]

    resultados_por_escenario = []

    for i in range(len(nombres_escenarios)):
        df_escenario = df.iloc[cortes[i]:cortes[i+1]]
        etas_unicos = df_escenario['eta'].unique()
        resultados = []

        print(f"\n--- {nombres_escenarios[i].upper()} ---")
        for eta in etas_unicos:
            df_eta = df_escenario[df_escenario['eta'] == eta].head(CANTIDAD_CORRIDAS)
            if df_eta.empty:
                continue

            va_promedio = df_eta['va_mean'].mean()
            if USAR_DESVIO_ESTANDAR:
                va_desvio = df_eta['va_mean'].std(ddof=1)
            else:
                va_desvio = df_eta['va_mean'].std(ddof=1) / np.sqrt(len(df_eta))
            if np.isnan(va_desvio):
                va_desvio = 0.0

            resultados.append({
                'eta': eta,
                'va_promedio': va_promedio,
                'va_desvio': va_desvio,
            })

            texto = truncar_y_formatear(va_promedio, va_desvio)
            print(f"Eta: {eta:.2f} -> Polarización: {texto}")

        resultados_por_escenario.append(resultados)

    # --- GRÁFICO COMBINADO ---
    plt.figure(figsize=(9, 6))
    colores = ['blue', 'green', 'red']
    marcadores = ['o', 's', '^']

    for i, resultados in enumerate(resultados_por_escenario):
        etas = [r['eta'] for r in resultados]
        vas = [r['va_promedio'] for r in resultados]
        errores = [r['va_desvio'] for r in resultados]

        plt.errorbar(
            etas,
            vas,
            yerr=errores,
            fmt=f'{marcadores[i % len(marcadores)]}-',
            capsize=4,
            markersize=3,
            color=colores[i % len(colores)],
            label=nombres_escenarios[i],
        )

    plt.xlabel(r'Ruido, $\eta$')
    plt.ylabel(r'Polarización, $v_a$')
    plt.title('Polarización en función del ruido')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    comparativa_path = output_dir / "polarization_noise.png"
    plt.savefig(comparativa_path)
    plt.close()

    print(f"Grafico comparativo generado: {comparativa_path}")


repo_root = Path(__file__).resolve().parents[4]
analysis_path = repo_root / "tp2-output" / "analysis_50_runs.csv"
if not analysis_path.exists():
    analysis_path = repo_root / "tp2-visual" / "src" / "main" / "python" / "analysis.csv"

output_dir = repo_root / "tp2-visual" / "graphs"

procesar_y_graficar(analysis_path, output_dir)
