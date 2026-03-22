import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --- CONSTANTES ---
CANTIDAD_CORRIDAS = 10

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


def procesar_y_graficar(archivo_csv):
    df = pd.read_csv(archivo_csv, sep=';')

    nombres_escenarios = ['Sin líder', 'Líder dirección fija', 'Líder circular']
    cortes = [0] + df.index[df['eta'].diff() < 0].tolist() + [len(df)]

    if len(cortes) != len(nombres_escenarios) + 1:
        raise ValueError("Cantidad de escenarios incorrecta")

    resultados_por_escenario = []

    for i in range(len(nombres_escenarios)):
        df_escenario = df.iloc[cortes[i]:cortes[i+1]]
        etas_unicos = df_escenario['eta'].unique()
        resultados = []

        print(f"\n--- {nombres_escenarios[i].upper()} ---")
        for eta in etas_unicos:
            df_eta = df_escenario[df_escenario['eta'] == eta].head(CANTIDAD_CORRIDAS)

            va_promedio = df_eta['va_mean'].mean()
            va_desvio = df_eta['va_mean'].std() / np.sqrt(len(df_eta))

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

        plt.errorbar(etas, vas, yerr=errores, fmt=f'{marcadores[i]}-', capsize=4, markersize=3,
                     color=colores[i], label=nombres_escenarios[i])

    plt.xlabel(r'Ruido, $\eta$')
    plt.ylabel(r'Polarización, $v_a$')
    plt.title('Polarización en función del ruido para los 3 escenarios')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparativa_polarizacion.pdf')
    plt.show()

    # --- GRÁFICOS SEPARADOS ---
    for i, resultados in enumerate(resultados_por_escenario):
        plt.figure(figsize=(8, 5))

        etas = [r['eta'] for r in resultados]
        vas = [r['va_promedio'] for r in resultados]
        errores = [r['va_desvio'] for r in resultados]

        plt.errorbar(etas, vas, yerr=errores, fmt=f'{marcadores[i]}-', capsize=4, markersize=3,
                     color=colores[i])

        plt.xlabel(r'Ruido, $\eta$')
        plt.ylabel(r'Polarización, $v_a$')
        plt.title(f'Polarización vs ruido - {nombres_escenarios[i]}')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        nombre_archivo = nombres_escenarios[i].lower().replace(' ', '_') + ".pdf"
        plt.savefig(nombre_archivo)
        plt.show()


from pathlib import Path

# Buscar analysis.csv relativo al repo como antes
repo_root = Path(__file__).resolve().parents[4]
analysis_path = repo_root / "tp2-output" / "analysis.csv"

procesar_y_graficar(analysis_path)
