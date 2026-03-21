import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --- CONSTANTES ---
CANTIDAD_CORRIDAS = 10 

def truncar_y_formatear(media, desvio):
    """
    Trunca y formatea el promedio y el desvío estándar según la posición 
    del primer dígito distinto de cero del error.
    """
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
    # 1. Cargar datos
    df = pd.read_csv(archivo_csv, sep=';')
    
    # 2. Detectar automáticamente los cortes entre los 3 escenarios
    # (buscamos dónde el valor de 'eta' actual es menor al anterior, indicando un reinicio a 0)
    nombres_escenarios = ['Sin líder', 'Líder dirección fija', 'Líder circular']
    cortes = [0] + df.index[df['eta'].diff() < 0].tolist() + [len(df)]

    if len(cortes) != len(nombres_escenarios) + 1:
        raise ValueError(
            f"Se esperaban {len(nombres_escenarios)} escenarios, pero se detectaron {len(cortes) - 1} bloques en {archivo_csv}."
        )

    resultados_por_escenario = []
    
    # 3. Procesar datos para cada escenario
    for i in range(len(nombres_escenarios)):
        # Recortar el DataFrame para el escenario actual
        df_escenario = df.iloc[cortes[i]:cortes[i+1]]
        etas_unicos = df_escenario['eta'].unique()
        resultados = []
        
        print(f"\n--- PROCESANDO: {nombres_escenarios[i].upper()} ---")
        for eta in etas_unicos:
            # Filtrar por eta y limitar a las corridas pedidas
            df_eta = df_escenario[df_escenario['eta'] == eta].head(CANTIDAD_CORRIDAS)

            va_promedio = df_eta['va_mean'].mean()
            va_desvio = df_eta['va_mean'].std() / np.sqrt(len(df_eta)) # Error estándar del promedio (barras de error)
            
            resultados.append({
                'eta': eta,
                'va_promedio': va_promedio,
                'va_desvio': va_desvio,
            })
            
            # Imprimir consola con la regla de truncamiento
            texto_formateado = truncar_y_formatear(va_promedio, va_desvio)
            print(f"Eta: {eta:.2f} -> Polarización: {texto_formateado}")
            
        resultados_por_escenario.append(resultados)

    # 4. Configurar el Gráfico Comparativo (Inciso D)
    plt.figure(figsize=(9, 6))
    
    # Colores y marcadores distintos para cada escenario para cumplir con la guía de destacar los datos
    colores = ['blue', 'green', 'red']
    marcadores = ['o', 's', '^'] # círculo, cuadrado, triángulo
    
    # Graficar cada escenario en la misma figura
    for i, resultados in enumerate(resultados_por_escenario):
        etas = [r['eta'] for r in resultados]
        vas = [r['va_promedio'] for r in resultados]
        errores = [r['va_desvio'] for r in resultados]
        
        # fmt incluye el marcador y la línea recta '-' sin interpolar
        plt.errorbar(etas, vas, yerr=errores, fmt=f'{marcadores[i]}-', capsize=4, 
                     color=colores[i], label=nombres_escenarios[i])

    # Leyendas, convenciones de tipografía para escalares y formato del gráfico
    plt.xlabel(r'Ruido, $\eta$', fontsize=12)
    plt.ylabel(r'Polarización, $v_a$', fontsize=12)
    plt.title('Polarización en función del ruido para los 3 escenarios', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Guardar y mostrar
    plt.savefig('comparativa_polarizacion.pdf')
    plt.show()

# Ejecutar la función
procesar_y_graficar('analysis.csv')
