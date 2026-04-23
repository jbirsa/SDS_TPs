# TP3 Context

## Repo overview

- `tp1-code`, `tp1-visual`, `tp1-output`: TP1.
- `tp2-code`, `tp2-visual`, `tp2-output`: TP2.
- `tp3-sds`: motor de simulacion del TP3.
- `tp3-visual`: visualizador Python del TP3 que genera GIFs.
- `tp3-output`: carpeta esperada para los outputs del TP3, al mismo nivel que `tp1-output` y `tp2-output`.

## Objetivo del TP3

Este TP implementa el Sistema 2 del enunciado: filas inteligentes en un recinto de `30 x 30 m` con servidores sobre un borde, llegadas Poisson con tiempo medio `t1`, tiempos de servicio exponenciales con media `t2` y simulacion dirigida por eventos.

La simulacion debe escribir snapshots del sistema en un archivo de texto. La animacion debe correr aparte a partir de esos archivos.

## Como correrlo

Desde `tp3-sds`:

```bash
javac -d out src/simulation/*.java
java -cp out simulation.Main
```

Los archivos se escriben en `../tp3-output/` cuando la ejecucion arranca desde `tp3-sds`, o en `tp3-output/` si se corre desde la raiz del repo.

Se puede overridear el directorio con la property Java `output.dir`.

## Como visualizarlo

Desde la raiz del repo:

```bash
python3 tp3-visual/src/main/python/visualizer.py --input tp3-output/archivo.txt
```

Si no se pasa `--input`, el script toma el `.txt` mas reciente de `tp3-output` y genera un GIF en `tp3-visual/graphs/`.

## Arquitectura actual

### Clases principales

- `src/simulation/Main.java`: arma y ejecuta los casos de estudio.
- `src/simulation/Simulation.java`: motor principal de eventos.
- `src/simulation/SimulationConfig.java`: configuracion inmutable de una corrida.
- `src/simulation/OutputWriter.java`: escribe snapshots y estadisticas.
- `src/simulation/Statistics.java`: promedios temporales de cola y tiempos de permanencia.

### Entidades y soporte

- `Client.java`, `ClientState.java`
- `Server.java`
- `Position.java`
- `Queue.java`
- `QueueAdvancement.java`
- `ServerAssigner.java`

### Tipos de evento

- `ClientArrivalEvent`
- `ClientArrivesAtQueueSpotEvent`
- `ClientAdvancesInQueueEvent`
- `ClientArrivesAtServerEvent`
- `ServiceCompletionEvent`

## Requisitos del TP y estado actual

### Modalidades

- Modalidad A: una cola por servidor.
- Modalidad B: una cola compartida para todos los servidores.

Estado: implementado en `Simulation.java`.

### Tipos de fila

- `SERPENTINE`: fila guiada en zig-zag.
- `LINE`: hoy representa la fila libre heuristica, no una linea vertical fija.

Estado:

- La fila guiada existe en `GuidedSerpentineQueue.java`.
- La fila libre ahora arma los puestos incrementalmente:
  - el primer cliente queda recto al servidor
  - cada nuevo puesto sale a 1 m del anterior
  - la direccion se obtiene rotando el segmento previo con un angulo aleatorio acotado
  - se mantiene dentro del recinto y con separacion minima respecto de los puestos ya generados

## Distancia entre clientes en la fila

Si. La distancia base entre clientes se modela con `1.0 m` entre puestos consecutivos.

### Como se calcula hoy

- En fila guiada, los puestos estan definidos sobre una grilla de 1 m.
- En fila libre, cada nuevo puesto se genera a distancia exacta `1.0 m` del anterior.
- El tiempo de desplazamiento siempre se calcula como:

```text
tiempo = distancia euclidea / walkingSpeed
```

con `walkingSpeed = 1.0 m/s` por defecto.

Esto aplica a:

- cliente nuevo -> posicion actual a tomar de la cola asignada
- primer cliente -> servidor
- avance de cada cliente al puesto anterior cuando se libera un lugar

Ademas, el motor actualiza la posicion interpolada de todos los clientes que estan caminando en cada tiempo de evento antes de procesarlo. Asi los outputs ya muestran movimiento parcial de toda la fila entre eventos.

Importante: la cola fisica y los clientes aproximandose ahora se modelan por separado. Un cliente no ocupa un `queueSpotIndex` al crearse; solo lo recibe cuando llega efectivamente a la `posicion a tomar` de su cola. Cada vez que esa posicion cambia, se replanifica el trayecto de todos los clientes que todavia estan aproximandose a esa cola.

## Asignacion de servidores

La asignacion de servidor esta en `ServerAssigner.java`.

Score actual por servidor `i`:

```text
score_i = 0.6 * distanciaNormalizada_i + 0.4 * cargaNormalizada_i
```

Se elige en forma deterministica el servidor con menor score. La carga incluye la cola fisicamente ocupada, los clientes ya asignados pero todavia aproximandose a esa cola y un extra cuando el servidor esta reservado u ocupado.

## Salidas

Cada corrida genera un archivo `out_*.txt` en `tp3-output/`.

El archivo contiene:

- un frame por evento
- posiciones y estado de clientes
- estado de servidores
- bloque final `STATS`
- lista de tiempos de permanencia

Los numeros se escriben con `Locale.US` para mantener punto decimal estable.

## Casos de estudio

`Main.java` corre:

- 2.1 variando `t2`
- 2.2 variando `t1`
- 2.3 variando `k`
- 2.4 repitiendo lo mismo para la modalidad B

Importante: ahora A y B usan el mismo `QueueType` para que la comparacion entre modalidades no mezcle tambien el tipo de fila. Ese tipo de fila se controla con la constante `STUDY_QUEUE_TYPE`.

Ademas se generan dos corridas de animacion de `300 s` para mostrar ambos tipos de fila.

## Pendientes reales del TP

Estos puntos siguen abiertos o deben validarse mejor antes de una entrega final:

1. La geometria guiada esta hardcodeada. El enunciado dice que deberia tratarse como input.
2. La visualizacion existe en `tp3-visual`, pero todavia no hay postprocesamiento avanzado ni exportes adicionales mas alla del GIF principal.
3. El codigo escribe largos promedio de cola, pero no detecta automaticamente si el sistema llego a estacionario o si hay que reportar tasa de crecimiento.
4. Falta postprocesamiento para comparar distribuciones y promedios entre corridas de forma automatica.
5. Conviene validar visualmente que la heuristica de fila libre no genere formas indeseadas en escenarios extremos.

## Checklist rapido para cualquier cambio futuro

1. Compilar con `javac -d out src/simulation/*.java`.
2. Correr un caso corto y verificar que aparezcan archivos en `tp3-output/`.
3. Confirmar que modalidad A siga siendo una cola por servidor.
4. Confirmar que modalidad B siga siendo una cola compartida.
5. Confirmar que la fila libre mantenga `1 m` entre puestos consecutivos.
6. Confirmar que los tiempos en output usen punto decimal.
7. Si se toca estadistica, revisar el promedio temporal del largo de cola con un caso simple y trazable.
