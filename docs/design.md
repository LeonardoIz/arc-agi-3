# Diseño del agente y bitácora de decisiones

Documento autoritativo del proyecto: qué sabemos del entorno (verificado en
código, no supuesto), qué contrato nos autoimpusimos, qué arquitectura sale
de él, y en qué orden construirlo.

`docs/model.md` describe el scaffold de `model/` **tal como está hoy**; parte
de esa arquitectura queda superada por este documento (ver §4.3).

---

## 1. Objetivo

Maximizar el score en ARC Prize 2026 / ARC-AGI-3. El benchmark son juegos
interactivos que hay que completar **sin instrucciones, sin objetivos
declarados y sin tutorial**. El set de juegos de la evaluación oculta es
**distinto** de los 25 públicos, así que el problema no es "aprender a jugar
estos juegos" sino "aprender a descifrar un juego desconocido rápido".

---

## 2. Hallazgos del entorno

Todo lo de esta sección está verificado leyendo el código instalado, con
referencia al archivo y la línea. No es documentación oficial ni suposición.

### 2.1 Fórmula de scoring

De `arc_agi/scorecard.py:170` y `:475-491`:

```
score_nivel = min( (baseline_actions / acciones_usadas)² × 100 , 115 )

score_juego = Σ(score_nivel_i × peso_i) / Σ(pesos_i)      con peso_i = i (1..N)
              acotado superiormente por
              (Σ pesos de niveles completados / Σ pesos) × 100

score_total = promedio simple sobre todos los juegos      (scorecard.py:614)
```

Detalles que importan:

- El bucle recorre **todos** los niveles del juego (`for level_idx in
  range(len(env_info.baseline_actions))`), no solo los intentados. Los
  niveles nunca alcanzados entran con score 0 y suman al denominador. No
  existe "no intentar" como forma de evitar la penalización.
- `level_index = level_idx + 1`, así que los pesos son 1, 2, 3… N: **los
  niveles tardíos valen mucho más**.
- `baseline_actions` sale de `EnvironmentInfo` y es un valor por nivel.

### 2.2 Implicaciones estratégicas

**La profundidad marca el techo; la eficiencia es un multiplicador ≤ 1.**
El `min(score, max_score)` impide que superar el baseline sume puntos extra.
El techo real de un juego es la fracción ponderada de niveles completados;
la ineficiencia solo puede bajarte de ahí, nunca subirte.

**La penalización por ineficiencia es cuadrática.** Usar 3× el baseline en un
nivel deja 11 puntos de 100.

**Explorar es barato al principio y caro al final.** En LS20 (7 niveles,
Σpesos = 28) el nivel 1 pesa 1/28 = 3,6 % del juego, y el nivel 7 pesa
7/28 = 25 %. Como el conocimiento del juego se adquiere una vez y sirve para
todos los niveles, el scoring premia estructuralmente una curva de
"aprender temprano y barato, ejecutar tarde y limpio". Ejemplos con LS20:

| Qué logra el agente | Score del juego |
|---|---|
| Nivel 1 completado exactamente en el baseline | 3,6 |
| Niveles 1-3 completados en el baseline | 21,4 |
| Nivel 1 completado usando 3× el baseline | 0,4 |
| Los 7 niveles en el baseline | 100 |

### 2.3 Semántica de RESET

`arcengine/base_game.py:305-316`: después de la primera acción, `RESET`
ejecuta **`level_reset`, no `full_reset`**. Morir y reiniciar **no pierde el
progreso de niveles anteriores**; solo cuesta acciones, que se acumulan al
nivel en curso y bajan su score cuadráticamente. Reintentar es viable; es
caro solo en eficiencia.

### 2.4 Qué observa el agente

`FrameData` (`arcengine/enums.py:131`) entrega por paso:

| Campo | Contenido |
|---|---|
| `frame` | Lista de grids 64×64 de índices de color (0-15). Es una lista porque una acción puede producir varios frames de animación; el último es el estado asentado. |
| `state` | `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER` |
| `levels_completed` | Contador de progreso. **Es la única señal de recompensa que existe.** |
| `win_levels` | Total de niveles del juego |
| `available_actions` | Qué acciones son legales en este juego |
| `full_reset` | Si el último reset fue completo |

Cada juego declara `available_actions` en su constructor y viene en cada
frame (`base_game.py:248`). Ejemplos reales: LS20 = `[1,2,3,4]` (solo
direccionales), FT09/LP85/R11L = `[6]` (solo click). Es un prior gratuito
que poda el espacio de acciones.

El motor renderiza con escalado entero + letterbox (`arcengine/camera.py`,
`MAX_DIMENSION = 64`), así que el frame de 64×64 puede ser el upscale de un
tablero lógico más chico. FT09 por ejemplo es un grid 16×16 escalado ×4.

### 2.5 Inventario de juegos públicos

25 juegos, 6 a 10 niveles cada uno:

| Juego | Niveles | Σ baseline | Modalidad |
|---|---|---|---|
| ar25 | 8 | 748 | keyboard_click |
| bp35 | 9 | 651 | keyboard_click |
| cd82 | 6 | 171 | keyboard_click |
| cn04 | 6 | 789 | keyboard_click |
| dc22 | 6 | 1228 | keyboard_click |
| ft09 | 6 | 208 | (sin tag) |
| g50t | 7 | 879 | keyboard |
| ka59 | 7 | 730 | keyboard_click |
| lf52 | 10 | 1339 | click |
| lp85 | 8 | 388 | click |
| ls20 | 7 | 776 | keyboard |
| m0r0 | 6 | 1107 | keyboard_click |
| r11l | 6 | 233 | click |
| re86 | 8 | 1255 | keyboard_click |
| s5i5 | 8 | 638 | click |
| sb26 | 8 | 213 | keyboard_click |
| sc25 | 6 | 350 | keyboard_click |
| sk48 | 8 | 1070 | keyboard_click |
| sp80 | 6 | 518 | keyboard_click |
| su15 | 9 | 361 | click |
| tn36 | 7 | 317 | click |
| tr87 | 6 | 414 | keyboard |
| tu93 | 9 | 462 | keyboard_click |
| vc33 | 7 | 447 | click |
| wa30 | 9 | 1843 | keyboard |

El tamaño del código fuente va de 777 líneas (cd82) a 41.463 (ka59). No son
juguetes: `ka59`, `lp85` (21k) y `dc22` (11k) tienen mecánicas profundas.

### 2.6 Incertidumbres no resueltas

- **No encontré un tope global de acciones** en el paquete `arc_agi` local.
  El límite en Kaggle es probablemente el tiempo de ejecución del notebook.
  `MAX_ACTIONS` en el framework es autoimpuesto por el agente, no del
  servidor. **Hay que medirlo empíricamente.**
- No verifiqué si la UI web le muestra al jugador humano el conjunto de
  acciones disponibles. Asumimos que sí (§3, C1).

---

## 3. El contrato

Nos autoimponemos estas reglas. Dos motivos: que el agente juegue con la
misma información que un humano en la web, y que la inteligencia esté en el
modelo y no en el código que lo rodea.

**Principio rector — plomería sí, decisiones no.** Es legítimo el código que
traduce (píxeles → tensor, salida → `GameAction`) y el que transporta. Es
ilegítimo el código que *elige*: qué probar, cuándo dejar de explorar, qué
priorizar, en qué orden.

**Test de cumplimiento:** si borrás el modelo, el agente no debe poder jugar
en absoluto. Si sin modelo todavía "explora razonablemente", la exploración
estaba en el código y violamos el contrato.

| # | Cláusula |
|---|---|
| **C1** | **Entrada.** El modelo consume exclusivamente `FrameData`: grid 64×64, `state`, `levels_completed`, `win_levels`, `available_actions`. Nada de `Sprite`, estado interno del motor, código fuente del juego ni `EnvironmentInfo`. |
| **C2** | **Sin identidad.** El modelo nunca recibe `game_id` ni ningún identificador de juego. Memorizar por juego queda estructuralmente imposible. |
| **C3** | **El modelo decide.** Toda elección de acción sale del modelo, incluida la estrategia de exploración. Cero secuencias cableadas, cero reglas tipo "si nada cambió, probá otra acción". |
| **C4** | **Memoria en el modelo.** Lo aprendido durante el episodio vive en el estado del modelo (contexto o recurrencia), no en estructuras que mantiene código externo. |
| **C5** | **Sin simulador en inferencia.** Prohibido clonar o forkear el entorno para probar futuros. En Kaggle es imposible igual (solo hay gateway HTTP), pero no dependemos de eso. |
| **C6** | **Entrenamiento ≠ inferencia.** Entrenar jugando los públicos y los sintéticos es legítimo. Usar el simulador como maestro privilegiado *durante el entrenamiento* también, siempre que el alumno que corre en test solo vea C1. |
| **C7** | **Validación honesta.** N juegos públicos reservados, nunca entrenados, cuyo código fuente no se usa ni para diseñar el generador procedural. Es la única métrica que nos creemos. |
| **C8** | **`baseline_actions` excluido.** Al jugador web no le muestran cuántas acciones "debería" llevar un nivel; es metadata de scoring. No se usa ni para presupuestar. |
| **C9** | **Sin búsqueda en inferencia.** El modelo emite la acción directamente. No hay planificador, ni MCTS, ni rollouts. *(Decisión tomada explícitamente; ver §8 el costo que tiene.)* |
| **C10** | **Percepción aprendida.** El modelo aprende a descomponer la escena en objetos (slots). Prohibida la segmentación cableada a mano que le entregue objetos ya formados. |

---

## 4. Arquitectura

### 4.1 El agente: una capa fina

`agent/my_agent.py` **no contiene inteligencia**. Es un adaptador entre el
framework del juego y un backend de decisión intercambiable:

```
ARC-AGI-3 framework
        │  FrameData
        ▼
   MyAgent  ── solo traduce y transporta ──►  Policy (backend)
        ▲                                        │
        └────────── GameAction ◄─────────────────┘
```

Backends que debe soportar la misma interfaz:

- `ModelPolicy` — el modelo PyTorch entrenado localmente.
- Un backend contra una API de LLM grande (útil como referencia temprana y
  como baseline no-cero mientras el modelo propio no existe).
- Cualquier otro, sin tocar el agente.

Interfaz mínima:

```python
class Policy(Protocol):
    def act(self, frames, latest_frame) -> GameAction: ...
    def reset(self) -> None: ...   # limpia la memoria del episodio
```

`reset()` es **nuevo y obligatorio** por C4: como el modelo acumula memoria
del episodio, hay que limpiarla entre juegos. El `ModelPolicy` actual no lo
tiene porque no tenía memoria.

### 4.2 El modelo

Por C3 + C4 + C9 + C10, la forma queda determinada:

```
frame 64×64 (índices de color)
    → [aprendido] encoder de slots: la escena se descompone en K objetos
    → [aprendido] modelo secuencial sobre el episodio:
                  contexto = historial de (slots, acción previa,
                             Δlevels_completed, flags de estado)
    → [aprendido] distribución sobre acciones legales + (x,y) para ACTION6
    → muestrear / argmax → GameAction
```

Todo lo aprendido está dentro del modelo. La estrategia de exploración —
probar acciones al principio, notar qué cambió, cambiar a explotación — no
se programa: **tiene que emerger del entrenamiento**.

### 4.3 Qué sobrevive del scaffold actual

| Módulo | Estado |
|---|---|
| `model/config.py` | **Hecho.** Parámetros de slots y del transformer agregados. |
| `model/adapters.py` | **Hecho.** Sigue siendo la única frontera con `arcengine`; construye la `Observation` sobre el episodio completo (no una ventana fija). |
| `model/inference.py` | **Hecho.** `Policy` extraído como interfaz (`act` + `reset()`); `ModelPolicy.reset()` es no-op hasta que el modelo necesite limpiar estado propio. |
| `model/data.py` | Sobrevive sin cambios. |
| `model/network.py` | **Hecho.** Encoder de slots (Slot Attention) + transformer causal con RoPE sobre el episodio. Sin `ValueHead`: el método de entrenamiento (§5.2) es behavior cloning puro, nada consume un value estimate. Ver el docstring del módulo para el detalle de cada pieza. |
| `agent/my_agent.py` | **Hecho.** Se sacó el fork por `game_id` de LS20 (violaba C2). También filtra el fallback aleatorio por `available_actions` (ver hallazgo abajo) y parchea un bug del framework vendorizado. |
| `model/train.py` | **Hecho para el entregable de esta fase.** `policy_loss` (behavior cloning, cross-entropy enmascarada) y el loop de `main()` implementados. Entrena de a un episodio grabado por paso — sin batching entre episodios todavía (Fase 3). |

**Entregable de Fase 2 medido:** el modelo sobreajusta un episodio grabado de
`ls20` (80 pasos, baseline aleatorio) — loss cae de 1.50 a 0.0001 en 500
pasos, y en modo determinista reproduce el 100% de las acciones de la
trayectoria memorizada. Arquitectura confirmada entrenable de punta a punta.

**Hallazgo:** `agents.agent.Agent._convert_raw_frame_data` (framework
vendorizado) descarta `action_input` al construir el `FrameData` que ve el
agente, aunque `arcengine` sí lo rellena — sin este dato, `model/adapters.py`
no puede saber qué acción produjo cada frame (todo se ve como "inicio de
episodio"), y toda grabación queda con ese hueco. Parcheado en tiempo de
import desde `agent/my_agent.py` (no en `vendor/`, que `make setup`
regenera y que ni siquiera existe en Kaggle) — ver el docstring de
`_patch_frame_data_action_input` ahí.

**Hallazgo:** el fallback aleatorio de `my_agent.py` elegía entre las 7
acciones sin filtrar por `available_actions`, a diferencia de la red (que sí
enmascara). Una trayectoria grabada con ese fallback contiene acciones que
el juego declaró ilegales en ese paso, y entrenar contra ellas pide
log-verosimilitud de una clase enmascarada a `-inf` → loss infinita. Corregido
en el fallback (ahora filtra por `available_actions`, igual que la red) y
además `build_training_example` descarta defensivamente cualquier target
ilegal que se cuele por otra vía.

### 4.4 La longitud del episodio — resuelto

Con baselines de hasta 1843 acciones por juego (wa30), atención completa
sobre el episodio entero es cara. Se evaluaron tres opciones: estado
recurrente (GRU), SSM tipo Mamba, y transformer causal con atención
completa. Se descartó GRU (capacidad de memoria más limitada en tareas de
contexto largo) y Mamba (dependencia de kernels CUDA específicos, riesgo de
fricción de instalación ya visto una vez con Blackwell/WSL, y su ventaja
frente a atención completa es de costo — no de calidad).

**Decisión: transformer causal con atención completa y RoPE** (sin
ventaneo). Es también la arquitectura que usa Algorithm Distillation en su
formulación original (Laskin et al.), el método de entrenamiento ya elegido
en §5.2. Justificado porque, en este proyecto, ni el tiempo de entrenamiento
ni la VRAM en inferencia son un factor limitante (si la GPU actual — RTX
5060, 8 GB — se vuelve el cuello de botella, la vía de escape es upgradear
hardware, no recortar arquitectura). RoPE en vez de posiciones absolutas
aprendidas porque no fija una longitud máxima de contexto de antemano.

Sin KV-cache por ahora: cada paso de inferencia recalcula el forward pass
sobre el episodio completo hasta ese punto — O(T) trabajo redundante por
paso, aceptable mientras el objetivo sea corrección antes que velocidad. Es
una optimización aislada en `model/network.py` si hace falta después.

---

## 5. Entrenamiento

### 5.1 Por qué la generación procedural es un prerequisito

C3 exige que el modelo decida *cómo explorar*. Eso significa que tiene que
**haber aprendido a explorar**, y no se aprende con 25 juegos: con 25, la red
memoriza cómo abordar esos 25. Para que emerja una estrategia general hace
falta una distribución de tareas lo bastante amplia como para que memorizar
sea peor que adaptarse.

Tenemos el motor `arcengine` y 25 juegos como referencia, así que podemos
generar cientos o miles de mini-juegos variando sprites, layouts, mecánicas y
modalidad. **Esto pasa de ser una optimización a ser la ruta crítica del
proyecto.**

### 5.2 Método propuesto

Algorithm Distillation: entrenar el modelo secuencial sobre las *trayectorias
de aprendizaje* de agentes resolviendo tareas de la distribución, de modo que
el modelo aprenda el operador de mejora en contexto en vez de una política
fija. C6 permite usar el simulador como maestro privilegiado offline para
generar esas trayectorias, siempre que el modelo desplegado solo vea C1.

### 5.3 Validación

Reservar juegos públicos que **nunca** se entrenan y cuyo código no se mira
ni para diseñar el generador. Si el modelo puntúa ahí, transfiere de verdad.
Si solo puntúa en los de entrenamiento, nos estábamos autoengañando y lo
detectamos antes de gastar una submission.

---

## 6. Entorno de desarrollo

- **GPU:** RTX 5060 Laptop, 8 GB VRAM, Blackwell (`sm_120`).
- **WSL2** Ubuntu 26.04, con passthrough de GPU funcionando.
- **PyTorch:** `2.13.0+cu129`, verificado con matmul en GPU. cu129 es
  necesario porque Blackwell requiere CUDA ≥ 12.8.
- **Python 3.14** en el venv del proyecto.

**Trampa encontrada, ya arreglada en el Makefile:** en WSL, `/tmp` es un
`tmpfs` de 2 GB en RAM. pip estaciona ahí los ~3,5 GB de wheels de CUDA antes
de instalarlos, y la instalación muere a mitad con `ENOSPC` — que pip reporta
como "problema de conectividad de red", diagnóstico incorrecto que hace
perder tiempo. El Makefile ahora usa `TMPDIR` en disco real y fija
`TORCH_INDEX` a cu129 (`make setup TORCH_INDEX=.../cpu` para máquinas sin
GPU).

**Hecho:** el repo se migró a `~/arc-agi-3` dentro de WSL (ext4 nativo, no
`/mnt/c` vía drvfs). `.kaggle/access_token` copiado a mano;
`environment_files/` se re-descargó sola en el primer `make play-local`.

---

## 7. Plan por fases

Cada fase tiene que terminar en un número medido, no en una sensación.

**Fase 0 — Instrumentación. ✅ Hecho.** Evaluador offline (`scripts/evaluate.py`)
que reproduce el score oficial por nivel reutilizando la fórmula de
`arc_agi.scorecard`. `Policy` refactorizado para backends intercambiables.
Entregable medido: score del baseline aleatorio en los 25 juegos públicos =
**0.00**.

**Fase 1 — Generador procedural. ✅ Hecho** (ver `docs/procedural.md` para el
diseño completo). Motor determinístico propio (`procedural/`, no LLM por
ahora) con 12 mecánicas primarias (9 direccionales + 3 por click) y 3
modificadores de peligro opcionales, validadas por un solver BFS antes de
aceptarlas. Entregable medido: 200 juegos generados, las 12 mecánicas
usadas, 34 combinaciones (mecánica × peligro) distintas, 139
direccionales / 61 por click, todos verificados jugables tanto por el
loader real de `arc_agi` como por `agent/my_agent.py` sin ningún cambio ni
caso especial.

**5 juegos públicos reservados para validación honesta (C7)**, nunca
entrenados ni usados para diseñar el generador: `wa30`, `lf52`, `cd82`,
`r11l`, `ka59` (se excluyó `ls20` del pool de reserva porque ya se usó para
el sanity check de sobreajuste de Fase 2).

**Fase 2 — Arquitectura del modelo. ✅ Hecho** (adelantada antes que Fase 1,
a pedido explícito: la elección de arquitectura no dependía del generador).
Encoder de slots + transformer causal; §4.4 resuelto. Entregable medido: el
modelo sobreajusta un episodio grabado de `ls20` (loss 1.50 → 0.0001 en 500
pasos, 100% de acciones reproducidas en modo determinista).

**Fase 3 — Entrenamiento a escala** sobre la distribución de juegos.
Entregable: score en los juegos held-out.

**Fase 4 — Integración con Kaggle** y medición del presupuesto real de
acciones (§2.6).

---

## 8. Riesgos

**El sesgo se muda al generador.** Con C10 y C3 saqué mis suposiciones del
código del agente, pero entran igual por la puerta de atrás: si el generador
solo produce "avatar que se mueve y recolecta", el modelo aprende a explorar
esa familia y se rompe con los puzzles del set oculto. **La diversidad del
generador es ahora la variable más importante del proyecto**, más que la
arquitectura de la red.

**C9 cuesta eficiencia, y el score la penaliza al cuadrado.** Sin búsqueda,
una política directa difícilmente ejecute cerca del baseline. Es una decisión
tomada a conciencia; si al medir vemos que la ineficiencia nos come los
puntos, hay que revisitarla.

**El horizonte es largo.** Este camino tiene techo más alto que un híbrido
con heurísticas, pero probablemente da **0 puntos durante bastante más
tiempo**. No esperar señal temprana.

**Los juegos complejos pueden ser inalcanzables.** `ka59` (41k líneas),
`lp85` (21k), `dc22` (11k) tienen mecánicas profundas. Expectativa realista a
mediano plazo: primeros niveles de los juegos simples.

**Recompensa ultra-esparsa.** `levels_completed` es la única señal. Con
espacio de acción de 4096 opciones al clickear, el entrenamiento por RL puro
sobre esto es notoriamente difícil; de ahí la propuesta de destilación en
§5.2 en vez de RL desde cero.
