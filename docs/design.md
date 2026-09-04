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
| `model/config.py` | Sobrevive; hay que agregar los parámetros de slots y de longitud de contexto. |
| `model/adapters.py` | Sobrevive. Sigue siendo la única frontera con `arcengine`. Se le quita cualquier tentación de segmentar (C10). |
| `model/inference.py` | Sobrevive con cambios: extraer una interfaz `Policy` y agregarle `reset()`. |
| `model/data.py` | Sobrevive. |
| `model/network.py` | **Se rehace.** Hoy es una CNN sobre un frame único con policy head y value head. Necesita ser encoder de slots + modelo secuencial sobre el episodio. La `ValueHead` queda solo si el método de entrenamiento la necesita (actor-critic); con C9 no hay búsqueda que la use. |
| `agent/my_agent.py` | Se adelgaza. Hay que sacar el fork por `game_id` de LS20 (viola C2 y de todos modos da 0 en el set oculto). |

### 4.4 Problema técnico abierto: la longitud del episodio

Con baselines de hasta 1843 acciones por juego (wa30), y K slots por paso, un
transformer con atención completa sobre el episodio entero es caro. Opciones
a evaluar: estado recurrente (GRU / SSM tipo Mamba), ventana deslizante con
tokens de resumen, o compresión aprendida del historial. **No está decidido**
y es una de las primeras cosas a resolver en la fase de arquitectura.

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

**Pendiente:** el repo vive en `/mnt/c`, al que WSL accede por drvfs, más
lento que ext4 nativo. Plan: clonar en `~/arc-agi-3` dentro de WSL. Hay que
copiar a mano `.kaggle/access_token` (gitignoreado) y `environment_files/`
se re-descarga sola en el primer `make play-local`.

---

## 7. Plan por fases

Cada fase tiene que terminar en un número medido, no en una sensación.

**Fase 0 — Instrumentación.** Evaluador offline que reproduzca el score
oficial por nivel (acciones usadas vs. baseline) reutilizando la fórmula de
`arc_agi.scorecard`. Refactor de `Policy` para backends intercambiables.
*Sin esto todo lo demás es fe ciega.* Entregable: el score real del baseline
aleatorio actual.

**Fase 1 — Generador procedural.** Ruta crítica. Entregable: N juegos
sintéticos con diversidad medible de mecánicas.

**Fase 2 — Arquitectura del modelo.** Encoder de slots + modelo secuencial;
resolver §4.4. Entregable: el modelo puede sobreajustar un solo juego
(prueba de cordura de que la arquitectura aprende algo).

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
