# CC0F4 - Tercera Práctica Calificada
### ¿Conviene que el modelo decida y recuerde? Workflow, agente controlado y memoria verificable para un asistente de revisión científica

**Curso:** CC0F4 - Tópicos de Ciencia de la Computación I · **Periodo:** 2026-2 · **Modalidad:** individual
**Estudiante:** César Augusto Sánchez Malaspina
**Tema (registrado en PC1, continuado en el parcial):** agente de revisión científica que identifica acuerdos y desacuerdos entre artículos.
**Cuaderno principal:** [`notebooks/PC3_CC0F4.ipynb`](notebooks/PC3_CC0F4.ipynb) (se entrega ejecutado) · **Diapositivas:** [`slides/PC3_CC0F4.pdf`](slides/PC3_CC0F4.pdf)

---

## La idea en un párrafo

Queremos un asistente que ayude a revisar literatura científica: le preguntas si dos líneas de investigación coinciden,
**busca** los fragmentos de los artículos que hablan del tema, **contrasta** lo que dice cada uno, **verifica** si una cita
entre dos artículos es posible por sus fechas, y **entrega** una respuesta con su evidencia, o **se abstiene** si no la hay.
En el examen parcial construimos esas piezas por separado. En esta práctica las conectamos de dos maneras —con una **receta fija**
y con un **agente** que decide los pasos— y le damos al agente **cuatro tipos de memoria**. El objetivo no es tener el sistema
más autónomo, sino **medir** si la autonomía y la memoria mejoran algo de verdad, y qué errores nuevos traen.

---

## 1. Problema y continuidad con las entregas anteriores

| Entrega | Qué dejó | Cómo se usa aquí |
|---|---|---|
| **PC1** | Qwen2.5-0.5B clasifica `agree/disagree/neutral` con JSON validado | el mismo contrato de salida (`RAGAnswer`) |
| **Examen parcial** (repositorio `Topicos1-PCI1`, carpeta `parcial/`) | 36 artículos reales de arXiv (135 passages), 22 consultas con qrels, recuperación C (E5+FAISS, BM25, RRF, Cross-Encoder), RAG con citas, herramienta `check_temporal_validity` con contratos | se reutiliza **todo**, sin modificar sus resultados (`src/baseline.py`) |
| **PC3** (este repositorio) | — | workflow (W0), agente (W1), controlador con límites, memorias M0-M3, dos experimentos |

El parcial terminó con dos hallazgos que motivan esta práctica: **la recuperación no era el cuello de botella** (la evidencia correcta
llegaba al contexto en 10 de 10 preguntas) y **el modelo de 0,5B no razonaba bien sobre posturas opuestas** (respondía `agree` en 12 de 12 casos).
Ahora la pregunta es otra: **¿cómo coordinar varias operaciones y cómo recordar resultados anteriores sin confundir recuerdos con hechos actuales?**

**Cambios respecto del parcial (documentados):**
- Los datos se **copiaron** a este repositorio y el cuaderno verifica que sus huellas SHA-256 son las mismas del parcial (`verify_same_corpus_as_parcial`).
- Se añadió un filtro por fecha (`as_of`) **fuera** del recuperador, para simular qué artículos existían en cada sesión del experimento de memoria.
- No se corrigió ningún resultado del parcial ni se reemplazó su baseline.

## 2. Tarea multi-paso seleccionada

> **Revisión de una afirmación científica con verificación de cita.** Dada una pregunta sobre un debate de ML/NLP y, opcionalmente,
> una cita "el artículo A cita al artículo B", el sistema debe (1) **recuperar** evidencia del corpus, (2) **contrastar** lo que dicen
> los artículos (`agree / disagree / neutral / insufficient_evidence`), (3) **verificar** si la cita es temporalmente posible y
> (4) **responder** en un objeto estructurado con evidencia y una etiqueta de terminación verificable, o abstenerse.

Requiere tres operaciones distintas: búsqueda, análisis con LLM y consulta a una herramienta determinista.

## 3. Arquitectura del workflow y del agente

```text
W0 · workflow determinista (src/workflow.py)
entrada validada -> plan fijo -> search_evidence(pregunta) -> [abstain si max_score < 2,0]
                 -> compare_evidence -> check_citation (si la tarea la pide) -> finish -> validación final

W1 · agente controlado por LLM (src/agent.py), ciclo ReAct
entrada validada -> estado inicial -> Thought (LLM) -> Action ∈ {search, compare, check_citation, finish, abstain} (LLM)
                 -> herramienta vía controlador -> observación validada -> actualización del estado -> nueva decisión o terminación
                 -> respuesta estructurada (validada por el controlador)
```

**Qué decide el LLM y qué decide el controlador**

| Lo decide el LLM (solo en W1) | Lo decide el controlador determinista (W0 y W1) |
|---|---|
| El pensamiento (*Thought*) de cada paso, que queda en la traza | Si la acción propuesta se ejecuta, se bloquea o se rechaza |
| La siguiente acción, entre 5 opciones permitidas | Contratos, reintentos y timeouts de cada herramienta |
| El texto de la búsqueda y los arXiv ID de la cita | Qué entra al prompt (estado, historial, memoria) y con qué presupuesto de tokens |
| Cuándo terminar (`finish`) o abstenerse (`abstain`) | La etiqueta `stop_reason` y la validación de la respuesta final |

La acción se elige con **decodificación restringida**: se calcula la log-probabilidad de cada una de las cinco acciones y se toma la mayor.
El modelo no puede emitir comandos fuera de la lista, y la distribución de probabilidades queda en cada paso de la traza como justificación.

**Patrones de la Semana 9 en este diseño:** W0 es *prompt chaining* con *routing* por reglas (abstenerse o continuar);
`compare_evidence` + la validación final forman un par *evaluator* (el controlador evalúa y rechaza respuestas sin evidencia);
W1 es un agente ReAct con un solo trabajador por paso. No se usó *parallelization* ni *orchestrator-workers*: las tareas son pequeñas y secuenciales,
y agregarlos no respondería ninguna pregunta experimental.

## 4. Herramientas, contratos y controlador

| Herramienta | Origen | Contrato de entrada | Errores tipados |
|---|---|---|---|
| `search_evidence` | condición C del parcial (§6) | `query` de 8 a 300 caracteres, `extra="forbid"` | `validation`, `transient`, `timeout`, `output_contract` |
| `compare_evidence` | RAG del parcial (§8): mismo prompt, Qwen 0.5B, greedy, 220 tokens | pregunta + 1 a 8 `passage_id` **ya observados en la tarea** | `validation`, `domain` (evidencia no observada), `output_contract` (JSON inválido) |
| `check_citation` | herramienta del parcial (§10), sin cambios | dos arXiv ID `^\d{4}\.\d{4,5}$` | `validation`, `domain`, `transient`, `timeout`, `output_contract` |

Todas pasan por `ToolRuntime.execute` (`src/tools.py`): allowlist → contrato de entrada → dominio → ejecución con timeout y reintentos
(solo `transient`/`timeout`, hasta 3 intentos, backoff 20/40 ms, igual que el parcial) → contrato de salida → observación.
**Las observaciones se tratan como datos:** al prompt solo llegan el tipo y el código del error (`error kind=timeout code=attempt_timeout`),
nunca su mensaje libre, para que un fallo no se convierta en una instrucción.

**Límites del controlador** (`Limits` en `src/controller.py`, fijados antes de los experimentos e iguales para W0, W1 y M0-M3):

| Límite | Valor |
|---|---|
| `max_steps` | 8 iteraciones |
| `max_tool_calls` | 6 llamadas reales |
| `timeout_per_tool` | search 5 s · compare 60 s · citation 0,4 s (deadline interno del parcial) |
| `total_deadline` | 180 s por tarea |
| `max_input_tokens` (presupuesto de contexto) | 1 800 por llamada al LLM, de los cuales 350 como máximo para memoria |
| `max_output_tokens` | 40 (Thought) · 32 (consulta) · 16 (un arXiv ID) · 220 (análisis) |
| detección de bucles | `no_progress_window = 3`, `max_repeats = 2` |

**Estados de terminación** (definiciones verificables; cada uno tiene una prueba en `tests/test_controller.py`):

| `stop_reason` | Definición |
|---|---|
| `SUCCESS` | `finish` y la respuesta final pasa la validación (salidas requeridas presentes; relación con al menos una cita a evidencia observada) |
| `ABSTAIN` | la política eligió `abstain`, o el análisis concluyó `insufficient_evidence` |
| `MAX_STEPS` | se alcanzó `max_steps` o `max_tool_calls` sin acción terminal |
| `TIMEOUT` | se superó `total_deadline` |
| `TOOL_ERROR` | `finish` con una salida requerida ausente porque su herramienta falló sin recuperarse |
| `INVALID_OUTPUT` | dos decisiones inválidas seguidas; o salida de herramienta que viola su contrato; o `finish` prematuro |
| `NO_PROGRESS` | 3 pasos seguidos sin actualizar el estado |
| `REPEATED_ACTION` | 2 intentos de repetir una acción idéntica ya ejecutada (la repetición no se ejecuta) |

## 5. Modelo y configuración

| Rol | Modelo | Configuración |
|---|---|---|
| Analista (`compare_evidence`), **igual en W0 y W1** | `Qwen/Qwen2.5-0.5B-Instruct` (rev. `7ae5576`, la del parcial) | fp32, CPU, greedy (`do_sample=False`, `repetition_penalty=1.0`), 220 tokens |
| Política del agente W1 | `Qwen/Qwen2.5-1.5B-Instruct` (rev. `989aa79`) | fp32, CPU, greedy; acción por decodificación restringida |
| Encoder denso y memoria semántica | `intfloat/multilingual-e5-small` + FAISS `IndexFlatIP` | el del parcial (Semana 4) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | el del parcial |

**¿Por qué la política del agente usa 1,5B?** Primero se probó Qwen 0,5B como política **solo con tareas de desarrollo**
(`data/dev_tasks.jsonl` y `data/dev_memory_scenarios.jsonl`, temas que no están en la evaluación). Terminó bien **0 de 5**:
nunca eligió `check_citation` aunque fuera la única salida pendiente, y cerraba con `finish` prematuro o abstención.
Con 1,5B terminó bien **4 de 5**. Las trazas de ambos están en `results/dev/`. Para que la comparación siga siendo justa,
el analista (el "modelo generador" del parcial) es el mismo en W0 y W1; solo cambia *quién decide el siguiente paso*.

**Iteraciones de desarrollo del prompt** (todas con tareas de desarrollo; el benchmark de evaluación no se modificó):
v1 → v2 agrega una lista de salidas requeridas con estado `DONE/PENDING/FAILED`; v3 agregó un ejemplo *few-shot* y **empeoró**
en desarrollo, así que se descartó. Se corrigió además un bug: el límite de 10 tokens truncaba los arXiv ID (Qwen tokeniza cada dígito).
Versiones en `prompts/dev_history/`. **Umbral de evidencia** (`max_score ≥ 2,0` del Cross-Encoder): fijado con preguntas de desarrollo
(fuera del corpus ≤ 1,24; dentro ≥ 2,52).

## 6. Benchmark de tareas y criterios de éxito

`data/tasks.jsonl` (12 tareas, generadas por `src/build_data.py` y **fijadas antes de ejecutar las comparaciones**) y
`data/expected_results.jsonl`. Cada tarea tiene `task_id`, `objetivo`, `entrada`, `operaciones_necesarias`, `fuentes_permitidas`,
`resultado_esperado`, `criterio_de_exito` y `condicion_de_terminacion`. Los resultados esperados **no dependen del agente**:
la relación correcta viene de `reference_answers.jsonl` del parcial, la evidencia relevante de sus qrels y la validez de cada cita
de las fechas de arXiv.

| Tareas | Tipo | Detalle |
|---|---|---|
| T01-T03, T05, T06 | ordinarias | buscar + contrastar (+ verificar una cita válida en 4 de ellas) |
| T04, T07 | ordinarias de **rechazo** | la cita propuesta es imposible por fechas (DPO 2023 no puede citar un estudio de 2024; LoRA 2021 no puede citar uno de 2024) |
| T08 | fallo inyectado | la búsqueda falla una vez (error transitorio) → debe recuperarse y terminar en `SUCCESS` |
| T09 | fallo inyectado | la herramienta de citas no responde nunca (timeout) → `TOOL_ERROR` (o `ABSTAIN`), sin inventar el veredicto |
| T10 | fallo inyectado | el analista devuelve texto que no es JSON → `INVALID_OUTPUT` (o `ABSTAIN`) |
| T11, T12 | abstención | la respuesta no está en el corpus (Mamba, Llama 3) → `ABSTAIN` |

**Criterio de éxito (`task_success`):** si se espera `SUCCESS`, deben cumplirse a la vez: `stop_reason = SUCCESS`, relación igual a la de referencia,
al menos una cita a un passage con qrel ≥ 1 y, si hay cita, el par y su veredicto correctos. Si se espera otra terminación, debe estar en la lista de aceptables.
Los fallos inyectados están marcados en los datos (`faults`) y en cada paso de la traza (`injected_fault`).

## 7. Experimento 1: workflow (W0) frente a agente (W1)

**Pregunta:** ¿mejora algo dejar que el modelo decida los pasos, comparado con una receta fija?
Se usaron las mismas 12 tareas, las mismas herramientas, los mismos límites y el mismo analista (Qwen 0.5B) en ambos casos.
La ejecución completa está en `results/exp1/` (`per_task.csv`, `summary.csv`, `results.jsonl` y `run_config.json`, con versiones y hashes)
y las trazas en `results/traces/exp1_traces.jsonl`.

| Métrica | W0 workflow | W1 agente | Cómo se cuenta |
|---|---|---|---|
| `task_success` | **0,50** (6/12) | 0,08 (1/12) | tareas que cumplen el criterio de la §6 |
| `termination_correct` | **0,75** (9/12) | 0,17 (2/12) | el `stop_reason` está entre los aceptables |
| relación correcta | 0,25 | 0,13 | tareas con relación de referencia |
| cita correcta | 0,57 | 0,00 | tareas con cita a verificar |
| pasos por tarea | 3,2 | 3,6 | media |
| llamadas a herramientas | 2,2 | 2,6 | media |
| `tool_errors` | 4 | 5 | total |
| `repeated_actions` | 0 | 4 | total (bloqueadas por el controlador, no ejecutadas) |
| `latency_ms` | **6 136** | 59 058 | media por tarea (CPU) |
| `tokens` | **671** | 6 222 | media por tarea (entrada + salida) |

**Terminaciones:** W0 terminó 5 veces en SUCCESS, 3 en ABSTAIN, 3 en INVALID_OUTPUT y 1 en TOOL_ERROR. W1 terminó 4 veces en SUCCESS, 4 en INVALID_OUTPUT, 3 en NO_PROGRESS y 1 en REPEATED_ACTION.

**Lectura en lenguaje simple.** En este problema, la receta fija resolvió seis veces más tareas, fue diez veces más rápida y gastó nueve veces menos tokens.
El agente sí elige entre cinco acciones y justifica cada paso en la traza (`thought` y probabilidad de cada acción), pero comete errores típicos de un modelo pequeño:

* **Termina antes de tiempo.** En T03, T04, T06 y T07 llamó a `finish` sin verificar la cita, aunque la lista de salidas requeridas la marcaba como PENDIENTE.
  El controlador no aceptó esas respuestas (`INVALID_OUTPUT`).
* **No sabe abstenerse.** En T11 y T12 (temas fuera del corpus) respondió `agree` con evidencia por debajo del umbral. W0 sí se abstuvo.
* **Se traba cuando falla una herramienta.** En T08, T09 y T10 repitió búsquedas y terminó en `NO_PROGRESS`. W0 terminó con la etiqueta exacta del error.

**Caso donde W1 sí es mejor: T01** (Kaplan frente a Chinchilla). W0 buscó con la pregunta literal y el mejor puntaje fue 0,53, por debajo del umbral de 2,0, así que **se abstuvo por error**.
W1 reformuló la consulta ("Kaplan-style scaling laws and Chinchilla study split compute"), encontró los dos artículos y terminó en `SUCCESS`, que era la terminación esperada.
Aun así, la tarea no cuenta como resuelta: invirtió el orden de la cita (puso a 2001.08361 citando a 2203.15556) y el analista eligió la relación equivocada.
Es la única ventaja real del agente: **adaptar la búsqueda**.

**Caso donde W1 no es mejor: T03** (atención multi-consulta). W0 hizo búsqueda, comparación, verificación de cita y `finish`, y acertó todo.
W1 hizo búsqueda y comparación, y luego pensó *"The evidence is already found and agreed upon. No further action is needed."* y llamó a `finish`.
El controlador lo detuvo con `INVALID_OUTPUT` (falta la cita). Pagó 4,6 mil tokens para terminar peor que la receta.

## 8. Experimento 2: memoria del agente (M0-M3)

**Pregunta:** ¿qué tipo de memoria ayuda al agente cuando la tarea continúa en otra sesión, y qué errores introduce?
Son 3 secuencias con 11 interacciones (`data/memory_scenarios.jsonl`). Se corrió el mismo agente W1 con cuatro memorias: sin memoria (M0), de trabajo (M1),
episódica en disco (M2) y semántica con fuente, fecha y estado (M3). Los resultados están en `results/exp2/`
(`per_interaction.csv`, `summary.csv`, `memory_writes.jsonl` y el estado final de los almacenes en `memory_store/`).

| Métrica | M0 | M1 | M2 | M3 | Denominador |
|---|---|---|---|---|---|
| `task_success` | 0,18 (2/11) | 0,18 (2/11) | **0,27** (3/11) | **0,27** (3/11) | 11 interacciones |
| `memory_recall_accuracy` | 0/6 | 0/6 | 0/6 | 0/6 | interacciones que necesitan un recuerdo |
| `cross_session_retention` | 0/3 | 0/3 | 0/3 | 0/3 | necesitan un recuerdo de otra sesión |
| `stale_memory_error_rate` | - | 0/1 | 0/1 | - | con un recuerdo obsoleto en el contexto |
| `wrong_memory_use_rate` | - | 0/5 | 0/10 | 0/6 | con algún recuerdo en el contexto |
| interacciones resueltas buscando (`solved_by_tools`) | 0 | 1 | 0 | 1 | |
| `context_tokens` (media) | 7 934 | 9 362 | 10 687 | 10 808 | por interacción |
| tokens de memoria (media) | 0 | 92 | 229 | 154 | por interacción |
| `latency_ms` (media) | 68 624 | 67 443 | 75 887 | 74 598 | por interacción |

La métrica `memory_recall_accuracy` se definió **antes** de ejecutar y es estricta: solo cuenta si el agente usó el recuerdo **y** la búsqueda de esa misma
interacción no encontró el documento. Como el agente siempre volvió a buscar, nunca se pudo atribuir el acierto solo a la memoria (0/6).
Los tres casos de análisis se describen con la traza:

* **(a) La memoria ayuda: M3, S2.3.** Hay que verificar si 2304.15004 puede citar a *Emergent Abilities* (2206.07682), que se revisó en **otra sesión**.
  Con M3, la ficha `F-2206.07682` (fuente, fecha y versión) estaba en el contexto: el agente verificó el par correcto en 3 pasos y acertó.
  Con M0 y M1 citó un artículo equivocado (1706.03762) y falló. Algo parecido pasó con M1 en S3.2: resolvió en la misma sesión lo que M0 no pudo.
* **(b) La memoria no mejora: S1.2, S1.4, S3.3.** El recuerdo correcto estaba en el contexto (M1/M2), pero el agente se trabó igual (`NO_PROGRESS` o `REPEATED_ACTION`).
  El cuello de botella es la política, no el recuerdo.
* **(c) La memoria confunde: M2, S2.3 (interferencia).** El diario episódico trajo dos episodios de la sesión anterior: *emergent abilities* y *LLM.int8*.
  El agente mezcló los dos temas en la búsqueda ("emergent abilities … and 8-bit") y verificó **2208.07339** en lugar de 2206.07682.
  La métrica estricta `wrong_memory_use` no lo cuenta porque la búsqueda también devolvió ese documento, pero la traza muestra que la confusión viene del recuerdo.
* **Recuerdo obsoleto.** En S1.4, la memoria M1/M2 trajo el resumen viejo (Kaplan) junto al nuevo (Chinchilla). El agente no llegó a citar, así que no hubo error
  por obsolescencia (0/1). La invalidación de M3 (`SUPERSEDED by …`) está probada en `tests/test_memory.py`, pero en la evaluación no se activó:
  el agente no resolvió S1.1 ni S1.3 con M3, y la memoria semántica **solo guarda tareas exitosas**. Sí se observó una **actualización**: `F-2208.07339` pasó a la versión 2.

**Costo:** la memoria aumentó el contexto entre un 18 % (M1) y un 36 % (M3) y no redujo los pasos.
Con 11 interacciones y una sola corrida, la diferencia 2/11 frente a 3/11 **no es concluyente**.

## 9. Pruebas de robustez y terminación

Cada prueba tiene evidencia en las trazas reales (cuaderno §8) y una prueba unitaria con política programada (`tests/`, 19 pruebas que pasan).

| Prueba | Dónde | Qué pasó | Prueba unitaria |
|---|---|---|---|
| Error transitorio | T08: la búsqueda falla una vez | W0 reintentó (2 intentos en la traza) y siguió. W1 terminó en `NO_PROGRESS` | `test_transient_error_recorded_in_trace` |
| Salida inválida | T10: el analista devuelve texto no JSON | W0 terminó en `INVALID_OUTPUT`. W1 en `NO_PROGRESS`. Ninguno inventó la relación | `test_invalid_tool_output` |
| Acción repetida | todas las corridas de W1 | 18 repeticiones bloqueadas antes de ejecutarse; 15 corridas terminaron en `REPEATED_ACTION` | `test_repeated_action_is_blocked_then_stops` |
| Agotamiento de pasos | `max_steps=8`, `max_tool_calls=6` | 4 corridas del Exp. 2 terminaron en `MAX_STEPS` | `test_max_steps`, `test_max_tool_calls_maps_to_max_steps` |
| Evidencia insuficiente | T11, T12 | W0 se abstuvo (`ABSTAIN`). W1 respondió sin evidencia suficiente (ver §11) | `test_abstain` |
| Recuerdo obsoleto frente a fuente nueva | S1.4 (Kaplan superado por Chinchilla) | el recuerdo viejo estuvo en el contexto y no se usó; marcado `SUPERSEDED` | `test_semantic_invalidation_update_and_delete` |

Además, T09 (la herramienta de citas no responde): W0 hizo 3 intentos con timeout y terminó en `TOOL_ERROR` **sin inventar el veredicto**.
En las 68 ejecuciones de los experimentos (24 + 44), ninguna que terminara con error devolvió una respuesta final; el cuaderno lo verifica con un `assert`.

## 10. Resultados comparativos (resumen)

| Decisión | Costo | Calidad | Robustez |
|---|---|---|---|
| W0 workflow | bajo (671 tokens, 6 s) | 6/12 tareas | termina con la etiqueta correcta en 9/12 |
| W1 agente | alto (6 222 tokens, 59 s) | 1/12 tareas | 2/12; se traba ante errores |
| M1 trabajo | +18 % contexto | igual que M0 | bajo riesgo: se borra al cambiar de sesión |
| M2 episódica | +35 % contexto | +1 interacción | provocó interferencia (S2.3) |
| M3 semántica | +36 % contexto | +1 interacción | con fuente, fecha y estado; solo guarda tareas validadas |

**Recomendación técnica:** para esta tarea conviene **la receta fija (W0) con el LLM solo como analista**, más una mejora pequeña:
reformular la búsqueda una vez antes de abstenerse, que es lo único que el agente hizo mejor (T01). Si se necesita continuidad entre sesiones,
usar **memoria semántica con estado y fuente (M3)** y no un diario de texto libre (M2), porque M2 fue el que provocó la interferencia.

## 11. Análisis de trazas y errores

| Error observado | Dónde | Causa | Qué hizo el controlador |
|---|---|---|---|
| `finish` prematuro | W1: T03, T04, T06, T07 | el modelo cree que "ya terminó" aunque falte la cita | no aceptó la respuesta (`INVALID_OUTPUT`) |
| responde sin evidencia | W1: T11, T12 | el agente ignora "evidence_found: no" | **la aceptó**: la validación revisa que las citas sean observadas, no que superen el umbral. Es un hueco del controlador (ver §12) |
| cita invertida | W1: T01 | intercambió `citing_id` y `cited_id` | ejecutó la herramienta; el error se detecta en la evaluación |
| analista no cumple el contrato | W0: T02, T04 (sin fallo inyectado) | Qwen 0.5B devuelve JSON inválido | `INVALID_OUTPUT`, no inventa la relación |
| relación equivocada | W0: T06, T07, T08 | el analista dice `agree` cuando la referencia es `disagree` (debilidad heredada del parcial) | nada que hacer: la salida es válida en forma |
| búsquedas repetidas | W1: T08, T09, T10 y Exp. 2 | ante un error, vuelve a buscar lo mismo | bloqueó la repetición y cortó con `REPEATED_ACTION` o `NO_PROGRESS` |
| interferencia de memoria | M2: S2.3 | dos episodios de temas parecidos en el contexto | ninguna; la memoria no tiene control de pertinencia |

Las observaciones de herramientas se tratan como datos: al prompt solo llega el tipo y el código del error (por ejemplo `timeout:attempt_timeout`), nunca su texto libre.

## 12. Limitaciones

* **Modelos pequeños en CPU.** La política es Qwen 1.5B y el analista Qwen 0.5B. Con 0.5B como política, 0/5 tareas de desarrollo terminaron bien (`results/dev/`).
  Los resultados hablan de *este* tamaño de modelo, no de agentes en general.
* **Muestras pequeñas y una sola corrida greedy:** 12 tareas y 11 interacciones. Las diferencias de una tarea no son significativas.
* **El controlador no exige el umbral de evidencia al terminar.** Por eso W1 pudo responder T11 y T12. No se corrigió después de ver los resultados
  para no ajustar el sistema a la evaluación; queda como mejora.
* **La métrica de recuperación de memoria es estricta** y dio 0/6 en todas las configuraciones; los beneficios se describen con trazas.
* **La invalidación de M3 no llegó a activarse en la evaluación** (solo en pruebas unitarias), porque depende de que el agente resuelva primero las revisiones.
* **El analista se equivoca en la relación** en varias tareas (herencia del parcial); afecta por igual a W0 y W1.
* **El timeout es cooperativo**: se mide después de la llamada, no la interrumpe.

## 13. Fuentes técnicas y matriz de trazabilidad

**Fuentes**

1. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR. arXiv:2210.03629.
2. Anthropic (2024). *Building effective agents*. https://www.anthropic.com/research/building-effective-agents
3. Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS. arXiv:2303.11366.
4. Park, J. S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST. arXiv:2304.03442.
5. Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.
6. Sumers, T. et al. (2024). *Cognitive Architectures for Language Agents (CoALA)*. TMLR. arXiv:2309.02427.
7. Wadden, D. et al. (2020). *Fact or Fiction: Verifying Scientific Claims (SciFact)*. EMNLP. arXiv:2004.14974.
8. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401.

**Matriz de trazabilidad**

| Semana | Recurso estudiado | Concepto | Archivo o función | Evidencia de ejecución | Limitación propia |
|---|---|---|---|---|---|
| **9. Workflows, planificación y ReAct** | Yao et al. (ReAct); Anthropic, *Building effective agents*; Shinn et al. (Reflexion) | workflow con reglas frente a agente; ciclo Thought → Action → Observation; controlador con límites y estados de parada | `src/workflow.py` (`DeterministicWorkflow`), `src/agent.py` (`LLMAgent.decide`), `src/controller.py` (`run_task`, `Limits`, `StopReason`) | `results/exp1/`, `results/traces/exp1_traces.jsonl`, cuaderno §6 y §9 | el agente 1.5B termina antes de tiempo y no sabe abstenerse; no se implementó la autorreflexión de Reflexion |
| **10. Estado y memoria de agentes** | Park et al. (recencia, relevancia e importancia); Packer et al. (MemGPT: memoria por niveles); Sumers et al. (CoALA: memoria de trabajo, episódica, semántica y procedimental) | M0-M3, persistencia entre sesiones, actualización, invalidación (`SUPERSEDED`), presupuesto de tokens | `src/memory.py` (`WorkingMemory`, `EpisodicMemory`, `SemanticMemory`, `make_memory`), `src/evaluation.py` (`eval_memory`) | `results/exp2/`, `results/exp2/memory_store/`, `tests/test_memory.py`, cuaderno §7 | la recuperación no filtra por pertinencia (interferencia en S2.3); la invalidación no se activó en la evaluación |
| 4. Recuperación (parcial) | Lewis et al. (RAG) | búsqueda híbrida, condición C del parcial | `src/baseline.py` (`Retrievers.condition_c`), `src/tools.py` (`search_evidence`) | hashes SHA-256 iguales a los del parcial (cuaderno §2) | la pregunta literal a veces no supera el umbral (T01) |
| 6. Contratos y herramientas (parcial) | contratos Pydantic del parcial | entradas y salidas validadas; errores tipados; reintentos | `src/tools.py` (`ToolRuntime.execute`), `src/baseline.py` (`validate_output`, `RetryPolicy`) | T08 a T10 en `exp1_traces.jsonl` | timeout cooperativo |
| 7. Métricas (parcial) | Wadden et al. (SciFact: verificación con evidencia) | relación correcta, evidencia con qrel, cita correcta | `src/evaluation.py` (`eval_exp1`, `summarize_exp1`) | `results/exp1/summary.csv` | muestra pequeña, una sola corrida |

## 14. Reproducción

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name cc0f4-pc3
pytest -q tests                    # 19 pruebas, sin LLM
jupyter lab notebooks/PC3_CC0F4.ipynb
```

* Con `RUN_EXPERIMENTS = False` (valor por defecto), el cuaderno carga los resultados guardados y solo ejecuta las pruebas en vivo y la demo (unos 5-10 min en CPU).
* Con `RUN_EXPERIMENTS = True`, repite los dos experimentos: unos 40-60 minutos en un portátil con CPU (Apple Silicon).
* Los modelos se descargan solos de Hugging Face la primera vez. El modelo de la política se fija por revisión (`AGENT_MODEL_REVISION` en `src/baseline.py`).
* Para regenerar los datos: `python src/build_data.py`. Para regenerar las diapositivas: `python slides/build_slides.py` (necesita Google Chrome).
* Versiones exactas, fecha, hashes de los datos y límites usados: `results/exp1/run_config.json` y `results/exp2/run_config.json`.

**Commit evaluado:** `COMMIT_EVALUADO` (se obtiene con `git rev-parse HEAD` después del primer commit).

