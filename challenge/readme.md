# Diccionario de datos de los archivos CSV

Este documento describe el significado de cada columna presente en los archivos CSV del challenge. La variable objetivo del histórico es `sla_breached`.

## Archivos usados en cada caso

- Clasificacion: `historical_executions.csv`
- Optimizacion exacta: `weekly_instance_medium.csv` + `analysts.csv` + `analyst_availability.csv` + `eligibility_medium.csv`.
- Heuristica/metaheuristica: `weekly_instance_large.csv` + `analysts.csv` + `analyst_availability.csv` + `eligibility_large.csv`.

## analysts.csv

| Columna | Significado |
|---|---|
| `analyst_id` | Identificador único del analista. |
| `seniority` | Nivel de seniority del analista. |
| `team` | Equipo al que pertenece el analista. |
| `contract_hours_week` | Horas semanales contractuales del analista (capacidad teórica). |
| `cost_per_hour` | Costo por hora del analista. |
| `efficiency_score` | Factor de eficiencia del analista usado para reflejar productividad relativa. |
| `skill_data` | Indicador binario de dominio de la skill `data`; vale 1 si la posee y 0 si no. |
| `skill_ops` | Indicador binario de dominio de la skill `ops`; vale 1 si la posee y 0 si no. |
| `skill_risk` | Indicador binario de dominio de la skill `risk`; vale 1 si la posee y 0 si no. |
| `skill_support` | Indicador binario de dominio de la skill `support`; vale 1 si la posee y 0 si no. |
| `max_hours_week` | Horas efectivamente disponibles en la semana (suma de `available_hours` en `analyst_availability.csv`); puede ser menor o igual a `contract_hours_week`. |

## analyst_availability.csv

| Columna | Significado |
|---|---|
| `analyst_id` | Identificador del analista al que corresponde el registro de disponibilidad. |
| `week_id` | Identificador de la semana a la que corresponde la disponibilidad. |
| `day` | Día de la semana del registro. |
| `available_hours` | Cantidad de horas disponibles del analista en ese día. |

## historical_executions.csv

| Columna | Significado |
|---|---|
| `execution_id` | Identificador único de la ejecución histórica. |
| `task_id` | Identificador de la tarea ejecutada. |
| `analyst_id` | Identificador del analista asignado a la ejecución. |
| `assigned_day` | Día de la semana en que la tarea fue asignada. |
| `queue_load` | Nivel de carga de trabajo observado al momento de la ejecución. |
| `skill_match` | Indicador binario de compatibilidad entre la skill requerida por la tarea y las skills del analista; vale 1 si hay compatibilidad y 0 si no. |
| `estimated_hours` | Horas estimadas de la tarea en el registro histórico. |
| `actual_hours` | Horas efectivamente consumidas en la ejecución. |
| `slack_hours` | Holgura en horas entre el deadline de la tarea y las horas efectivas requeridas. |
| `reassignment_count` | Número de reasignaciones asociadas a la ejecución. |
| `overtime_flag` | Indicador binario de condición de sobrecarga operativa durante la ejecución; vale 1 si se activa y 0 si no. |
| `sla_breached` | Variable objetivo del histórico; indica si la ejecución incumplió el SLA, con 1 para incumplimiento y 0 para cumplimiento. |
| `required_skill` | Skill requerida por la tarea histórica. |
| `priority` | Nivel de prioridad de la tarea histórica. |
| `priority_weight` | Peso numérico de la prioridad en el registro histórico. |
| `complexity` | Nivel de complejidad de la tarea histórica. |
| `deadline_hours` | Tiempo límite de la tarea expresado en horas. |
| `client_criticality` | Nivel de criticidad del cliente en la tarea histórica. |
| `requires_pairing` | Indicador binario de si la tarea histórica requiere trabajo en pareja; vale 1 si lo requiere y 0 si no. |
| `analyst_seniority` | Nivel de seniority del analista asignado. |
| `analyst_efficiency_score` | Puntaje de eficiencia del analista asignado. |
| `analyst_cost_per_hour` | Costo por hora del analista asignado. |
| `skill_data` | Indicador binario de si el analista asignado posee la skill `data`; vale 1 si la posee y 0 si no. |
| `skill_ops` | Indicador binario de si el analista asignado posee la skill `ops`; vale 1 si la posee y 0 si no. |
| `skill_risk` | Indicador binario de si el analista asignado posee la skill `risk`; vale 1 si la posee y 0 si no. |
| `skill_support` | Indicador binario de si el analista asignado posee la skill `support`; vale 1 si la posee y 0 si no. |

## weekly_instance_medium.csv y weekly_instance_large.csv

Los dos archivos comparten la misma estructura. La diferencia entre ellos es el tamaño de la instancia.

| Columna | Significado |
|---|---|
| `week_id` | Identificador de la semana de planificación. |
| `task_id` | Identificador único de la tarea incluida en la instancia semanal. |
| `required_skill` | Skill requerida para ejecutar la tarea. |
| `priority` | Nivel de prioridad de la tarea. |
| `priority_weight` | Peso numérico asociado a la prioridad de la tarea. |
| `complexity` | Nivel de complejidad de la tarea. |
| `estimated_hours` | Estimación de horas necesarias para ejecutar la tarea. |
| `deadline_hours` | Tiempo disponible en horas hasta el vencimiento de la tarea. |
| `deadline_day` | Día de la semana en que vence la tarea. |
| `deadline_slot` | Slot o bloque temporal dentro del día de vencimiento. |
| `risk_score` | Puntaje de riesgo asociado a la tarea en la instancia semanal. |
| `client_criticality` | Nivel de criticidad del cliente asociado a la tarea. |
| `client_criticality_score` | Valor numérico asociado a la criticidad del cliente. |
| `requires_pairing` | Indicador binario de si la tarea requiere trabajo en pareja; vale 1 si lo requiere y 0 si no. |
| `instance_size` | Etiqueta del tamaño de la instancia (`small`, `medium` o `large`). |

## eligibility_medium.csv y eligibility_large.csv

Los dos archivos comparten la misma estructura. Cada fila representa una combinación tarea-analista considerada elegible.

| Columna | Significado |
|---|---|
| `task_id` | Identificador de la tarea. |
| `analyst_id` | Identificador del analista elegible para esa tarea. |
| `eligible` | Indicador binario de elegibilidad; vale 1 cuando la combinación tarea-analista es elegible. |
| `estimated_hours` | Horas estimadas de la tarea para esa combinación. |
| `cost_per_hour` | Costo por hora del analista en esa combinación. |
| `priority_weight` | Peso numérico de prioridad de la tarea en esa combinación. |
| `risk_score` | Puntaje de riesgo asociado a la tarea en esa combinación. |