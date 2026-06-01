========================================================================================================================================================================================================================
CHALLENGE TÉCNICO: DATA SCIENCE & OPTIMIZACIÓN DE OPERACIONES
========================================================================================================================================================================================================================

Este repositorio contiene la solución completa al desafío técnico de asignación de analistas a tareas operativas. La solución está dividida en tres partes interconectadas que abordan desde el análisis predictivo hasta la optimización matemática exacta y heurística a gran escala.

1. REQUISITOS Y ENTORNO

El código está desarrollado en Python 3 y requiere las siguientes librerías para su correcta ejecución:

    * pandas & numpy (Manipulación de datos)
    * scikit-learn (Modelamiento predictivo y preprocesamiento)
    * imbalanced-learn (Balanceo de clases - SMOTE)
    * PuLP (Optimización matemática lineal / MILP)
    * matplotlib (Visualización opcional)

Se pueden instalar las dependencias principales ejecutando:
pip install pandas numpy scikit-learn imbalanced-learn pulp

*Nota: Asegúrarse de verificar y ajustar las rutas absolutas del dataset en la variable 'ruta' al inicio de cada script antes de ejecutar.*

2. ESTRUCTURA DEL PROYECTO Y COMPONENTES

========================================================================================================================================================================================================================
PARTE 1: CLASIFICACIÓN BINARIA (Parte1.py)
========================================================================================================================================================================================================================
Objetivo: Estimar la probabilidad de incumplimiento de un SLA (variable 'sla_breached') utilizando el histórico 'historical_executions.csv'.

Aspectos clave de la implementación:
* Limpieza y Blindaje: Se eliminan variables de ruido y fugas de datos (data leakage) como 'actual_hours', 'slack_hours' y 'overtime_flag', las cuales solo se conocen "después" de que la tarea se ejecutó.
* Control de Desbalance: Se aplica SMOTE (Synthetic Minority Over-sampling Technique) exclusivamente sobre el set de entrenamiento para resolver el severo desbalance de la variable objetivo.
* Modelos Implementados: 
  - Regresión Logística (Modelo Baseline).
  - Random Forest Classifier (Modelo Final) optimizado mediante GridSearchCV con validación cruzada de 5 pliegues maximizando F1-Score.
* Enfoque de Negocio (Umbral Económico): En lugar de utilizar el umbral estándar de 0.5, el script implementa una función de pérdida monetaria:
  Costos: Falso Positivo (Falsa alarma) = $50 | Falso Negativo (Incumplir SLA) = $300 El algoritmo barre automáticamente los umbrales probabilísticos para encontrar el punto óptimo que minimiza el impacto financiero total.

Instrucciones de ejecución:
- Ejecutar Parte1.py

========================================================================================================================================================================================================================
PARTE 2: PROGRAMACIÓN LINEAL ENTERA MIXTA (Parte2.py)
========================================================================================================================================================================================================================
Objetivo: Resolver la asignación óptima de la instancia mediana ('weekly_instance_medium.csv') garantizando el cumplimiento estricto de las restricciones operativas.

Aspectos clave de la implementación:
* Eficiencia de Memoria (Sparsity): En lugar de crear una matriz densa de variables para cada combinación posible, el script solo declara variables binarias x[t,a] para los pares explícitamente factibles en 'eligibility_medium.csv',
  reduciendo drásticamente el espacio de búsqueda del solver.
* Función Objetivo Balanceada: Diseñada algebraicamente de forma lineal para maximizar el valor de las prioridades, restar los costos operativos (horas estimadas x costo por hora) y mitigar el riesgo 
  transformando el castigo por tareas no asignadas en un coeficiente positivo de "ahorro de riesgo".
* Restricciones Duras Incorporadas:
  a) Unicidad: Máximo 1 analista asignado por tarea.
  b) Capacidad Acumulada por Deadline: Controla de forma acumulativa que si una tarea vence el día D (ej. Miércoles), su carga horaria debe absorberse con la disponibilidad acumulada entre Lunes y Miércoles.
  c) Límite de Volumen: Máximo 8 tareas por analista a la semana.
  d) Cobertura Crítica: Obligatoriedad de cubrir al menos el 90% de las tareas catalogadas como prioridad Alta.

Instrucciones de ejecución:
python Parte2.py
Salida: Genera el archivo 'solution_exact_medium.csv' con los emparejamientos.

========================================================================================================================================================================================================================
PARTE 3: OPTIMIZACIÓN HEURÍSTICA A GRAN ESCALA (Parte3.py)
========================================================================================================================================================================================================================
Objetivo: Resolver la asignación de la instancia grande 
('weekly_instance_large.csv') en un plazo menor a 1 minuto, donde los 
enfoques exactos se vuelven computacionalmente inviables.

Aspectos clave de la implementación:
* Algoritmo Diseñado: Greedy Search (Búsqueda Golosa) optimizada.
* Métrica de Densidad de Valor (Score Goloso): Por cada combinación 
  tarea-analista, se calcula un índice de eficiencia:
    Score = (Peso Prioridad + Score Riesgo - Costo Total) / Horas Estimadas
  Esto prioriza las asignaciones que aportan el mayor valor al negocio 
  mientras consumen la menor cantidad de tiempo posible del analista.
* Arquitectura Flexible (Panel de Control): El script incluye interruptores 
  booleanos en la parte superior para activar o relajar restricciones 
  del problema exacto de manera sencilla:
    - USAR_LIMITE_8_TAREAS
    - USAR_DEADLINE_DIARIO
    - USAR_CUOTA_90_PCT (Fase prioritaria que asegura la meta alta)
* Mecánica en Dos Fases: Si se activa la cuota del 90%, el motor realiza 
  primero una pasada exclusiva para blindar las tareas de alta prioridad 
  y luego ejecuta una fase general sobre el resto de las tareas usando el 
  orden decreciente del Score Goloso.

Instrucciones de ejecución:
python Parte3.py
Salida: Genera el archivo 'solution_heuristic_large.csv' en pocos segundos.

------------------------------------------------------------------------
3. REPRODUCIBILIDAD Y DISEÑO DE CÓDIGO
------------------------------------------------------------------------
* Semillas fijas: Todos los componentes estocásticos (como SMOTE, 
  train_test_split y Random Forest) utilizan 'random_state=42' para 
  asegurar que los resultados métricos sean idénticos en cada ejecución.
* Rendimiento computacional: En los scripts de optimización se evitó el 
  uso de bucles pesados sobre DataFrames de Pandas, utilizando en su 
  lugar estructuras basadas en 'itertuples()' y diccionarios indexados 
  (Tablas Hash de complejidad O(1)), garantizando ejecuciones fluidas.
========================================================================