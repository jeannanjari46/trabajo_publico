# IMPORTACIÓN DE MÓDULOS
# ==========================================================================================================================
import pandas as pd
import pulp
from collections import defaultdict

def resolver_optimizacion_exacta():
    
    # Carga de datos
    # ==========================================================================================================================
    ruta = r'C:\Users\PC RST\Downloads\Evaluación_Expertice_IT\Challenge_Jean_Luca_Nanjari_Pacheco\dataset' #Ajustar la ruta

    print("Cargando datasets para la instancia mediana...")
    df_instance = pd.read_csv(f'{ruta}\\weekly_instance_medium.csv')
    df_analysts = pd.read_csv(f'{ruta}\\analysts.csv')
    df_availability = pd.read_csv(f'{ruta}\\analyst_availability.csv')
    df_eligibility = pd.read_csv(f'{ruta}\\eligibility_medium.csv')

    # Diccionario para homogeneizar días a índices numéricos (Ej: Lunes=1, Martes=2, ...) y convertimos los días a números en los dataframes correspondientes
    DIAS_MAP = {'Lunes': 1, 'Martes': 2, 'Miércoles': 3, 'Jueves': 4, 'Viernes': 5, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}
    
    # 
    df_instance['deadline_day_num'] = df_instance['deadline_day'].map(DIAS_MAP)
    df_availability['day_num'] = df_availability['day'].map(DIAS_MAP)

    # Inicializar el problema de maximización usando pulp para luego preprocesar parámetros y conjuntos
    # ==========================================================================================================================
    prob = pulp.LpProblem("Asignacion_Semanal_Mediana", pulp.LpMaximize)

    # Mapeo del día de deadline de cada tarea e identificamos el conjunto de tareas de prioridad alta
    dict_deadline_tarea = dict(zip(df_instance['task_id'], df_instance['deadline_day_num']))
    tareas_alta_set = set(df_instance[df_instance['priority'].str.lower().isin(['alta', 'high'])]['task_id'])
    
    # Calcular la disponibilidad acumulada por analista por cada día de la semana
    disponibilidad_acum = defaultdict(dict)
    for a_id, group in df_availability.groupby('analyst_id'):
        horas_acumuladas = 0
        for d in range(1, 6):  # Se usa este rango para simular los dias de lunes a viernes
            horas_dia = group[group['day_num'] == d]['available_hours'].sum()
            horas_acumuladas += horas_dia
            disponibilidad_acum[a_id][d] = horas_acumuladas

    # Definición de variables de decisión binarias
    # ==========================================================================================================================
    
    # Filtramos la elegibilidad para crear solo las variables estrictamente válidas y preaparaamos las estructuras que guardaran información de
    # las tareas y analistas asociados
    df_filtrado = df_eligibility[df_eligibility['eligible'] == 1]
    
    pares_elegibles = []
    analistas_por_tarea = defaultdict(list)
    tareas_por_analista = defaultdict(list)
    
    # Diccionarios de parámetros indexados por la tupla (tarea, analista)
    horas_est = {}
    costo_operativo = {}
    peso_prioridad = {}
    score_riesgo = {}
    
    for row in df_filtrado.itertuples():
        t, a = row.task_id, row.analyst_id
        pares_elegibles.append((t, a))
        analistas_por_tarea[t].append(a)
        tareas_por_analista[a].append(t)
        horas_est[(t, a)] = row.estimated_hours
        costo_operativo[(t, a)] = row.estimated_hours * row.cost_per_hour
        peso_prioridad[(t, a)] = row.priority_weight
        score_riesgo[(t, a)] = row.risk_score

    x = pulp.LpVariable.dicts("assign", pares_elegibles, cat='Binary')

    # Formulación de la función objetivo
    # ==========================================================================================================================
    # Maximizamos: peso prioridad - costo operativo + score riesgo.
    # Al sumar el score de riesgo al coeficiente de x[t,a], modelamos el "ahorro" 
    # de mitigar el riesgo, equivalente a minimizar la penalización por dejarla vacía.
    prob += pulp.lpSum([(peso_prioridad[(t, a)] - costo_operativo[(t, a)] + score_riesgo[(t, a)]) * x[(t, a)]
        for (t, a) in pares_elegibles]), "Funcion_Objetivo_Balanceada"

    # Restricciones del Negocio
    # ==========================================================================================================================
    
    # a) Cada tarea puede ser asignada a lo más a un analista
    all_tasks = df_instance['task_id'].unique()
    for t in all_tasks:
        if t in analistas_por_tarea:
            prob += pulp.lpSum([x[(t, a)] for a in analistas_por_tarea[t]]) <= 1

    # b) Límite de carga por día (capacidad acumulada del deadline)
    # El día 5 (viernes) restringe automáticamente el 100% de la capacidad semanal total.
    all_analysts = df_analysts['analyst_id'].unique()
    for a in all_analysts:
        for d in range(1, 6):
            # Tareas asignables a 'a' cuyo deadline vence en o antes del día d
            tareas_antes_d = [t for t in tareas_por_analista[a] if dict_deadline_tarea.get(t, 6) <= d]
            capacidad_disponible = disponibilidad_acum[a].get(d, 0)
            
            prob += pulp.lpSum([horas_est[(t, a)] * x[(t, a)] for t in tareas_antes_d]) <= capacidad_disponible

    # c) Máximo 8 tareas por analista durante la semana
    for a in all_analysts:
        prob += pulp.lpSum([x[(t, a)] for t in tareas_por_analista[a]]) <= 8

    # d) Cubrir obligatoriamente al menos el 90% de las tareas de prioridad alta
    tareas_alta_instancia = [t for t in all_tasks if t in tareas_alta_set]
    if len(tareas_alta_instancia) > 0:
        prob += pulp.lpSum([
            x[(t, a)] for t in tareas_alta_instancia for a in analistas_por_tarea[t]
        ]) >= 0.90 * len(tareas_alta_instancia)

    # 7. Resolución del Modelo
    # ==========================================================================================================================
    print("Ejecutando el solver lineal...")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    print(f"\nEstado del solver: {pulp.LpStatus[status]}")
    
    # Procesamiento y exportación de resultados
    ## ==========================================================================================================================

    if status == pulp.LpStatusOptimal:
        asignaciones = []
        for (t, a) in pares_elegibles:
            if x[(t, a)].varValue is not None and x[(t, a)].varValue > 0.5:
                asignaciones.append({'task_id': t, 'analyst_id': a, 'estimated_hours': horas_est[(t, a)], 
                                     'cost_per_hour': costo_operativo[(t, a)] / horas_est[(t, a)]
                                     })
        
        df_solucion = pd.DataFrame(asignaciones)
        print(f"¡Éxito! se lograron asignar {len(df_solucion)} tareas.")
        
        # Validación de KPI de prioridad alta
        alta_asignadas = df_solucion[df_solucion['task_id'].isin(tareas_alta_set)].shape[0]
        cobertura_alta = (alta_asignadas / len(tareas_alta_instancia)) * 100 if tareas_alta_instancia else 0
        print(f"Cobertura final de prioridad alta: {cobertura_alta:.2f}% ({alta_asignadas}/{len(tareas_alta_instancia)})")
        
        # Exportar resultado requerido
        df_solucion[['task_id', 'analyst_id']].to_csv('solution_exact_medium.csv', index=False)
        print("Resultados exportados correctamente a 'solution_exact_medium.csv'.")
    else:
        print("ERROR: El modelo es infactible o no encontró solución óptima.")

resolver_optimizacion_exacta()