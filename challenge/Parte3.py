# IMPORTACIÓN DE MÓDULOS
# ==========================================================================================================================
import pandas as pd
import time
from collections import defaultdict


# PANEL DE CONTROL (INTERRUPTORES DE RELAJACIÓN)
# ==========================================================================================================================
# Podemos encender (True) o apagar (False) estas restricciones según la estrategia que escojamos
USAR_LIMITE_8_TAREAS = True   # Restricción original de cantidad máxima 
USAR_DEADLINE_DIARIO = False  # Control cronológico acumulado por día 
USAR_CUOTA_90_PCT    = False # Fase de asignación prioritaria para alta prioridad

# Esta es nuestra función principal para realizar la optimización relajada con la heurística conocida como GREEDY SEARCH o BÚSQUEDA GOLOSA
def ejecutar_optimizacion_flexible():
    start_time = time.time()
    print("==================================================")
    print("        INICIANDO HEURÍSTICA GREEDY SEARCH       ")
    print("==================================================")
    
  
    # Carga de datos
    # ==========================================================================================================================
    ruta = r'C:\Users\PC RST\Downloads\Evaluación_Expertice_IT\Challenge_Jean_Luca_Nanjari_Pacheco\dataset' # Ajustar la ruta
    
    df_analysts = pd.read_csv(f'{ruta}\\analysts.csv')
    df_eligibility = pd.read_csv(f'{ruta}\\eligibility_large.csv')
    df_instance_large = pd.read_csv(f'{ruta}\\weekly_instance_large.csv')
    df_availability = pd.read_csv(f'{ruta}\\analyst_availability.csv')

    # Diccionario blindado, pues, está preparado para recibir disintos valores en los datos.
    DIAS_MAP = {
        'lunes': 1, 'martes': 2, 'miércoles': 3, 'miercoles': 3, 'jueves': 4, 'viernes': 5,
        'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 'friday': 5
    }
    
    # Mapeos rápidos desde la instancia
    # Convertir días a números limpiando espacios en blanco y pasando todo a minúsculas (.str.lower())
    df_instance_large['deadline_day_num'] = df_instance_large['deadline_day'].astype(str).str.strip().str.lower().map(DIAS_MAP)
    df_instance_large['is_high_priority'] = df_instance_large['priority'].str.lower().isin(['alta', 'high']) 
    dict_deadline = dict(zip(df_instance_large['task_id'], df_instance_large['deadline_day_num']))
    dict_high_prio = dict(zip(df_instance_large['task_id'], df_instance_large['is_high_priority']))


    # Inicialización de las estructuras de estado
    # ==========================================================================================================================
    tareas_asignadas = set()
    asignacion_final = []
    
    # Estos seran nuestros rastreadores de uso por analista. Esto, tanto para contar las horas de trabajo y tareas que estos irán acumulando
    conteo_tareas_analista = {a: 0 for a in df_analysts['analyst_id']}
    horas_asignadas_cum = {a: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for a in df_analysts['analyst_id']}
    
    # Inicializar disponibilidad acumulada (día viernes por defecto contiene el total semanal)
    disponibilidad_acum = {
        a: {1: 0, 2: 0, 3: 0, 4: 0, 5: df_analysts.loc[df_analysts['analyst_id'] == a, 'max_hours_week'].values[0]} 
        for a in df_analysts['analyst_id']
    }
    
    # Si se activa el deadline diario, estructuramos las capacidades acumuladas reales por día
    if USAR_DEADLINE_DIARIO:
        df_availability['day_num'] = df_availability['day'].map(DIAS_MAP)
        for a_id, group in df_availability.groupby('analyst_id'):
            horas_acumuladas = 0
            for d in range(1, 6):
                horas_dia = group[group['day_num'] == d]['available_hours'].sum()
                horas_acumuladas += horas_dia
                if a_id in disponibilidad_acum:
                    disponibilidad_acum[a_id][d] = horas_acumuladas

    # Estas son nuestras funciones auxiliares para gestionar los estados, son muy importantes para el parte final donde asignamos 
    # las tareas a los analistas
    # ==========================================================================================================================
    def es_asignacion_valida(tarea_id, analista_id, horas_req, deadline_d):
        # Primer filtro: Límite superior de tareas semanales por analista
        if USAR_LIMITE_8_TAREAS:
            if conteo_tareas_analista[analista_id] >= 8:
                return False
        
        # Segundo filtro: Capacidad disponible según ventana temporal configurada
        if USAR_DEADLINE_DIARIO:
            # Debe caber de forma acumulada en todos los días desde su vencimiento en adelante
            for k in range(deadline_d, 6):
                if horas_asignadas_cum[analista_id][k] + horas_req > disponibilidad_acum[analista_id][k]:
                    return False
        else:
            # Solo controlamos que no sature la bolsa total de la semana entera (día 5)
            if horas_asignadas_cum[analista_id][5] + horas_req > disponibilidad_acum[analista_id][5]:
                return False                
        return True

    def registrar_asignacion(tarea_id, analista_id, horas_req, deadline_d):
        asignacion_final.append({'task_id': tarea_id, 'analyst_id': analista_id})
        tareas_asignadas.add(tarea_id)
        conteo_tareas_analista[analista_id] += 1
        # Al asignar, añadimos las horas a todos los cortes acumulados posteriores al deadline
        for k in range(deadline_d, 6):
            horas_asignadas_cum[analista_id][k] += horas_req

    # Preprocesamiento y cálculo del score goloso
    # ==========================================================================================================================
    df_validos = df_eligibility[df_eligibility['eligible'] == 1].copy()
    
    # Filtrar para quedarnos SOLO con las tareas de ESTA semana
    df_validos = df_validos[df_validos['task_id'].isin(df_instance_large['task_id'])]
    df_validos['costo_total'] = df_validos['cost_per_hour'] * df_validos['estimated_hours']
    df_validos['score_goloso'] = (df_validos['priority_weight'] + df_validos['risk_score'] - df_validos['costo_total']) / df_validos['estimated_hours']
    
    # Enlazar datos temporales y de criticidad a las combinaciones elegibles 
    df_validos['deadline_day_num'] = df_validos['task_id'].map(dict_deadline)
    df_validos['is_high_priority'] = df_validos['task_id'].map(dict_high_prio)
    

    # Por seguridad, eliminamos filas con días no válidos o vacíos (NaN)
    # =========================================================
    df_validos = df_validos.dropna(subset=['deadline_day_num'])
    
    # Forzamos el tipo entero para evitar errores y ordenamos usando el score goloso
    df_validos['deadline_day_num'] = df_validos['deadline_day_num'].astype(int)
    print(f"-> Datos preparados para optimización: {len(df_validos)} combinaciones elegibles detectadas.")
    df_ordenado = df_validos.sort_values(by='score_goloso', ascending=False)


    # Ejecución del motor
    # ==========================================================================================================================
    
    # Primera parte: Cobertura acelerada del 90% en prioridad alta (si es que está activa)
    if USAR_CUOTA_90_PCT:
        print("-> Ejecutando primera fase: Asignación prioritaria para alta prioridad...")
        df_alta = df_ordenado[df_ordenado['is_high_priority'] == True]
        total_alta_instancia = df_instance_large[df_instance_large['is_high_priority'] == True]['task_id'].nunique()
        meta_90_pct = total_alta_instancia * 0.90
        contador_alta = 0
        
        for row in df_alta.itertuples(index=False):
            if contador_alta >= meta_90_pct:
                print(f"   Meta del 90% de prioridad alta alcanzada ({contador_alta} tareas).")
                break
                
            if row.task_id in tareas_asignadas:
                continue
                
            if es_asignacion_valida(row.task_id, row.analyst_id, row.estimated_hours, row.deadline_day_num):
                registrar_asignacion(row.task_id, row.analyst_id, row.estimated_hours, row.deadline_day_num)
                contador_alta += 1

    # Segunda parte (general): Asignación general (o Única Fase si la cuota está desactivada)
    print("-> Ejecutando fase general de distribución por densidad de valor...")
    for row in df_ordenado.itertuples(index=False):
        if row.task_id in tareas_asignadas:
            continue
            
        if es_asignacion_valida(row.task_id, row.analyst_id, row.estimated_hours, row.deadline_day_num):
            registrar_asignacion(row.task_id, row.analyst_id, row.estimated_hours, row.deadline_day_num)

    # ---------------------------------------------------------
    # 6. Reporte de KPIs y Exportación
    # ---------------------------------------------------------
    # Si la lista tiene datos, crea el DataFrame normal; si no, lo inicializa vacío con sus columnas estructurales
    if asignacion_final:
        df_solucion = pd.DataFrame(asignacion_final)
    else:
        df_solucion = pd.DataFrame(columns=['task_id', 'analyst_id'])
    execution_time = time.time() - start_time
    
    print("\n==================================================")
    print("         MÉTRICAS DE LA SOLUCIÓN GENERADA         ")
    print("==================================================")
    print(f"Configuración aplicada:")
    print(f"  - Control Máximo 8 Tareas : {'ENCENDIDO' if USAR_LIMITE_8_TAREAS else 'APAGADO'}")
    print(f"  - Control Deadline Diario : {'ENCENDIDO' if USAR_DEADLINE_DIARIO else 'APAGADO'}")
    print(f"  - Estrategia Cuota 90%    : {'ENCENDIDO' if USAR_CUOTA_90_PCT else 'APAGADO'}")
    print(f"--------------------------------------------------")
    print(f"Tiempo de ejecución: {execution_time:.4f} segundos.")
    print(f"Tareas asignadas   : {len(df_solucion)} de {df_instance_large['task_id'].nunique()}")
    
    # Calcular cobertura real de alta prioridad para control analítico
    tareas_alta_set = set(df_instance_large[df_instance_large['is_high_priority'] == True]['task_id'])
    if tareas_alta_set:
        alta_asignadas = df_solucion[df_solucion['task_id'].isin(tareas_alta_set)].shape[0]
        pct_alta = (alta_asignadas / len(tareas_alta_set)) * 100
        print(f"Cobertura real de prioridad alta: {pct_alta:.2f}% ({alta_asignadas}/{len(tareas_alta_set)})")
        
    df_solucion.to_csv('solution_heuristic_large.csv', index=False)
    print("Archivo 'solution_heuristic_large.csv' exportado.")
    print("==================================================")

if __name__ == "__main__":
    ejecutar_optimizacion_flexible()