# IMPORTACIÓN DE MÓDULOS
# ==========================================

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ANÁLISIS EXPLORATORIO BREVE (EDA)
# ==========================================

# Cargamos nuestro dataset que contiene el historial de ejecuciones de las tareas

ruta = r'C:\Users\PC RST\Downloads\Evaluación_Expertice_IT\Challenge_Jean_Luca_Nanjari_Pacheco\dataset\historical_executions.csv' #Ajustar la ruta

df = pd.read_csv(ruta)

# Hacemos un Vistazo rápido a los datos
print("--- Información General ---")
print(df.info()) 
print("\n--- Estadísticas Descriptivas ---")
print(df.describe())
print("\n--- Balance de la Variable Objetivo ---")
print(df['sla_breached'].value_counts(normalize=True) * 100)


# LIMPIEZA Y PREPARACIÓN DE DATOS

# Eliminar ruido y fuga de datos. Son datos que NO aportarán al entrenamiento de nuestros modelos de aprendizaje, por lo tanto, son excluidos.
columnas_a_eliminar = ['execution_id', 'task_id', 'analyst_id', 'actual_hours', 'slack_hours', 'overtime_flag', 'reassignment_count']
df_limpio = df.drop(columns=columnas_a_eliminar)

# Sacamos los valores nulos, preparado los datos para el escalamiento 
df_limpio = df_limpio.dropna()

# Identificamos las variables categóricas de nuestros datos y las convertimos a números (0 y 1) con get_dummies
columnas_categoricas = ['required_skill', 'analyst_seniority', 'assigned_day', 'priority', 'complexity', 'client_criticality']
df_preparado = pd.get_dummies(df_limpio, columns=columnas_categoricas, drop_first=True)

# Separar características (X) y etiqueta (y). Además, inicializamos nuestros datos de entrenamiento y prueba.
X = df_preparado.drop(columns=['sla_breached'])
y = df_preparado['sla_breached']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Estandarización (escalado de datos). Esto es para evitar que que el modelo le de mas peso a características con valores más altos en relación a otras.
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# BALANCEO CON SMOTE
# ==========================================

# Aplicamos SMOTE solo a los datos de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_escalado, y_train)

# Comparamos el efecto de aplicar SMOTE  
print(f"Antes de SMOTE: {y_train.value_counts().to_dict()}")
print(f"Después de SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ENTRENAMIENTO DEL MODELO
# ==========================================

# Entrenamos la regresión logística con los datos balanceados. Además, calculamos las predicciones sobre el set de prueba 
# que esta desbalanceado porque no le aplicamos SMOTE
modelo_log = LogisticRegression(random_state=42)
modelo_log.fit(X_train_res, y_train_res)
y_pred = modelo_log.predict(X_test_escalado)
print("\n--- Reporte de Clasificación de Regresión Logística con SMOTE ---")
print(classification_report(y_test, y_pred))

# MODELO FINAL (RANDOM FOREST)
# ==========================================

# Definimos la cuadrícula de parámetros (grid) para aplicar un tuning simple e inicializamos el random forest base configurando con GridSearchCV
parametros_rf = { 'max_depth': [5, 10, 20, None], 'min_samples_leaf': [5, 10, 15, 20], 'n_estimators': [50, 100, 150, 200] }
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

busqueda = GridSearchCV(
    estimator=rf_base, 
    param_grid=parametros_rf, 
    scoring='f1', # Aca usamos la métrica f1 para maximizar el equilibrio entre precisión y recall
    cv=5, 
    n_jobs=-1
)

# Entrenamiento con validación cruzada usando los datos balanceados (reciclados del modelo de regresión logística) para obtener el mejor modelo balanceado
print("Iniciando búsqueda de hiperparámetros...")
busqueda.fit(X_train_res, y_train_res)
modelo_rf_perfecto = busqueda.best_estimator_
print(f"Mejores parámetros: {busqueda.best_params_}")

# Definir los costos del negocio (el criterio de la empresa)
COSTO_FP = 50   # Costo por falsa alarma (tiempo del analista)
COSTO_FN = 300  # Costo por fallo de SLA (multa o pérdida de cliente)

# Obtenemos las probabilidades crudas e iteramos para encontrar el umbral que minimiza la pérdida total
probs_test = modelo_rf_perfecto.predict_proba(X_test_escalado)[:, 1]

umbrales_a_probar = np.arange(0.25, 0.51, 0.01)
mejor_umbral = 0.5
minima_perdida = float('inf')
mejor_y_pred = None

print("Analizando impacto financiero por umbral...")

for u in umbrales_a_probar:
    y_pred_u = (probs_test >= u).astype(int)
    cm = confusion_matrix(y_test, y_pred_u)
    
    # Extraer errores: cm = [[TN, FP], [FN, TP]]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    # Función de impacto: Loss = (FP * C_fp) + (FN * C_fn)
    perdida_total = (fp * COSTO_FP) + (fn * COSTO_FN)
    
    if perdida_total < minima_perdida:
        minima_perdida = perdida_total
        mejor_umbral = u
        mejor_y_pred = y_pred_u

# Resultados del umbral económico
print(f"\n==================================================")
print(f"UMBRAL ECONÓMICO ÓPTIMO: {mejor_umbral:.2f}")
print(f"PÉRDIDA MÍNIMA ESTIMADA: ${minima_perdida:,}")
print(f"==================================================")
print(f"\n--- Reporte Final de Clasificación RF con Umbral Económico ---")
print(classification_report(y_test, mejor_y_pred))
print("\n--- Matriz de Confusión Final ---")
final_cm = confusion_matrix(y_test, mejor_y_pred)
print(final_cm)
# ROC-AUC (global del modelo)
print(f"\nROC-AUC Score global: {roc_auc_score(y_test, probs_test):.4f}")