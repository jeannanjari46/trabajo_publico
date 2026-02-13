### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 9918c2c8-6da2-4ee8-989c-856b58921681
begin
	using DataFrames
	using CSV
	using Statistics
	using Printf
	using GLM
	using DecisionTree
	using Random
	using Dates
	using Plots
	
	# =============================================================================
	# 1. FUNCIÓN DE PARSEO DE TAGS (NÚCLEO GENÉRICO)
	# =============================================================================
	
	function parse_tag_string(tag::AbstractString)
	    # Diccionario para guardar lo que encontremos
	    params = Dict{Symbol, Any}()

		tag_str = String(tag)
		
	    # 1. Dividir por guiones bajos
	    tokens = split(tag_str, "_")
		
	    regex_kv = r"^([a-zA-Z]+)([\d\.]+)$"
	    
	    for token in tokens
	        m = match(regex_kv, String(token))
	        
	        if m !== nothing
				
	            key = m.captures[1]
	            val = m.captures[2]
	            
	            params[Symbol(key)] = parse(Float64, val)
				
			else
	            if all(isletter, token)
					
	                params[Symbol(token)] = String(token)
	            end
	        end
	    end
	    
	    return params
	end
	
	# =============================================================================
	# 2. FUNCIONES DE PROCESAMIENTO DE DATAFRAME
	# =============================================================================
	
	function clean_redundant_columns!(df::DataFrame)
		
	    redundant_cols = [:Run, :InstanceSeed]
	    
	    # Filtramos solo las que existen en el DF actual
	    cols_to_remove = intersect(redundant_cols, propertynames(df))
	    
	    if !isempty(cols_to_remove)
	        select!(df, Not(cols_to_remove))
	        println("Columnas redundantes eliminadas: $cols_to_remove")
	    end
	    return df
	end
	
	function unpack_instance_tags!(df::DataFrame)
	    if !("InstanceTag" in names(df))
	        error("❌ No se encontró la columna 'InstanceTag' en el CSV.")
	    end
	
	    println("Desempaquetando tags de instancia...")
	
	    # 1. Analizar todos los tags únicos para descubrir qué columnas nuevas necesitamos
	    unique_tags = unique(df.InstanceTag)
	    discovered_params = Set{Symbol}()
	    
	    # Pre-parseamos los tags únicos para eficiencia
	    tag_map = Dict{String, Dict{Symbol, Any}}()
	    
	    for t in unique_tags
	        p = parse_tag_string(t)
	        tag_map[t] = p
	        union!(discovered_params, keys(p))
	    end
	    
	    # Quitamos 'n' si ya existe como columna en el CSV original (para no duplicar)
	    if :n in propertynames(df)
	        delete!(discovered_params, :n)
	    end
	
	    println("Parámetros descubiertos: $(collect(discovered_params))")
	
		# 2. Crear las nuevas columnas en el DataFrame con el tipo correcto
	    for param in discovered_params
			
	        found_values = [tag_map[t][param] for t in keys(tag_map) if haskey(tag_map[t], param)]
			
	        is_string_col = any(x -> x isa AbstractString, found_values)
	        
	        if is_string_col
	
	            df[!, param] = Vector{Union{String, Missing}}(missing, nrow(df))
	        else
	            # Caso Numérico: Inicializamos con NaN
	            df[!, param] = Vector{Float64}(undef, nrow(df))
	            df[!, param] .= NaN
	        end
	    end
	
	    # 3. Llenar los datos
	    # Iteramos fila por fila
	    for row in eachrow(df)
	        tag = row.InstanceTag
	        if haskey(tag_map, tag)
	            params = tag_map[tag]
	            for (k, v) in params
	                if k in discovered_params # Solo si es una columna nueva
	                    row[k] = v
	                end
	            end
	        end
	    end
	    
	    return df
	end
	
	# =============================================================================
	# 3. FUNCIÓN MAESTRA (WRAPPER)
	# =============================================================================
	
	function load_and_process_csv(filepath::String)
	    println("\n📂 Cargando archivo: $filepath")
	    
	    if !isfile(filepath)
	        error("El archivo no existe.")
	    end
	    
	    # Cargar CSV
	    df = CSV.read(filepath, DataFrame)
	    
	    # 1. Limpieza
	    clean_redundant_columns!(df)
	    
	    # 2. Parsing Dinámico
	    unpack_instance_tags!(df)
	    
	    println("✅ Procesamiento inicial completado. Dimensiones: $(size(df))")
	    return df
	end
end

# ╔═╡ 0601c443-3968-48a1-b4c0-345e74e6b84f
begin
		ruta_archivo = "C:\\Users\\PC RST\\Documents\\tesis_pruebas\\Resultados\\PPA_vs_EPA_WINS_Cuadratica.csv"  
		df_procesado = load_and_process_csv(ruta_archivo)
			
		df_procesado
end

# ╔═╡ 297fc9af-c587-4616-b47b-722f8911abc0
function analisis_completo_final(df::DataFrame)
    # 1. Configuración
    fixed_cols_set = Set(["n", "InstanceTag", "γ", "Tol", "SeedX0", "Iter_Classic", "Iter_Epi"])
    targets = ["Iter_Classic", "Iter_Epi"]
    
    # 2. Detectar columnas dinámicamente
    all_cols = names(df)
    other_cols = [c for c in all_cols if !(c in fixed_cols_set)]
    
    cat_grouping = String[]
    num_features = String[]
    
    for col in other_cols
        T = eltype(df[!, col])
        if T <: AbstractString || T <: Union{Missing, AbstractString}
            push!(cat_grouping, col)
        elseif T <: Number || T <: Union{Missing, Number}
            push!(num_features, col)
        end
    end
    
    # Variables X (Gamma + Numéricas caja negra)
    features_X = vcat("γ", num_features)
    
    println("--- Iniciando Análisis ---")
    println("Agrupando por: n, ", join(cat_grouping, ", "))
    println("Modelando efectos de: ", join(features_X, ", "))
    println("------------------------\n")

    # 3. Agrupación
    group_keys = vcat("n", cat_grouping)
    gdf = groupby(df, group_keys)
    
    resultados = Dict()

    for (key, subdf) in pairs(gdf)
        # Limpiamos datos para Targets y Features
        cols_needed = vcat(targets, features_X)
        data_clean = dropmissing(subdf[:, cols_needed])
        
        n_rows = size(data_clean, 1)
        n_vars = length(features_X)
        
        if n_rows > n_vars + 1
            # --- PARTE A: Correlación entre Targets (Lo que faltaba) ---
            # Si la desviación estándar es 0, cor() da NaN. Lo manejamos.
            c_targets = NaN
            try
                c_targets = cor(data_clean.Iter_Classic, data_clean.Iter_Epi)
            catch
                c_targets = NaN
            end

            # --- PARTE B: Regresión Lineal (Solución C) ---
            res_modelos = Dict()
            
            for target in targets
                try
                    # Fórmula: Target ~ Feature1 + Feature2...
                    f = term(Symbol(target)) ~ sum(term.(Symbol.(features_X)))
                    model = lm(f, data_clean)
                    
                    # Extraer coeficientes
                    ct = coeftable(model)
                    df_coefs = DataFrame(
                        Variable = ct.rownms,
                        Beta = round.(ct.cols[1], digits=4),
                        P_Value = round.(ct.cols[4], digits=4),
                        Signif = ct.cols[4] .< 0.05
                    )
                    
                    res_modelos[target] = (
                        r2 = round(r2(model), digits=4),
                        coefs = df_coefs
                    )
                catch e
                    res_modelos[target] = "Error Ajuste"
                end
            end
            
            # GUARDAMOS TODO EN LA ESTRUCTURA FINAL
            resultados[values(key)] = (
                n_muestras = n_rows,
                corr_entre_iters = c_targets,  # <--- ¡AQUÍ ESTÁ EL CAMPO QUE FALTABA!
                modelos = res_modelos
            )
            
        else
            resultados[values(key)] = "Datos insuficientes ($n_rows filas)"
        end
    end

    return resultados
end

# ╔═╡ fcafd135-db87-4df3-83c5-141806e7ca4c
#res = analisis_completo_final(df_procesado)

# ╔═╡ 5c438831-6dbc-4502-83f2-4e657807ec14
# ╠═╡ disabled = true
#=╠═╡
for (grupo, data) in res
    println("\n========================================")
    println("GRUPO: $grupo")
    
    if data isa String
        println("  -> $data") # Mensaje de error o datos insuficientes
    else
        println("  Muestras: $(data.n_muestras)")
        
        # Ahora sí existe el campo corr_entre_iters
        val_corr = isnan(data.corr_entre_iters) ? "NaN (Ctes)" : round(data.corr_entre_iters, digits=4)
        println("  Corr(Iter_Classic, Iter_Epi): $val_corr")
        
        println("\n  --- Resultados de Regresión ---")
        for (target, modelo) in data.modelos
            println("  TARGET: $target")
            if modelo isa String
                println("    Error: $modelo")
            else
                println("    R²: $(modelo.r2)")
                # Mostramos solo filas significativas para limpiar la vista (opcional)
                display(modelo.coefs) 
            end
            println("")
        end
    end
end
  ╠═╡ =#

# ╔═╡ b161a616-a48e-44cc-9f32-86b7ef07eaa0
function detectar_tendencia(model, X_base, col_idx, col_values)
    # 1. Creamos 20 puntos de prueba desde el min hasta el max de la variable
    min_v, max_v = minimum(col_values), maximum(col_values)
    if min_v == max_v return "Constante" end
    
    steps = range(min_v, max_v, length=20)
    
    # 2. Creamos una matriz simulada
    # Repetimos la fila promedio de X_base 20 veces
    # (Calculamos la mediana de cada columna para tener un caso base "típico")
    X_mediana = median(X_base, dims=1)
    X_sim = repeat(X_mediana, 20)
    
    # 3. Solo variamos la columna de interés
    X_sim[:, col_idx] .= steps
    
    # 4. Predecimos qué pasaría
    preds = apply_forest(model, X_sim)
    
    # 5. Calculamos la correlación entre el cambio de la variable y la predicción
    # Esto nos dice la dirección global
    c = cor(collect(steps), preds)
    
    if isnan(c) return "Plana/Sin Efecto" end
    
    # Interpretación heurística
    if c > 0.6
        return "Creciente (Sube)"    # Si sube X, sube Y
    elseif c < -0.6
        return "Decreciente (Baja)"  # Si sube X, baja Y
    elseif abs(c) <= 0.6
        # Si la correlación es baja pero la importancia es alta, es una curva (U o n)
        return "No Lineal / Compleja"
    else
        return "Indefinida"
    end
end

# ╔═╡ a3908066-6523-4d03-a4a2-a35ecade0794
function analisis_rf_con_tendencia(df::DataFrame)
    fixed_cols_set = Set(["n", "InstanceTag", "γ", "Tol", "SeedX0", "Iter_Classic", "Iter_Epi"])
    targets = ["Iter_Classic", "Iter_Epi"]
    
    all_cols = names(df)
    other_cols = [c for c in all_cols if !(c in fixed_cols_set)]
    
    cat_grouping = String[]
    num_features = String[]
    for col in other_cols
        T = eltype(df[!, col])
        if T <: AbstractString || T <: Union{Missing, AbstractString}
            push!(cat_grouping, col)
        elseif T <: Number || T <: Union{Missing, Number}
            push!(num_features, col)
        end
    end
    features_X_names = vcat("γ", num_features)
    
    println("--- Análisis RF con Tendencia ---")
    println("Variables: ", join(features_X_names, ", "))
    println("--------------------------------\n")

    group_keys = vcat("n", cat_grouping)
    gdf = groupby(df, group_keys)
    resultados = Dict()

    for (key, subdf) in pairs(gdf)
        cols_needed = vcat(targets, features_X_names)
        data_clean = dropmissing(subdf[:, cols_needed])
        n_rows = size(data_clean, 1)
        
        if n_rows > 10 
            res_modelos = Dict()
            X = Matrix(data_clean[:, features_X_names])
            
            for target in targets
                y = Vector(data_clean[!, target])
                model = build_forest(y, X, -1, 50, 0.7, -1)
                
                # R2
                preds = apply_forest(model, X)
                ss_res = sum((y .- preds).^2)
                ss_tot = sum((y .- mean(y)).^2)
                r2_score = ss_tot ≈ 0 ? 0.0 : (1 - (ss_res / ss_tot))
                
                # Importancia
                imp = split_importance(model)
                feat_imp_pairs = []
                if length(imp) == length(features_X_names)
                    for (i, val) in enumerate(imp)
                        push!(feat_imp_pairs, (i, features_X_names[i], val)) # Guardamos índice también
                    end
                end
                
                # Normalizar importancia
                total_gain = sum([x[3] for x in feat_imp_pairs])
                if total_gain == 0 total_gain = 1.0 end
                
                # --- AQUÍ CALCULAMOS LA TENDENCIA ---
                vec_nombres = String[]
                vec_import  = Float64[]
                vec_tenden  = String[]
                
                for (idx, nombre, val) in feat_imp_pairs
                    imp_norm = val / total_gain
                    
                    # Solo calculamos tendencia si la variable importa algo (> 1%)
                    # para ahorrar tiempo de cómputo
                    tendencia = "Irrelevante"
                    if imp_norm > 0.01
                        # Pasamos la columna de datos reales para saber min/max
                        col_vals = X[:, idx]
                        tendencia = detectar_tendencia(model, X, idx, col_vals)
                    end
                    
                    push!(vec_nombres, nombre)
                    push!(vec_import, imp_norm)
                    push!(vec_tenden, tendencia)
                end
                
                df_res = DataFrame(
                    Variable = vec_nombres,
                    Importancia = vec_import,
                    Efecto = vec_tenden # <--- NUEVA COLUMNA
                )
                
                # Ordenar por importancia
                sort!(df_res, :Importancia, rev=true)
                
                res_modelos[target] = (r2 = round(r2_score, digits=4), info = df_res)
            end
            resultados[values(key)] = (n=n_rows, modelos=res_modelos)
        else
            resultados[values(key)] = "Insuficientes datos"
        end
    end
    return resultados
end

# ╔═╡ 6bf42edb-0e1a-4fec-8add-2bf26a35e4e3
#res_rf = analisis_rf_con_tendencia(df_procesado)

# ╔═╡ 1e132f3e-4983-4db1-a154-f3cffd34fe45
# ╠═╡ disabled = true
#=╠═╡
for (grupo, data) in res_rf
    println("\n========================================")
    println("GRUPO: $grupo")
    if !(data isa String)
        for (target, info) in data.modelos
            println("\n  TARGET: $target (R²: $(info.r2))")
            println("  ------------------------------------------------")
            # Imprimimos formateado
            display(first(info.info, 6)) # Top 6 variables
        end
    else
        println(data)
    end
end
  ╠═╡ =#

# ╔═╡ 257cebe4-49c1-4a39-a19f-789f81fdd0f4
function exportar_reporte_txt(df::DataFrame, nombre_archivo::String="reporte_analisis.txt")
    
    # 1. Ejecutar análisis
    println("Calculando modelos lineales...")
    res_lineal = analisis_completo_final(df)
    
    println("Calculando modelos no lineales (Random Forest)...")
    res_rf = analisis_rf_con_tendencia(df)
    
    println("Escribiendo resultados en: $nombre_archivo")

    # 2. Escribir archivo
    open(nombre_archivo, "w") do io
        
        println(io, "==========================================================")
        println(io, "REPORTE DE ANÁLISIS DE DATOS: OPTIMIZACIÓN")
        println(io, "Fecha: $(now())")
        println(io, "==========================================================\n")
        
        for (grupo, data_lin) in res_lineal
            if !haskey(res_rf, grupo)
                println("Aviso: Grupo $grupo no encontrado en RF")
                continue
            end
            data_rf = res_rf[grupo]
            
            println(io, "__________________________________________________________")
            println(io, "GRUPO: $grupo")
            
            if data_lin isa String
                println(io, "  [!] Error/Datos Insuficientes: $data_lin")
                continue
            end

            println(io, "  Muestras (N): $(data_lin.n_muestras)")
            val_corr = isnan(data_lin.corr_entre_iters) ? "NaN" : round(data_lin.corr_entre_iters, digits=4)
            println(io, "  Correlación (Iter_Classic vs Iter_Epi): $val_corr")
            println(io, "")

            targets = keys(data_lin.modelos) 
            
            for target in targets
                println(io, "  >> TARGET: $target")
                
                # --- A) LINEAL ---
                modelo_l = data_lin.modelos[target]
                println(io, "    [ENFOQUE LINEAL]")
                if modelo_l isa String
                    println(io, "      Error: $modelo_l")
                else
                    println(io, "      R²: $(modelo_l.r2)")
                    println(io, "      Coeficientes Significativos (P < 0.05):")
                    
                    df_filtrado = filter(:Signif => x -> x == true, modelo_l.coefs)
                    if nrow(df_filtrado) > 0
                        # Sintaxis corregida para DataFrames.jl
                        show(io, MIME("text/plain"), df_filtrado; summary=false, show_row_number=false)
                    else
                        println(io, "      (Ninguna variable resultó significativa)")
                    end
                    println(io, "")
                end
                
                # --- B) NO LINEAL (RF) ---
                modelo_rf = data_rf.modelos[target]
                println(io, "    [ENFOQUE NO-LINEAL (RF)]")
                if modelo_rf isa String
                    println(io, "      Error: $modelo_rf")
                else
                    # CORRECCIÓN 1: Accedemos directo a .r2 (no dentro de .info)
                    println(io, "      R²: $(modelo_rf.r2)") 
                    
                    println(io, "      Importancia y Tendencia (Top 5):")
                    
                    # CORRECCIÓN 2: El DataFrame está en .info
                    top_vars = first(modelo_rf.info, 5) 
                    
                    # Sintaxis corregida
                    show(io, MIME("text/plain"), top_vars; summary=false, show_row_number=false)
                    println(io, "")
                end
                println(io, "    ------------------------------------")
            end
            println(io, "\n")
        end
    end
    println("¡Listo! Archivo guardado correctamente.")
end

# ╔═╡ e1321e1e-1122-43fa-a112-5ca040f380e0
#exportar_reporte_txt(df_procesado, "C:\\Users\\PC RST\\Documents\\tesis_pruebas\\Resultados\\Estadísticas.txt")

# ╔═╡ b8ad1009-cf9e-46ea-be17-ef9cc2e5bf80
function guardar_mejores_casos(df::DataFrame, ruta_directorio::String)
    # 1. Filtramos donde Epi gana estrictamente a Classic
    df_filtrado = filter(row -> row.Iter_Epi < row.Iter_Classic, df)
    
    # 2. Construimos la ruta completa
    nombre_archivo = "Casos_Epi_Ganadores.csv"
    ruta_completa = joinpath(ruta_directorio, nombre_archivo)
    
    # 3. Guardamos
    # Verificamos si el directorio existe, si no, intentamos crearlo (opcional, pero recomendado)
    if !isdir(ruta_directorio)
        mkpath(ruta_directorio)
    end
    
    CSV.write(ruta_completa, df_filtrado)
    
    println("Tarea 1 Completada: Archivo guardado en $ruta_completa")
    println("  -> Total filas filtradas: $(nrow(df_filtrado))")
    
    return df_filtrado # Lo retornamos por si queremos usarlo luego
end

# ╔═╡ a0073338-9624-45bd-b682-30559c743171
function generar_resumen_general(df_input::DataFrame, 
                                 ruta_salida::String, 
                                 params_ignorar::Vector{String}, 
                                 val_penalizacion::Float64)

    println("--- Generando Resumen General (Grilla Cerrada) ---")

    # 1. IDENTIFICACIÓN DE COLUMNAS
    # Definimos las columnas que NO son parámetros estructurales del problema
    # Según tu descripción: InstanceTag se ignora/borra.
    cols_metricas = ["Iter_Classic", "Iter_Epi"] 
    # Agrega tiempos si los tienes: ["Time_Classic", "Time_Epi"]
    
    col_run = "SeedX0"  # Esta es tu columna sagrada de iteraciones/runs
    cols_basura = ["InstanceTag"] 

    # Todas las columnas disponibles
    all_cols = names(df_input)

    # A. Identificar Parámetros Fijos (Los que definirán cada fila de la tabla final)
    # Son: Todo - Métricas - Run - Ignorados - Basura
    cols_a_excluir = vcat(cols_metricas, [col_run], params_ignorar, cols_basura)
    params_fijos = setdiff(all_cols, cols_a_excluir)

    println(" -> Fila definida por: $(join(params_fijos, ", "))")
    println(" -> Promediando sobre: $col_run + $(join(params_ignorar, ", "))")

    # 2. CONSTRUCCIÓN DE LA GRILLA PERFECTA (Producto Cartesiano)
    
    # Parte A: Tabla de combinaciones únicas de los parámetros FIJOS
    df_fijos = unique(df_input[:, params_fijos])

    # Parte B: Universo de 'SeedX0' (Todos los runs posibles)
    # OJO: Asumimos que quieres usar TODOS los seeds únicos encontrados en toda la tabla.
    # Si cada 'n' tiene seeds distintos, el crossjoin generará muchas filas inválidas.
    # Si los seeds son "globales" (ej. hash dependiente de n), esto está bien.
    # Para seguridad máxima en tesis: Usamos los seeds únicos presentes en el DF global.
    df_runs = DataFrame(col_run => unique(df_input[:, col_run]))

    # Parte C: Universo de Parámetros Ignorados
    # Por cada parámetro ignorado, sacamos sus valores únicos GLOBALES.
    grid_components = Any[df_fijos, df_runs]
    
    for p in params_ignorar
        vals = unique(df_input[:, p])
        push!(grid_components, DataFrame(Symbol(p) => vals))
    end

    # D. EL CRUCE FINAL (Aquí se crean los huecos que faltaban)
    df_grid_full = crossjoin(grid_components...)

    # 3. FUSIÓN Y PENALIZACIÓN
    
    # Unimos la grilla perfecta con los datos reales
    # Las llaves de unión son: Fijos + Ignorados + SeedX0
    join_keys = intersect(names(df_grid_full), names(df_input))
    
    df_merged = leftjoin(df_grid_full, df_input, on = join_keys)

    # Rellenamos los Missing con la Penalización
    # Si no hubo match, es porque esa simulación se saltó o filtró.
    df_merged.Iter_Classic = coalesce.(df_merged.Iter_Classic, val_penalizacion)
    df_merged.Iter_Epi     = coalesce.(df_merged.Iter_Epi, val_penalizacion)
    
    # 4. AGRUPACIÓN Y PROMEDIO
    
    gdf = groupby(df_merged, params_fijos)
    
    resumen = combine(gdf,
        :Iter_Classic => mean => :Promedio_Classic,
        :Iter_Epi     => mean => :Promedio_Epi,
        nrow => :Cant_Instancias_Promediadas # Esto debe dar igual para todos
    )

    # Redondeo
    resumen.Promedio_Classic = round.(resumen.Promedio_Classic, digits=2)
    resumen.Promedio_Epi     = round.(resumen.Promedio_Epi, digits=2)

    # Ordenar por n si existe, o por lo que haya
    if "n" in names(resumen)
        sort!(resumen, ["n"])
    else
        sort!(resumen, params_fijos)
    end

    # 5. GUARDADO
    if !isdir(ruta_salida) mkpath(ruta_salida) end
    f_out = joinpath(ruta_salida, "Resumen_General.csv")
    CSV.write(f_out, resumen)
    
    println(" -> Guardado en: $f_out")
    return df_merged
end

# ╔═╡ 568946c7-2850-4a4c-a7bf-5ac0388f4374
function generar_resumen_cascada(df::DataFrame, ruta_salida::String, limite_filas::Int, incluir_gamma::Bool, decimales::Int, params_ignorar::AbstractVector=String[])
    
    # 1. Preparación de columnas de agrupación
    cols_sistema = Set(["Iter_Classic", "Iter_Epi", "SeedX0", "InstanceTag", "n", "γ"])
    all_cols = names(df)
    
    black_box_params = sort([c for c in all_cols if !(c in cols_sistema)])
    
    parametros_orden = String["n"]
    if incluir_gamma 
        push!(parametros_orden, "γ") 
    end
    append!(parametros_orden, black_box_params)
    
    # --- FILTRADO DE PARÁMETROS ---
    filter!(p -> !(String(p) in string.(params_ignorar)), parametros_orden)
    
    println("--- Generación de Resumen (Muestreo Uniforme) ---")
    println("Agrupando por: $(join(parametros_orden, " + "))")
    if !isempty(params_ignorar)
        println("Ignorando: $(join(params_ignorar, ", "))")
    end
    
    # 2. CÁLCULO DE ESTADÍSTICAS GLOBALES
    gdf = groupby(df, parametros_orden)
    
    pool_global = combine(gdf,
        :Iter_Classic => mean => :Promedio_Classic,
        :Iter_Classic => std  => :DesvStd_Classic,
        :Iter_Epi     => mean => :Promedio_Epi,
        :Iter_Epi     => std  => :DesvStd_Epi,
        nrow => :Cantidad_Instancias
    )

    # Redondeo
    cols_num = [:Promedio_Classic, :DesvStd_Classic, :Promedio_Epi, :DesvStd_Epi]
	
    for c in cols_num 
        pool_global[!, c] = round.(pool_global[!, c], digits=decimales) 
    end

    # 3. SEPARACIÓN Y ORDENAMIENTO PREVIO
    # Es vital ordenar ANTES de muestrear para que el salto tenga sentido físico (barrer n, gamma, etc.)
    pool_ganadores  = filter(row -> row.Promedio_Epi < row.Promedio_Classic, pool_global)
    pool_perdedores = filter(row -> row.Promedio_Epi >= row.Promedio_Classic, pool_global)

    sort!(pool_ganadores, parametros_orden)
    sort!(pool_perdedores, parametros_orden)

    # --- FUNCIÓN DE MUESTREO UNIFORME (TU NUEVA LÓGICA) ---
    function obtener_muestreo_uniforme(pool::DataFrame, limite::Int)
        total_filas = nrow(pool)
        
        # Caso A: Si tenemos menos (o igual) combinaciones que el límite, mostramos TODO.
        if total_filas <= limite
            return pool
        end
        
        # Caso B: Si hay más combinaciones que el límite, seleccionamos índices equidistantes.
        # Ejemplo: Total 300, Limite 10 -> indices: 1, 34, 67, 100... aprox.
        # range(1, total, length=limite) nos da los puntos flotantes exactos.
        # round.(Int, ...) los convierte a índices de fila válidos.
        indices = unique(round.(Int, range(1, total_filas, length=limite)))
        
        return pool[indices, :]
    end

    # 4. APLICACIÓN DEL MUESTREO
    tabla_ganadores = obtener_muestreo_uniforme(pool_ganadores, limite_filas)
    tabla_perdedores = obtener_muestreo_uniforme(pool_perdedores, limite_filas)

    # 5. GUARDADO
    if !isdir(ruta_salida) mkpath(ruta_salida) end
    
    f_win = joinpath(ruta_salida, "Resumen_Ganadores.csv")
    f_lose = joinpath(ruta_salida, "Resumen_NoGanadores.csv")
    
    if nrow(tabla_ganadores) > 0 
        CSV.write(f_win, tabla_ganadores) 
        println("  -> Guardado Ganadores: $f_win")
        println("     (Mostrando $(nrow(tabla_ganadores)) filas representativas de un total de $(nrow(pool_ganadores)))")
    else
        println("  -> [AVISO] No hay configuraciones donde Epi gane en promedio.")
    end

    if nrow(tabla_perdedores) > 0 
        CSV.write(f_lose, tabla_perdedores) 
        println("  -> Guardado No Ganadores: $f_lose")
        println("     (Mostrando $(nrow(tabla_perdedores)) filas representativas de un total de $(nrow(pool_perdedores)))")
    end
end

# ╔═╡ 00de7894-6f53-471b-b89a-c4381313c83d
function imprimir_mejores_diferencias_relativas(df::DataFrame, ruta_salida::String, top_n::Int=5, guardar_csv::Bool=false)
    
    if nrow(df) == 0
        println("El DataFrame está vacío. No hay instancias para mostrar.")
        return
    end

    # 1. Copia y cálculo de métrica
    # Usamos (Classic - Epi) / Max(Classic, Epi)
    df_temp = copy(df)
    
    # Manejamos el caso de división por cero si max es 0 (raro en iteraciones, pero posible por bugs)
    df_temp.Mejora_Relativa = [
        let m = max(row.Iter_Classic, row.Iter_Epi)
            m > 0 ? (row.Iter_Classic - row.Iter_Epi) / m : 0.0
        end 
        for row in eachrow(df_temp)
    ]
    
    # 2. Ordenar de Mayor a Menor (Las mejores victorias primero)
    sort!(df_temp, :Mejora_Relativa, rev=true)
    
    # 3. Recortar el Top N (o lo que haya disponible)
    cantidad_a_mostrar = min(top_n, nrow(df_temp))
    df_top = df_temp[1:cantidad_a_mostrar, :] # Este es el sub-dataframe ganador
    
    # --- IMPRESIÓN EN CONSOLA ---
    println("\n=== TOP $cantidad_a_mostrar INSTANCIAS CON MAYOR DIFERENCIA RELATIVA ===")
    
    for i in 1:cantidad_a_mostrar
        fila = df_top[i, :]
        pct = round(fila.Mejora_Relativa * 100, digits=2)
        
        println("\n[#$i] Mejora: $pct% (Classic=$(fila.Iter_Classic) -> Epi=$(fila.Iter_Epi))")
        println("     Config: n=$(fila.n) | γ=$(fila.γ) | Tag=$(fila.InstanceTag)")
        println("     Semilla: $(fila.SeedX0)")
    end
    println("\n==========================================================")

    # --- GUARDADO EN CSV (Opcional) ---
    if guardar_csv
        # Verificamos que la ruta no esté vacía
        if isempty(ruta_salida)
            println("⚠️ [AVISO] Se pidió guardar CSV pero 'ruta_salida' está vacía. No se guardó.")
        else
            if !isdir(ruta_salida) mkpath(ruta_salida) end
            
            nombre_archivo = "Top_$(cantidad_a_mostrar)_Mejores_Instancias.csv"
            path_completo = joinpath(ruta_salida, nombre_archivo)
            
            # Guardamos el subconjunto top
            CSV.write(path_completo, df_top)
            println("✅ Tabla Top guardada exitosamente en: $path_completo")
        end
    end
end

# ╔═╡ 5c8c6f4f-301f-4cec-886d-f46e58b93a44
function generar_mosaicos_comparativos(df::DataFrame, ruta_salida::String, num_mosaicos::Int, params_ignorar::AbstractVector=String[])
    
    # 1. IDENTIFICACIÓN DE COLUMNAS
    cols_sistema = Set(["Iter_Classic", "Iter_Epi", "SeedX0", "InstanceTag", "n", "γ"])
    all_cols = names(df)
    black_box_params = sort([c for c in all_cols if !(c in cols_sistema)])
    
    clave_instancia = String["n"]
    append!(clave_instancia, black_box_params)
    filter!(p -> !(String(p) in string.(params_ignorar)), clave_instancia)
    
    clave_punto = copy(clave_instancia)
    push!(clave_punto, "γ")
    
    println("\n--- Generando Mosaicos Comparativos (Estilizados) ---")
    println("Agrupando instancias por: $(join(clave_instancia, " + "))")

    # 2. AGREGACIÓN DE DATOS
    gdf_puntos = groupby(df, clave_punto)
    
    df_curvas = combine(gdf_puntos, 
        :Iter_Classic => (x -> mean(skipmissing(x))) => :Mean_Classic,
        :Iter_Epi => (x -> mean(skipmissing(x))) => :Mean_Epi
    )
    
    # Sanitización de NaNs
    filter!(row -> !isnan(row.Mean_Classic) && !isnan(row.Mean_Epi) && !isnan(row.γ), df_curvas)
    
    # 3. CLASIFICACIÓN
    gdf_instancias = groupby(df_curvas, clave_instancia)
    
    lista_ganadores = []
    lista_perdedores = []
    
    function obtener_valores_clave(subdf)
        return values(subdf[1, clave_instancia])
    end

    for key in keys(gdf_instancias)
        subdf = gdf_instancias[key]
        if nrow(subdf) == 0 continue end
        
        avg_classic = mean(subdf.Mean_Classic)
        avg_epi = mean(subdf.Mean_Epi)
        
        item = (data=DataFrame(subdf), sort_vals=obtener_valores_clave(subdf))
        
        if avg_epi < avg_classic
            push!(lista_ganadores, item)
        else
            push!(lista_perdedores, item)
        end
    end
    
    # 4. ORDENAMIENTO Y MUESTREO
    function ordenar_lista!(lista)
        sort!(lista, by = x -> x.sort_vals)
    end
    
    ordenar_lista!(lista_ganadores)
    ordenar_lista!(lista_perdedores)
    
    function muestrear_lista(lista, limite)
        total = length(lista)
        if total == 0 return [] end
        if total <= limite return lista end
        indices = unique(round.(Int, range(1, total, length=limite)))
        return lista[indices]
    end
    
    seleccion_ganadores = muestrear_lista(lista_ganadores, num_mosaicos)
    seleccion_perdedores = muestrear_lista(lista_perdedores, num_mosaicos)
    
    # 5. GENERACIÓN DE GRÁFICOS (ESTILO MEJORADO)
    function crear_mosaico(lista_seleccion, titulo_archivo)
        if isempty(lista_seleccion)
            println("  -> [Skip] No hay datos válidos para $titulo_archivo")
            return
        end
        
        plots_array = []
        
        for item in lista_seleccion
            datos = item.data
            sort!(datos, :γ)
            
            if nrow(datos) == 0 continue end
            
            vals = item.sort_vals
            txt_params = [string(col, "=", val) for (col, val) in zip(clave_instancia, vals)]
            titulo_plot = join(txt_params, ", ")
            
            # --- AJUSTES VISUALES ---
            p = plot(datos.γ, datos.Mean_Classic, label="Classic", lw=2, linestyle=:dash, color=:red)
            plot!(p, datos.γ, datos.Mean_Epi, label="Epi", lw=2, color=:blue)
            
            plot!(p, 
                title = titulo_plot,
                titlefontsize = 6,        # Título más pequeño
                xrotation = 45,           # Rotación de etiquetas en X
                xlabel = "γ",
                ylabel = "Iter",
                margin = 5Plots.mm,       # Margen extra para evitar solapamiento
                legend = :topright,
                legendfontsize = 5
            )
            
            push!(plots_array, p)
        end
        
        n_plots = length(plots_array)
        if n_plots == 0 return end
        
        n_cols = ceil(Int, sqrt(n_plots))
        n_rows = ceil(Int, n_plots / n_cols)
        
        try
            # Aumenté un poco el tamaño vertical (size) para compensar los márgenes extra
            mosaico = plot(plots_array..., layout=(n_rows, n_cols), size=(1100, 900))
            full_path = joinpath(ruta_salida, "$(titulo_archivo).png")
            savefig(mosaico, full_path)
            println("  -> Mosaico guardado: $full_path ($n_plots gráficos)")
        catch e
            println("  -> [ERROR GRAFICANDO] $titulo_archivo: $e")
        end
    end

    if !isdir(ruta_salida) mkpath(ruta_salida) end

    println("Generando mosaico de GANADORES...")
    crear_mosaico(seleccion_ganadores, "Mosaico_Ganadores")
    
    println("Generando mosaico de NO GANADORES...")
    crear_mosaico(seleccion_perdedores, "Mosaico_NoGanadores")
end

# ╔═╡ 0af2a2aa-f1a2-42b5-8089-f96fd36057a1
begin
	# --- EJECUCIÓN --
	ruta_resultados = "C:\\Users\\PC RST\\Documents\\tesis_pruebas\\Resultados"
	
	# Guardar CSV filtrad
	#guardar_mejores_casos(df_procesado, ruta_resultados)

	const INCLUIR_GAMMA = true
	
    #const MAX_ITERS = 1000.0 

	generar_resumen_cascada(df_procesado, ruta_resultados, 6, INCLUIR_GAMMA, 2, ["I"])

	#df_resumen = generar_resumen_general(df_procesado, ruta_resultados, ["I"], MAX_ITERS)
	
	const GUARDAR = false
	
	#imprimir_mejores_diferencias_relativas(df_procesado, ruta_resultados, 100, GUARDAR)

	generar_mosaicos_comparativos(df_procesado, ruta_resultados, 6, ["I"])
end