### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 87954af0-7276-414b-8338-60de58ed2009
begin
	### 1. 📦 Importación de Paquetes
	using PlutoUI
	using DataFrames
	using LinearAlgebra
	using Statistics
	using BenchmarkTools # Usaremos @timed para el tiempo por ejecución
	using Roots 
	using Plots
    using CSV
	using Random
	using Printf
	using ProgressLogging
end

# ╔═╡ 6b151046-70bb-47d6-b7af-935579045806
begin
    include("ArenaCuadraticaL1.jl")
    include("SearchL1Cuadratica.jl")

    using .ArenaCuadraticaL1
    using .SearchL1Cuadratica 

	const ProblemModule = ArenaCuadraticaL1

	const Search = SearchL1Cuadratica

    const MAX_NUM_PARAMS = Search.MAX_NUM_PARAMS

    const NOMBRE_ARENA = "Cuadratica_g0"

	const MAX_ITER_PPA = 0
	const MAX_ITER_EPA = 0
	const MAX_ITER_PGA = 5000
	const MAX_ITER_EPGA = 5000
	const MAX_ITER_DRA = 0
	const MAX_ITER_EDRA = 0
	const patience_limit = 5
end

# ╔═╡ 32afa4de-87e3-444f-b53d-7e7dceddac88
# ocultar esta celda en Pluto
begin
	# Definimos un estilo para las tablas
	PlutoUI.html"""
	<style>
		table {
			width: 100%;
			border-collapse: collapse;
			margin-bottom: 20px;
		}
		th, td {
			border: 1px solid #ddd;
			padding: 8px;
			text-align: left;
		}
		th {
			background-color: #f2f2f2;
		}
	</style>
	"""
end

# ╔═╡ 656001fb-bd56-4716-bc70-d0fbb4220dac
function recover_param_tuple(n::Int, target_tag::String)
    
    for param_idx in 1:MAX_NUM_PARAMS
        # Llama a la función que genera la tupla y el tag
        param_tuple, instance_tag = Search.generate_parameters_and_tag(n, param_idx)
        
        if instance_tag == target_tag
            return param_tuple
        end
    end
    
    # Si la ejecución no llega a encontrar el tag (no debería pasar si los datos son consistentes)
    error("No se pudo encontrar la tupla de parámetros para InstanceTag: $target_tag (n=$n).")
end

# ╔═╡ 93872d67-e391-4eaf-9ffe-9f56ac042cc0
begin
	### 3. 🛠️ Funciones Auxiliares (Rutas, Gráficos, Raíces y Puntos Iniciales)

	### Configuración de Resultados
    # Definir carpeta para guardar los CSVs.
    # Nota: Usa doble barra inversa (\\) o barra simple (/) para rutas en Windows.
    const RUTA_RESULTADOS = "C:\\Users\\PC RST\\Documents\\tesis_pruebas\\Resultados"

	# --- Función Helper para Graficar ---
	function save_plot_mosaic(plot_obj, category_name)
        if plot_obj === nothing
            @info "No hay gráficos de mosaico para guardar en $category_name"
            return
        end
        
        filename = "mosaic_$(category_name)_$(NOMBRE_ARENA).png"
        fullpath = joinpath(RUTA_RESULTADOS, filename)
        
        try
            mkpath(dirname(fullpath))
            savefig(plot_obj, fullpath)
            @info "Mosaico guardado correctamente: $fullpath"
        catch e
            @error "Error al guardar $filename: $e"
        end
    end	
	
    # --- Función Helper para Guardar CSV ---
    function save_winning_instances(df_wins, prefix_name)
        if isempty(df_wins)
            # Usamos @info en lugar de @warn para no alarmar si es esperado
            @info "No hay instancias ganadoras para guardar en $prefix_name"
            return
        end
        
        # Construir ruta completa: "RUTA\prefix_NOMBREARENA.csv"
        filename = "$(prefix_name)_$(NOMBRE_ARENA).csv"
        fullpath = joinpath(RUTA_RESULTADOS, filename)
        
        try
            # mkpath crea el directorio si no existe (evita errores)
            mkpath(dirname(fullpath))
            CSV.write(fullpath, df_wins)
            @info "Archivo guardado correctamente: $fullpath"
        catch e
            @error "Error al guardar $filename: $e"
        end
    end
	
	
	# Plan B: El solver por defecto que usa Roots.findzero()
	function default_lambda_finder(f, prox_f, y, η, γ; search_max=100.0)
	    
	    # h(λ) = λ - f(prox_f(y, λ)) + η - γ
	    h = λ -> λ - f(prox_f(y, λ)) + (η - γ)
	    
	    # Encontrar un intervalo [a, b] que contenga la raíz
	    a = 1e-9 # Empezar justo por encima de 0
	    b = 1e-3
	    while h(b) < 0 && b < search_max
	        b *= 2
	    end
	    
	    if b >= search_max
	        @warn "No se encontró límite superior para la raíz de λ"
	        return 0.0
	    end
	
	    # Usar findzero en el intervalo
	    try
	        return findzero(h, (a, b))
	    catch e
	        @warn "findzero falló: $e"
	        return 0.0 # Fallback
	    end
	end
	
	# Función "Despachadora" (Dispatcher)
	# Esta es la que llamarán los algoritmos.
	function find_lambda(f, prox_f, y, η, γ, custom_finder=nothing; search_max=1000.0)
	    
	    # 1. Calcular f(y) directamente (más barato que prox_f)
	    local f_at_y = f(y)
	
	    if !isfinite(f_at_y)
			# No podemos hacer el chequeo de h_at_zero.
			# Se salta al Plan B sin hacer nada.
	    else
	        # --- CASO ANTIGUO (f(y) es finito) ---
	        # Ahora es seguro calcular h(0)
	        local h_at_zero = η - γ - f_at_y
	        
	        if h_at_zero >= 0
	            # h(0) >= 0. El punto (y, η-γ) está en el epígrafe.
	            # La raíz es 0.
	            return 0.0
	        end
	        # Si h_at_zero < 0, el punto está fuera.
	        # Continuamos al Plan B.
	    end
	
	    # --- PLAN B (Calcular λ > 0) ---
	    # Llegamos aquí si f(y)=Inf O si f(y) es finito pero h_at_zero < 0
	    
	    if custom_finder !== nothing
	        # Usar el solver eficiente del módulo
	        return custom_finder(y, η, γ)
	    else
	        # Usar el solver por defecto (Roots.findzero)
	        return default_lambda_finder(f, prox_f, y, η, γ, search_max=search_max)
	    end
	end
end

# ╔═╡ a23c5be1-a547-4745-9c94-53140fb62285
begin
	### 4. 🏁 Implementación de Algoritmos (Los Competidores) - CORREGIDO
	
	# Cada función retorna un NamedTuple: (solution, iterations, time)

	# --- Función Auxiliar para Criterio de Parada ---
	function calculate_error(x_k, solution)
	    if solution.type == :value
	        # Distancia al punto solución
	        return norm(x_k .- solution.value)
	    elseif solution.type == :set
	        # Distancia al conjunto solución
	        x_proj = solution.projector(x_k, 0.0) # El 0.0 es un dummy
	        return norm(x_k - x_proj)
	    elseif solution.type == :unbounded
	        return Inf # Nunca converge
	    else
	        @warn "Tipo de solución desconocido: $(solution.type)"
	        return Inf
	    end
	end	
	# --- Categoría 1: PPA vs EPA (usa f, x_star_f) ---
	
	function run_PPA(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_PPA, patience_limit=patience_limit)
	    x_k = copy(x0)
	    local iters = max_iter
		local patience = 0
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
	                x_prev = x_k
	                x_k = funcs.prox_f(x_prev, γ)
                	
	                if x_star.type == :nothing
	                    # Criterio 1: Residuo Relativo con Paciencia
	                    # diff = ||x_k - x_prev||
	                    diff_norm = norm(x_k - x_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
	                    # Criterio 2: Distancia a la solución conocida
	                    if calculate_error(x_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
	            end
	        else
	            iters = 0
	        end
	    end
	    
	    # Construimos el retorno FUERA del @timed
	    return (solution=x_k, iterations=iters, time=timed_val.time)
	end
	
	function run_EPA(funcs, x0, η0, γ, tol, x_star; max_iter=MAX_ITER_EPA, patience_limit=patience_limit)
		local patience = 0
	    x_k = copy(x0)
	    η_k = copy(η0)
	    local iters = max_iter
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					x_prev = x_k 
	                λ_k = find_lambda(funcs.f, funcs.prox_f, x_k, η_k, γ, funcs.find_lambda_f)
	                
	                x_k = funcs.prox_f(x_k, λ_k)
	                η_k = η_k - γ + λ_k
	                
	                if x_star.type == :nothing
	                    # Criterio 1: Residuo Relativo con Paciencia
	                    # diff = ||x_k - x_prev||
	                    diff_norm = norm(x_k - x_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
	                    # Criterio 2: Distancia a la solución conocida
	                    if calculate_error(x_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
	            end
	        else
	            iters = 0
	        end
	    end
	    
	    return (solution=x_k, iterations=iters, time=timed_val.time)
	end
	
	# --- Categoría 2: PGA vs EPGA (usa f, g, grad_g, x_star_fg) ---
	
	function run_PGA(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_PGA, patience_limit=patience_limit)
	    x_k = copy(x0)
	    local iters = max_iter
		local patience = 0
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					x_prev = x_k
	                y_k = x_k - γ * funcs.grad_g(x_k)
	                x_k = funcs.prox_f(y_k, γ)
	                
					# --- CRITERIO DE PARADA ---
	                if x_star.type == :nothing
	                    # Criterio 1: Residuo Relativo con Paciencia
	                    # diff = ||x_k - x_prev||
	                    diff_norm = norm(x_k - x_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
	                    # Criterio 2: Distancia a la solución conocida
	                    if calculate_error(x_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
	            end
	        else
	            iters = 0
	        end
	    end
	    
	    return (solution=x_k, iterations=iters, time=timed_val.time)
	end
	
	function run_EPGA(funcs, x0, η0, γ, tol, x_star; max_iter=MAX_ITER_EPGA, patience_limit=patience_limit)
	    x_k = copy(x0)
	    η_k = copy(η0)
	    iters = max_iter
	    local patience = 0
	    
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
	                x_prev = x_k
	                # Paso del algoritmo
	                y_k = x_k - γ * funcs.grad_g(x_k)
	                λ_k = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
	                x_k = funcs.prox_f(y_k, λ_k)
	                η_k = η_k - γ + λ_k
	                
	                # --- CRITERIO DE PARADA ---
	                if x_star.type == :nothing
	        
	                    diff_norm = norm(x_k - x_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	if calculate_error(x_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
	            end
	        else
	            iters = 0
	        end
	    end
	    return (solution=x_k, iterations=iters, time=timed_val.time)
	end
	
	# --- Categoría 3: DRA vs EDRA Acoplado (usa f, g, prox_f, prox_g, x_star_fg) ---
	
	function run_DRA(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_DRA, patience_limit=patience_limit)
	    x_k = copy(x0) # Variable auxiliar
	    local y_k = funcs.prox_g(x_k, γ) # Solución
	    local iters = max_iter
		local patience = 0
	    
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					y_prev = y_k
	                y_k = funcs.prox_g(x_k, γ) # Punto de la solución
					
	                # --- CRITERIO DE PARADA ---
	                if x_star.type == :nothing
	        
	                    diff_norm = norm(y_k - y_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	if calculate_error(y_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
					
	                q_k = 2*y_k - x_k
	                p_k = funcs.prox_f(q_k, γ)
	                x_k = x_k + p_k - y_k 
	            end
	        else
	            iters = 0
	        end
	    end
	    
	    return (solution=y_k, iterations=iters, time=timed_val.time)
	end
	
	function run_EDRA_coupled(funcs, x0, η0, ρ0, γ, tol, x_star; max_iter=MAX_ITER_EDRA, patience_limit=patience_limit)
	    x_k = copy(x0); η_k = copy(η0); ρ_k = copy(ρ0) # Auxiliares

		local patience = 0
	    
	    # Inicializar w_k (la solución)
	    λ_2_init = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
	    local w_k = funcs.prox_g(x_k, λ_2_init)
	    local iters = max_iter
	
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					w_prev = w_k
	                λ_2 = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
	                w_k = funcs.prox_g(x_k, λ_2) # Solución
					
	                # --- CRITERIO DE PARADA ---
	                if x_star.type == :nothing
	        
	                    diff_norm = norm(w_k - w_prev)
	                    rel_err = diff_norm 
	                    
	                    if rel_err < tol
	                        patience += 1
	                    else
	                        patience = 0
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	if calculate_error(w_k, x_star) < tol
	                        iters = i
	                        break
	                    end
	                end
					
	                y_k = 2*w_k - x_k
	                λ_1 = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
	                p_k = funcs.prox_f(y_k, λ_1)
	                
	                x_k = x_k - w_k + p_k
	                η_k = η_k - γ + λ_1
	                ρ_k = ρ_k - γ + λ_2
	                
	            end
	        else
	            iters = 0
	        end
	    end
	    
	    return (solution=w_k, iterations=iters, time=timed_val.time)
	end
end

# ╔═╡ e13a95f0-9a4c-4f6c-8a8e-5dbdf077aa8a
begin
	### 5. 🚀 ¡La Competición! (Script Base Principal)
	
	# --- Parámetros de la simulación ---

	const n_valores = [2,5,10,15]
	
	const betas_lipschitz = []
	
	const betas_raw = [1,11]
	
	const grid_config = [
    [(b, :lipschitz) for b in betas_lipschitz];
    [(b, :raw) for b in betas_raw] 
	]
	const tol_valores = [1e-10]
	const num_puntos_iniciales = 10 # Cuántos puntos iniciales aleatorios probar
	
	# --- Inicializar los DataFrames para los resultados ---
	begin
		df_ppa_epa = DataFrame(
			n=Int[],
			InstanceTag=String[],
			γ=Float64[], 
			Tol=Float64[], 
			Run=Int[], 
			Algoritmo=String[], 
			Iteraciones=Int[], 
			Tiempo=Float64[],
			InstanceSeed=UInt64[],
    		SeedX0=UInt64[]
		)
		
		df_pga_epga = DataFrame(
			n=Int[],
			InstanceTag=String[],
			γ=Float64[], 
			Tol=Float64[], 
			Run=Int[], 
			Algoritmo=String[], 
			Iteraciones=Int[], 
			Tiempo=Float64[],
			InstanceSeed=UInt64[],
    		SeedX0=UInt64[]
		)

		df_pga_edra = DataFrame(
			n=Int[],
			InstanceTag=String[],
			γ=Float64[], 
			Tol=Float64[], 
			Run=Int[], 
			Algoritmo=String[], 
			Iteraciones=Int[], 
			Tiempo=Float64[],
			InstanceSeed=UInt64[],
    		SeedX0=UInt64[]
		)

		df_dra_epga = DataFrame(
			n=Int[],
			InstanceTag=String[],
			γ=Float64[], 
			Tol=Float64[], 
			Run=Int[], 
			Algoritmo=String[], 
			Iteraciones=Int[], 
			Tiempo=Float64[],
			InstanceSeed=UInt64[],
    		SeedX0=UInt64[]
		)	
		
		df_dra_edra = DataFrame(
			n=Int[],
			InstanceTag=String[],
			γ=Float64[], 
			Tol=Float64[], 
			Run=Int[], 
			Algoritmo=String[], 
			Iteraciones=Int[], 
			Tiempo=Float64[],
			InstanceSeed=UInt64[],
    		SeedX0=UInt64[]
		)
	end; # Usamos ; para que Pluto no muestre la salida de esta celda
end

# ╔═╡ c2f8f693-4aca-42c7-b318-2a738392d060
# 1. Calculamos el total estimado de iteraciones (el 100% de la barra)
#    Multiplicamos el tamaño de todos los rangos para saber cuántos "átomos" hay.
total_steps_est = length(n_valores) * MAX_NUM_PARAMS * length(grid_config) * length(tol_valores) * num_puntos_iniciales

# ╔═╡ f22a1dd4-b6e8-4c6f-a44f-b5d1ad0bf0de
# Bucle principal
@withprogress name="Simulación Global" begin
    # --- ZONA SEGURA: Definimos el contador AQUÍ para evitar errores ---
    global_step = 0 
    
    # Reiniciamos los DataFrames
    empty!(df_ppa_epa)
    empty!(df_pga_epga)
    empty!(df_pga_edra)
    empty!(df_dra_epga)
    empty!(df_dra_edra)

    println("Iniciando simulación con aprox. $total_steps_est iteraciones...")

    for n in n_valores
        for param_idx in 1:MAX_NUM_PARAMS

			local param_tuple, instance_tag = Search.generate_parameters_and_tag(n, param_idx)
            
            # SI SALTAMOS: Debemos sumar al contador lo que "no hicimos" 
            # para que la barra no se atrase.
            if instance_tag == "SKIP"
				skipped_steps = length(grid_config) * length(tol_valores) * num_puntos_iniciales
				global_step += skipped_steps
				@logprogress global_step/total_steps_est message="Saltando configuración inválida..."
                continue
            end
            
            local instance_seed = hash((n, param_tuple))
            local problem = ProblemModule.setup_problem(n, param_tuple)
            local L = problem.L
            local limit_lipschitz = (isfinite(L) && L > 0) ? (2.0 / L) : 1.0

            for (beta_val, tipo_gamma) in grid_config
                
                local γ = 0.0
                if tipo_gamma == :lipschitz
                    γ = beta_val * limit_lipschitz
                else 
                    γ = beta_val 
                end
                
                for tol in tol_valores
                    for run in 1:num_puntos_iniciales
                        
                        # --- INICIO DEL NIVEL ATÓMICO ---
						# Aquí incrementamos el contador global por cada simulación 	real
                        global_step += 1
                        
                        # Mensaje dinámico: Muestra n, tag y gamma actual
						msg_atomico = "n=$n | Tag=$instance_tag | γ=$(round(γ, digits=4)) | Run=$run"
                        @logprogress global_step/total_steps_est message=msg_atomico
                        
                        # --------------------------------
                        
                        seed_x0 = hash((n, param_tuple, γ, run))
                        local (x0, η0, ρ0) = problem.generate_initial_points(seed_x0)
                        local current_solution_fg
    
                        if problem.solution_fg.type == :dynamic
                            ref_val = problem.solution_fg.solver(x0,γ)
                            current_solution_fg = (type = :value, value = ref_val)
                        else
                            current_solution_fg = problem.solution_fg
                        end
                        
                        # [TUS EJECUCIONES DE SOLVERS AQUÍ ABAJO...]
                        # (El resto de tu código sigue igual)
                        
                        res_ppa = run_PPA(problem, x0, γ, tol, problem.solution_f)
						push!(df_ppa_epa, (n, instance_tag, γ, tol, run, "PPA", res_ppa.iterations, res_ppa.time, instance_seed, seed_x0))
                        
						res_epa = run_EPA(problem, x0, η0, γ, tol, problem.solution_f)
						push!(df_ppa_epa, (n, instance_tag, γ, tol, run, "EPA", res_epa.iterations, res_epa.time, instance_seed, seed_x0))
    
                        res_dra = run_DRA(problem, x0, γ, tol, current_solution_fg)
						push!(df_dra_edra, (n, instance_tag, γ, tol, run, "DRA", res_dra.iterations, res_dra.time, instance_seed, seed_x0))
                        
						res_edra = run_EDRA_coupled(problem, x0, η0, ρ0, γ, tol, current_solution_fg)
						push!(df_dra_edra, (n, instance_tag, γ, tol, run, "EDRA", res_edra.iterations, res_edra.time, instance_seed, seed_x0))
    
                        if problem.g_is_L_lipschitz && (γ < limit_lipschitz)
							res_pga = run_PGA(problem, x0, γ, tol, current_solution_fg)
							push!(df_pga_edra, (n, instance_tag, γ, tol, run, "PGA", res_pga.iterations, res_pga.time, instance_seed, seed_x0))
							push!(df_pga_edra, (n, instance_tag, γ, tol, run, "EDRA", res_edra.iterations, res_edra.time, instance_seed, seed_x0)) # Ojo: aquí repites EDRA, ¿es intencional?
                            
							res_epga = run_EPGA(problem, x0, η0, γ, tol, current_solution_fg)
							push!(df_pga_epga, (n, instance_tag, γ, tol, run, "PGA", res_pga.iterations, res_pga.time, instance_seed, seed_x0))
							push!(df_pga_epga, (n, instance_tag, γ, tol, run, "EPGA", res_epga.iterations, res_epga.time, instance_seed, seed_x0))

							push!(df_dra_epga, (n, instance_tag, γ, tol, run, "DRA", res_dra.iterations, res_dra.time, instance_seed, seed_x0))
							push!(df_dra_epga, (n, instance_tag, γ, tol, run, "EPGA", res_epga.iterations, res_epga.time, instance_seed, seed_x0))
                        end
                    end
                end
            end
        end
    end
    println("¡Simulación completada!")
end

# ╔═╡ d2b07275-a733-4f1a-9891-f9709cb7c6d2
begin
	### 7. 📈 Funciones de Análisis y Gráfico
	
	# --- Implementaciones de Algoritmos (Versión con Historial) ---
	# (Estas funciones son más lentas y solo se usan para el gráfico final)

	function run_PPA_with_history(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_PPA; patience_limit = patience_limit)
		local patience = 0
	    x_k = copy(x0); local iters = max_iter
		error_history = Float64[]
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					x_prev = x_k
	                x_k = funcs.prox_f(x_k, γ)
					if x_star.type == :nothing
	                    # Criterio 1: Residuo Relativo con Paciencia
	                    # diff = ||x_k - x_prev||
	                    diff_norm = norm(x_k - x_prev)
	                    err = diff_norm 
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
	                    # Criterio 2: Distancia a la solución conocida
	                    err = calculate_error(x_k, x_star)
						if err < tol
							iters = i
							break
						end
	                end	 
					push!(error_history, err)
	            end
	        else; iters = 0; end
	    end
	    return (history=error_history,)
	end
	
	function run_EPA_with_history(funcs, x0, η0, γ, tol, x_star; max_iter=MAX_ITER_EPA; patience_limit = patience_limit)
		local patience = 0
	    x_k = copy(x0); η_k = copy(η0); local iters = max_iter
		error_history = Float64[]
	    timed_val = @timed begin
	        if max_iter > 0
	             for i in 1:max_iter
					x_prev = x_k
	                λ_k = find_lambda(funcs.f, funcs.prox_f, x_k, η_k, γ, funcs.find_lambda_f)
	                x_k = funcs.prox_f(x_k, λ_k)
	                η_k = η_k - γ + λ_k
	                
					if x_star.type == :nothing
	                    # Criterio 1: Residuo Relativo con Paciencia
	                    # diff = ||x_k - x_prev||
	                    diff_norm = norm(x_k - x_prev)
	                    err = diff_norm 
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
	                    # Criterio 2: Distancia a la solución conocida
	                    err = calculate_error(x_k, x_star)
						if err < tol
							iters = i
							break
						end
	                end
					 
					push!(error_history, err)
	            end
	         else; iters = 0; end
	    end
	    return (history=error_history,)
	end
	
	function run_PGA_with_history(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_PGA, patience_limit = patience_limit)
	    x_k = copy(x0); local iters = max_iter
		error_history = Float64[]
		local patience = 0
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					x_prev = x_k
	                y_k = x_k - γ * funcs.grad_g(x_k)
	                x_k = funcs.prox_f(y_k, γ)
	                if x_star.type == :nothing
	                    diff_norm = norm(x_k - x_prev)
	                    err = diff_norm 
						push!(error_history, err)
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	err = calculate_error(x_k, x_star)
						push!(error_history, err)
						
	                end
					
	            end
	         else; iters = 0; end
	    end
	    return (history=error_history,)
	end
	
	function run_EPGA_with_history(funcs, x0, η0, γ, tol, x_star; max_iter=MAX_ITER_EPGA, patience_limit = patience_limit)
	    x_k = copy(x0); η_k = copy(η0); local iters = max_iter
		error_history = Float64[]
		local patience = 0
	    timed_val = @timed begin
	        if max_iter > 0
	             for i in 1:max_iter
					x_prev = x_k
	                y_k = x_k - γ * funcs.grad_g(x_k)
	                λ_k = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
	                x_k = funcs.prox_f(y_k, λ_k)
	                η_k = η_k - γ + λ_k
					 
	                if x_star.type == :nothing
	                    diff_norm = norm(x_k - x_prev)
	                    err = diff_norm 
						push!(error_history, err)
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	err = calculate_error(x_k, x_star)
						push!(error_history, err)
						
	                end
	             end
	        else; iters = 0; end
	    end
	    return (history=error_history,)
	end
	
	function run_DRA_with_history(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_DRA, patience_limit = patience_limit)
	    x_k = copy(x0); local y_k = funcs.prox_g(x_k, γ); local iters = max_iter
		error_history = Float64[]
		local patience = 0
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					y_prev = y_k
	                y_k = funcs.prox_g(x_k, γ)
					
	                if x_star.type == :nothing
	                    diff_norm = norm(y_k - y_prev)
	                    err = diff_norm 
						push!(error_history, err)
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	err = calculate_error(y_k, x_star)
						push!(error_history, err)
						
	                end
					
	                q_k = 2*y_k - x_k
	                p_k = funcs.prox_f(q_k, γ)
	                x_k = x_k + p_k - y_k 
	            end
	        else; iters = 0; end
	    end
	    return (history=error_history,)
	end
	
	function run_EDRA_coupled_with_history(funcs, x0, η0, ρ0, γ, tol, x_star; max_iter=MAX_ITER_EDRA, patience_limit = patience_limit)
	    x_k = copy(x0); η_k = copy(η0); ρ_k = copy(ρ0); local iters = max_iter
		local patience = 0
		error_history = Float64[]
	    λ_2_init = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
	    local w_k = funcs.prox_g(x_k, λ_2_init)
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					w_prev = w_k
	                λ_2 = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
	                w_k = funcs.prox_g(x_k, λ_2) # Solución
					
	                if x_star.type == :nothing
	                    diff_norm = norm(w_k - w_prev)
	                    err = diff_norm 
						push!(error_history, err)
	                    
	                    if err < tol
	                        patience += 1
	                    else
	                        patience = 0 # Reiniciar si damos un paso grande
	                    end
	                    
	                    if patience >= patience_limit
	                        iters = i
							iters = iters - patience_limit + 1
	                        break
	                    end
	                else
					 	err = calculate_error(w_k, x_star)
						push!(error_history, err)
						
	                end
					
	                y_k = 2*w_k - x_k
	                λ_1 = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
	                p_k = funcs.prox_f(y_k, λ_1)
	                x_k = x_k - w_k + p_k
	                η_k = η_k - γ + λ_1
	                ρ_k = ρ_k - γ + λ_2
	            end
	         else; iters = 0; end
	    end
	    return (history=error_history,)
	end
end

# ╔═╡ 407ed391-a0d3-4606-8707-95edcbc312e1
### 8. 📊 Función de Análisis Principal

function run_custom_analysis(df_raw, classic_name, epi_name, category_name, problem_module, run_classic_func, run_epi_func, solution_type_flag)
    
    if isempty(df_raw)
        return md"No hay datos para **$category_name**."
    end
    
    # --- 0. Preparar Datos ---
    # MODIFICACIÓN 1: Reemplazar :n con :InstanceTag para el unstack.
    df_wide = unstack(df_raw, 
                  [:n, :InstanceTag, :γ, :Tol, :Run, :InstanceSeed, :SeedX0], 
                  :Algoritmo, 
                  :Iteraciones,
			      combine=first)
    
    if !(classic_name in names(df_wide)) || !(epi_name in names(df_wide))
        return md"Datos incompletos para **$category_name**."
    end
    rename!(df_wide, classic_name => :Iter_Classic, epi_name => :Iter_Epi)
    
    # --- 1. Porcentaje de Victorias ---
    epi_better = filter(r -> r.Iter_Epi < r.Iter_Classic, df_wide)
    total_runs = nrow(df_wide)
    win_pct = (nrow(epi_better) / total_runs) * 100.0
	
	if !isempty(df_wide)
        df_wins_to_save = select(df_wide, 
                                 [:n, :InstanceTag, :γ, :Tol, :Run, 
                                  :InstanceSeed, :SeedX0, 
                                  :Iter_Classic, :Iter_Epi])
        
        # Llamamos a la función auxiliar para guardar el CSV
        save_winning_instances(df_wins_to_save, "$(category_name)_WINS")
    end	
	
    println("\n=== ANÁLISIS: $category_name ===")
    @printf("1. Porcentaje de Victorias (Iter): %.2f%% (%d de %d)\n", win_pct, nrow(epi_better), total_runs)

    # --- 3. Mejor Instancia (Re-ejecución) ---
    candidates_net_winners = filter(r -> r.Iter_Epi >= 5, epi_better)
    
    if isempty(candidates_net_winners)
	    df_ranked = filter(r -> r.Iter_Epi >= 5, df_wide)
	    
	    # Marcamos el estado para el mensaje final.
	    local status = "PERDEDOR NETO (Menor Pérdida Relativa)"
	else
	    # Caso 2: Sí hay ganadores netos.
	    # El ranking solo se hace sobre el subconjunto de ganadores netos.
	    df_ranked = candidates_net_winners
	    
	    local status = "GANADOR NETO (Mejor Ganancia Relativa)"
	end
	
	# Si df_ranked NO pasó el filtro >= 5)
	if isempty(df_ranked)
	    return md"Análisis completado. (Sin gráfico de convergencia: Ninguna corrida superó las 5 iteraciones)."
	end
    
    # Maximizar ganancia relativa
	df_ranked.Score = (df_ranked.Iter_Classic .- df_ranked.Iter_Epi) ./ max.(df_ranked.Iter_Epi, df_ranked.Iter_Classic)
	
	# Buscar el índice del Score máximo
	best_idx = findmax(df_ranked.Score)[2]
	best = df_ranked[best_idx, :]
    
    println("\n3. Mejor Instancia Encontrada:")
	println("   • Estado: $status")
    # MODIFICACIÓN 8: Imprimir InstanceTag (y n para contexto).
    println("   • Configuración: InstanceTag=$(best.InstanceTag), n=$(best.n), γ=$(best.γ)")
    println("   • Punto Inicial (Run): #$(best.Run)")
	println("   • Puntuación (Score): $(round(best.Score, digits=4))")

    # --- Re-ejecutar para obtener datos detallados ---
    
    # 1. Recuperar Problema
    # Se mantiene best.n para configurar el problema (setup_problem espera la dimensión).
    # Usa la función auxiliar para obtener la tupla de parámetros del InstanceTag.
    param_tuple_recov = recover_param_tuple(best.n, best.InstanceTag)
    
    # Se mantiene best.n para configurar el problema (setup_problem espera la dimensión).
    Random.seed!(hash(best.n))
    
    # Llama a setup_problem con la dimensión 'n' Y la tupla de parámetros.
    prob = problem_module.setup_problem(best.n, param_tuple_recov)
    
	# 2. Recuperar el Gamma Exacto y Punto Inicial
    L_local = prob.L
    limit_lipschitz = (isfinite(L_local) && L_local > 0) ? (2.0/L_local) : 1.0
	
    possible_gammas = Float64[]
    
    for (val, tipo) in grid_config
        if tipo == :lipschitz
            push!(possible_gammas, val * limit_lipschitz)
        else # :raw
            push!(possible_gammas, val)
        end
    end

    diffs = abs.(possible_gammas .- best.γ)
    val_min, g_idx = findmin(diffs)
    
    gamma_exact = possible_gammas[g_idx]
    
    # Verificación de seguridad (opcional)
    if val_min > 1e-10
		@warn "El gamma recuperado difiere mucho del registrado: $(best.γ) vs 		$gamma_exact"
    end
    
    seed_x0 = hash((best.n, param_tuple_recov, gamma_exact, best.Run))
    
    # Generamos el punto
    (x0, η0, ρ0) = prob.generate_initial_points(seed_x0)
    
    # --- IMPRIMIR DETALLES ---
    println("   • Norma ||x0||: $(norm(x0))")
    
    if hasproperty(prob, :print_details)
        prob.print_details() # Llama a la función de la Arena
    else
        println("    (Detalles visuales de funciones no disponibles)")
    end
    
    # Ejecución con historial
    sol_obj = (solution_type_flag == :f) ? prob.solution_f : prob.solution_fg

	local current_solution_fg

	if sol_obj.type == :dynamic
		
		ref_val = sol_obj.solver(x0, best.γ)
		
		current_solution_fg = (type = :value, value = ref_val)
	else
		current_solution_fg = sol_obj
	end
	
    res_c = run_classic_func(prob, x0, η0, ρ0, best.γ, best.Tol, current_solution_fg)
    res_e = run_epi_func(prob, x0, η0, ρ0, best.γ, best.Tol, current_solution_fg)

    # Reemplazamos 0.0 con un valor muy pequeño para que el logaritmo no explote
    safe_history_c = replace(res_c.history, 0.0 => 1e-20)
    safe_history_e = replace(res_e.history, 0.0 => 1e-20)

    max_iters_plot = max(length(safe_history_c), length(safe_history_e))
    p_conv = plot(
        # MODIFICACIÓN 9: Imprimir InstanceTag en el título.
        title="Mejor Victoria Relativa ($category_name)\nInstanceTag=$(best.InstanceTag), γ=$(best.γ), Run=$(best.Run)", 
        yscale=:log10, 
        xlabel="Iteración", 
        ylabel="Error",
        legend=:topright,
        xlims=(1, max_iters_plot)
    )
    plot!(p_conv, safe_history_c, label=classic_name, lw=2)
    plot!(p_conv, safe_history_e, label=epi_name, lw=2, linestyle=:dash)
    return p_conv
end

# ╔═╡ 146067da-862d-4a52-ac85-99eb69f68baf
### 9. 📊 Resultados: Categoría 1 (PPA vs EPA)
# Agrupamos por parámetros y calculamos la media de iteraciones y tiempo
begin
	# Definir funciones de historial para esta categoría
	# (Ignoran η0, ρ0 ya que PPA/EPA no los usan)
	run_classic_hist_ppa = (p, x, η, ρ, γ, t, s) -> run_PPA_with_history(p, x, γ, t, s)
	run_epi_hist_epa = (p, x, η, ρ, γ, t, s) -> run_EPA_with_history(p, x, η, γ, t, s)
	
	run_custom_analysis(
		df_ppa_epa, 
		"PPA", 
		"EPA", 
		"PPA_vs_EPA", 
		ProblemModule,
		run_classic_hist_ppa,
		run_epi_hist_epa,
		:f # Flag: Usar problem.solution_f
	)
end

# ╔═╡ 2fbe630c-7558-442e-8699-5cd623975896
### 10. 📊 Resultados: Categoría 2 (PGA vs EPGA)
begin
	# Definir funciones de historial
	run_classic_hist_pga = (p, x, η, ρ, γ, t, s) -> run_PGA_with_history(p, x, γ, t, s)
	run_epi_hist_epga = (p, x, η, ρ, γ, t, s) -> run_EPGA_with_history(p, x, η, γ, t, s)
	
	run_custom_analysis(
		df_pga_epga, 
		"PGA", 
		"EPGA", 
		"PGA_vs_EPGA", 
		ProblemModule,
		run_classic_hist_pga,
		run_epi_hist_epga,
		:fg # Flag: Usar problem.solution_fg
	)
end

# ╔═╡ da370bf9-2b7e-4138-96be-ec0caa57a3d4
### 11. 📊 Resultados: Categoría 3 (DRA vs EDRA)
begin
	# Definir funciones de historial
	run_classic_hist_dra = (p, x, η, ρ, γ, t, s) -> run_DRA_with_history(p, x, γ, t, s)
	run_epi_hist_edra = (p, x, η, ρ, γ, t, s) -> run_EDRA_coupled_with_history(p, x, η, ρ, γ, t, s)
	
	run_custom_analysis(
		df_dra_edra, 
		"DRA", 
		"EDRA", 
		"DRA_vs_EDRA", 
		ProblemModule,
		run_classic_hist_dra,
		run_epi_hist_edra,
		:fg # Flag: Usar problem.solution_fg
	)
end

# ╔═╡ 0da768f8-2c3e-49de-b869-50528f72ff16
### 12. 📊 Resultados: Categoría 4 (DRA vs EPGA)
begin
	run_custom_analysis(
		df_dra_epga, 
		"DRA",           # Algoritmo Clásico 
		"EPGA",          # Algoritmo Epigráfico
		"EPGA_vs_DRA", 
		ProblemModule,
		# Se usa la función para DRA y la función para EPGA
		run_classic_hist_dra, 
		run_epi_hist_epga,    
		:fg # Flag: Usar problem.solution_fg
	)
end

# ╔═╡ 161b02dc-68a8-4c56-a14e-20b0de0f3bb9
### 13. 📊 Resultados: Categoría 5 (PGA vs EDRA)
begin
	run_custom_analysis(
		df_pga_edra, 
		"PGA",           # Algoritmo Clásico
		"EDRA",          # Algoritmo Epigráfico
		"EDRA_vs_PGA", 
		ProblemModule,
		# Se usa la función para PGA y la función para EDRA
		run_classic_hist_pga, 
		run_epi_hist_edra,
		:fg # Flag: Usar problem.solution_fg
	)
end
