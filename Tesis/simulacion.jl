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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.6.3"
CSV = "~0.10.15"
DataFrames = "~1.8.1"
Plots = "~1.41.1"
PlutoUI = "~0.7.73"
ProgressLogging = "~0.1.5"
Roots = "~2.2.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "cde96db46c016567cd3e297dd23a6bd6a17d3836"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccc81ba5e42497f4e76553a5545665eed577a663"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.0+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f52c27dd921390146624f3aab95f4e8614ad6531"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.18"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b0406b866ea9fdbaf1148bc9c0b887e59f9af68"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.18+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "386b47442468acfb1add94bf2d85365dea10cbab"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f19301ae653233bc88b1810ae908194f07f8db9d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8a433b1ede5e9be9a7ba5b1cc6698daa8d718f1d"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.10"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─87954af0-7276-414b-8338-60de58ed2009
# ╟─32afa4de-87e3-444f-b53d-7e7dceddac88
# ╠═6b151046-70bb-47d6-b7af-935579045806
# ╟─656001fb-bd56-4716-bc70-d0fbb4220dac
# ╟─93872d67-e391-4eaf-9ffe-9f56ac042cc0
# ╟─a23c5be1-a547-4745-9c94-53140fb62285
# ╠═e13a95f0-9a4c-4f6c-8a8e-5dbdf077aa8a
# ╟─c2f8f693-4aca-42c7-b318-2a738392d060
# ╟─f22a1dd4-b6e8-4c6f-a44f-b5d1ad0bf0de
# ╟─d2b07275-a733-4f1a-9891-f9709cb7c6d2
# ╟─407ed391-a0d3-4606-8707-95edcbc312e1
# ╠═146067da-862d-4a52-ac85-99eb69f68baf
# ╠═2fbe630c-7558-442e-8699-5cd623975896
# ╠═da370bf9-2b7e-4138-96be-ec0caa57a3d4
# ╠═0da768f8-2c3e-49de-b869-50528f72ff16
# ╠═161b02dc-68a8-4c56-a14e-20b0de0f3bb9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
