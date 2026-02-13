### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 501ee62a-94c8-4433-9525-e6926dd2ff9a
begin
    # 1. 📦 Importación de Paquetes
    using PlutoUI, DataFrames, LinearAlgebra, Random, Printf, Plots
	using Roots
    
    include("ArenaL1Cuadratica.jl")
    include("SearchL1Cuadratica.jl")

    const ProblemModule = ArenaL1Cuadratica
    const SearchModule = SearchL1Cuadratica
end

# ╔═╡ e13c44a4-f197-430b-86a7-5cd72196631c
lambda_history = Float64[]

# ╔═╡ ad8dc1d8-b2c3-4bf7-b369-8017fb8952a9
begin
	lambdaf_history = Float64[]
	lambdag_history = Float64[]
end

# ╔═╡ d688f816-bcd1-472b-aa05-395fd9959a3d
# ----------------------------------------------------
## 🛠️ PARÁMETROS DE REPRODUCCIÓN (ENTRADA MANUAL) 
# Rellenar con una fila del CSV de instancias ganadoras.

const N_REPRODUCIR = 10

# ╔═╡ 8ef95e14-8e5e-42e7-b16a-8c393b88a69e
const GAMMA_PASO = 1

# ╔═╡ d7176de3-acd8-49e3-ad0d-20cd8f038c47
const TOLERANCIA = 1e-10

# ╔═╡ f4739a3d-5d76-4aef-b2b0-6b59268be7f5
begin
	const MAX_ITER_PGA = 0
	const MAX_ITER_EPGA = 0
	const MAX_ITER_DRA = 5000
	const MAX_ITER_EDRA = 5000
	const patience_limit = 5
end

# ╔═╡ f7164752-310c-416d-9500-cf5f61493a31
# Semillas guardadas en la tabla de resultados (Reemplazar con valores UInt64 del CSV)
const INSTANCE_TAG_REPRODUCIR = "n10_m25_k10_L10_Sing"

# ╔═╡ e4ae0a11-f047-42bf-81e7-6c000c609b36
const SEED_X0 = 2611150579553030000 # Semilla que define el punto inicial (x0, eta0, rho0)


# ╔═╡ dfeba1d2-13d8-4afd-8664-b416465cc610
# ----------------------------------------------------
## 🔍 1. PREPARACIÓN DE LA INSTANCIA

# 1.1. Lógica para recuperar la tupla de parámetros internos (del Search Module)
function recover_param_tuple(n::Int, target_tag::String)
    for param_idx in 1:SearchModule.MAX_NUM_PARAMS
        param_tuple, instance_tag = SearchModule.generate_parameters_and_tag(n, param_idx)
        if instance_tag == target_tag
            return param_tuple
        end
    end
    error("No se pudo encontrar la tupla de parámetros para InstanceTag: $target_tag (n=$n).")
end

# ╔═╡ 3f896575-233c-4a67-9157-ff148750002f
const PARAM_TUPLE = recover_param_tuple(N_REPRODUCIR, INSTANCE_TAG_REPRODUCIR)

# ╔═╡ 9b8fa1e9-81a1-46cb-b2ad-3ffcc1980cef
# 1.2. Recreación del problema con la semilla de instancia
# NOTA: La Arena debe usar INSTANCE_SEED internamente o aquí debemos forzarla.
# Asumiremos que la Arena USA INSTANCE_SEED internamente.

const problem = ProblemModule.setup_problem(N_REPRODUCIR, PARAM_TUPLE)

# ╔═╡ ac5c175b-2b1c-45e6-8b13-ec4258f7680b
# 1.3. Recreación del punto inicial con la SeedX0

const (x0, η0, ρ0) = problem.generate_initial_points(SEED_X0)

# ╔═╡ 7aaa0d90-fe7e-46ad-a6c3-6f7d15772955
# Preparamos la solución correcta para este estudio específico
const SOLUTION_FOR_STUDY = begin
    if problem.solution_fg.type == :dynamic
        println("Calculando solución de referencia dinámica para x0 actual...")
        val = problem.solution_fg.solver(x0, GAMMA_PASO)
        (type=:value, value=val)
    else
        problem.solution_fg
    end
end

# ╔═╡ 0edb8a50-040e-4922-b102-5cb616fcd28b
@printf("Problema de dimensión n=%d con γ=%.4f listo. ||x0|| = %.2f\n", 
         N_REPRODUCIR, GAMMA_PASO, norm(x0))

# ╔═╡ ccf6d9eb-567a-4ad5-b2c2-31970b3946fe
	function calculate_error(x_k, solution)
	    if solution.type == :value
	        # Distancia al punto solución
	        return norm(x_k .- solution.value)
	    elseif solution.type == :set
	        # Distancia al conjunto solución
	        x_proj = solution.projector(x_k, 0.0) # El 0.0 es un dummy
	        return norm(x_k - x_proj)
	    elseif solution.type == :optval
			# Distancia al valor óptimo
			current_val = solution.f(x_k) + solution.g(x_k)
			return abs(current_val - solution.optval)
	    elseif solution.type == :unbounded
	        return Inf # Nunca converge
	    else
	        @warn "Tipo de solución desconocido: $(solution.type)"
	        return Inf
	    end
	end	

# ╔═╡ 92b17a5c-0e5c-40b0-8ef0-8d0e81b6d6de
# -------------------------------------------------------------------------
# A. FUNCIÓN AUXILIAR PARA ENCONTRAR LA RAÍZ (NECESITA using Roots)
# -------------------------------------------------------------------------

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
        @warn "No se encontró límite superior para la raíz de λ (default_lambda_finder)"
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

# -------------------------------------------------------------------------
# B. FUNCIÓN PRINCIPAL DE DESPACHO (LA QUE RESUELVE λ)
# -------------------------------------------------------------------------

# ╔═╡ 6bd88502-037e-49a5-9775-5319c2db91e1
function find_lambda(f, prox_f, y, η, γ, custom_finder=nothing; search_max=100.0)
    
    # 1. Calcular f(y) para la comprobación inicial
    local f_at_y = f(y)

    if !isfinite(f_at_y)
        # Si f(y) no es finito, se asume que el punto está fuera del epígrafo.
    else
        # --- CASO DE COMPROBACIÓN RÁPIDA (Si λ=0) ---
        # h(0) = η - γ - f(y). Si h(0) >= 0, el punto está en el epígrafe, y λ=0.
        local h_at_zero = η - γ - f_at_y
        
        if h_at_zero >= 0
            return 0.0
        end
        # Si h_at_zero < 0, continuamos al cálculo de λ > 0 (Plan B).
    end

    # --- PLAN B (Calcular λ > 0) ---
    
    if custom_finder !== nothing
        # Usar el solver eficiente del módulo (e.g., bisection_solver de la Arena)
        return custom_finder(y, η, γ)
    else
        # Usar el solver por defecto (Roots.findzero)
        return default_lambda_finder(f, prox_f, y, η, γ, search_max=search_max)
    end
end

# ╔═╡ cf08a140-9736-4a91-a86f-09cc54687f96
function run_EPGA_with_history(funcs, x0, η0, γ, tol, x_star; max_iter=MAX_ITER_EPGA, patience_limit = patience_limit)
    x_k = copy(x0); η_k = copy(η0); local iters = max_iter
    
    # MODIFICACIÓN 1: Inicializar los historiales necesarios
    x_history = typeof(x0)[] # Almacena los vectores x_k
    obj_history = Float64[]  # Almacena los valores objetivo F(x_k)
    eta_history = Float64[]
	local patience = 0 
    
    timed_val = @timed begin
        if max_iter > 0
            for i in 1:max_iter
				x_prev = x_k
                y_k = x_k - γ * funcs.grad_g(x_k)
                λ_k = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
				push!(lambda_history, λ_k)
                x_k = funcs.prox_f(y_k, λ_k)
                η_k = η_k - γ + λ_k
				
                F_k = funcs.f(x_k) + funcs.g(x_k)
				
				push!(eta_history, η_k)
                push!(x_history, x_k)
                push!(obj_history, F_k)
				
				if x_star.type == :nothing
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
					err = calculate_error(x_k, x_star)
					
					if err < tol
						iters = i
						break
					end
				
				end
            end
        else; iters = 0; end
    end
    
    # MODIFICACIÓN 3: Retornar los nuevos historiales
    return (x_history=x_history, obj_history=obj_history, iterations=iters, time=timed_val.time, lambda_history=lambda_history, gamma_value=γ, eta_history)
end

# ╔═╡ 6e81d820-8520-4b1e-9fc7-5c1eff7d4049
	function run_PGA_with_history(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_PGA, patience_limit = patience_limit)
	    x_k = copy(x0); local iters = max_iter
		
		x_history = typeof(x0)[]
		obj_history = Float64[]
		
		local patience = 0
		
	    timed_val = @timed begin
	        if max_iter > 0
	            for i in 1:max_iter
					x_prev = x_k
	                y_k = x_k - γ * funcs.grad_g(x_k)
	                x_k = funcs.prox_f(y_k, γ)
					
					F_k = funcs.f(x_k) + funcs.g(x_k)
					
	                push!(x_history, x_k)
	                push!(obj_history, F_k)
					
	                if x_star.type == :nothing
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
					 	err = calculate_error(x_k, x_star)
						
						if err < tol
							iters = i
							break
						end
						
	                end
					
	            end
	         else; iters = 0; end
	    end
	    return (x_history=x_history, obj_history=obj_history, iterations=iters, time=timed_val.time, gamma_value=γ)
	end

# ╔═╡ ab90de98-d8fe-4b82-90ca-93a250c8ee08
function run_EDRA_coupled_with_history(funcs, x0, η0, ρ0, γ, tol, x_star; max_iter=MAX_ITER_EDRA, patience_limit = patience_limit)
    x_k = copy(x0); η_k = copy(η0); ρ_k = copy(ρ0); local iters = max_iter
    local patience = 0
    # MODIFICACIÓN 1: Inicializar todos los historiales
    x_history = typeof(x0)[] 
    obj_history = Float64[]
    lambdaf_history = Float64[] # Nuevo historial para lambda_f
    lambdag_history = Float64[] # Nuevo historial para lambda_g
    eta_history = Float64[]
	rho_history = Float64[]
	
	λ_2 = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
	w_k = funcs.prox_g(x_k, λ_2)
		
    timed_val = @timed begin
        if max_iter > 0
            for i in 1:max_iter
				w_prev = w_k
                # Paso 1: Cálculo de λ_g (la iteración actual)
                λ_2 = find_lambda(funcs.g, funcs.prox_g, x_k, ρ_k, γ, funcs.find_lambda_g)
                w_k = funcs.prox_g(x_k, λ_2) # Solución w_k
                ρ_k = ρ_k - γ + λ_2 
                
                # Capturar historial (usamos w_k para x y F)
                F_k = funcs.f(w_k) + funcs.g(w_k)
				
                # Paso 2: Pasos Douglas-Rachford
                y_k = 2*w_k - x_k
                λ_1 = find_lambda(funcs.f, funcs.prox_f, y_k, η_k, γ, funcs.find_lambda_f)
				
                p_k = funcs.prox_f(y_k, λ_1)
                x_k = x_k - w_k + p_k
                η_k = η_k - γ + λ_1
                ρ_k = ρ_k - γ + λ_2

				push!(lambdaf_history, λ_1)
				push!(lambdag_history, λ_2)
				push!(eta_history, η_k)
				push!(rho_history, ρ_k)
                push!(x_history, w_k)
                push!(obj_history, F_k)
				
				if x_star.type == :nothing
					diff_norm = norm(w_k - w_prev)
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
					err = calculate_error(w_k, x_star)
					
					if err < tol
						iters = i
						break
					end
				end
				
            end
        else; iters = 0; end
    end
    
    return (x_history=x_history, obj_history=obj_history, iterations=iters, time=timed_val.time, 
            lambdag_history=lambdag_history, lambdaf_history=lambdaf_history, gamma_value=γ, eta_history, rho_history)
end

# ╔═╡ 1ca9375a-eaf9-47fa-ab9b-8e12a2107962
function run_DRA_with_history_for_comparison(funcs, x0, γ, tol, x_star; max_iter=MAX_ITER_DRA, patience_limit = patience_limit)
	local patience = 0
    x_k = copy(x0) # Variable auxiliar z_k
    
    # Inicialización de la solución principal (y_k) y el historial
    y_history = typeof(x0)[]
	obj_history = Float64[]
	
    y_k = funcs.prox_g(x_k, γ) 
    
    local iters = max_iter
    timed_val = @timed begin
        if max_iter > 0
            for i in 1:max_iter
                y_prev = y_k
            
                y_k = funcs.prox_g(x_k, γ)
                r_k = 2 * y_k - x_k
                w_k = funcs.prox_f(r_k, γ)
                x_k = x_k + w_k - y_k
                
				F_k = funcs.f(y_k) + funcs.g(y_k)
				push!(obj_history, F_k)
                push!(y_history, y_k)
				
				if x_star.type == :nothing
					diff_norm = norm(y_k - y_prev)
					err = diff_norm 
					
					if err < tol
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
					err = calculate_error(y_k, x_star)
					
					if err < tol
						iters = i
						break
					end	
				end
            end
        else; iters = 0; end
    end
    
    return (y_history=y_history, obj_history = obj_history, iterations=iters, time=timed_val.time)
end

# ╔═╡ 4ae04edd-d154-4745-8bb9-c4d8ae025c61
function run_and_analyze_history(alg_name::String)
    
    # --- 1. EJECUCIÓN ---
    if alg_name == "EPGA"
        res_epga = run_EPGA_with_history(problem, x0, η0, GAMMA_PASO, TOLERANCIA, SOLUTION_FOR_STUDY)

		res_pga = run_PGA_with_history(problem, x0, GAMMA_PASO, TOLERANCIA, SOLUTION_FOR_STUDY)

		res = res_epga
		
    elseif alg_name == "EDRA"
	    res_edra = run_EDRA_coupled_with_history(problem, x0, η0, ρ0, GAMMA_PASO, TOLERANCIA, SOLUTION_FOR_STUDY)
	    

	    res_dra = run_DRA_with_history_for_comparison(problem, x0, GAMMA_PASO, TOLERANCIA, SOLUTION_FOR_STUDY)
	
	    res = res_edra 
		
    else
        error("Algoritmo no soportado")
    end

    # Extraemos los historiales base
    local x_history = res.x_history
    local obj_history = res.obj_history
    local eta_history = res.eta_history
    local rho_history

    if alg_name == "EDRA"
        rho_history = res.rho_history
    else 
        rho_history = [problem.g(x) for x in x_history]
    end	
    
    # --- 2. CONSTRUCCIÓN DEL ESTADO COMPLETO Z_K ---
    # z_k incluye las variables primal y auxiliar (epígrafo)
    
    local z_history = []
    for k in 1:length(x_history)
        xk = x_history[k]
		
		zk_vec = vcat(xk, [eta_history[k], rho_history[k]])
		
        push!(z_history, zk_vec)
    end
    
    # --- 3. CÁLCULO DE MÉTRICAS ---
    
    local iterations = 1:(length(z_history)-1)
    local EPSILON = 1e-15

    # A. Cambio Relativo del Estado (Estabilidad)
    local delta_z_relativo = Float64[]
    for k in 2:length(z_history)
        diff = norm(z_history[k] - z_history[k-1])
        norm_z = max(norm(z_history[k]), EPSILON)
        push!(delta_z_relativo, diff)
    end
    
    # B. Métricas Originales (Objetivo y Variable Primal)
    local delta_obj = abs.(obj_history[2:end] .- obj_history[1:end-1])
    local delta_x = [norm(x_history[k] - x_history[k-1]) for k in 2:length(x_history)]

    # --- 4. GRAFICACIÓN ---

    # PLOT 1: Dinámica de la Función Objetivo y Primal
	if alg_name == "EDRA"
	    p_conv = plot(iterations, max.(delta_obj, EPSILON), 
	             yscale=:log10, label="|(f+g)(w_k) - (f+g)(w_{k-1})|", lw=2,
	             title="Error v/s Iteración (DRA v/s $alg_name)", xlabel="k", ylabel="Log",    		         linestyle=:dash)
	else 
	    p_conv = plot(iterations, max.(delta_obj, EPSILON), 
	             yscale=:log10, label="|(f+g)(x_k) - (f+g)(x_{k-1})|", lw=2,
	             title="Error v/s Iteración (PGA v/s $alg_name)", xlabel="k", ylabel="Log",    		         linestyle=:dash)
	end
	
	if (alg_name == "EDRA") && (SOLUTION_FOR_STUDY.type == :value)
		local x_star = SOLUTION_FOR_STUDY.value
		local primal_error = [norm(xk - x_star) for xk in x_history]
		local y_history = res_dra.y_history
		local y_primal_error = [norm(yk - x_star) for yk in y_history]
		
		# Usamos max(..., 1e-20) para evitar log(0) en gráficos logarítmicos
        plot!(p_conv, 1:length(primal_error), max.(primal_error, EPSILON),
              label="||w_k - x*|| EDRA",
              lw=2, 
              color=:green, 
              linestyle=:solid)
		
		plot!(p_conv, 1:length(y_primal_error), max.(y_primal_error, EPSILON),
              label="||y_k - x*|| DRA",
              lw=2, 
              color=:purple, 
              linestyle=:dash)

	elseif (alg_name == "EDRA")
		local y_history = res_dra.y_history
		local x_error = [norm(x_history[k] - x_history[k-1]) for k in 2:length(x_history)]
		local y_error = [norm(y_history[k] - y_history[k-1]) for k in 2:length(y_history)]
		
        plot!(p_conv, 1:length(x_error), max.(x_error, EPSILON),
              label="||w_k - w_{k-1}|| EDRA",
              lw=2, 
              color=:green, 
              linestyle=:solid)
		
		plot!(p_conv, 1:length(y_error), max.(y_error, EPSILON),
              label="||y_k - y_{k-1}|| DRA",
              lw=2, 
              color=:purple, 
              linestyle=:dash)
		
	end


	if (alg_name == "EPGA") && (SOLUTION_FOR_STUDY.type == :value)
		local x_star = SOLUTION_FOR_STUDY.value
		local primal_error = [norm(xk - x_star) for xk in x_history]
		local y_history = res_pga.x_history
		local y_primal_error = [norm(yk - x_star) for yk in y_history]
		
        plot!(p_conv, 1:length(primal_error), max.(primal_error, EPSILON),
              label="||x_k - x*|| EPGA",
              lw=2, 
              color=:green, 
              linestyle=:solid)
		
		plot!(p_conv, 1:length(y_primal_error), max.(y_primal_error, EPSILON),
              label="||y_k - x*|| PGA",
              lw=2, 
              color=:purple, 
              linestyle=:dash)
	elseif (alg_name == "EPGA")
		local y_history = res_pga.x_history
		local x_error = [norm(x_history[k] - x_history[k-1]) for k in 2:length(x_history)]
		local y_error = [norm(y_history[k] - y_history[k-1]) for k in 2:length(y_history)]
		
        plot!(p_conv, 1:length(x_error), max.(x_error, EPSILON),
              label="||x_k - x_{k-1}|| EPGA",
              lw=2, 
              color=:green, 
              linestyle=:solid)
		
		plot!(p_conv, 1:length(y_error), max.(y_error, EPSILON),
              label="||y_k - y_{k-1}|| PGA",
              lw=2, 
              color=:purple, 
              linestyle=:dash)
		
	end
	
    p_relativo = plot(iterations, max.(delta_z_relativo, EPSILON),
                      yscale=:log10, 
                      label="Estabilidad ||z_k - z_{k-1}||", 
                      title="Dinámica del Estado Z: $alg_name",
                      xlabel="k", ylabel="Error Log", lw=2, color=:blue)
    
    if SOLUTION_FOR_STUDY.type == :value
        local x_star = SOLUTION_FOR_STUDY.value
        local z_star = nothing
		
        if alg_name == "EPGA"
            # z* = (x*, f(x*))
            z_star = vcat(x_star, [problem.f(x_star), problem.g(x_star)])
        elseif alg_name == "EDRA"
            # z* = (x*, f(x*), g(x*))
            z_star = vcat(x_star, [problem.f(x_star), problem.g(x_star)])
        end
        
        # 5.2 Calcular error absoluto ||z_k - z*||
        local dist_to_opt = [norm(zk - z_star) for zk in z_history]
        
        plot!(p_relativo, 1:length(dist_to_opt), max.(dist_to_opt, EPSILON),
              label="Error Absoluto ||z_k - z*||",
              lw=2, color=:red, linestyle=:dot)
    end

    return p_conv, p_relativo, res
end

# ╔═╡ b52999f1-ee56-4011-97dc-fe955c84c8d6
## 🚀 3. SELECCIÓN Y EJECUCIÓN (Ejecución real)

# Verificar la estabilidad EPGA (Recomendado antes de correr)
const L_constant = problem.L

# ╔═╡ 521fa06b-0ac8-4013-a603-095117b36e37
if GAMMA_PASO >= 2.0 / L_constant
    @warn "El paso γ (%.4f) es inestable (2/L = %.4f)." GAMMA_PASO 2.0/L_constant
end

# ╔═╡ c3481402-5596-4902-8647-8d27bf9c95ba
begin
	p_epga_conv, p_epga_relativo, res_epga = run_and_analyze_history("EPGA")
	p_epga_conv # Muestra el gráfico de convergencia
end

# ╔═╡ 4ddb30b7-04e4-4540-b17f-13a9cda4e3c4
p_epga_relativo

# ╔═╡ b1eaf7e1-42ac-468e-9cd9-b2c49f8c707f
function plot_lambda_vs_gamma(lambda_history::AbstractVector{<:Real}, gamma_value::Real, alg_name::String, n::Int)
    
    local iterations = 1:length(lambda_history)
    
    # Crea el gráfico principal con la historia de lambda
    p = plot(iterations, lambda_history,
             title="λ_k vs. γ (fijo) ($alg_name | n=$n)",
             xlabel="Iteración (k)",
             ylabel="Valor de Paso (λ / γ)",
             label="λ_k (Paso Dinámico)",
             legend=:topright,
             lw=2)
             
    # Agrega la línea horizontal para γ (el paso fijo)
    hline!(p, [gamma_value], 
           label="γ Constante",
           line=(:dash, 2, :red))
           
    # Agrega la línea para 2γ (Un punto de referencia importante en optimización)
    # Solo si 2γ es mayor que γ.
    if 2 * gamma_value > gamma_value
        hline!(p, [2 * gamma_value], 
               label="2γ",
               line=(:dot, 1, :gray))
    end
           
    return p
end

# ╔═╡ 18e8f7a6-35f2-474d-865e-cf0da11cbc04
begin
	p_epga_lambda = plot_lambda_vs_gamma(res_epga.lambda_history, 
	                                     GAMMA_PASO, 
	                                     "EPGA", 
	                                     N_REPRODUCIR)
	p_epga_lambda # Muestra el gráfico de λ_f vs γ
end

# ╔═╡ f6b0ece8-115e-49a1-a582-e73208537d78
begin
	p_edra_conv, p_edra_relativo, res_edra = run_and_analyze_history("EDRA")
	p_edra_conv # Muestra el gráfico de convergencia
end

# ╔═╡ 63b512b7-f8d8-4e93-922c-0e2355176d4b
p_edra_relativo

# ╔═╡ 770ddc87-1c5e-4fb9-a3ad-f645768b9fa6
function plot_lambda_edra(lambdag_history::AbstractVector{<:Real}, lambdaf_history::AbstractVector{<:Real}, gamma_value::Real, alg_name::String, n::Int)
    
    local iterations = 1:min(length(lambdag_history), length(lambdaf_history))
    
    p = plot(iterations, lambdag_history[iterations],
             title="Variación de λ_f y λ_g vs. γ ($alg_name | n=$n)",
             xlabel="Iteración (k)",
             ylabel="Valor de Paso (λ / γ)",
             label="λ_g (para g)",
             legend=:topright,
             lw=2)
             
    plot!(p, iterations, lambdaf_history[iterations], 
          label="λ_f (para f)",
          line=(:solid, 2, :blue))
          
    # Líneas de referencia
    hline!(p, [gamma_value], 
           label="γ Constante",
           line=(:dash, 2, :red))
           
    hline!(p, [2 * gamma_value], 
           label="2γ",
           line=(:dot, 1, :gray))
           
    return p
end

# ╔═╡ 731d798f-eadf-468e-9db1-1bfcae2aa59f
begin
	p_edra_lambda = plot_lambda_edra(res_edra.lambdag_history, 
	                                 res_edra.lambdaf_history, 
	                                 GAMMA_PASO, 
	                                 "EDRA", 
	                                 N_REPRODUCIR)
	p_edra_lambda # Muestra el gráfico de λ_f y λ_g vs γ
end
