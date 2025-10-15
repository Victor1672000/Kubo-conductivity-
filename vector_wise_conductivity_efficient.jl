using QuadGK
using Random
using LinearAlgebra
using Statistics
using Plots
using SparseArrays
using KrylovKit
using Unitful
using PhysicalConstants
import PhysicalConstants.CODATA2018: h, ħ
import PhysicalConstants.CODATA2018: e
import PhysicalConstants.CODATA2018:  k_B
using PhysicalConstants.CODATA2018: m_e,m_p
using DelimitedFiles
using BenchmarkTools
using CSV, DataFrames
using FastGaussQuadrature
using Base.Threads
using MPI

"""Analytical expression for the prefactor tanh, w is the frequency given Hz, T in Kelvin and V is the volume"""
function tanhc(w::Float64,T::Float64,V::Float64)
    h_val= ustrip(h)/ustrip(e)
    if T==0 
        beta=1
        if w == 0  
            return 1
        else
            return (pi)/(abs(w)*h_val*V*h_val)
        end
    else
        beta= ustrip(e)/(T*ustrip(k_B))
        if w == 0  
            return (pi*beta*0.5)/(V*h_val)
        else
            return (pi*tanh(w*h_val*beta *0.5))/(w*h_val*V*h_val)
        end
    end
end

"""Analytical expression for the Chebyshev coefficients"""
function chebyshev_polynomials(x::Float64, n::Int)
    if abs(x)>1
        return 0
    else
        return (cos(n * acos(x)))
    end
end



"""Analytical expression of the Jackson Kernel"""
function jackson_kernel_elem(m, M)
    
    g_m = ((M + 1 - m) * cos(m * pi / (M + 1)) + sin(m * pi / (M + 1)) * cot(pi / (M + 1))) / (M + 1)
    
    return g_m
end

"""Analytical expression of the Fermi Dirac distribution, E_n is the normalised energy and u_n is the normalised fermi energy"""
function Fermi_function(E_n::Float64, beta_n::Float64, u_n::Float64)
    if beta_n==Inf
        if E_n>u_n
            return 0
        else
            return 1
        end
    else
        return (1)/(exp(beta_n*(E_n-u_n))+1)
    end
end



"""f(E)*(1-f(E+w)) Fermi functions analytical energies in eV, u_n_quasi is the normalised quasi fermi energy u+w"""
function Pauli_function_e(E_n::Float64, beta_n::Float64, u_n::Float64, u_n_quasi::Float64)
    return (1)/(exp(beta_n*(E_n-u_n))+1)*(1- (1)/(exp(beta_n*(E_n-u_n_quasi))+1))
    #return Fermi_function(E_n,beta_n,u_n) * (1-Fermi_function(E_n,beta_n,u_n_quasi))
end

"""f(E+w)(1-f(E)) Fermi functions analytical energies in eV"""
function Pauli_function_h(E_n::Float64, beta_n::Float64, u_n::Float64, u_n_quasi::Float64)
    return (1)/(exp(beta_n*(E_n-u_n_quasi))+1)*(1- (1)/(exp(beta_n*(E_n-u_n))+1))
    #return Fermi_function(E_n,beta_n,u_n_quasi) * (1-Fermi_function(E_n,beta_n, u_n))
end


#########Expansion coefficients for Chebyshev polynomials#######

"""Calculates coefficients for the chebyshev expansion of the DOS projector"""
function DOS_projector(E_n::Float64,M::Int)
    chebyshev_series=Vector{ComplexF64}(undef, M) 
    @inbounds for m in range(1,M)
        g_m= jackson_kernel_elem(m-1, M)
        if m-1==0
            factor= 1
        else
            factor=2
        end
    if abs(E_n)>=1
        chebyshev_series[m]=0
    else
        d_m= (factor)/(pi* sqrt(1-E_n^2))
        T_m= chebyshev_polynomials(E_n, m-1)
        s_m= g_m*T_m*d_m
        chebyshev_series[m]=s_m
    end
    end
return chebyshev_series
end

"""Calculates coefficients for the chebyshev expansion of the JDOS projector"""
function JDOS_projector(E_n::Float64,w_n::Float64,M::Int)
    chebyshev_series=Vector{ComplexF64}(undef, M) 
    @inbounds for m in range(1,M)
        g_m= jackson_kernel_elem(m-1, M)
        if m-1==0
            factor= 1
        else
            factor=2
        end
    if abs(E_n-w_n)>=1
        chebyshev_series[m]=0
    else
        d_m= (factor)/(pi* sqrt(1-(E_n-w_n)^2))
        T_m= chebyshev_polynomials(E_n-w_n, m-1)
        s_m= g_m*T_m*d_m
        chebyshev_series[m]=s_m
    end
    end
return chebyshev_series
end

##### Kernel Polynomial method and stochastic trace approximation

"""Cheybshev expansion for the Hamiltonian, takes a vector v and calculates the power series expansion of the a matrix H via Matrix vector multiplication"""
function vector_wise_Chebyshev_2_optimized_2(H_n::SparseMatrixCSC{ComplexF64, Int64}, v::Vector{ComplexF64}, M::Int)
    
    v_0 = copy(v)                            
    v_1 = similar(v)                          
    mul!(v_1, H_n, v)

    v_i = similar(v)                          
    Chebyshev_series = Vector{ComplexF64}(undef, M)

    Chebyshev_series[1] = dot(v, v_0)
    Chebyshev_series[2] = dot(v, v_1)

    @inbounds for m in 3:M
        mul!(v_i, H_n, v_1)                   
        @inbounds @. v_i = 2 * v_i - v_0      
        Chebyshev_series[m] = dot(v, v_i)
        tmp = v_0
        v_0 = v_1
        v_1 = v_i
        v_i = tmp                          
    end

    return Chebyshev_series
end

"""Cheybshev expansion for the Hamiltonian"""
function vector_wise_Chebyshev_new(H_n::SparseMatrixCSC{ComplexF64, Int64}, v::Vector{ComplexF64}, M::Int,Chebyshev_series::Vector{Vector{ComplexF64}})
    v_0 = copy(v)
    v_1 = H_n*v_0

    Chebyshev_series[1]= v_0
    Chebyshev_series[2] = v_1

    v_i = similar(v)
    v_temp=similar(v)
    @inbounds for m in 3:M
        mul!(v_temp, H_n, v_1)
        v_i = 2 .* v_temp .- v_0
        Chebyshev_series[m]= v_i
        v_0=v_1
        v_1=v_i
    end
    return nothing
end

"""Cheybshev expansion for the Hamiltonian"""
function vector_wise_Chebyshev_optimized(H_n::SparseMatrixCSC{ComplexF64, Int64}, v::Vector{ComplexF64}, M::Int)
    Chebyshev_series = Vector{Vector{ComplexF64}}(undef, M)

    v_0 = copy(v)
    v_1 = H_n*v_0
    v_temp = similar(v)
    v_i=similar(v)

    #mul!(v_1,H_n,v_0)

    Chebyshev_series[1]= v_0
    Chebyshev_series[2]= v_1

    
    @inbounds for m in 3:M
        mul!(v_temp, H_n, v_1)
        v_i = 2 .* v_temp .- v_0
        Chebyshev_series[m]=v_i

        # Rotate buffers without allocation
        v_0=v_1
        v_1=v_i
    end
    return Chebyshev_series
end

"""Calculates the current current correlation function using the stochastic trace approximation and the Chebyshev polynomial expansion"""
function Current_Current_correlation_optimized(H_n::SparseMatrixCSC{ComplexF64, Int64}, v::Vector{ComplexF64}, M::Int, J_1::SparseMatrixCSC{ComplexF64, Int64}, J_2::SparseMatrixCSC{ComplexF64, Int64})
    Chebyshev_series_0 = vector_wise_Chebyshev_optimized(H_n, v , M)

    @inbounds for m in 1:M
        Chebyshev_series_0[m]= J_1* Chebyshev_series_0[m]  # In-place matrix-vector
    end
    tmp_v = similar(v)
    mul!(tmp_v,J_2,v)

    Chebyshev_series_1 = vector_wise_Chebyshev_optimized(H_n, tmp_v, M)

    trace_Matrix = Matrix{ComplexF64}(undef, M, M)
    @inbounds for i in 1:M
        @inbounds for j in 1:M
         trace_Matrix[i, j] = Chebyshev_series_0[i]' * Chebyshev_series_1[j]
        end
    end
    return trace_Matrix
end


"""Calculates the current current correlation function using the stochastic trace approximation and the Chebyshev polynomial expansion"""
function Current_Current_correlation_new(H_n::SparseMatrixCSC{ComplexF64, Int64},  v::Vector{ComplexF64},  M::Int,  J_1::SparseMatrixCSC{ComplexF64, Int64},  J_2::SparseMatrixCSC{ComplexF64, Int64},Chebyshev_series_0::Vector{Vector{ComplexF64}},Chebyshev_series_1::Vector{Vector{ComplexF64}})
    vector_wise_Chebyshev_new(H_n, v, M,Chebyshev_series_0)
    @inbounds for m in range(1,M)
        Chebyshev_series_0[m] = J_1*Chebyshev_series_0[m]
    end

    vector_wise_Chebyshev_new(H_n, J_2 * v  , M, Chebyshev_series_1)
    
    trace_Matrix = Matrix{ComplexF64}(undef, M, M)

    @inbounds for i in range(1,M) 
        @inbounds for j in range(1,M)
            trace_Matrix[j ,i] = Chebyshev_series_0[j]' * Chebyshev_series_1[i]
        end
    end
    return trace_Matrix
end





#####vector wise kernel polynomial method
"""Calculates the DOS using a set of stochastic vectors v"""
function vector_wise_DOS(E_n::Float64,v::Vector{ComplexF64},M::Int)
    DOS=DOS_projector(E_n,M)
    Chebyshev_series= transpose(DOS) * v
    return Chebyshev_series
end


"""Calculates the JDOS using a set of stochastic vectors v"""
function vector_wise_JDOS(E_n::Float64,w_n::Float64,v::Vector{ComplexF64},M::Int)
    JDOS = JDOS_projector(E_n,w_n,M)
    Chebyshev_series= transpose(JDOS) * v
    return Chebyshev_series
end



#integrands for conductivity calculations
"""Calculates one integrand of the conductivity, CM is an array that contains all the Chebyhev coefficients"""
function integrand_conductivty_e(E_n::Float64, w_n::Float64, u_n::Float64, u_n_quasi::Float64, beta_n::Float64, M::Int, C_m::Matrix{ComplexF64})
    DOS=DOS_projector(E_n, M)
    JDOS=JDOS_projector(E_n, w_n, M)
    Integrand_e= Pauli_function_e(E_n, beta_n, u_n, u_n_quasi) .* (DOS'* (C_m*JDOS))
    return Integrand_e
end



"""Calculates one integrand of the conductivity, CM is an array that contains all the Chebyhev coefficients"""
function integrand_conductivty_h(E_n::Float64,w_n::Float64,u_n::Float64,u_n_quasi::Float64,beta_n::Float64,M::Int,C_m::Matrix{ComplexF64})
    DOS=DOS_projector(E_n,M)
    JDOS=JDOS_projector(E_n,w_n,M)
    Integrand_h= Pauli_function_h(E_n, beta_n, u_n, u_n_quasi) .* (DOS' *(C_m*JDOS))
    return Integrand_h
end


"""Calculates the optical conductivtiy"""
function conductivity_new(w_n::Float64, u_n::Float64, u_n_quasi::Float64, beta_n::Float64, M::Int, C_m::Matrix{ComplexF64}, tol::Float64, maxvals::Int) 
    conduc_e, error_e= quadgk(E_n->integrand_conductivty_e(E_n,  w_n,  u_n,  u_n_quasi  ,beta_n,  M,  C_m),-1,1, rtol=tol,maxevals=maxvals)
    conduc_h, error_h= quadgk(E_n->integrand_conductivty_h(E_n,  w_n,  u_n,  u_n_quasi  ,beta_n,  M,  C_m),-1,1, rtol=tol,maxevals=maxvals)
    conductivity_tot= (conduc_e+conduc_h)
    return real(conductivity_tot) , real(conduc_h), real(conduc_e) 
end






"""Calculates the spectral bounds: highest and lowest eigenvalue"""
function get_spectral_bounds(hamiltonian::SparseMatrixCSC{ComplexF64, Int64})
    E_max = real(eigsolve(hamiltonian, 1, :LR, ishermitian=true)[1][1])::Float64
    E_min = real(eigsolve(hamiltonian, 1, :SR, ishermitian=true)[1][1])::Float64
    return E_max, E_min
end

"""A split count function for MPI and parallelisation"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q+1 : q for i in 1:n]
end



function vector_wise_Chebyshev_2(H_n::SparseMatrixCSC{ComplexF64},v::Vector{ComplexF64},M::Int)
    v_0 = v + zeros(ComplexF64, length(v))
    v_1 = H_n * v
    Chebyshev_series=[v' * v_0 ,v'* v_1]
    v_i = zeros(ComplexF32, length(v))
    for m in range(3,M)
        v_i= 2 .* (H_n * v_1) .- v_0
        push!(Chebyshev_series,v' * v_i)
        v_0 = v_1
        v_1 = v_i

    end
    return Chebyshev_series
end





function generate_random_vector(n)
    return [exp(im * 2pi * rand()) for _ in 1:n]
end

"""Calculates the charge carrier density using the DOS"""
function charge_density(v::Vector{ComplexF64},M::Int,beta_n::Float64,u_n::Float64,V::Float64,tol::Float64,maxvals::Int)
    p,error= quadgk(E_n->(vector_wise_DOS(E_n,v,M)*Fermi_function(E_n, beta_n, u_n)),u_n,1, rtol=tol,maxevals=maxvals)
    return real(p)/V
end

