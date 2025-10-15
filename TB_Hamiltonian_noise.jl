

using SparseArrays
using Distributions
using Random

function noise_hamiltonian!(hamiltonian::SparseMatrixCSC{ComplexF64}, Anderson_noise::Float64, DD_noise::Float64, t::Int)

    if Anderson_noise > 0.0
        dim = size(hamiltonian, 1)
        Random.seed!(1)
        hamiltonian .+= spdiagm(0 => (rand(Uniform(-Anderson_noise, Anderson_noise), dim)))
    end

    if DD_noise > 0.0
        Random.seed!(420 + t)
        z = length(hamiltonian.nzval)
        hamiltonian.nzval .+= DD_noise .* randn(Float64, z)
        hamiltonian .+= adjoint(hamiltonian)
        hamiltonian ./= 2
    end

end

function noise_current!(current::Vector{SparseMatrixCSC{ComplexF64}}, DD_noise::Float64, t::Int)

    if DD_noise > 0.0
        for d in 1:3
            Random.seed!(420 + t)
            z = length(current[d].nzval)
            current[d].nzval .+= 1im * DD_noise .* randn(Float64, z)
            current[d] .+= adjoint(current[d])
            current[d] ./= 2    
        end
    end

end


function noise!(hamiltonian::SparseMatrixCSC{ComplexF64}, current::Vector{SparseMatrixCSC{ComplexF64}}, Anderson_noise::Float64, DD_noise::Float64, t::Int)

    noise_hamiltonian!(hamiltonian, Anderson_noise, DD_noise, t)

    noise_current!(current, DD_noise, t)

end