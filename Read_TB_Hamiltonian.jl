
##########################################################################
########## TB
##########################################################################

function construct_TB_matrices(row::Vector{Int64}, col::Vector{Int64}, H_elem::Vector{ComplexF64}, traj::Array{Float64}, ħ_eVfs::Float64)

    ### initialize empty matrices
    current0 = Vector{SparseMatrixCSC{ComplexF64}}(undef, 3)
    
    ### fill hamiltonian
    hamiltonian0 = sparse(row, col, H_elem)

    ### fill current
    onsites = row .== col
    hopping = .!onsites

    dropzeros!(hamiltonian0)

    #rows = vcat(row[onsites], row[hopping])
    #cols = vcat(col[onsites], col[hopping])


    for d in 1:3

        #C_elem_onsite = 0 
        #C_elem_onsite = traj[onsites, d]
            
        #C_elem_hopping = -1im/ħ_eVfs * H_elem[hopping] #.* traj[hopping, d]
        C_elem_hopping = -1im/ħ_eVfs * H_elem[hopping] .* traj[hopping, d]

        #C_elem = -1im/ħ_eVfs * H_elem .* traj[:, d]
        #C_elem = vcat(C_elem_onsite, C_elem_hopping)

        #current0[d] = sparse(row, col, C_elem)
        current0[d] = sparse(row[hopping], col[hopping], C_elem_hopping)
        dropzeros!(current0[d])

    end

    return hamiltonian0, current0
end

### extract intitial hamiltonian, current matrix, and commutator of current and polarization matrix for snapshot t_start
function get_TB_matrices(TB_path::String, ħ_eVfs::Float64, t::Int, Anderson_noise::Float64, DD_noise::Float64)

    row, col, H_elem, traj = read_TB_params(t, TB_path)

    hamiltonian0, current0 = construct_TB_matrices(row, col, H_elem, traj, ħ_eVfs)

    noise!(hamiltonian0, current0, Anderson_noise, DD_noise, t)

    return hamiltonian0, current0
end



### extract hamiltonian for snapshot t
function get_H_TB(TB_path::String, t::Int, Anderson_noise::Float64, DD_noise::Float64)
   
    #hamiltonian = SparseMatrixCSC{ComplexF64}

    row, col, H_elem, _ = read_TB_params(t, TB_path)

    hamiltonian = sparse(row, col, H_elem)
    dropzeros!(hamiltonian)

    noise_hamiltonian!(hamiltonian, Anderson_noise, DD_noise, t)

    return hamiltonian
end

### overwrite hamiltonian of snapshot t to hamiltonian variable
function get_H_TB!(hamiltonian::SparseMatrixCSC{ComplexF64}, TB_path::String, t::Int, Anderson_noise::Float64, DD_noise::Float64)

    fill!(hamiltonian.nzval, 0.0 + 1im * 0.0)

    row, col, H_elem, _ = read_TB_params(t, TB_path)

    hamiltonian .+= sparse(row, col, H_elem)

    #dropzeros!(hamiltonian)

    noise_hamiltonian!(hamiltonian, Anderson_noise, DD_noise, t)

end


### extract current for snapshot t
function get_C_TB(TB_path::String, ħ_eVfs::Float64, t::Int64, DD_noise::Float64)
   
    current = Vector{SparseMatrixCSC{ComplexF64}}(undef, 3)

    row, col, H_elem, traj = read_TB_params(t, TB_path)

    onsites = row .== col
    hopping = .!onsites

    #rows = vcat(row[onsites], row[hopping])
    #cols = vcat(col[onsites], col[hopping])
        
    for d in 1:3

        #C_elem_onsite  = traj[onsites, d]
            
        C_elem_hopping = -1im/ħ_eVfs .* H_elem[hopping] .* traj[hopping, d]

        #C_elem = vcat(C_elem_onsite, C_elem_hopping)

        #current[d] = sparse(rows, cols, C_elem_hopping)
        current[d] = sparse(row[hopping], col[hopping], C_elem_hopping)
        dropzeros!(current[d])
            
    end

    noise_current!(current, DD_noise, t)

    return current
end

### overwrite current of snapshot t to current variable
function get_C_TB!(current::Vector{SparseMatrixCSC{ComplexF64}}, TB_path::String, ħ_eVfs::Float64, t::Int64, DD_noise::Float64)
    
    for d in 1:3
        fill!(current[d].nzval, 0.0 + 1im * 0.0)
    end

    row, col, H_elem, traj = read_TB_params(t, TB_path)

    onsites = row .== col
    hopping = .!onsites

    #rows = vcat(row[onsites], row[hopping])
    #cols = vcat(col[onsites], col[hopping])
        
    for d in 1:3

        #C_elem_onsite  = traj[onsites, d]
            
        C_elem_hopping = -1im/ħ_eVfs .* H_elem[hopping] .* traj[hopping, d]

        #C_elem = vcat(C_elem_onsite, C_elem_hopping)

        #current[d] = sparse(rows, cols, C_elem_hopping)
        current[d] .+= sparse(row[hopping], col[hopping], C_elem_hopping)
        dropzeros!(current[d])
            
    end

    #for d in 1:3
    #    dropzeros!(current[d])
    #end

    noise_current!(current, DD_noise, t)

end



### calculates the spectral bounds for num_H hamiltonians
function calc_spectral_bounds(TB_path::String, num_H::Int, mod_H::Int, rank::Int, rank_size::Int, t_start::Int, Anderson_noise::Float64, DD_noise::Float64)

    ### initialization of energy bound arrays
    E_max = Vector{Float64}(undef, num_H)
    E_min = Vector{Float64}(undef, num_H)

    #hamiltonian = get_H_TB(TB_path, t_start + 1)

    @time begin

        tforeach(1:num_H; chunksize=1) do t
        #for t in 1:num_H

            GC.gc()

            ### build hamiltonian
            #get_H_TB!(hamiltonian, TB_path, t_start + rank * num_H + t)
            hamiltonian = get_H_TB(TB_path, t_start + rank * num_H + t, Anderson_noise, DD_noise) 

            ### calculate bounds for hamiltonian
            E_max[t], E_min[t] = get_spectral_bounds(hamiltonian)

            #@show Sys.free_memory() / 2^20
            GC.gc()

        end

    end

    mean_E, ΔE = transform_band_center_and_width(E_max, E_min)
    
    @time begin

        ### special treatment for last rank if total number of hamiltonians divided by rank_size leads to a remainder mod_H
        if rank < mod_H

            #GC.gc()

            #get_H_TB!(hamiltonian, TB_path, t_start + rank_size * num_H + rank)
            hamiltonian = get_H_TB(TB_path, t_start + rank_size * num_H + rank, Anderson_noise, DD_noise) 

            E_max1, E_min1 = get_spectral_bounds(hamiltonian)

            mean_E1, ΔE1 = transform_band_center_and_width(E_max1, E_min1)

        else
            mean_E1 = 0.0
            ΔE1 = 0.0
        end

    end

    return mean_E, ΔE, mean_E1, ΔE1
end

function read_TB_params(t::Int, TB_path::String)
    file_path = TB_path * "TB_" * string(t) * ".txt"
    
    row = Int[]
    col = Int[]
    H_elem = Complex{Float64}[]
    traj = Vector{Vector{Float64}}(undef, 0)
    
    open(file_path, "r") do io
        for line in eachline(io)
            data = parse.(Float64, split(line))
            push!(row, Int(data[1]))
            push!(col, Int(data[2]))
            push!(H_elem, Complex{Float64}(data[3], data[4]))
            push!(traj, data[5:end])
        end
    end
    
    traj = permutedims(hcat(traj...), [2,1])
    
    return row, col, H_elem, traj
end

##########################################################################
########## static_avg_TB
##########################################################################

### extract intitial hamiltonian, current matrix, and commutator of current and polarization matrix by averaging over all considered snapshots
function get_avg_TB_matrices0(TB_path::String, l::Int, ħ_eVfs::Float64, t_start::Int, Anderson_noise::Float64, DD_noise::Float64)
    
    for t in t_start:(l - 1 + t_start)

        if t == t_start
            row, col, H_elem, traj = read_TB_params(t, TB_path)
        else
            row0, col0, H_elem0, traj0 = read_TB_params(t, TB_path)
            row += row0
            col += col0
            H_elem += H_elem0
            traj += traj0
        end

    end

    row /= l
    col /= l
    H_elem /= l
    traj /= l

    hamiltonian0, current0 = construct_TB_matrices(row, col, H_elem, traj, ħ_eVfs)

    noise!(hamiltonian0, current0, Anderson_noise, DD_noise, t_start)

    return hamiltonian0, current0
end

