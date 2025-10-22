#example file for a conductivity calculation with MPI

#import functions needed
path_functions= joinpath("Path_to_file", "vector_wise_conductivity_efficient.jl")
path_reading=joinpath("Path_to_file", "Read_TB_Hamiltonian.jl") 
path_noise= joinpath("Path_to_file","TB_Hamiltonian_noise.jl")  

using MPI
include(path_functions)
include(path_reading)
include(path_noise)


#import Hamiltonians
test_path= "Path_to_TB_Hamiltonians"
h_val= ustrip(ħ)/ustrip(e)
Hamiltonian_perovskite,current=get_TB_matrices(test_path, h_val, 0, 0.0, 0.0)
Current_1= current[1]
Current_2=current[2]
Current_3=current[3]

E_max,E_min= get_spectral_bounds(Hamiltonian_perovskite) #calculate the spectral bounds
#Define simulation parameters
Chebyshev_num=1000 #Number of Chebyhsev coefficients
Stochastic_trace_num=100 #Number of stochastic vectors
T=300.0 #Temperature in Kelvin
beta= ustrip(e)/(T *ustrip(k_B))
u=1.2 #Fermi energy
V=1 #Volume of cell
tol=1e-8 #Tolerance of error for numerical integration
maxval=1000 #Maximum value and cutoff for numerical integration
n=size(Hamiltonian_perovskite)[1]

#Rescaling of parameters and Hamiltonian for Chebyhsev expansion
Emean = (E_max_huckel + E_min_huckel) / 2
Ediff = (E_max_huckel - E_min_huckel) / (2 - 0.01)
u_n=(u-Emean)/(Ediff)
beta_n= beta*Ediff


for i in 1:n
    Hamiltonian_perovskite[i, i] = (Hamiltonian_perovskite[i, i] - Emean)
end
Hamiltonian_perovskite .= Hamiltonian_perovskite ./ Ediff

#Initialise MPI
MPI.Init()
start_mpi= MPI.Wtime() #for measuring computation time
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)
root = 0

#Distributing tasks on different ranks
a = split_count(Stochastic_trace_num, comm_size)
local_size = a[rank + 1]
local_start = sum(a[1:rank]) + 1
local_end = local_start + local_size - 1

# Local computation of Kernel Polynomial Method
local_sum = zeros(ComplexF64, Chebyshev_num, Chebyshev_num)
local_vec= zeros(ComplexF64, Chebyshev_num)
v=Vector{ComplexF64}(undef, n)
Chebyshev_series_0 = Vector{Vector{ComplexF64}}(undef, Chebyshev_num)
Chebyshev_series_1 = Vector{Vector{ComplexF64}}(undef, Chebyshev_num)


for i in local_start:local_end
    copy!(v ,generate_random_vector(n)) #generate random vector in place
    local_sum .+= Current_Current_correlation_new(Hamiltonian_perovskite, v, num_cheby, current[1], current[1],Chebyshev_series_0 ,Chebyshev_series_1) #calculating coefficients for Chebyhsev expansion
end
MPI.Barrier(comm)


for i in local_start:local_end
    copy!(v ,generate_random_vector(n))
    local_vec .+= vector_wise_Chebyshev_2_optimized_2(Hamiltonian_perovskite, v, num_cheby) #calculating coefficients for DOS calculation
end
MPI.Barrier(comm)

    
# Global reduction 
global_sum = zeros(ComplexF64, Chebyshev_num, Chebyshev_num)
global_vec = zeros(ComplexF64, Chebyshev_num)
MPI.Reduce!(local_sum, global_sum, +, root, comm)
MPI.Reduce!(local_vec, global_vec, +, root, comm)
    
MPI.Barrier(comm)
    
K_m = rank == root ? global_sum ./ num_vec : nothing
v_h = rank == root ? global_vec ./ num_vec : nothing
    
MPI.Barrier(comm)


parallel_end= MPI.Wtime()
#Calculation of DOS
#Distribution of Energy spectrum over ranks
E_n=1000
b= split_count(E_n, comm_size)
global_start_E= -2
global_end_E= 2 #Set boundary for normalised energy
full_interval_E = range(global_start_E, global_end_E, E_n)

local_size_E = b[rank + 1]
local_start_E = sum(b[1:rank]) + 1
local_end_E = local_start_E + local_size_E - 1

local_xs_E = collect(full_interval_E[local_start_E:local_end_E])



local_DOS = Vector{Float64}(undef, local_size_E)



MPI.Bcast!(v_h, root, comm)

DOS_start_time=MPI.Wtime()
 for idx in 1:local_size_E
    local_E_val = local_xs_E[idx]
    local_DOS[idx]= real( vector_wise_DOS(local_E_val,  v_h,  Chebyshev_num) )
end
MPI.Barrier(comm)
DOS_end_time=MPI.Wtime()


global_DOS = rank == root ? Vector{Float64}(undef, E_n) : Vector{Float64}(undef, 0)


if rank == root
    recvbuf = MPI.VBuffer(global_DOS, b)
else
    recvbuf = nothing
end


MPI.Barrier(comm)

MPI.Gatherv!(local_DOS, recvbuf, root, comm)

MPI.Barrier(comm)


#Write the results for the DOS calculation as CSV dataframe
if rank==0
output_dataframe_DOS= DataFrame(Energy= [E for E in range(E_min,E_max,E_n)], Density_of_states=global_DOS) 
outputpath_DOS= joinpath(pwd(), "DOS_$(Chebyshev_num).csv")
CSV.write(outputpath_DOS, output_dataframe_DOS, delim=',')
end
MPI.Barrier(comm)


MPI.Barrier(comm)

######parallel integral calculation for Kubo conductivity

global_start=-10*10e16 #define frequency range
global_end=10*10e16
N = 1000 #Number of evaluated points
E_list=[] #energy in eV
wn_list=[] #normalised frequency given in eV
u_list=[] #nomralised quasi fermi energy 
for w in range(global_start,global_end,N)
    E=ustrip(h)*w/(ustrip(e))
    w_n= (E)/(Ediff)
    u_quasi= u+E
    u_n_quasi=(u_quasi-Emean)/(Ediff)
    push!(E_list, E)
    push!(wn_list,w_n)
    push!(u_list,u_n_quasi)
end
frequ= [w for w in range(global_start,global_end,N)]
#Distribution of different frequencies on different ranks
a = split_count(N, comm_size)
local_size = a[rank + 1]
local_start = sum(a[1:rank]) + 1
local_end = local_start + local_size - 1

conduc= rank == root ? Vector{Float64}(undef, N) : Vector{Float64}(undef, 0)
hole= rank == root ? Vector{Float64}(undef, N) : Vector{Float64}(undef, 0)
electron= rank == root ? Vector{Float64}(undef, N) : Vector{Float64}(undef, 0)
local_xs = collect(frequ[local_start:local_end])



local_conduc = Vector{Float64}(undef, local_size)
local_hole = Vector{Float64}(undef, local_size)
local_electron = Vector{Float64}(undef, local_size)


if rank == root
    recvbuf = MPI.VBuffer(conduc, a)
    recvbuf1 = MPI.VBuffer(hole, a)
    recvbuf2 = MPI.VBuffer(electron, a)
else
    recvbuf = nothing
    recvbuf1 = nothing
    recvbuf2 = nothing
end


#Local calculation of kubo conductivity via numerical integration
 for idx in 1:local_size 
    w_val = local_xs[idx]
    c,h,e =conductivity_new(wn_list[idx], u_n, u_list[idx], beta_n, Chebyshev_num, K_m, tol, maxval) 
    local_conduc[idx]= tanhc(w_val ,T,V)*real(c)
    local_hole[idx]= tanhc(w_val ,T,V)*real(h)
    local_electron[idx]= tanhc(w_val ,T,V)*real(e) 
end


MPI.Barrier(comm)


MPI.Gatherv!(local_conduc, recvbuf, root, comm)
MPI.Gatherv!(local_hole, recvbuf1, root, comm)
MPI.Gatherv!(local_electron, recvbuf2, root, comm)


MPI.Bcast!(conduc, root, comm)
MPI.Bcast!(hole, root, comm)
MPI.Bcast!(electron, root, comm)

MPI.Barrier(comm)
end_mpi=MPI.Wtime()

#####writing results and performance to file
if rank==root
elpased_mpi=end_mpi-start_mpi


DOS_time=DOS_end_time-DOS_start_time

output_dataframe_conduc= DataFrame(Frequency= frequ ,Conductivity= conduc,hole= hole, electron= electron) #writing conductivity as CSV
outputpath_conduc= joinpath(pwd(), "Perovskite_$(num_cheby).csv")
CSV.write(outputpath_conduc, output_dataframe_conduc, delim=',')

open(pwd() * "/performance_$(n).txt", "w") do file
    write(file, "### Simulation parameters \n \n")
    write(file, "Dimension of matrix: $(n)  \n")
    write(file, "Number of Chebyshev polynomials: $(Chebyshev_num)  \n")
    write(file, "Number of vectors: $(Stochastic_trace_num)  \n")
    write(file, "Integral accuracy: $(tol)  \n")
    write(file, "Temperature: $(T)  \n")
    write(file, "Fermi energy: $(u)  \n")
    write(file, "### Benchmarking \n \n")
    write(file, "MPI time: $(elpased_mpi)  \n")
    write(file, "DOS time : $(DOS_time)  \n")
    write(file, "### parallelization: \n \n")
    write(file, "number of ranks: $(comm_size) \n")
    write(file, "number of threads: $(nthreads()) \n")
end
end
MPI.Finalize()