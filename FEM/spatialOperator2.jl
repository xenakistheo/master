using Gridap
using Gridap.FESpaces
using LinearAlgebra

using GridapMakie, GLMakie

#=
We are interested in solving the PDE

  Lu :=  (κ^2 - ∇²)u = f
with zero Neumann BC's. 
As κ > 0 and f can be varying, we want to construct 
the problem such that we can limit the repetition of 
assembling matrices. 
For that, we define L = κ^2 C + G, and assemble C, G once. 

=#
# Domain and mesh
Nx, Ny = 20, 20
domain = (0.0, 1.0, 0.0, 1.0)
partition = (Nx-1, Ny-1)

# Define triangular mesh, and measure for integration
model_quad = CartesianDiscreteModel(domain, partition)
model = simplexify(model_quad)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)

# Define Finite Element spaces
reffe = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe; conformity=:H1)
U = TrialFESpace(V)

# Mass and stiffness forms
c(u,v) = ∫(v*u) * dΩ
g(u,v) = ∫(∇(v) ⋅ ∇(u)) * dΩ

# Assemble mass and stiffness matrices
C = assemble_matrix(c, U, V) #mass
G = assemble_matrix(g, U, V) #stiffness


# # Test functions for the right-hand side
# f1(x) = sin(π*x[1]) * sin(π*x[2])
# f2(x) = exp(-50*((x[1]-0.5)^2 + (x[2]-0.5)^2))

# L1 = 1.0^2 * C + G
# L2 = 1.0^2 * C + G

# l(v) = ∫(v*f1) * dΩ
# b = assemble_vector(l, V)

# u = L1 \ b # solve A \ b
# uh = FEFunction(U, u)

# fig = plot(uh) # plot mesh
# wireframe!(Ω, color=:black, linewidth=1)

####
E = eigen(Matrix(G), Matrix(C))
ξ = E.values
V = E.vectors

for k in 1:10
    println(k, ": ", ξ[k])
end