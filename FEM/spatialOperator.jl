using Gridap
using GridapMakie, GLMakie


Nx, Ny = 20, 20

domain = (0.0, 1.0, 0.0, 1.0)
partition = (Nx-1, Ny-1)

# Define triangular mesh. 
model_quad = CartesianDiscreteModel(domain, partition)
model = simplexify(model_quad)
Ω = Triangulation(model)

# Plot mesh
fig = plot(Ω) # plot mesh
wireframe!(Ω, color=:black, linewidth=1)
scatter!(Ω, markersize=4)
fig

# Define boundary
Γ = BoundaryTriangulation(model)

# Define measures
degree = 1
dΩ = Measure(Ω, 1)

# FE-space, tell Gridap what FE-space uh and vh live in
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model, reffe; conformity=:H1)
U = TrialFESpace(V)

# Define weak form 
f(x) = sin(π*x[1])*sin(π*x[2])

κ = 1.0
a(u, v) = ∫(v * u)*dΩ + ∫( ∇(v) ⋅ ∇(u) )*dΩ
b(v) = ∫( v*f )*dΩ



# Build FE problem 
op = AffineFEOperator(a, b, U, V)

# Solve linear system 
ls = LUSolver() # select solver
solver = LinearFESolver(ls)
uh = solve(solver,op)

# Plot solution
fig = plot(uh) # plot solution

#### Approach nr. 2
#= 
As we are interested in solving for varying κ, we simply extract the matrices
    C, G such that the linear system can be written as 
    L = κ^2 C + G
=# 
mass(u, v) = ∫(v * u)*dΩ
stiffness(u, v) = ∫( ∇(v) ⋅ ∇(u) )*dΩ

zero_rhs(v) = 0.0

op_C = AffineFEOperator(mass, zero_rhs, U, V)
op_G = AffineFEOperator(stiffness, zero_rhs, U, V)

C = get_matrix(op_C)
G = get_matrix(op_G)