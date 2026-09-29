using Gridap


# Load geometry from JSON File
model = DiscreteModelFromFile("./FEM/models/model.json")
# fieldnames(typeof(model))

# these lines generate new files for vertices edges, faces, and cells
mkpath("output_path")
writevtk(model,"output_path/model")

# We can visualise using `Paraview` - a software for visualising scientific data. 


### Generate a discrete approx. of the FE-spaces.  (std. conforming Lagrangian FE-space)
# V0
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="sides")

# Ug
g(x) = 2.0
Ug = TrialFESpace(V0, g)

### Define setup for numerical integration. I.e. create quadrature
# Interior
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

# Boundaries
neumanntags = ["circle", "triangle", "square"]
Γ = BoundaryTriangulation(model, tags=neumanntags)
dΓ = Measure(Γ,degree)

### Weak form 
f(x) = 1.0
h(x) = 3.0
a(u,v) = ∫( ∇(v) ⋅ ∇(u) )*dΩ
b(v) = ∫( v*f )*dΩ + ∫( v*h )*dΓ

### Build the FE problem
op = AffineFEOperator(a,b,Ug,V0)


# Solver phase
ls = LUSolver() # select solver
solver = LinearFESolver(ls)

# solve the linear system
uh = solve(solver,op)

# we can inspet the result by writing it to a VTK file
writevtk(Ω,"output_path/results",cellfields=["uh"=>uh])

### Visualize
using GridapMakie, GLMakie
fig = plot(uh) # plot solution

fig = plot(Ω) # plot mesh

fig = plot(Γ, uh, colormap=:rainbow) # plot solution on boundary