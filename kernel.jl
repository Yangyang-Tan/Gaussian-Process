using KernelFunctions
using Plots, Stheno

struct SpecKernel{T} <: Kernel
    p_grid
    σ::T
    l::T
end
struct PropKernel{T} <: Kernel
    σ::T
    l::T
end
SpecKernel(; p_grid, σ=1.0, l=1.0) = SpecKernel(p_grid, σ, l)
PropKernel(; σ=1.0, l=1.0) = PropKernel(σ, l)
function (k::SpecKernel)(x, y)
    p_grid = k.p_grid
    σ = k.σ
    l = k.l
    return spec_kernel_fun(x, y; p_grid=p_grid, σ=σ, l=l)
end


function (k::PropKernel)(x, y)
    σ = k.σ
    l = k.l
    return Ctilde_RBF_fun(x, y; σ=σ, l=l)
end

# A simple standardised squared-exponential / exponentiated-quadratic kernel.
p0_BW = gluon_lat_T121[2:end, 1]
G_BW = gluon_lat_T121[2:end, 1]

Prop_BW_fun.(p0_BW; A=1, Γ=1, M=1)
k₁ = MyKernel()
K₁ = kernelmatrix(k₁, x)
heatmap(K₁; yflip=true, colorbar=false, aspect_ratio=1, size=(600, 600))

x = GPPPInput(:f1, exp.(range(log(0.01), log(40); length=30)));
f = @gppp let
    f1 = GP(
        x -> spec_mean_fun.(x; p_grid=p0_BW, G_grid=G_BW, σ=1.0, l=1 / 1.0),
        SpecKernel(; p_grid=p0_BW, σ=1.0, l=1 / 1.0),
    )
end

fx = f(x, 2e-3)
# y = rand(fx);
collect(mean(fx))
y = sin.(collect(range(-5.0, 5.0; length=100)))
plot(x.x, y; color=:red, label="", seriestype=:scatter)
x_plot = range(-7.0, 7.0; length=100)
xp = GPPPInput(:f1, x_plot);
x = GPPPInput(:f3, collect(range(-5.0, 5.0; length=100)));

plot(
    x.x, f(x); ribbon_scale=1, label="", color=:blue, fillalpha=0.2, linewidth=2, xaxis=:log
)

f_posterior = posterior(fx, y)

x_plot = range(-7.0, 7.0; length=1000)
xp = GPPPInput(:f1, x_plot);

fp = f_posterior(xp)

plot!(x_plot, mean.(marginals(fp)); label="mean")

(
    σ -> plot!(
        x_plot,
        mean(fp) .+ 3 * σ;
        fillrange=mean(fp) .- 3 * σ,
        fillalpha=0.2,
        label="std",
    )
)(
    sqrt.(var(fp))
)

marginals(fp)
mean.(marginals(fp))
plot!(
    x_plot,
    f_posterior(xp);
    ribbon_scale=3,
    label="",
    color=:blue,
    fillalpha=0.2,
    linewidth=2,
)

plot!(
    x_plot,
    f_posterior(xp);
    ribbon_scale=3,
    label="",
    color=:blue,
    fillalpha=0.2,
    linewidth=2,
)

plot!(
    x_plot,
    rand(f_posterior(xp, 1e-9), 10);
    samples=10,
    markersize=1,
    alpha=0.3,
    label="",
    color=:blue,
)

fp(x_plot)
