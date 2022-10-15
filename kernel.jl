using QuadGK
using KernelFunctions
using Plots, Stheno
function Kernel_KL(p0::T, ω::T) where {T}
    return ω / ((p0^2 + ω^2) * π)
end

function Kernel_RBF(ω1::T, ω2::T; σ=1, l=1) where {T}
    return σ^2 * exp(-((ω1 - ω2)^2) / (2 * l^2))
end

function KernelKC_KLRBF(p0, ω; σ=1, l=1)
    return quadgk(η -> Kernel_KL(p0, η) * Kernel_RBF(η, ω, σ=σ, l=l), 0, Inf)[1]
end

KernelKC_KLRBF(0.0, 0.0)

struct MyKernel <: KernelFunctions.Kernel end

(::MyKernel)(x, y; σ=1, l=1) = KernelKC_KLRBF(x, y, σ=σ, l=l)
x = range(0.0001, 1.0; length=100)

# A simple standardised squared-exponential / exponentiated-quadratic kernel.
k₁ = MyKernel()
K₁ = kernelmatrix(k₁, x)
heatmap(K₁; yflip=true, colorbar=false, aspect_ratio=1, size=(600, 600))
x = GPPPInput(:f1, collect(range(-5.0, 5.0; length=100)));
f = @gppp let
    f1 = GP(SqExponentialKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end
fx = f(x, 2e-3)
# y = rand(fx);
collect(mean(fx))
y = sin.(collect(range(-5.0, 5.0; length=10)))

f_posterior = posterior(fx, y)
plot(x.x, y; color=:red, label="", seriestype=:scatter)
x_plot = range(-7.0, 7.0; length=1000)
xp = GPPPInput(:f1, x_plot);

fp = f_posterior(xp)

plot!(x_plot, mean.(marginals(fp)); label="mean")

var(fp).|>sqrt |> (σ -> plot!(x_plot, mean(fp) .+ 3*σ, fillrange=mean(fp) .- 3*σ, fillalpha=0.2, label="std"))

marginals(fp)
mean.(marginals(fp))
plot!(
    x_plot, f_posterior(xp);
    ribbon_scale=3, label="", color=:blue, fillalpha=0.2, linewidth=2
)

plot!(
    x_plot, f_posterior(xp);
    ribbon_scale=3, label="", color=:blue, fillalpha=0.2, linewidth=2
)

plot!(
    x_plot, rand(f_posterior(xp, 1e-9), 10);
    samples=10, markersize=1, alpha=0.3, label="", color=:blue
)

fp(x_plot)
