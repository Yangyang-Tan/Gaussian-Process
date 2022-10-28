
@everywhere using DistributedArrays, OffsetArrays, LinearAlgebra, QuadGK

function KCK_RBF_polar(p1, p2, r, θ; σ=1, l=1)
    return (r^3 * σ^2 * cos(θ) * exp(-1 / 2 * (r * cos(θ) - r * sin(θ))^2 / l^2) * sin(θ)) /
           (pi^2 * (p1^2 + r^2 * cos(θ)^2) * (p2^2 + r^2 * sin(θ)^2))
end

function Kernel_KL(p0::T, ω::T) where {T}
    return ω / ((p0^2 + ω^2) * π)
end

function Kernel_RBF(ω1::T, ω2::T; σ=1, l=1) where {T}
    return σ^2 * exp(-((ω1 - ω2)^2) / (2 * l^2))
end

function KernelKC_KLRBF(p0, ω; σ=1, l=1)
    return quadgk(η -> Kernel_KL(p0, η) * Kernel_RBF(η, ω; σ=σ, l=l), 0, Inf)[1]
end

function Ctilde_RBF_fun(p1, p2; σ=1, l=1)
    return quadgk(
        θ -> quadgk(
            r -> KCK_RBF_polar(p1, p2, r, θ; σ=σ, l=l),
            0.0,
            Inf;
            order=50,
            atol=1e-7,
            rtol=1e-7,
        )[1],
        0,
        π / 4,
        π / 2;
        atol=1e-7,
        rtol=1e-7,
        order=10,
    )[1]
end
function KCint_RBF_fun(p1, p2; σ=1, l=1)
    return quadgk(
        θ -> quadgk(
            r -> KCK_RBF_polar(p1, p2, r, θ; σ=σ, l=l),
            0.0,
            Inf;
            order=50,
            atol=1e-7,
            rtol=1e-7,
        )[1],
        0,
        π / 4,
        π / 2;
        atol=1e-7,
        rtol=1e-7,
        order=10,
    )[1]
end

function spec_mean_fun(ω; p_grid, G_grid, σ=1, l=1)
    invkck = inv(
        sitp_Ctilde_fun.(p_grid', p_grid) +
        0.01 .* Matrix(I, length(p_grid), length(p_grid)),
    )
    kc = KernelKC_KLRBF.(p_grid, ω; σ=σ, l=l)
    return ⋅(kc', invkck, G_grid)
end

function spec_kernel_fun(ω1, ω2; p_grid, σ=1, l=1)
    invkck = inv(
        sitp_Ctilde_fun.(p_grid', p_grid) +
        0.01 .* Matrix(I, length(p_grid), length(p_grid)),
    )
    kc1 = KernelKC_KLRBF.(p_grid, ω1; σ=σ, l=l)
    kc2 = KernelKC_KLRBF.(p_grid, ω2; σ=σ, l=l)
    return Kernel_RBF(ω1, ω2; σ=σ, l=l) - ⋅(kc1', invkck, kc2)
end
