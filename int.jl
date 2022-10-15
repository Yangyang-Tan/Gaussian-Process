using HCubature, FastGaussQuadrature

function KCK_RBF_polar(p1, p2, r, θ; σ=1, l=1)
    return (r^3 * σ^2 * cos(θ) * exp(-1 / 2 * (r * cos(θ) - r * sin(θ))^2 / l^2) * sin(θ)) /
           (pi^2 * (p1^2 + r^2 * cos(θ)^2) * (p2^2 + r^2 * sin(θ)^2))
end

function intfun_kck2(t, t2, σc, l)
    return (t * t2 * σc^2 * exp(-1 / 2 * (t - t2)^2 / l^2)) /
           (pi^2 * (1^2 + t^2) * (1^2 + t2^2))
end

function intfun_kck3(t, t2, σc, l)
    return (t * t2 * σc^2 * exp(-1 / 2 * (t - t2)^2 / l^2 + t + t2)) /
           (pi^2 * (1^2 + t^2) * (1^2 + t2^2))
end

hcubature(
    x -> KCK_RBF_polar(1.0, 1.0, x[1],x[2]; σ=10, l=1 / 10),
    [0.0, 0.0],
    [100.0, π/2];
    # initdiv=1000,
    atol=1e-14,
    rtol=1e-14,
    maxevals=2000000000,
)[1]

using QuadGK

quadgk(
    r -> quadgk(
        θ -> KCK_RBF_polar(1.0/40, 70.2, r, θ; σ=10, l=1 / 10), 0, π / 4, π / 2; order=500
    )[1],
    0.0,
    Inf;
    order=20,
)[1]

quadgk(
    θ -> quadgk(r -> KCK_RBF_polar(1.0, 1.0, r, θ; σ=10, l=1 / 10), 0.0,Inf; order=10,atol=1e-8, rtol=1e-8)[1],
    0,
    π / 4,
    π / 2;atol=1e-8, rtol=1e-8,
    order=10,
)


quadgk(
    θ -> quadgk(r -> KCK_RBF_polar(10.0, 1.0, r, θ; σ=10, l=1 / 10), 0.0,Inf; order=10,atol=1e-5, rtol=1e-5)[1],
    0,
    π / 4,
    π / 2;atol=1e-5, rtol=1e-5
    # order=10,
)[1]

using LinearAlgebra

function QuadraIntegral(f; npoints=100)
    x, w = gausslegendre(2 * npoints)
    # x1, w1 = gausslegendre(npoints)
    # x2, w2 = gausslegendre(2*npoints)
    x1, w1 = x[1:2:(2 * npoints)], w[1:2:(2 * npoints)]
    x2, w2 = x[2:2:(2 * npoints)], w[2:2:(2 * npoints)]
    # x, w = x[(npoints + 1):(2 * npoints)], w[(npoints + 1):(2 * npoints)]
    function g(t1, t2)
        return 4 / (1 - t1)^2 * 1 / (1 - t2)^2 * f((1 + t1) / (1 - t1), (1 + t2) / (1 - t2))
    end
    fv = Matrix{Float64}(undef, npoints, npoints)
    fv .= g.(x1, x2')
    return ⋅(w1, fv, w2) * 4
end

function QuadraIntegral_polar(f; npoints=100)
    # x, w = gausslegendre(8 * npoints)
    x1, w1 = gausslegendre(1 * npoints)
    x2, w2 = gausslegendre(128*npoints)
    # x1, w1 = x[1:2:2*npoints], w[1:2:2*npoints]
    # x2, w2 = x[2:2:2*npoints], w[2:2:2*npoints]
    # x, w = x[(npoints + 1):(8 * npoints)], w[(npoints + 1):(8 * npoints)]
    function g(t1, t2)
        return (pi * f(1/(2 - 2 * t1), (pi * (1 + t2)) / 4)) / (8 * (-1 + t1)^2)
    end
    fv = Matrix{Float64}(undef, 1 * npoints, 128*npoints)
    fv .= g.(x1, x2')
    int1 = ⋅(w1, fv, w2)
    return int1
end

QuadraIntegral_polar((r, θ) -> KCK_RBF_polar(1.0, 1.0, r, θ; σ=10, l=1 / 10); npoints=1000)


QuadraIntegral((x, y) -> intfun_kck3(x, y, 10.0, 1 / 10); npoints=100)

function intfun_kck3(t, t2, σc, l)
    return (t * t2 * σc^2 * exp(-1 / 2 * (t - t2)^2 / l^2)) /
           (pi^2 * (1^2 + t^2) * (1^2 + t2^2))
    # return exp(-1 / 2 * (t - t2)^2 / l^2)
end

intfun_kck3(1000, 1000, 10.0, 1 / 10)
@time QuadraIntegral((x, y) -> intfun_kck3(x, y, 10.0, 1 / 10); npoints=10000)

@time QuadraIntegral((x, y) -> exp(-(x - y)^2) / (x^2 + 1); npoints=20000)
@time QuadraIntegral2((x, y) -> exp(-(x - y)^2) / (x^2 + 1); npoints=6000)
quadgk(
    y -> quadgk(x -> exp(-(x - y)^2) / (x^2 + 1), 0, Inf; order=600)[1], 0, Inf; order=600
)
