using AbstractGPs
using Plots
using QuadGK
using KernelFunctions
using LinearAlgebra
using Random
using Stheno

import AbstractGPs: AbstractGP, mean, cov, var

using Stheno: DerivedGP

convolve(f::AbstractGP) = DerivedGP((convolve, f), f.gpc)

const conv_args = Tuple{typeof(convolve),AbstractGP}

mean((_, f)::conv_args, x::AbstractVector{<:Real}) = zeros(length(x))
cov(args::conv_args, x::AbstractVector{<:Real}) = cov(args, x, x)
var(args::conv_args, x::AbstractVector{<:Real}) = var(args, x, x)
function var(args::conv_args, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    return diag(cov(args, x, x′))
end

_quadrature(f, xs, ws) = sum(map((x, w) -> w * f(x), xs, ws))

function cov((_, f)::conv_args, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    cols_of_C = map(x′) do x′n
        col_elements = map(x) do xn
            _quadrature(
                x -> _quadrature(x′ -> only(cov(f, [xn - x], [x′n - x′])), xs, ws),
                xs,
                ws,
            )
        end
    end
    return reduce(hcat, cols_of_C)
end
