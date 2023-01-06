module Numerical

export integrate,trapz, cumtrapz, differentiate, interp, quadarea, differentiate_first_dim

import ..RealArr1D, ..RealArr2D

function integrate(x::RealArr1D, y::RealArr1D, f::RealArr2D)
    # Compute the differences between adjacent elements in the x and y arrays
    dx = diff(x)
    dy = diff(y)
    
    # Use zip and map to apply the integration formula to each element of the f array
    integral = sum(map((xi, yi, fi) -> (xi * yi * fi), zip(dx, dy, f[1:end-1, 1:end-1])))
    
    return integral
end

function trapz(x::RealArr1D, f::RealArr1D)
    @assert length(x) == length(f)
    @assert !isempty(x)
    return sum(@. 0.5 * (f[1:end-1] + f[2:end]) * (x[2:end] - x[1:end-1]))
end

function trapz(x::RealArr2D, f::RealArr2D, dim::Integer)
    @assert size(x) == size(f)
    @assert dim > 0 && dim <= ndims(f)
    @assert !isempty(f)
    lower = Any[:, :]
    lower[dim] = 1:size(f)[dim]-1
    upper = Any[:, :]
    upper[dim] = 2:size(f)[dim]
    return vec(
        sum(@. 0.5 * (f[lower...] + f[upper...]) * (x[upper...] - x[lower...]); dims = dim),
    )
end

function trapz(x::RealArr1D, y::RealArr1D, f::RealArr2D)
    @assert (length(x), length(y)) == size(f)
    @assert !isempty(x)
    @assert !isempty(y)
    return sum(
        quadarea(x, y) .* @. 0.25 *
           (f[1:end-1, 1:end-1] + f[2:end, 1:end-1] + f[2:end, 1:end-1] + f[2:end, 2:end])
    )
end

function trapz(x::RealArr2D, y::RealArr2D, f::RealArr2D)
    @assert size(x) == size(f)
    @assert size(y) == size(f)
    @assert !isempty(f)
    return sum(
        quadarea(x, y) .* @. 0.25 *
           (f[1:end-1, 1:end-1] + f[2:end, 1:end-1] + f[2:end, 1:end-1] + f[2:end, 2:end])
    )
end

function quadarea(x::RealArr1D, y::RealArr1D)
    return quadarea.(
        x[1:end-1],
        y[1:end-1]',
        x[2:end],
        y[1:end-1]',
        x[2:end],
        y[2:end]',
        x[1:end-1],
        y[2:end]',
    )
end

function quadarea(x::RealArr2D, y::RealArr2D)
    return quadarea.(
        x[1:end-1, 1:end-1],
        y[1:end-1, 1:end-1],
        x[2:end, 1:end-1],
        y[2:end, 1:end-1],
        x[2:end, 2:end],
        y[2:end, 2:end],
        x[1:end-1, 2:end],
        y[1:end-1, 2:end],
    )
end

function quadarea(x1, y1, x2, y2, x3, y3, x4, y4)
    return 0.5 * abs(
        (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)
    )
end

function cumtrapz(x::RealArr1D, y::RealArr1D)
    @assert length(x) == length(y)
    @assert !isempty(x)
    y_int = similar(y)
    y_int[1] = 0.0
    y_int[2:end] = cumsum(@. 0.5 * (y[1:end-1] + y[2:end]) * (x[2:end] - x[1:end-1]))
    return y_int
end

function differentiate(x::RealArr1D, y::RealArr1D)
    @assert length(x) == length(y)
    @assert length(x) >= 2
    dydx = similar(y)
    dydx[1] = (y[2] - y[1]) / (x[2] - x[1])
    dydx[2:end-1] = (y[3:end] .- y[1:end-2]) ./ (x[3:end] .- x[1:end-2])
    dydx[end] = (y[end] - y[end-1]) / (x[end] - x[end-1])
    return dydx
end

function differentiate_first_dim(x::RealArr2D, y::RealArr1D)
    @assert size(x, 1) == length(y)
    @assert length(y) >= 2
    dydx = similar(x)
    dydx[1, :] = (y[2] - y[1]) / (x[2, :] - x[1, :])
    dydx[2:end-1, :] = (y[3:end] .- y[1:end-2]) ./ (x[3:end, :] .- x[1:end-2, :])
    dydx[end, :] = (y[end] - y[end-1]) / (x[end, :] - x[end-1, :])
    return dydx
end

function interp(x_points::RealArr1D, f_points::RealArr1D, x::Real)
    @assert length(x_points) == length(f_points)
    @assert length(x_points) >= 2

    i_lower = max(1, searchsortedlast(x_points[1:end-1], x))

    x_start = x_points[i_lower]
    x_end = x_points[i_lower+1]

    f_x =
        (f_points[i_lower] * (x_end - x) + f_points[i_lower+1] * (x - x_start)) /
        (x_end - x_start)

    return f_x
end

function interp(
    x_points::RealArr1D,
    f_points::RealArr1D,
    x::Real,
    f_lower::Real,
    f_upper::Real,
)
    @assert length(x_points) == length(f_points)
    @assert length(x_points) >= 2

    return if x < x_points[1]
        f_lower
    elseif x > x_points[end]
        f_upper
    else
        interp(x_points, f_points, x)
    end

end

function interp(
    x_points::RealArr1D,
    y_points::RealArr1D,
    f_points::RealArr2D,
    x::Real,
    y::Real,
)
    @assert length(x_points) == size(f_points, 1)
    @assert length(y_points) == size(f_points, 2)
    @assert length(x_points) >= 2
    @assert length(y_points) >= 2

    i_lower = max(1, searchsortedlast(x_points[1:end-1], x))
    j_lower = max(1, searchsortedlast(y_points[1:end-1], y))

    x_start = x_points[i_lower]
    x_end = x_points[i_lower+1]
    y_start = y_points[j_lower]
    y_end = y_points[j_lower+1]

    f_x_y_start =
        (
            f_points[i_lower, j_lower] * (x_end - x) +
            f_points[i_lower+1, j_lower] * (x - x_start)
        ) / (x_end - x_start)
    f_x_y_end =
        (
            f_points[i_lower, j_lower+1] * (x_end - x) +
            f_points[i_lower+1, j_lower+1] * (x - x_start)
        ) / (x_end - x_start)

    f_x_y = (f_x_y_start * (y_end - y) + f_x_y_end * (y - y_start)) / (y_end - y_start)

    return f_x_y
end

end
