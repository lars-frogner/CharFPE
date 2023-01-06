module CharFPE

RealArr1D = AbstractVector{<:Real}
RealArr2D = AbstractMatrix{<:Real}

include("Numerical.jl")
include("Const.jl")
include("Atmosphere.jl")
include("Transport.jl")
include("Beams.jl")
include("Solvers.jl")

end
