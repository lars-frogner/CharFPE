### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab99e056-6007-11ed-0ec5-27044884ba2c
begin
    using Revise # Don't move this line, or Revise won't work
    import Pkg
    Pkg.activate(".")

    using CharFPE.Const
    using CharFPE.Numerical
    using CharFPE.Atmosphere
    using CharFPE.Beams
    using CharFPE.Solvers
    using CharFPE.Transport
end

# ╔═╡ ebc04802-c456-4085-8a38-3b8b22af5a3c
begin
    using CairoMakie
    set_theme!(theme_dark())
end

# ╔═╡ cb35b024-30b2-4795-9469-d3bedf8bd0ed
using SpecialFunctions

# ╔═╡ 47dbf79d-a78b-4386-ba9c-730b13e46c0d
using Printf

# ╔═╡ c0c5ecd9-f4b9-4be9-a039-42c1b0ff0399
using NPZ

# ╔═╡ 4bba2be0-9e74-4fb4-b6dd-86206a2a3e61
import PlutoUI

# ╔═╡ 487aa234-4f78-485c-ac57-f02938341ad7
function plotdistr(E, μ, n; axis = (;), colorrange = nothing)
    limits = (nothing, nothing, -0.1, 1.0)
    axis = (; xscale = log10, xlabel = "E [keV]", ylabel = "μ", limits = limits, axis...)
    fig, ax, sc = scatter(
        E ./ KEV_TO_ERG,
        μ,
        color = n,
        markersize = 6,
        colorrange = colorrange,
        axis = axis,
    )
    Colorbar(fig[1, 2], sc; label = "log₁₀(n)")
    return fig, ax
end

# ╔═╡ 6bcf7702-0467-4de0-8923-d73731da09ec
beam_data = npzread("data/beam.npz")

# ╔═╡ 4baa16cb-b086-4deb-8258-fff558d3ff49
function compute_dlnBds(s, B)
	dlnBds = zeros(length(s))
	mean_B = @. (B[1:end-1] + B[2:end]) / 2
	dlnBds[2:end] = @. (B[2:end] - B[1:end-1]) / (mean_B * (s[2:end] - s[1:end-1]))
	dlnBds[2:end][mean_B .< 0.1] .= 0.0
	return dlnBds
end

# ╔═╡ 63954b95-eed0-4508-b7e1-78c74a8a6a6c
begin
    s = beam_data["s"] .* 1e8
    nH = beam_data["total_hydrogen_density"]
    x = beam_data["ionization_fraction"]
    dlnBds = compute_dlnBds(s, beam_data["magnetic_flux_density"])
    ℰ = zeros(length(nH))
end

# ╔═╡ 592b6144-a4fe-4c82-b57d-f2e8d662ec6c
begin
    Ec = beam_data["lower_cutoff_energy"] * KEV_TO_ERG
    δ = 4.0
    ℱ_tot = 1.0
    beam = Beam(Ec, δ, ℱ_tot)
end

# ╔═╡ 6b248a0c-bb97-461a-8495-2ffe1362b9ae
begin
	E_min = 0.05 * Ec
	E_max = 120.0 * Ec
    log₁₀E = range(log10(E_min), log10(E_max), 200)
    E = 10 .^ log₁₀E
end

# ╔═╡ e43d4d68-21dd-47b6-8dd6-c8de4e406fa2
min_n_steps_to_thermalize = 2

# ╔═╡ bc882e48-cdf5-4cfd-b066-f92d276a19e8
max_n_steps_to_thermalize = 10

# ╔═╡ 9f86e437-7e20-484e-8079-7cbb6fcfa3db
use_analytical = false

# ╔═╡ 67affd08-62ae-471f-9b4d-9451dbb3df82
μ₀ = beam_data["initial_pitch_angle_cosine"]

# ╔═╡ 2163f87c-9d96-452e-badc-be339ecf281a
Q_an = compute_analytical_heating(beam, μ₀, s, nH, x)

# ╔═╡ 62b5e343-968c-4470-83d4-58a460e28e9f
E_l, μ_l, n_l, E₀_l, Q_l, E_sub, μ_sub, n_sub = propagate_lagrange_discrete(beam, s, log₁₀E, μ₀, nH, x, dlnBds, ℰ, use_analytical, min_n_steps_to_thermalize=min_n_steps_to_thermalize, max_n_steps_to_thermalize=max_n_steps_to_thermalize)

# ╔═╡ 2ea58545-f79b-4da6-8152-7ad2527511a4
@bind k PlutoUI.Slider(1:length(s))

# ╔═╡ 4e4e1a52-5d52-45c9-82df-5fb675774e7b
let
	n2 = copy(n_sub)
	n2[n2.<0.0] .= NaN
	fig, ax = plotdistr(E_sub[:, k], μ_sub[:, k], log10.(n2[:, k]), colorrange=(-1.0, 8.0))
	fig
end

# ╔═╡ 974bb463-08fd-4d4a-afe4-766e9b42a99e
let
	n2 = copy(n_l)
	n2[n2.<0.0] .= NaN
	fig, ax = plotdistr(E_l[:, k], μ_l[:, k], log10.(n2[:, k]), colorrange=(-1.0, 8.0))
	fig
end

# ╔═╡ fa0b1725-1d0e-41f1-ac60-2f56007362e2
let
	n2 = copy(n_l)
	n2[n2.<0.0] .= NaN
	fig, ax = plotdistr(E₀_l[:, k], μ_l[:, k], log10.(n2[:, k]), colorrange=(-1.0, 8.0))
	fig
end

# ╔═╡ 89e28caa-bad8-43e0-90d5-a241377cb661
k

# ╔═╡ 488de288-f4a9-484a-b6ab-667d393d04f4
let
    fig, ax, _ = lines(s .* 1e-8, Q_an, axis = (; yscale = log10, limits=(0.0, nothing, 1e-15, nothing)))
	scatter!(ax, s[Q_l .> 1e-20] .* 1e-8, Q_l[Q_l.>1e-20], markersize=2, color=:orange)
    vlines!(ax, s[k] .* 1e-8, color = :gray)
    fig
end

# ╔═╡ Cell order:
# ╠═ab99e056-6007-11ed-0ec5-27044884ba2c
# ╠═ebc04802-c456-4085-8a38-3b8b22af5a3c
# ╠═cb35b024-30b2-4795-9469-d3bedf8bd0ed
# ╠═47dbf79d-a78b-4386-ba9c-730b13e46c0d
# ╠═c0c5ecd9-f4b9-4be9-a039-42c1b0ff0399
# ╠═4bba2be0-9e74-4fb4-b6dd-86206a2a3e61
# ╠═487aa234-4f78-485c-ac57-f02938341ad7
# ╠═6bcf7702-0467-4de0-8923-d73731da09ec
# ╠═4baa16cb-b086-4deb-8258-fff558d3ff49
# ╠═63954b95-eed0-4508-b7e1-78c74a8a6a6c
# ╠═592b6144-a4fe-4c82-b57d-f2e8d662ec6c
# ╠═6b248a0c-bb97-461a-8495-2ffe1362b9ae
# ╠═e43d4d68-21dd-47b6-8dd6-c8de4e406fa2
# ╠═bc882e48-cdf5-4cfd-b066-f92d276a19e8
# ╠═9f86e437-7e20-484e-8079-7cbb6fcfa3db
# ╠═67affd08-62ae-471f-9b4d-9451dbb3df82
# ╠═2163f87c-9d96-452e-badc-be339ecf281a
# ╠═62b5e343-968c-4470-83d4-58a460e28e9f
# ╠═4e4e1a52-5d52-45c9-82df-5fb675774e7b
# ╠═974bb463-08fd-4d4a-afe4-766e9b42a99e
# ╠═fa0b1725-1d0e-41f1-ac60-2f56007362e2
# ╠═2ea58545-f79b-4da6-8152-7ad2527511a4
# ╠═89e28caa-bad8-43e0-90d5-a241377cb661
# ╠═488de288-f4a9-484a-b6ab-667d393d04f4
