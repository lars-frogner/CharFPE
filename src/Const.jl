module Const

export c, α, e, mₑ, mₚ, kB, χ, K, KEV_TO_ERG

const c = 2.99792458e10 # Speed of light in vacuum [cm/s]
const α = 0.00729735257 # Fine structure constant
const e = 4.80325e-10 # Electron charge [esu]
const mₑ = 9.1093897e-28 # Electron mass [g]
const mₚ = 1.6726219e-24 # Proton mass
const kB = 1.380658e-16 # Boltzmann constant [erg/K]
const χ = 13.595 # Ionization energy of an hydrogen atom [eV]
const K = 2π * e^4
const KEV_TO_ERG = 1.60217733e-9

end
