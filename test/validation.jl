using Test
using Distributions
using MarkovMelding
import MarkovMelding as MM

@testset "Link Prior Validation" begin

    @testset "Valid priors" begin
        @test MM.validate_link_priors([Normal(0, 1), Normal(2, 5)]) === nothing
        @test MM.validate_link_priors([Gamma(2, 1), Exponential(1.0)]) === nothing # same (0, Inf) support
        @test MM.validate_link_priors([Normal(0, 1)]) === nothing
    end

    @testset "Invalid priors: ($d1, $d2)" for (d1, d2) in [
        (Normal(0, 1), Poisson(3)),  # Continuous vs. Discrete mismatch
        (Normal(0, 1), Gamma(2, 1)),  # Constrained vs. Unconstrained support mismatch
        (Beta(1, 1), Gamma(2, 1)),  # Constrained vs. Unconstrained support mismatch
        (Beta(2, 2), Gamma(2, 1)),  # Bounds mismatch
    ] 
        @test_throws ArgumentError MM.validate_link_priors([d1, d2])
    end

    @testset "Type stability" begin
        link = @inferred MM.SharedLink([:phi_1, :phi_2], [Gamma(2, 1), Gamma(1, 2)], exp)
        @test link.inv_b === exp
    end
end

@testset "Dictatorial Bounds" begin
    @testset "Lower Bounds - Dictator: $d" for (d) in [-1, 0]
        @test_throws ArgumentError Dictatorial(d)
    end
    @test_throws ArgumentError MM.validate_pooling(Dictatorial(3), 2)
end

@testset "Extracting Priors" begin
    @testset "Extract priors" begin
        @test MM.extract_prior(MODEL1, :mu) == Normal(0, 1)
        @test MM.extract_prior.([MODEL1, MODEL2], [:mu, :mu]) == [Normal(0,1), Logistic(5, 1)]
    end

    @testset "Prior not found" begin
        @test_throws KeyError MM.extract_prior(MODEL1, :phi)
    end
end

@testset "Melding Validation" begin
    @testset "No submodels" begin
        @test_throws ArgumentError meld([], [], ProductOfExperts())
    end

    @testset "Submodel-Link length mismatch" begin
        @test_throws ArgumentError meld([MODEL1, MODEL2], [:mu], ProductOfExperts())
    end


end
