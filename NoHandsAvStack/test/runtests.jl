using Test
using example_project


@testset "all tests" begin

    include("perception_test01.jl")
    include("perception_test02.jl")

end
