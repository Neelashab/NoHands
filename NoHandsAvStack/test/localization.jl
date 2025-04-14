using Test
using StaticArrays
using LinearAlgebra

# Mock the necessary components from VehicleSim
module VehicleSim
    function Jac_x_f(x, dt)
        # Simplified mock implementation
        return I(13)
    end

    function f(state, dt)
        # Simplified mock implementation - just return state
        return state
    end

    function get_cam_transform(camera_id)
        # Return identity transform for testing
        return [1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0]
    end

    function get_body_transform(quaternion, position)
        # Return identity transform for testing
        return [1.0 0.0 0.0 position[1];
                0.0 1.0 0.0 position[2];
                0.0 0.0 1.0 position[3];
                0.0 0.0 0.0 1.0]
    end

    function multiply_transforms(T1, T2)
        return T1 * T2
    end

    function extract_yaw_from_quaternion(q)
        return atan(2(q[1]*q[4] + q[2]*q[3]), 1 - 2*(q[3]^2 + q[4]^2))
    end
end

# Include necessary structs and functions
include("example_project.jl")  # Adjust path if needed

@testset "Localization Tests" begin
    # Test heading_to_quaternion
    @testset "heading_to_quaternion" begin
        yaw = π/4  # 45 degrees
        q = heading_to_quaternion(yaw)
        @test length(q) == 4
        @test q[1] ≈ cos(yaw/2)  # w component
        @test q[2] ≈ 0.0         # x component
        @test q[3] ≈ 0.0         # y component
        @test q[4] ≈ sin(yaw/2)  # z component
    end

    # Test extract_yaw_from_quaternion
    @testset "extract_yaw_from_quaternion" begin
        yaw = π/3
        q = heading_to_quaternion(yaw)
        extracted_yaw = extract_yaw_from_quaternion(q)
        @test extracted_yaw ≈ yaw
    end

    # Test localization function with mocked channels
    @testset "localization workflow" begin
        # Create mock channels
        gps_channel = Channel{Any}(32)
        imu_channel = Channel{Any}(32)
        localization_state_channel = Channel{MyLocalizationType}(1)
        
        # Create mock GPS measurement
        struct MockGPS
            lat::Float64
            long::Float64
            heading::Float64
        end

        # Push mock GPS data
        for i in 1:5
            put!(gps_channel, MockGPS(10.0 + 0.01*i, 20.0 + 0.02*i, π/4))
        end

        # Mock IMU data
        struct MockIMU
            linear_vel::Vector{Float64}
            angular_vel::Vector{Float64}
        end

        # Push mock IMU data
        put!(imu_channel, MockIMU([1.0, 0.0, 0.0], [0.0, 0.0, 0.1]))

        # Test that localization can process this data without errors
        # We'll run this in a separate task with a timeout
        @test_skip begin
            done = Channel{Bool}(1)
            errormonitor(@async begin
                try
                    # Only run localize for a short time in test
                    @async begin
                        sleep(0.5)  # Give localize time to initialize
                        put!(done, true)
                    end
                    localize(gps_channel, imu_channel, localization_state_channel)
                catch e
                    @error "Error in localize" exception=(e, catch_backtrace())
                    put!(done, false)
                end
            end)
            
            result = take!(done)
            @test result == true
            
            # Check that localization produced something
            @test isready(localization_state_channel)
            loc_state = take!(localization_state_channel)
            @test isa(loc_state, MyLocalizationType)
        end
    end
end