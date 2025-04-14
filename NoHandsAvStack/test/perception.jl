using Test
using StaticArrays
using LinearAlgebra

# Include necessary structs and functions
include("example_project.jl")  # Adjust path if needed

# Mock the necessary components from VehicleSim
module VehicleSim
    function f(x, dt)
        # For EKF prediction step
        return x
    end

    function Jac_x_f(x, dt)
        # Jacobian for EKF
        return I(13)
    end

    function get_cam_transform(camera_id)
        # Return a transform based on camera ID
        if camera_id == 1
            return [1.0 0.0 0.0 2.0;
                    0.0 1.0 0.0 0.0;
                    0.0 0.0 1.0 0.0;
                    0.0 0.0 0.0 1.0]
        else
            return [1.0 0.0 0.0 -2.0;
                    0.0 1.0 0.0 0.0;
                    0.0 0.0 1.0 0.0;
                    0.0 0.0 0.0 1.0]
        end
    end

    function get_body_transform(quaternion, position)
        # Simplified transform
        yaw = extract_yaw_from_quaternion(quaternion)
        return [cos(yaw) -sin(yaw) 0.0 position[1];
                sin(yaw)  cos(yaw) 0.0 position[2];
                0.0       0.0      1.0 position[3];
                0.0       0.0      0.0 1.0]
    end

    function multiply_transforms(T1, T2)
        return T1 * T2
    end

    function extract_yaw_from_quaternion(q)
        return atan(2(q[1]*q[4] + q[2]*q[3]), 1 - 2*(q[3]^2 + q[4]^2))
    end
end

@testset "Perception Tests" begin
    # Test EKF function
    @testset "EKF updating tracked object" begin
        # Create a test tracked object
        id = 1
        time = 0.0
        pos = SVector{3, Float64}(10.0, 20.0, 0.0)
        orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)  # No rotation
        vel = SVector{3, Float64}(1.0, 0.0, 0.0)  # Moving in x direction
        angular_vel = SVector{3, Float64}(0.0, 0.0, 0.0)  # No rotation
        P = SMatrix{13, 13, Float64}(I)  # Identity covariance
        
        track = TrackedObject(id, time, pos, orientation, vel, angular_vel, P)
        
        # Create a measurement (slightly different from predicted position)
        measurement = SVector{3, Float64}(10.5, 20.1, 0.0)
        
        # Time step
        dt = 0.1
        
        # Run EKF update
        updated_track = ekf(track, measurement, dt)
        
        # Test that the updated track has the expected properties
        @test updated_track.id == id
        @test updated_track.time ≈ time + dt
        
        # The updated position should be somewhere between the predicted position and the measurement
        # Predicted position would be pos + vel*dt = [10.1, 20.0, 0.0]
        # Measurement is [10.5, 20.1, 0.0]
        # So updated position should be between these
        @test updated_track.pos[1] > 10.1
        @test updated_track.pos[1] < 10.5
        @test updated_track.pos[2] > 20.0
        @test updated_track.pos[2] < 20.1
        
        # Covariance should have changed
        @test updated_track.P != P
    end

    # Test track_orientation_estimate function
    @testset "Track orientation estimate" begin
        # Test case: object in same lane
        ego_pos = SVector{3, Float64}(0.0, 0.0, 0.0)
        ego_quat = SVector{4, Float64}(cos(0.0/2), 0.0, 0.0, sin(0.0/2))  # Facing positive x
        
        # Object in front, same lane
        obj_pos = SVector{3, Float64}(5.0, 0.0, 0.0)
        orientation = track_orientation_estimate(ego_pos, ego_quat, obj_pos)
        
        # Should return same orientation as ego
        @test orientation ≈ ego_quat
        
        # Test with object in opposite lane
        obj_pos = SVector{3, Float64}(0.0, 11.0, 0.0)  # Assuming lane width is 10.0
        orientation = track_orientation_estimate(ego_pos, ego_quat, obj_pos)
        
        # Should return opposite orientation (π rotation)
        expected_quat = SVector{4, Float64}(cos(π/2), 0.0, 0.0, sin(π/2))
        @test orientation ≈ expected_quat
    end

    # Test perception process with mocked channels
    @testset "perception workflow" begin
        # Create mock channels
        cam_meas_channel = Channel{Any}(32)
        gt_channel = Channel{Any}(32)
        perception_state_channel = Channel{MyPerceptionType}(1)
        
        # Create mock camera measurement
        struct MockCameraMeasurement
            time::Float64
            camera_id::Int
            focal_length::Float64
            pixel_length::Float64
            image_width::Int
            image_height::Int
            bounding_boxes::Vector{Vector{Float64}}
        end
        
        # Create mock ground truth
        struct MockGroundTruth
            time::Float64
            vehicle_id::Int
            position::SVector{3, Float64}
            orientation::SVector{4, Float64}
            velocity::SVector{3, Float64}
            angular_velocity::SVector{3, Float64}
            size::SVector{3, Float64}
        end
        
        # Push mock camera data
        cam_data = MockCameraMeasurement(
            0.0,  # time
            1,    # camera_id
            10.0, # focal_length
            0.01, # pixel_length
            640,  # image_width
            480,  # image_height
            [[320.0, 240.0, 340.0, 260.0]]  # One bounding box: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        )
        put!(cam_meas_channel, cam_data)
        
        # Push mock GT data
        gt_data = MockGroundTruth(
            0.0,  # time
            1,    # vehicle_id
            SVector{3, Float64}(0.0, 0.0, 0.0),  # position
            SVector{4, Float64}(1.0, 0.0, 0.0, 0.0),  # orientation (no rotation)
            SVector{3, Float64}(1.0, 0.0, 0.0),  # velocity
            SVector{3, Float64}(0.0, 0.0, 0.0),  # angular_velocity
            SVector{3, Float64}(13.2, 5.7, 5.3)   # size
        )
        put!(gt_channel, gt_data)
        
        # Test that perception can process this data without errors
        # We'll run this in a separate task with a timeout
        @test_skip begin
            done = Channel{Bool}(1)
            errormonitor(@async begin
                try
                    # Only run perception for a short time in test
                    @async begin
                        sleep(0.5)  # Give perception time to initialize
                        put!(done, true)
                    end
                    perception(cam_meas_channel, gt_channel, perception_state_channel)
                catch e
                    @error "Error in perception" exception=(e, catch_backtrace())
                    put!(done, false)
                end
            end)
            
            result = take!(done)
            @test result == true
            
            # Check that perception produced something
            @test isready(perception_state_channel)
            perc_state = take!(perception_state_channel)
            @test isa(perc_state, MyPerceptionType)
        end
    end
end