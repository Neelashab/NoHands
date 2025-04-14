using Test
using StaticArrays
using LinearAlgebra

# Include necessary structs and functions
include("example_project.jl")  # Adjust path if needed

# Mock map segments for testing
function create_mock_map_segments()
    # Define what we need for the tests
    struct LaneBoundary
        pt_a::SVector{2, Float64}
        pt_b::SVector{2, Float64}
        curvature::Float64
    end
    
    struct MapSegment
        lane_boundaries::Vector{LaneBoundary}
        lane_types::Vector{Symbol}
        children::Vector{Int}
    end
    
    # Create a simple map with straight segments
    map_segments = Dict{Int, MapSegment}()
    
    # Segment 1: Starting segment
    map_segments[1] = MapSegment(
        [
            LaneBoundary(SVector(-10.0, -5.0), SVector(10.0, -5.0), 0.0),
            LaneBoundary(SVector(-10.0, 5.0), SVector(10.0, 5.0), 0.0)
        ],
        [:normal],
        [2]
    )
    
    # Segment 2: Middle segment
    map_segments[2] = MapSegment(
        [
            LaneBoundary(SVector(10.0, -5.0), SVector(30.0, -5.0), 0.0),
            LaneBoundary(SVector(10.0, 5.0), SVector(30.0, 5.0), 0.0)
        ],
        [:normal],
        [3]
    )
    
    # Segment 3: Curved segment
    map_segments[3] = MapSegment(
        [
            LaneBoundary(SVector(30.0, -5.0), SVector(50.0, 15.0), 0.1),
            LaneBoundary(SVector(30.0, 5.0), SVector(60.0, 25.0), 0.1)
        ],
        [:normal],
        [4]
    )
    
    # Segment 4: Loading zone
    map_segments[4] = MapSegment(
        [
            LaneBoundary(SVector(50.0, 15.0), SVector(70.0, 15.0), 0.0),
            LaneBoundary(SVector(60.0, 25.0), SVector(80.0, 25.0), 0.0),
            LaneBoundary(SVector(60.0, 35.0), SVector(80.0, 35.0), 0.0)
        ],
        [:normal, :loading_zone],
        []
    )
    
    return map_segments
end

# Mock VehicleSim's Rot_from_quat for testing
function Rot_from_quat(quaternion)
    # Extract yaw from quaternion
    yaw = extract_yaw_from_quaternion(quaternion)
    
    # Create rotation matrix for yaw
    return [cos(yaw) -sin(yaw) 0.0;
            sin(yaw)  cos(yaw) 0.0;
            0.0       0.0      1.0]
end

@testset "Decision Making Tests" begin
    map_segments = create_mock_map_segments()
    
    # Test target_velocity function
    @testset "Target Velocity Calculation" begin
        # Test normal case
        current_vel = 2.0
        distance_to_target = 100.0
        steering_angle = 0.1
        angular_velocity = 0.05
        veh_wid = 5.7
        poly_count = 10
        best_next = 5
        signed_dist = 1.0
        
        vel = target_velocity(
            current_vel, 
            distance_to_target,
            steering_angle,
            angular_velocity,
            veh_wid, 
            poly_count, 
            best_next, 
            signed_dist
        )
        
        # Should accelerate but stay below speed limit
        @test vel > current_vel
        @test vel <= 4.0
        
        # Test case with large lateral error
        vel_large_error = target_velocity(
            current_vel,
            distance_to_target,
            steering_angle,
            angular_velocity,
            veh_wid,
            poly_count,
            best_next,
            10.0  # Large signed distance
        )
        
        # Should slow down when far from center
        @test vel_large_error < current_vel
        
        # Test approaching target
        vel_near_target = target_velocity(
            current_vel,
            0.5 * veh_wid,  # Very close to target
            steering_angle,
            angular_velocity,
            veh_wid,
            poly_count,
            poly_count - 1,  # Last segment
            signed_dist
        )
        
        # Should stop when at target
        @test vel_near_target == 0.0
        
        # Test with sharp steering
        vel_sharp_turn = target_velocity(
            current_vel,
            distance_to_target,
            π/2,  # 90 degree turn
            0.2,
            veh_wid,
            poly_count,
            best_next,
            signed_dist
        )
        
        # Should slow down significantly for sharp turns
        @test vel_sharp_turn < 2.0
    end
    
    # Test helper functions
    @testset "Path Planning Helpers" begin
        # Test get_first_point
        seg = map_segments[1]
        first_point = get_first_point(seg)
        expected_first_point = (seg.lane_boundaries[1].pt_a + seg.lane_boundaries[2].pt_a) / 2
        @test first_point ≈ expected_first_point
        
        # Test get_middle_point for a straight segment
        straight_seg = map_segments[1]
        add_mid_straight, mid_straight = get_middle_point(straight_seg)
        # Should not add midpoint for short straight segment
        @test add_mid_straight == false
        
        # Test get_middle_point for a curved segment
        curved_seg = map_segments[3]
        add_mid_curved, mid_curved = get_middle_point(curved_seg)
        # Should add midpoint for curved segment
        @test add_mid_curved == true
        # Midpoint should be between start and end
        pt_a = (curved_seg.lane_boundaries[1].pt_a + curved_seg.lane_boundaries[2].pt_a) / 2
        pt_b = (curved_seg.lane_boundaries[1].pt_b + curved_seg.lane_boundaries[2].pt_b) / 2
        # Midpoint should be somewhere along the path, not just linear interpolation
        @test norm(mid_curved - pt_a) > 0
        @test norm(mid_curved - pt_b) > 0
    end
    
    # Test decision making integration (simplified)
    @testset "Decision Making Integration" begin
        # Mock necessary channels
        localization_state_channel = Channel{MyLocalizationType}(1)
        perception_state_channel = Channel{MyPerceptionType}(1)
        target_segment_channel = Channel{Int}(1)
        shutdown_channel = Channel{Bool}(1)
        
        # Mock socket
        struct MockSocket
            commands::Vector{Tuple}
        end
        
        function serialize(socket::MockSocket, cmd)
            push!(socket.commands, cmd)
        end
        
        socket = MockSocket([])
        
        # Set up initial state
        put!(target_segment_channel, 4)  # Target is segment 4
        put!(shutdown_channel, false)
        
        # Create and put localization state
        pos = SVector{3, Float64}(0.0, 0.0, 0.0)
        quat = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)  # No rotation
        vel = SVector{3, Float64}(1.0, 0.0, 0.0)  # Moving in x direction
        ang_vel = SVector{3, Float64}(0.0, 0.0, 0.0)  # No rotation
        size = SVector{3, Float64}(13.2, 5.7, 5.3)
        
        loc_state = MyLocalizationType(0.0, pos, quat, vel, ang_vel)
        put!(localization_state_channel, loc_state)
        
        # Empty perception state
        perc_state = MyPerceptionType(0.0, 1, [])
        put!(perception_state_channel, perc_state)
        
        # Test simplified decision making workflow
        @test_skip begin
            done = Channel{Bool}(1)
            errormonitor(@async begin
                try
                    # Only run decision_making for a short time in test
                    @async begin
                        sleep(0.5)  # Give it time to process
                        take!(shutdown_channel)
                        put!(shutdown_channel, true)
                        put!(done, true)
                    end
                    decision_making(
                        localization_state_channel,
                        perception_state_channel,
                        target_segment_channel,
                        shutdown_channel,
                        map_segments,
                        socket
                    )
                catch e
                    @error "Error in decision_making" exception=(e, catch_backtrace())
                    put!(done, false)
                end
            end)
            
            result = take!(done)
            @test result == true
            
            # Check that decision making sent commands to the socket
            @test length(socket.commands) > 0
            
            # Verify command structure
            cmd = socket.commands[1]
            @test length(cmd) == 3
            @test isa(cmd[1], Number)  # Steering angle
            @test isa(cmd[2], Number)  # Target velocity
            @test isa(cmd[3], Bool)    # Drive flag
        end
    end
end