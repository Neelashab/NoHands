using Test
using StaticArrays
using Graphs
using VehicleSim
using LinearAlgebra

using CSV
using DataFrames
using Plots

# Helper function to save test results to CSV
function save_test_results(filename, data)
    df = DataFrame(data)
    CSV.write(filename, df)
    println("Saved test results to $filename")
end

@testset "Routing and Decision Making Tests" begin
    map_segments = create_test_map_segments()
    
    # Initialize data collection structures
    route_test_data = Dict(
        :test_case => String[],
        :start_pos => Tuple{Float64,Float64}[],
        :target_segment => Int[],
        :expected_route => Vector{Int}[],
        :actual_route => Vector{Int}[],
        :path_length => Float64[],
        :optimal_length => Float64[],
        :is_optimal => Bool[]
    )
    
    velocity_test_data = Dict(
        :test_case => String[],
        :distance_to_target => Float64[],
        :found_stop_sign => Bool[],
        :distance_to_stop_sign => Float64[],
        :current_velocity => Float64[],
        :result_velocity => Float64[]
    )
    
    distance_test_data = Dict(
        :test_case => String[],
        :point => Tuple{Float64,Float64}[],
        :expected_distance => Float64[],
        :calculated_distance => Float64[],
        :difference => Float64[]
    )

    # Helper function to calculate path length
    function calculate_path_length(map_segments, route)
        total_length = 0.0
        for i in 1:length(route)-1
            current_seg = map_segments[route[i]]
            next_seg = map_segments[route[i+1]]
            
            # Get center points of segments
            current_center = (current_seg.lane_boundaries[1].pt_a + current_seg.lane_boundaries[2].pt_a) / 2
            next_center = (next_seg.lane_boundaries[1].pt_a + next_seg.lane_boundaries[2].pt_a) / 2
            
            total_length += norm(next_center - current_center)
        end
        return total_length
    end

    # Helper function to find all possible paths
    function find_all_paths(map_segments, start_id, target_id)
        graph = SimpleDiGraph()
        node_ids = collect(keys(map_segments))
        node_to_index = Dict(id => i for (i, id) in enumerate(node_ids))
        
        # Add nodes
        for _ in node_ids
            add_vertex!(graph)
        end
        
        # Add edges
        for (id, seg) in map_segments
            for child in seg.children
                add_edge!(graph, node_to_index[id], node_to_index[child])
            end
        end
        
        # Find all paths
        all_paths = []
        start_idx = node_to_index[start_id]
        target_idx = node_to_index[target_id]
        
        function dfs(current, path, visited)
            if current == target_idx
                push!(all_paths, [node_ids[i] for i in path])
                return
            end
            for neighbor in neighbors(graph, current)
                if !(neighbor in visited)
                    dfs(neighbor, [path; neighbor], [visited; neighbor])
                end
            end
        end
        
        dfs(start_idx, [start_idx], [start_idx])
        return all_paths
    end

    @testset "get_route - straight path" begin
        start_pos = SVector(50.0, 1.5)
        target_segment = 3
        route = get_route(map_segments, start_pos, target_segment)
        
        # Verify the route is correct
        @test route == [1, 2, 3]
        
        # Calculate path length
        path_length = calculate_path_length(map_segments, route)
        
        # Find all possible paths and their lengths
        start_id = get_pos_seg_id(map_segments, start_pos)
        all_paths = find_all_paths(map_segments, start_id, target_segment)
        path_lengths = [calculate_path_length(map_segments, p) for p in all_paths]
        optimal_length = minimum(path_lengths)
        
        # Record data
        push!(route_test_data[:test_case], "straight_path")
        push!(route_test_data[:start_pos], (start_pos[1], start_pos[2]))
        push!(route_test_data[:target_segment], target_segment)
        push!(route_test_data[:expected_route], [1, 2, 3])
        push!(route_test_data[:actual_route], route)
        push!(route_test_data[:path_length], path_length)
        push!(route_test_data[:optimal_length], optimal_length)
        push!(route_test_data[:is_optimal], isapprox(path_length, optimal_length, atol=0.1))
        
        # Test if the path is optimal
        @test isapprox(path_length, optimal_length, atol=0.1)
    end
    
    @testset "get_route - to loading zone" begin
        start_pos = SVector(20.0, 1.5)
        target_segment = 4
        route = get_route(map_segments, start_pos, target_segment)
        
        # Verify the route is correct
        @test route == [1, 2, 3, 4]
        
        # Calculate path length
        path_length = calculate_path_length(map_segments, route)
        
        # Find all possible paths and their lengths
        start_id = get_pos_seg_id(map_segments, start_pos)
        all_paths = find_all_paths(map_segments, start_id, target_segment)
        path_lengths = [calculate_path_length(map_segments, p) for p in all_paths]
        optimal_length = minimum(path_lengths)
        
        # Record data
        push!(route_test_data[:test_case], "loading_zone")
        push!(route_test_data[:start_pos], (start_pos[1], start_pos[2]))
        push!(route_test_data[:target_segment], target_segment)
        push!(route_test_data[:expected_route], [1, 2, 3, 4])
        push!(route_test_data[:actual_route], route)
        push!(route_test_data[:path_length], path_length)
        push!(route_test_data[:optimal_length], optimal_length)
        push!(route_test_data[:is_optimal], isapprox(path_length, optimal_length, atol=0.1))
        
        # Test if the path is optimal
        @test isapprox(path_length, optimal_length, atol=0.1)
    end
    
    @testset "get_route - optimal path verification" begin
        # Test multiple routes to ensure optimality
        test_cases = [
            (SVector(10.0, 1.5), 3),  # Simple straight path
            (SVector(80.0, 1.5), 4),  # Partial path to loading zone
            (SVector(130.0, 30.0), 4) # From curve to loading zone
        ]
        
        for (i, (start_pos, target_segment)) in enumerate(test_cases)
            route = get_route(map_segments, start_pos, target_segment)
            
            # Calculate path length
            path_length = calculate_path_length(map_segments, route)
            
            # Find all possible paths and their lengths
            start_id = get_pos_seg_id(map_segments, start_pos)
            all_paths = find_all_paths(map_segments, start_id, target_segment)
            path_lengths = [calculate_path_length(map_segments, p) for p in all_paths]
            optimal_length = minimum(path_lengths)
            
            # Record data
            push!(route_test_data[:test_case], "optimal_test_$i")
            push!(route_test_data[:start_pos], (start_pos[1], start_pos[2]))
            push!(route_test_data[:target_segment], target_segment)
            push!(route_test_data[:actual_route], route)
            push!(route_test_data[:path_length], path_length)
            push!(route_test_data[:optimal_length], optimal_length)
            push!(route_test_data[:is_optimal], isapprox(path_length, optimal_length, atol=0.1))
            
            # Test if the path is optimal
            @test isapprox(path_length, optimal_length, atol=0.1)
        end
    end
    
    # Rest of the test cases remain unchanged...
    @testset "signed_distance calculation" begin
        # Create a simple straight polyline using our test version
        poly = create_test_polyline()
        
        # Test points at various distances
        test_points = [
            SVector(10.0, 2.0),   # Right side
            SVector(10.0, -1.5),   # Left side
            SVector(5.0, 0.0),     # On the line
            SVector(15.0, 1.0),    # Right side, middle
            SVector(15.0, -0.5)    # Left side, middle
        ]
        
        expected_distances = [2.0, 1.5, 0.0, 1.0, 0.5]
        
        for (i, (point, expected)) in enumerate(zip(test_points, expected_distances))
            dist = test_signed_distance(poly, point, 1, 2)
            @test dist ≈ expected
            
            # Record data
            push!(distance_test_data[:test_case], "distance_test_$i")
            push!(distance_test_data[:point], (point[1], point[2]))
            push!(distance_test_data[:expected_distance], expected)
            push!(distance_test_data[:calculated_distance], dist)
            push!(distance_test_data[:difference], abs(dist - expected))
        end
    end

    @testset "Decision Making Tests" begin
        # Test target velocity calculation
        @testset "target_velocity" begin
            # Create test cases
            test_cases = [
                # (distance_to_target, found_stop_sign, distance_to_stop_sign, current_velocity)
                (50.0, false, 100.0, 5.0),  # Normal case
                (3.0, false, 100.0, 5.0),   # Near target
                (50.0, true, 5.0, 5.0),     # Near stop sign
                (50.0, false, 100.0, 10.0), # At speed limit
                (1.0, false, 100.0, 2.0)    # Very near target
            ]
            
            # Create a dummy channel for perception state
            perception_state_channel = Channel{MyPerceptionType}(1)
            
            for (i, (dist_to_target, has_stop, dist_to_stop, curr_vel)) in enumerate(test_cases)
                vel = target_velocity(
                    SVector(0.0, 0.0),  # veh_pos
                    10.0,               # avoid_collision_speed
                    curr_vel,
                    dist_to_target,
                    has_stop,
                    dist_to_stop,
                    0.1,                # steering_angle
                    0.1,                # angular_velocity_z
                    5.7,                # veh_wid
                    10,                 # poly_count
                    5,                  # best_next
                    0.5,                # signed_dist
                    perception_state_channel
                )
                
                # Record data
                push!(velocity_test_data[:test_case], "velocity_test_$i")
                push!(velocity_test_data[:distance_to_target], dist_to_target)
                push!(velocity_test_data[:found_stop_sign], has_stop)
                push!(velocity_test_data[:distance_to_stop_sign], dist_to_stop)
                push!(velocity_test_data[:current_velocity], curr_vel)
                push!(velocity_test_data[:result_velocity], vel)
                
                # Basic sanity checks
                @test vel >= 0
                @test vel <= 10.0
                if has_stop && dist_to_stop < 10
                    @test vel < curr_vel
                end
                if dist_to_target < 5
                    @test vel < curr_vel
                end
            end
        end
    end
    
    # Save all test data to CSV files
    save_test_results("route_test_results.csv", route_test_data)
    save_test_results("velocity_test_results.csv", velocity_test_data)
    save_test_results("distance_test_results.csv", distance_test_data)
end

# Mock map segments for testing that matches VehicleSim's structure
function create_test_map_segments()
    segments = Dict{Int, VehicleSim.RoadSegment}()
    
    # Straight segment 1
    segments[1] = VehicleSim.RoadSegment(
        1,
        [
            VehicleSim.LaneBoundary(SVector(0.0, 0.0), SVector(100.0, 0.0), 0.0, true, true),
            VehicleSim.LaneBoundary(SVector(0.0, 3.0), SVector(100.0, 3.0), 0.0, true, true)
        ],
        [VehicleSim.standard, VehicleSim.standard],
        10.0,  # speed_limit
        [2]    # children
    )
    
    # Curved segment 2 (right turn)
    segments[2] = VehicleSim.RoadSegment(
        2,
        [
            VehicleSim.LaneBoundary(SVector(100.0, 0.0), SVector(130.0, 30.0), 1/30.0, true, true),
            VehicleSim.LaneBoundary(SVector(100.0, 3.0), SVector(127.0, 33.0), 1/30.0, true, true)
        ],
        [VehicleSim.standard, VehicleSim.standard],
        10.0,
        [3]
    )
    
    # Straight segment 3 with stop sign
    segments[3] = VehicleSim.RoadSegment(
        3,
        [
            VehicleSim.LaneBoundary(SVector(130.0, 30.0), SVector(130.0, 60.0), 0.0, true, true),
            VehicleSim.LaneBoundary(SVector(127.0, 33.0), SVector(127.0, 63.0), 0.0, true, true)
        ],
        [VehicleSim.standard, VehicleSim.stop_sign],
        10.0,
        [4]
    )
    
    # Loading zone segment 4
    segments[4] = VehicleSim.RoadSegment(
        4,
        [
            VehicleSim.LaneBoundary(SVector(130.0, 60.0), SVector(130.0, 70.0), 0.0, true, true),
            VehicleSim.LaneBoundary(SVector(127.0, 63.0), SVector(127.0, 73.0), 0.0, true, true),
            VehicleSim.LaneBoundary(SVector(120.0, 70.0), SVector(120.0, 80.0), 0.0, true, true)
        ],
        [VehicleSim.standard, VehicleSim.stop_sign, VehicleSim.loading_zone],
        10.0,
        []  # No children - end of route
    )
    
    return segments
end

# Modified StandardSegment to use MVector for mutability
struct TestStandardSegment <: PolylineSegment
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
    tangent::SVector{2,Float64}
    normal::SVector{2,Float64}
    road::Int
    part::Int
    stop::Int
    function TestStandardSegment(p1, p2, road, part, stop)
        tangent = (p2 - p1) ./ norm(p2 - p1)
        normal = SVector(-tangent[2], tangent[1])
        new(p1, p2, tangent, normal, road, part, stop)
    end
end

# Modified Polyline to use our test segment type
struct TestPolyline
    segments::Vector{TestStandardSegment}
    function TestPolyline(points, roads, parts, stops)
        @assert length(points) ≥ 2 "Polyline needs at least 2 points"
        segments = Vector{TestStandardSegment}()
        for i in 1:length(points)-1
            seg = TestStandardSegment(points[i], points[i+1], roads[i], parts[i], stops[i])
            push!(segments, seg)
        end
        new(segments)
    end
end

# Helper function to create a test polyline
function create_test_polyline()
    points = [
        SVector(0.0, 0.0),
        SVector(10.0, 0.0),
        SVector(20.0, 0.0)
    ]
    roads = [1, 1, 1]
    parts = [1, 1, 1]
    stops = [0, 0, 0]
    return TestPolyline(points, roads, parts, stops)
end

# Modified signed_distance for our test polyline
function test_signed_distance(poly::TestPolyline, point, index_start, index_end)
    min_dist = Inf
    for i in index_start:min(index_end, length(poly.segments))
        seg = poly.segments[i]
        v = seg.p2 - seg.p1
        w = point - seg.p1
        c1 = dot(w, v)
        if c1 <= 0
            dist = norm(point - seg.p1)
        else
            c2 = dot(v, v)
            if c2 <= c1
                dist = norm(point - seg.p2)
            else
                b = c1 / c2
                pb = seg.p1 + b * v
                dist = norm(point - pb)
            end
        end
        min_dist = min(min_dist, dist)
    end
    return min_dist
end


# ... (keep all your existing code)

@testset "Routing Visualization" begin
    map_segments = create_test_map_segments()
    
    function visualize_route(map_segments, route, start_pos, target_pos; title="Route Visualization")
        # Create a new plot
        plt = plot(size=(800, 800), title=title, legend=:topleft)
        
        # Plot all road segments
        for (id, seg) in map_segments
            # Plot lane boundaries
            for lb in seg.lane_boundaries
                if lb.visualized
                    color = lb.hard_boundary ? :yellow : :orange
                    alpha = lb.hard_boundary ? 1.0 : 0.5
                    plot!([lb.pt_a[1], lb.pt_b[1]], [lb.pt_a[2], lb.pt_b[2]], 
                          color=color, linewidth=2, label="", alpha=alpha)
                end
            end
            
            # Plot lane centers
            if length(seg.lane_boundaries) > 1
                for i in 1:length(seg.lane_boundaries)-1
                    lb1 = seg.lane_boundaries[i]
                    lb2 = seg.lane_boundaries[i+1]
                    center_a = (lb1.pt_a + lb2.pt_a) / 2
                    center_b = (lb1.pt_b + lb2.pt_b) / 2
                    plot!([center_a[1], center_b[1]], [center_a[2], center_b[2]], 
                          color=:blue, linewidth=1, label="", linestyle=:dot)
                end
            end
        end
        
        # Highlight the optimal route
        for i in 1:length(route)-1
            seg = map_segments[route[i]]
            next_seg = map_segments[route[i+1]]
            
            # Get center points of current and next segments
            current_center_start = (seg.lane_boundaries[1].pt_a + seg.lane_boundaries[2].pt_a) / 2
            current_center_end = (seg.lane_boundaries[1].pt_b + seg.lane_boundaries[2].pt_b) / 2
            next_center_start = (next_seg.lane_boundaries[1].pt_a + next_seg.lane_boundaries[2].pt_a) / 2
            
            # Draw connection between segments
            plot!([current_center_end[1], next_center_start[1]], 
                  [current_center_end[2], next_center_start[2]], 
                  color=:green, linewidth=3, label="Optimal Path")
        end
        
        # Mark start and target positions
        scatter!([start_pos[1]], [start_pos[2]], color=:green, markersize=8, label="Start")
        scatter!([target_pos[1]], [target_pos[2]], color=:red, markersize=8, label="Target")
        
        # Add segment IDs
        for (id, seg) in map_segments
            center = (seg.lane_boundaries[1].pt_a + seg.lane_boundaries[1].pt_b +
                     seg.lane_boundaries[end].pt_a + seg.lane_boundaries[end].pt_b) / 4
            annotate!(center[1], center[2], text("$id", 8, :black, :center))
        end
        
        display(plt)
        return plt
    end

    @testset "Visualize Straight Path" begin
        start_pos = SVector(50.0, 1.5)
        target_segment = 3
        route = get_route(map_segments, start_pos, target_segment)
        target_pos = get_center(target_segment, map_segments, target_segment)
        
        plt = visualize_route(map_segments, route, start_pos, target_pos, 
                            title="Straight Path Visualization")
        @test route == [1, 2, 3]  # Still verify the route is correct
        
        # Save the plot
        savefig(plt, "straight_path_visualization.png")
    end
    
    @testset "Visualize Loading Zone Path" begin
        start_pos = SVector(20.0, 1.5)
        target_segment = 4
        route = get_route(map_segments, start_pos, target_segment)
        target_pos = get_center(target_segment, map_segments, target_segment)
        
        plt = visualize_route(map_segments, route, start_pos, target_pos,
                            title="Loading Zone Path Visualization")
        @test route == [1, 2, 3, 4]  # Verify the route is correct
        
        # Save the plot
        savefig(plt, "loading_zone_visualization.png")
    end
    
    @testset "Visualize All Paths Comparison" begin
        start_pos = SVector(80.0, 1.5)
        target_segment = 4
        start_id = get_pos_seg_id(map_segments, start_pos)
        
        # Get all possible paths
        all_paths = find_all_paths(map_segments, start_id, target_segment)
        path_lengths = [calculate_path_length(map_segments, p) for p in all_paths]
        optimal_idx = argmin(path_lengths)
        
        # Create comparison plot
        plt = plot(size=(1000, 800), title="All Possible Paths Comparison", legend=:topleft)
        
        # Plot all road segments (same as before)
        for (id, seg) in map_segments
            for lb in seg.lane_boundaries
                if lb.visualized
                    plot!([lb.pt_a[1], lb.pt_b[1]], [lb.pt_a[2], lb.pt_b[2]], 
                          color=:gray, linewidth=1, label="", alpha=0.3)
                end
            end
        end
        
        # Plot all paths with different colors
        colors = palette(:rainbow, length(all_paths))
        for (i, path) in enumerate(all_paths)
            path_length = calculate_path_length(map_segments, path)
            label = i == optimal_idx ? "Optimal ($(round(path_length, digits=2))" : 
                                      "Path $i ($(round(path_length, digits=2))"
            
            for j in 1:length(path)-1
                seg = map_segments[path[j]]
                next_seg = map_segments[path[j+1]]
                
                current_center_end = (seg.lane_boundaries[1].pt_b + seg.lane_boundaries[2].pt_b) / 2
                next_center_start = (next_seg.lane_boundaries[1].pt_a + next_seg.lane_boundaries[2].pt_a) / 2
                
                plot!([current_center_end[1], next_center_start[1]], 
                      [current_center_end[2], next_center_start[2]], 
                      color=colors[i], linewidth=2, label=j==1 ? label : "")
            end
        end
        
        # Mark start and target positions
        target_pos = get_center(target_segment, map_segments, target_segment)
        scatter!([start_pos[1]], [start_pos[2]], color=:green, markersize=8, label="Start")
        scatter!([target_pos[1]], [target_pos[2]], color=:red, markersize=8, label="Target")
        
        # Add segment IDs
        for (id, seg) in map_segments
            center = (seg.lane_boundaries[1].pt_a + seg.lane_boundaries[1].pt_b +
                     seg.lane_boundaries[end].pt_a + seg.lane_boundaries[end].pt_b) / 4
            annotate!(center[1], center[2], text("$id", 8, :black, :center))
        end
        
        display(plt)
        savefig(plt, "all_paths_comparison.png")
        
        # Verify the optimal path is the one found by get_route
        optimal_path = all_paths[optimal_idx]
        found_path = get_route(map_segments, start_pos, target_segment)
        @test found_path == optimal_path
    end
end