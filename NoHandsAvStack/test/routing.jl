using Test
using StaticArrays
using LinearAlgebra
using Graphs

# Include necessary structs and functions
include("example_project.jl")  # Adjust path if needed

# Create mock map segments for testing
function create_mock_map_segments()
    # Create lane boundary
    struct LaneBoundary
        pt_a::SVector{2, Float64}
        pt_b::SVector{2, Float64}
        curvature::Float64
    end
    
    # Create map segment
    struct MapSegment
        lane_boundaries::Vector{LaneBoundary}
        lane_types::Vector{Symbol}
        children::Vector{Int}
    end
    
    # Create a simple map with three segments
    map_segments = Dict{Int, MapSegment}()
    
    # Segment 1: Starting segment
    map_segments[1] = MapSegment(
        [
            LaneBoundary(SVector(-10.0, -10.0), SVector(10.0, -10.0), 0.0),
            LaneBoundary(SVector(-10.0, 0.0), SVector(10.0, 0.0), 0.0)
        ],
        [:normal],
        [2]
    )
    
    # Segment 2: Middle segment
    map_segments[2] = MapSegment(
        [
            LaneBoundary(SVector(10.0, -10.0), SVector(30.0, -10.0), 0.0),
            LaneBoundary(SVector(10.0, 0.0), SVector(30.0, 0.0), 0.0)
        ],
        [:normal],
        [3]
    )
    
    # Segment 3: End segment (loading zone)
    map_segments[3] = MapSegment(
        [
            LaneBoundary(SVector(30.0, -10.0), SVector(50.0, -10.0), 0.0),
            LaneBoundary(SVector(30.0, 0.0), SVector(50.0, 0.0), 0.0),
            LaneBoundary(SVector(30.0, 10.0), SVector(50.0, 10.0), 0.0)
        ],
        [:normal, :loading_zone],
        []
    )
    
    return map_segments
end

@testset "Routing Tests" begin
    map_segments = create_mock_map_segments()
    
    # Test utility functions
    @testset "Utility functions" begin
        # Test perp function
        v = [1.0, 0.0]
        v_perp = perp(v)
        @test v_perp ≈ [0.0, 1.0]
        
        # Test signOfDot0 function
        @test signOfDot0([1.0, 0.0], [2.0, 0.0]) ≈ 1.0  # Same direction
        @test signOfDot0([1.0, 0.0], [-2.0, 0.0]) ≈ -1.0  # Opposite direction
        @test signOfDot0([1.0, 0.0], [0.0, 1.0]) ≈ 0.0  # Perpendicular
    end
    
    # Test PolylineSegment functions
    @testset "PolylineSegment" begin
        # Test StandardSegment constructor
        p1 = SVector{2, Float64}(0.0, 0.0)
        p2 = SVector{2, Float64}(10.0, 0.0)
        road = 1
        part = 1
        seg = StandardSegment(p1, p2, road, part)
        
        @test seg.p1 ≈ p1
        @test seg.p2 ≈ p2
        @test seg.tangent ≈ SVector{2, Float64}(1.0, 0.0)
        @test seg.normal ≈ SVector{2, Float64}(0.0, 1.0)
        @test seg.road == 1
        @test seg.part == 1
        
        # Test Polyline constructor
        points = [SVector{2, Float64}(0.0, 0.0), SVector{2, Float64}(10.0, 0.0), SVector{2, Float64}(20.0, 10.0)]
        roads = [1, 2]
        parts = [1, 1]
        poly = Polyline(points, roads, parts)
        
        @test length(poly.segments) == 2
        @test poly.segments[1].p1 ≈ points[1]
        @test poly.segments[1].p2 ≈ points[2]
        @test poly.segments[2].p1 ≈ points[2]
        @test poly.segments[2].p2 ≈ points[3]
    end
    
    # Test signed_distance functions
    @testset "Signed Distance" begin
        # Create a simple polyline
        points = [SVector{2, Float64}(0.0, 0.0), SVector{2, Float64}(10.0, 0.0)]
        roads = [1]
        parts = [1]
        poly = Polyline(points, roads, parts)
        
        # Test points
        @test signed_distance_standard(poly.segments[1], SVector{2, Float64}(5.0, 1.0)) ≈ 1.0  # 1 unit above
        @test signed_distance_standard(poly.segments[1], SVector{2, Float64}(5.0, -1.0)) ≈ -1.0  # 1 unit below
        
        # Test polyline signed distance
        @test signed_distance(poly, SVector{2, Float64}(5.0, 1.0), 1, 1) ≈ 1.0
    end
    
    # Test get_route function
    @testset "Route Calculation" begin
        # Mock the necessary Graphs functions
        function dijkstra_shortest_paths(graph, start, distmx)
            # Simple mock that returns direct path 1 -> 2 -> 3
            struct PathResult
                parents::Vector{Int}
            end
            
            return PathResult([0, 1, 2])
        end
        
        # Test get_route
        start_pos = SVector{2, Float64}(0.0, -5.0)  # In segment 1
        target_segment = 3
        
        route = get_route(map_segments, start_pos, target_segment)
        
        @test route == [1, 2, 3]  # Expected path
    end
    
    # Test get_polyline function
    @testset "Polyline Generation" begin
        # Test get_polyline with mock route
        function get_route(map_segments, start_position, target_segment)
            return [1, 2, 3]  # Mock route
        end
        
        start_pos = SVector{2, Float64}(0.0, -5.0)
        target_segment = 3
        
        poly = get_polyline(map_segments, start_pos, target_segment)
        
        @test length(poly.segments) >= 3  # At least one segment per route point
        @test poly.segments[1].p1 ≈ start_pos  # First point should be start position
    end
end