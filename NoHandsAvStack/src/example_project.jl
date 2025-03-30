
struct MyLocalizationType
    field1::Int
    field2::Float64
end

struct MyPerceptionType
    field1::Int
    field2::Float64
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function iterative_closest_point(map_points, pointcloud, R, t; max_iters=10, visualize=false)
end


function update_point_associations!(point_associations, pointcloud, map_points, R, t)
end

function update_point_transform!(point_associations, pointcloud, map_points, R, t)
end



function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good

        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

# an initial ray extends infinitely after the first line
struct TerminalRay <: PolylineSegment
    point::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
    function TerminalRay(point, prev_point) # same as initial ray, but looks at the previous point
        tangent = point - prev_point
        tangent ./= norm(tangent)
        normal = perp(tangent)
        new(point, tangent, normal)
    end
end

# standard segment has a starting and an ending point that are both finite
struct StandardSegment <: PolylineSegment
    p1::SVector{2, Float64}
    p2::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
    function StandardSegment(p1, p2)
        tangent = p2 - p1
        tangent ./= norm(tangent)
        normal = perp(tangent)
        new(p1, p2, tangent, normal) 
    end
end

struct Polyline
    
end


"""
Decision making is the function where the car will decide whether to start, move forward, stop, or change direction (left, right). 

We will recieve a trajectory from the target_road_segment_id. We should define a polyline and connect the line. The polyline may have some straight lines and some curves. Define the bandwith


This will power the pure pursuit controller. We will attempt to match the current path of the vehicle to the given path with minimal error using the control law.

We will define a timestep/look ahead distance that will move the car forward X steps at a time. You will fetch the information at each timestep to make a decision about how to move forward.

The car will recieve information from latest_localization_state and latest_perception_state. latest_perception_state will help us understand where the car is in given input of the map. 

Avoid obstacles using heuristics. See stop sign = stop. See 

What exactly is given by localization_state_channel?

What exactly is given by perception_state_channel?


"""

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        cmd = (steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    #localization_state_channel = Channel{MyLocalizationType}(1)
    #perception_state_channel = Channel{MyPerceptionType}(1)
    target_segment_channel = Channel{Int}(1) # end point
    shutdown_channel = Channel{Bool}(1) # end point


    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, socket)
end

# Routing skeleton 
"""
    plan_route(start_position, goal_position, map_data; constraints=nothing)

Plans an optimal route from start to goal position considering map data and constraints.

# Arguments
- `start_position::Vector{Float64}`: Starting position coordinates [x, y]
- `goal_position::Vector{Float64}`: Goal position coordinates [x, y]
- `map_data::Dict`: Contains map information including:
  - `obstacles`: Vector of polygons representing no-go areas
  - `road_network`: Graph representation of available roads
  - `traffic_info`: Current traffic conditions 
- `constraints::Dict`: Additional routing constraints such as:
  - `max_distance`: Maximum allowable route distance
  - `preferred_road_types`: Vector of preferred road types (highway, local, etc.), maybe?
  - `avoid_areas`: Areas to avoid if possible
  - `time_constraints`: Time-based routing preferences

# Returns
- `route::Vector{Vector{Float64}}`: Sequence of [x, y] coordinates representing the planned route
- `metadata::Dict`: Additional information about the route:
  - `total_distance`: Estimated distance of the route
  - `estimated_time`: Estimated travel time
  - `complexity`: Routing complexity metrics
"""
function plan_route(start_position::Vector{Float64}, 
                    goal_position::Vector{Float64}, 
                    map_data::Dict; 
                    constraints::Union{Dict, Nothing}=nothing)
    # Initialize routing components
    route = Vector{Vector{Float64}}()
    metadata = Dict(
        "total_distance" => 0.0,
        "estimated_time" => 0.0,
        "complexity" => 0.0
    )
    
    # 1. Preprocess map data
    processed_map = preprocess_map(map_data)
    
    # 2. Apply constraints to create a cost function
    cost_function = create_cost_function(constraints)
    
    # 3. Select appropriate routing algorithm based on constraints and map size
    routing_algorithm = select_routing_algorithm(processed_map, constraints)
    
    # 4. Execute routing algorithm
    waypoints, raw_metadata = routing_algorithm(
        start_position, 
        goal_position, 
        processed_map, 
        cost_function
    )
    
    # 5. Post-process route (smoothing, simplification)
    route = post_process_route(waypoints)
    
    # 6. Calculate route metadata
    metadata = calculate_route_metadata(route, raw_metadata, map_data)
    
    # 7. Validate route against constraints
    if !validate_route(route, constraints)
        # Fallback planning if constraints can't be met
        route, metadata = plan_fallback_route(start_position, goal_position, map_data)
    end
    
    return route, metadata
end

"""
    preprocess_map(map_data::Dict)

Preprocess map data for efficient routing.
"""
function preprocess_map(map_data::Dict)
    # Implementation would convert map data into appropriate data structures
    # such as graphs, quadtrees, or grid representations
    # ...
    return map_data # Placeholder
end

"""
    create_cost_function(constraints::Union{Dict, Nothing})

Create a cost function based on routing constraints.
"""
function create_cost_function(constraints::Union{Dict, Nothing})
    # Implementation would create a function that scores route segments
    # based on distance, traffic, road type, etc.
    # ...
    return (a, b) -> norm(a - b) # Euclidean distance as placeholder
end

"""
    select_routing_algorithm(processed_map, constraints::Union{Dict, Nothing})

Select appropriate routing algorithm based on problem characteristics.
"""
function select_routing_algorithm(processed_map, constraints::Union{Dict, Nothing})
    # Could return implementations of:
    # - A* for efficient path finding
    # - Dijkstra's algorithm for optimal paths
    # - RRT for complex environments
    # - Hierarchical routing for large maps
    # ...
    
    # Placeholder: 
    return (start, goal, map_data, cost_func) -> simple_astar(start, goal, map_data, cost_func)
end

"""
    simple_astar(start, goal, map_data, cost_func)

Simple A* implementation as a placeholder.
"""
function simple_astar(start, goal, map_data, cost_func)
    # A basic implementation would go here
    # ...
    
    # Placeholder: just return a direct line
    return [start, goal], Dict("raw_cost" => cost_func(start, goal))
end

"""
    post_process_route(waypoints::Vector{Vector{Float64}})

Smooth and simplify the generated route.
"""
function post_process_route(waypoints::Vector{Vector{Float64}})
    # Implementation would:
    # - Remove unnecessary waypoints
    # - Smooth sharp turns
    # - Ensure feasibility for vehicle dynamics
    # ...
    return waypoints # Placeholder
end

"""
    calculate_route_metadata(route::Vector{Vector{Float64}}, raw_metadata::Dict, map_data::Dict)

Calculate detailed metadata about the route.
"""
function calculate_route_metadata(route::Vector{Vector{Float64}}, raw_metadata::Dict, map_data::Dict)
    # Implementation would calculate:
    # - Precise distance
    # - Estimated travel time based on speed limits and traffic
    # - Complexity metrics
    # ...
    
    total_distance = 0.0
    for i in 1:(length(route)-1)
        total_distance += norm(route[i+1] - route[i])
    end
    
    return Dict(
        "total_distance" => total_distance,
        "estimated_time" => total_distance / 10.0, # Assuming average speed of 10 units/time
        "complexity" => length(route)
    )
end

"""
    validate_route(route::Vector{Vector{Float64}}, constraints::Union{Dict, Nothing})

Validate that the route meets all specified constraints.
"""
function validate_route(route::Vector{Vector{Float64}}, constraints::Union{Dict, Nothing})
    # Implementation would check:
    # - Route doesn't exceed maximum distance
    # - Route avoids specified areas
    # - Route meets other constraints
    # ...
    return true # Placeholder
end

"""
    plan_fallback_route(start_position::Vector{Float64}, goal_position::Vector{Float64}, map_data::Dict)

Plan a fallback route when constraints cannot be satisfied.
"""
function plan_fallback_route(start_position::Vector{Float64}, goal_position::Vector{Float64}, map_data::Dict)
    # Implementation would use more relaxed constraints
    # to ensure a route can be found
    # ...
    return [start_position, goal_position], Dict(
        "total_distance" => norm(goal_position - start_position),
        "estimated_time" => norm(goal_position - start_position) / 5.0,
        "complexity" => 1,
        "fallback" => true
    )
end
