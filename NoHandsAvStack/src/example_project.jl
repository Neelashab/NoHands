import Statistics # used to average measurements

# seaprate this struct from gector used by EKF
struct MyLocalizationType
    x::Float64                   # position x
    y::Float64                   # position y
    z::Float64                   # elevation (? dont think we need this)
    yaw::Float64                 # heading
end

# update to include detailed state
# i.e. vector of vehicle state (bounding boxes, velocity, heading, acceleration, etc.)
struct MyPerceptionType
    field1::Int
    field2::Float64
end

function localize(gps_channel, imu_channel, localization_state_channel)

    # initialize state to just the few GPS measurements (maybe some IMU, but probably just initialize to 0) (warm up loop)
    # heading should not be initialized to 0

    # WARM UP LOOP 
    starting_gps = []
    start_time = time() 

    while time() - start_time < 1.0
        if isready(gps_channel)
            push!(gps_buffer, take!(gps_channel))
        end
    end

     # Estimate initial x, y, heading from GPS
     init_x = mean([m.lat for m in gps_buffer])
     init_y = mean([m.long for m in gps_buffer])
     init_yaw = mean([m.heading for m in gps_buffer])

     # Initialize state vector
    state = @SVector [
        init_x,   # x
        init_y,   # y
        0.0,      # z (assumed flat ground)
        init_yaw, # yaw
        0.0, 0.0, 0.0,  # vx, vy, vz
        0.0, 0.0, 0.0,  # wx, wy, wz
        0.0,           # acc
        time(),        # timestamp
        0.0            # dummy (13th element?)
    ]

    Σ = 0.1 * I(13) # initial covariance matrix
    dt = 0.1 # time step (in seconds)


    # ENTER MAIN LOOP

    # take most recent GPS and IMU measurements (alternate between both?) and trash the rest (for now)
    # keep track of dt = current measurement time - last measurement time for prediction step 
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

         # Use the most recent measurement from each
         latest_gps = isempty(fresh_gps_meas) ? nothing : last(fresh_gps_meas)
         latest_imu = isempty(fresh_imu_meas) ? nothing : last(fresh_imu_meas)
 
         # Compute dt
         now = time()
         dt = now - state[12]  # use last timestamp
         state = setindex(state, now, 12)

         # TODO - Where to grab process and measuremnet noise? I think its in the simulation code?
         Q = nothing 
         R = nothing

        # PREDICT: Where should we be given the previous state and how we expect the car to move?
        # TODO define motion model that accepts localization state and computes:
            # predicted state = f(previous state, current controls, dt)
            # predicted covariance = F (jacobian of current state) * previous covariance * F' + Q
            # * you can reuse the h in the measurement model in measurements.jl, check to see what the state of x is

            # Inject latest IMU measurements into state
            # TODO - check what F function accepts as argument
        if latest_imu !== nothing
            state = setindex(state, latest_imu.linear_vel[1], 5)
            state = setindex(state, latest_imu.linear_vel[2], 6)
            state = setindex(state, latest_imu.linear_vel[3], 7)
            state = setindex(state, latest_imu.angular_vel[1], 8)
            state = setindex(state, latest_imu.angular_vel[2], 9)
            state = setindex(state, latest_imu.angular_vel[3], 10)
        end

            F = Jac_x_f(state, dt)
            predicted_state = f(state, dt)
            Σ = F * Σ * F' + Q


        # CORRECT: Given new sensor measurements, how do we correct our prediction? 
        # We balance both our prediction given motion model and the actual sensor measurement using Kalman Gain
        # TODO define measurement model that accepts new GPS measurements and computes:
            # measurement prediction = h(predicted state)
            # y = difference between real measurement and predicted measurement (residual)
            # Kalman gain = predicted covariance * H' * (H * predicted covariance * H' + R)^-1
            # updated state = predicted state + Kalman gain * y
            # updated covariance = (I - Kalman gain * H) * predicted covariance
            #  you can reuse the g in the measurement model in measurements.jl, check to see what the state of x is

            if latest_gps !== nothing
                z = @SVector [latest_gps.lat, latest_gps.long, latest_gps.heading]
                H = Jac_h_gps(predicted_state)
                z_pred = h_gps(predicted_state)
                y = z - z_pred  # residual
                S = H * Σ * H' + R
                K = Σ * H' * inv(S)  # Kalman gain
                state = predicted_state + K * y
                Σ = (I - K * H) * Σ
            else
                state = predicted_state
            end
    
            localization_state = MyLocalizationType(state[1], state[2], state[3], state[4])

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end






function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end
        # step 1 -> get piping done and populate perception channel
        # for now you can do this with ground truth data

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good


        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end

    # object tracking that runs EKF for other cars using camera measurements
    # run EKF for each bounding box detected
    # note -> all cars are the same size so you can cheat a bit (find in rendering code)
    # just need to estimate position, velocity, heading and velocity (unknown state)
    # known state 0> z component, length, width, height

    # given bounding box, how do we estimate state? 
    
end

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

function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)
    # create an optional shut down channel to kill all threads once one fails

    #localization_state_channel = Channel{MyLocalizationType}(1)
    #perception_state_channel = Channel{MyPerceptionType}(1)

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
