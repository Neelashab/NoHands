#include("../../VehicleSim/src/VehicleSim.jl")
using Graphs
using Rotations
using StaticArrays
using LinearAlgebra
using Sockets
abstract type PolylineSegment end

struct MyLocalizationType
    time::Float64
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end


struct TrackedObject
    id::Int
    time::Float64
    pos::SVector{3, Float64}  # x, y, z
    orientation::SVector{4, Float64}
    vel::SVector{3, Float64}  # vx, vy, vz
    angular_velocity::SVector{3, Float64}
    P::SMatrix{13,13,Float64} # covariance matrix
end

struct MyPerceptionType
    time::Float64
    next_id::Int
    tracked_objs::Vector{TrackedObject}
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



function perception(cam_meas_channel, gt_channel, perception_state_channel, shutdown_channel)

   
    next_id = 1
    tracks = TrackedObject[]
    #perception_state = MyPerceptionType(0.0, next_id, tracks)
    #put!(perception_state_channel, perception_state)

    try
        while true
            #println("Loop iteration")
    
            if isready(shutdown_channel) && take!(shutdown_channel)
                break
            end

            # gather the needed information
    
            fresh_cam_meas = []
            while isready(cam_meas_channel)
                meas = take!(cam_meas_channel)
                push!(fresh_cam_meas, meas)
            end
    
            # this is to be replaced with localization
            fresh_gt_meas = []
            while isready(gt_channel)
                meas = take!(gt_channel)
                push!(fresh_gt_meas, meas)
            end
    
    
            if isempty(fresh_cam_meas) || isempty(fresh_gt_meas)
                if !isopen(cam_meas_channel) && !isopen(gt_channel)
                    break
                end
                ## no new measurments , take a break
                sleep(0.05)
                continue
            end
    
            latest_cam = last(fresh_cam_meas)
            latest_gt = last(fresh_gt_meas)

           
            if isempty(latest_cam.bounding_boxes)
                # no bouding boxes yet, take a break
                sleep(0.05)
                continue
            end
    
            # unpack GT state (will become localization)
            ego_position = latest_gt.position
            ego_quaternion = latest_gt.orientation
    
            # get transformation matrices
            camera_id = latest_cam.camera_id
            T_body_from_cam = VehicleSim.get_cam_transform(camera_id)
            T_cam_camrot = VehicleSim.get_rotated_camera_transform()
            T_body_camrot = VehicleSim.multiply_transforms(T_body_from_cam, T_cam_camrot)
            T_world_body = VehicleSim.get_body_transform(ego_quaternion, ego_position)
            T_world_camrot = VehicleSim.multiply_transforms(T_world_body, T_body_camrot)

    
            focal_len = latest_cam.focal_length
            px_len = latest_cam.pixel_length
            img_w = latest_cam.image_width + 1
            img_h = latest_cam.image_height + 1
            t = latest_cam.time
            vehicle_size = SVector(13.2, 5.7, 5.3)
    
            for box in latest_cam.bounding_boxes
                u = (box[1] + box[3]) / 2
                v = (box[2] + box[4]) / 2
                x = (u - img_w / 2) * px_len
                y = (v - img_h / 2) * px_len
                
                z = 15
                
                
                point_from_cam = SVector{4, Float64}(x, y, z, 1.0)
                pos = T_world_camrot * point_from_cam
                pos[3] = vehicle_size[3] / 2

    
                matched = false
                for (i, track) in enumerate(tracks)
                    dist = norm(track.pos[1:2] - pos[1:2])
                    if dist < 5
                       
                        Δt = t - track.time
                        updated_track = ekf(track, SVector{3, Float64}(pos[1:3]), Δt)
                        tracks[i] = updated_track
                        matched = true
                        break

                    end
                end
    
                if !matched
    
                    initial_orientation = track_orientation_estimate(
                        ego_position, ego_quaternion, SVector{3, Float64}(pos[1:3]))
    
                    initial_velocity = SVector(0.0, 0.0, 0.0)
                    initial_angular_velocity = SVector(0.0, 0.0, 0.0)
    
                    cov_diag = [
                        0.5, 0.5, 10.0, 
                        2.0, 2.0, 2.0, 2.0,
                        10.0, 10.0, 0.5,
                        10.0, 10.0, 0.5
                    ]
                    initial_covariance = Diagonal(SVector{13, Float64}(cov_diag))
    
                    new_track = TrackedObject(
                        next_id, t, pos,
                        initial_orientation,
                        initial_velocity,
                        initial_angular_velocity,
                        initial_covariance
                    )
    
                    push!(tracks, new_track)
                    next_id += 1
                end
            end
    
            perception_state = MyPerceptionType(t, next_id, tracks)
            put!(perception_state_channel, perception_state)
    
            sleep(0.05)
        end
    catch e
        println("ERROR in perception: ", e)
        println(sprint(showerror, e))
    end
      
    
end



function track_orientation_estimate(ego_pos::SVector{3, Float64}, ego_quat::SVector{4, Float64}, obj_pos::SVector{3, Float64})

    lane_width = 10.0       # based on the city_map defintions in map.jl

    # compute heading of the ego vehicle
    # TODO: check if neelasha already has "finding heading logic" from her function
    ego_yaw = VehicleSim.extract_yaw_from_quaternion(ego_quat)
    ego_heading = SVector(cos(ego_yaw), sin(ego_yaw))  # 2D heading vector

    # compute how sideways teh object is from ego
    delta = obj_pos[1:2] - ego_pos[1:2]
    lateral_offset = abs(det(hcat(ego_heading, delta)) / norm(ego_heading))

    # decide orientation of the object
    if lateral_offset < lane_width / 2
        # same lane = same heaidng
        return ego_quat
    elseif lateral_offset < lane_width * 1.5
        # opposite lane = flipped heading
        flipped_yaw = ego_yaw + π
        return SVector(cos(flipped_yaw/2), 0.0, 0.0, sin(flipped_yaw/2))
    else
        # TODO: UPDATE THIS SECTION ( IF TIME )
        # else 
        # use function to find what segement the object is in
        # use function to know what part of the lane the object is in (helps with dircetion)
        # make heading guess based on the direction 
        return ego_quat
        ## ^^ TODO: this is a placeholder, need to implement the logic


    end

end


function ekf(track::TrackedObject, z::SVector{3, Float64}, Δt::Float64)
    """
    From measurement.jl:
        Jac_x_f(x, Δt) - returns the Jacobian matrix of the process model
            * x is a state vector with the following info: position, quaternion, velocity, angular velocity
            * Δt is the time step
            * returns a 13x13 matrix describing how each component of the next state is affected 
              by each component of the current state
            * needed to calculate the covariance of the next state
    """

    """
    outline:
        * create the state vector of the object
        * predict the next state using the process model
        * predict the next covariance using the Jacobian of the process model

        * run EFK probability for updated track

    """

    # creating state vector for the object
    # creating state vector for the object
    x = vcat(track.pos,                     # 1:3 position
             track.orientation,             # 4:7 orientation quaternion
             track.vel,                     # 8:10 velocity
             track.angular_velocity)        # 11:13 angular velocity

    # setting up necessary noise matrices
    R = Diagonal(@SVector [1.0, 1.0, 50.0])
    # we trust x and y but not z 
 
    # TODO take data and calculate what noise is 
    Q = I(13) * 10 # noise matrix # im not sure if i did this correctly 
    P = track.P # covariance matrix of the object
    
    # predict the next state using the process model
    # using the previous state and some motion model, we predict where the object should be now.
    # F will linearize the nonlinear motion function f(x, Δt) around the current estimate of the state
    F = VehicleSim.Jac_x_f(x, Δt)
    x_pred = VehicleSim.f(x, Δt)                       # recall that x is "previous" state
    P_pred = F * P * F' + Q
   

    # finding the "actual state" using the measurement model
    # what actually ended up happening (use the position found by the camera)
    H = hcat(I(3), zeros(3, 10)) # linearizes this relationship around the current guess (x_pred) using the Jacobian
    y = z - x_pred[1:3] # residual = actual_measurement - expected_measurement
    S = H * P_pred * H' + R
    K = P_pred * H' * inv(S)                # the Kalman gain, how trustworthy was our prediction? 
                                            # kalman_gain = predicted_covariance * H' * inverse(residual_covariance)

    # now with the kalman gain K we can update the state and covariance
    x_updated = x_pred + K * y              # updated_state = predicted_state + kalman_gain * residual
    P_updated = (I - K * H) * P_pred        # updated_covariance = (I - kalman_gain * H) * predicted_covariance

    # build updated track structure
    ekf_updated_track = TrackedObject(track.id,
                                      track.time + Δt,
                                      x_updated[1:3], # position
                                      x_updated[4:7], # orientation
                                      x_updated[8:10], # velocity
                                      x_updated[11:13], # angular velocity
                                      P_updated)  # covariance matrix
    
    return ekf_updated_track
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
    segments::Vector{PolylineSegment}
    function Polyline(points, roads, parts, stops)
        segments = Vector{PolylineSegment}()
        N = length(points)
        @assert N ≥ 2
        for i = 1:N-1
            seg = StandardSegment(points[i], points[i+1],roads[i],parts[i], stops[i])
            push!(segments, seg)
        end
        new(segments)
    end
    function Polyline(points...)
        Polyline(points)
    end

    # default constructor 
    function Polyline()
        segments = Vector{PolylineSegment}()
    end
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
