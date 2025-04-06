
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



function aminita_perception(cam_meas_channel, gt_channel, perception_state_channel)

    # input information:

    """
    cam_meas_channel

    camera measurements:
        - time: timestamp of measurement
        - camera_id: 1 or 2 (indicating which camera produced the bounding boxes)
        - focal_length: focal length of camera
        - pixel_length: length of a pixel in meters
        - image_width: width of image in pixels
        - image_height: height of image in pixels
        - bounding_boxes: list of bounding boxes represented by top-left and bottom-right pixel coordinates
        "Each pixel corresponds to a pixel_len x pixel_len patch at focal_len away from a pinhole model."

    localization (for now we are reciecing from ground truth): 
        - time: timestamp of measurement
        - vehicle_id: id of vehicle (since we are only focusing on this car, it'll be constant)
        - position: 3D position of center of vehicle
        - orientation: represented as quaternion
        - velocity: 3D velocity of vehicle
        - angular_velocity: angular velocity around x,y,z axes
        - size: length, width, height of 3D bounding box centered at (position/orientation)
      
    perception: we will have to figure this out later, bc we still aren't exactly sure what we want
    our perception to measure. but we have to use previous perception info to build on the current one

    """

    # set up stuff
    # do i need to create some initial guesses for when the program starts?

    # grab most recent cam_measurement
        # like perception, should we do a buffer and select based on that?

    # grab the latest localization state (for now we will be using ground truth)
    # from the localization state, create the neceeary transformation matrices

    # process the bounding boxes
        # for each bounding box...
            # pre process: convert TL and BR coordinagets to get the whole box
            # note: all cars are the same size so you can cheat a bit, find information in render code
            # idenitfy which camera took the measurement (where is it on the car)
            # convert image pixel → 3D direction (using camera intrinsics: focal stuff etc. )
            # known state 0> z component, length, width, height
            # use localalization matrix and camera infor to convert boudning box to world coordinates

            # using EKF... (need to find vel, heading, veloctity ; and match it to a tack)
            # see if an exsisitng path track is close enough to  bounding box
            # if so, update the track
            # if not, create a new track
            # if a track is not updated for a while, delete it (it might have exited the frame)
    
    # once he have updated everything:
        # create a new perception state object
            # so far im thinking thet the perception structure infcludes the list of all the paths
        # put the new perception state into the perception_state_channel

    # ok not sure if this is perception or decion planning , but i was thinking of a signial
    # we should implenet a rule "keep one cars distance for eveyr 10 mph your going"
    # so if the car is going 20 mph, we should be 2 car lengths away from the car in front of us
        # so our in front cehck should and stop if we are too close to the car in front of us
        # should we make a flag/ or signal for this

    # initializing perception state
    next_id = 1
    tracked_objs = TrackedObject[]
    perception_state = MyPerceptionType(0.0, next_id, tracked_objs)
    put!(perception_state_channel, perception_state)


    # program
    while true

        # set up stuff
        vehicle_size = SVector(13.2, 5.7, 5.3) # retrieved from measurements.jl

        # fetching camera measurement 
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        #  during the development phase, we will be using the ground truth localization
        #  but in the future we will be using the localization function (dont forget to update the function header)
        fresh_gt_meas = []
        while isready(gt_channel)
            meas = take!(gt_channel)
            push!(fresh_gt_meas, meas)
        end

        # TODO: implement neelasha's method of combining buffer with grabbing the correct measurement
        # TODO: create necessary matrices
        curr_camera = fresh_cam_meas[1]
        curr_gt = fresh_gt_meas[1]

        # uncpack the latest camera measurement
        focal_len = curr_camera.focal_length
        px_len = curr_camera.pixel_length
        img_w = curr_camera.image_width
        img_h = curr_camera.image_height
        t = curr_camera.time
        camera_id = curr_camera.camera_id
        
        # get_cam_transform(camera_id) is camera → car
        T_body_from_cam = get_cam_transform(camera_id) 

    
        # unpack the latest localization state
        ego_position = curr_gt.position
        ego_quaternion = curr_gt.orientation


        for box in latest_cam_meas.bounding_boxes

            # pre processing the box
            top_left_x = box[1]
            top_left_y = box[2]
            bottom_right_x = box[3]
            bottom_right_y = box[4]

            # center of bounding box
            u = (top_left_x + bottom_right_x) / 2   # horizontal (x pixel)
            v = (top_left_y + bottom_right_y) / 2   # vertical (y pixel)

            # converting pixel to 3D point (with respect to camera)
            # using pinhole camera model
            x = (u - img_w / 2) * px_len
            y = (v - img_h / 2) * px_len
            real_vehicle_height = vehicle_size[3]
            pixel_height = bottom_right_y - top_left_y
            z = (focal_len * real_vehicle_height) / (pixel_height * px_len)
            point_from_cam = SVector{4, Float64}(x, y, z, 1.0)


            # use transforamtion matrices to convert middle of object from camera to world coordinates
            # get_body_transform(quat, loc) is car → world
            T_world_from_body = get_body_transform(ego_quaternion, ego_position)
            # get_body_transform(quat, loc) is car → world
            T_world_from_cam = multiply_transforms(T_world_from_body, T_body_from_cam)
            pos = T_world_from_cam * point_from_cam


            # try to match bounding box with existing track
            # if no match, create a new track
            matched = false
            for (i, track) in enumerate(tracks)
                dist = norm(track.pos[1:2] - pos[1:2])  
                if dist < 5
                    Δt = t - track.time
                    updated_track = ekf(track, pos, Δt)
                    tracks[i] = updated_track  
                    matched = true
                    break
                end
            end

            # TODO: should i have a better estimate for veloctity and heading?
            # beecause right now its 0 0 and how is the EKF update supposed to work with
            # for know i just set all values (excpet positon and ID) to zero and set 
            # the covairance to be very large (veyr uncertain)
            if !matched
                new_id += 1
                initial_orientation = SVector(1.0, 0.0, 0.0, 0.0)         # identity quaternion
                initial_velocity = SVector(0.0, 0.0, 0.0)                  # unknown, so assume zero
                initial_angular_velocity = SVector(0.0, 0.0, 0.0)          # same
                initial_covariance = I(13) * 10.0                          # large uncertainty

                new_track = TrackedObject(new_id,
                              t,
                              pos,
                              initial_orientation,
                              initial_velocity,
                              initial_angular_velocity,
                              initial_covariance)

                push!(tracks, new_track)
            end
        end

        perception_state = MyPerceptionType(t, new_id, tracks)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)   
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
        Jac_x_h(x) - returns the Jacobian matrix of the measurement model
            * x is a state vector with the following info: position, quaternion, velocity, angular velocity
            * correct our predicted state using new measurements
    """

    """
    outline:
        * create the state vector of the object
        * predict the next state using the process model
        * predict the next covariance using the Jacobian of the process model

        * run EFK probability for updated track

    """

    # creating state vector for the object
    x = vcat(track.pos,                     # 1:3 position
             track.orientation,             # 4:7 orientation quaternion
             track.vel,                     # 8:10 velocity
             track.angular_velocity)        # 11:13 angular velocity

    # setting up necessary noise matrices
    R = I(3) * 1.0 # the code in measurement.jl does not provide a noise matrix for camera like it does 
                  # for GPS and IMU, so we will just use a simple identity matrix ; this is equivalent to saying
                  # i trust x, y, z measurements equally
    Q = I(13) * 0.1 # noise matrix # im not sure if i did this correctly 
    P = track.P # covariance matrix of the object
    
    # predict the next state using the process model
    # using the previous state and some motion model, we predict where the object should be now.
    # F will linearize the nonlinear motion function f(x, Δt) around the current estimate of the state
    F = Jac_x_f(x, Δt)
    x_pred = f(x, Δt)                       # recall that x is "previous" state
    P_pred = F * P * F' + Q

    # finding the "actual state" using the measurement model
    # what actually ended up happening (use the position found by the camera)
    H = Jac_h_camera(x_pred) # linearizes this relationship around the current guess (x_pred) using the Jacobian
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
