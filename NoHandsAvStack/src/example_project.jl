using StaticArrays

struct MyLocalizationType
    time::Float64
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end

struct MyPerceptionType
    time::Float64
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end

function gt_state(input)
    MyLocalizationType(input.time, 
                    input.vehicle_id, 
                    input.position,
                    input.orientation,
                    input.velocity,
                    input.angular_velocity,
                    input.size)
end

function process_gt(
        gt_channel,
        shutdown_channel,
        localization_state_channel,
        perception_state_channel)

    initialized = false;
    while true
        fetch(shutdown_channel) && break

        fresh_gt_meas = []

        println("get ready to take measurements")

        # this code does not work, is isready causing problems?
        # while isready(gt_channel)
        #    meas = take!(gt_channel)
        #    push!(fresh_gt_meas, meas)
        # end
        
        meas = fetch(gt_channel)
        while meas.time > 0 && length(fresh_gt_meas)<10
            take!(gt_channel) # delete here
            push!(fresh_gt_meas, meas)
            meas = fetch(gt_channel) # fetch does not delete the prior measurement
        end

        #println("measurements fetched and taken")

        # perception_state
        L = length(fresh_gt_meas)
        new_localization_state_from_gt = gt_conversion(fresh_gt_meas[L-1])

        #println("new state: $new_localization_state_from_gt")

        if initialized == true
            take!(localization_state_channel)
        end 

        put!(localization_state_channel, new_localization_state_from_gt)
        
        #take!(perception_state_channel)
        #put!(perception_state_channel, new_perception_state_from_gt)
        initialized = true
        sleep(0.1)

    end
end

function localize(
        gps_channel, 
        imu_channel, 
        localization_state_channel, 
        shutdown_channel)
    # Set up algorithm / initialize variables
    while true

        fetch(shutdown_channel) && break

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
        take!(localization_state_channel)
        put!(localization_state_channel, localization_state)
    end 
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)
    # set up stuff
    while true
        
        fetch(shutdown_channel) && break

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

function get_center(road_id, map, loading_id)
    # same for straight and curved segments
    seg = map[road_id]
    i = road_id == loading_id ? 2 : 1
    A = seg.lane_boundaries[i].pt_a
    B = seg.lane_boundaries[i].pt_b
    C = seg.lane_boundaries[i+1].pt_a
    D = seg.lane_boundaries[i+1].pt_b
    (A + B + C + D)/4
end

# hardcoded for now
function get_route(map, my_location)
    [24,17,14,10,84,82,80]
end

# --- NEW CODE HERE ----- #

# --- Polyline Calculations --- #
# how to make it work for the input and curved line segments?

abstract type PolylineSegment end # defines a type called a polyline segment
# the structs are the implementations of this type

function perp(x)
    [-x[2], x[1]]
end

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

# curved segment?
struct CurvedSegment <: PolylineSegment
    p1::SVector{2, Float64}
    p2::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
    curvature::Float64  # 1 / radius of curvature

    function CurvedSegment(p1, p2, curvature::Float64)
        # Compute the tangent vector
        tangent = p2 - p1
        tangent ./= norm(tangent)
        
        # The normal vector is perpendicular to the tangent
        normal = perp(tangent)
        
        # Adjust the segment based on curvature
        # The curvature adjusts how much we deviate from the straight line.
        # A positive curvature will make the segment curve in one direction, negative for the other.
        
        # The offset distance for curvature: offset = curvature * distance
        distance = norm(p2 - p1)
        offset = curvature * distance  # Larger curvature -> larger offset
        
        # Move the p2 point based on the curvature
        p2_curved = p2 + offset * normal
        
        # Return the curved segment
        new(p1, p2_curved, tangent, normal, curvature)
    end
end

# remove starting and terminal rays
struct Polyline
    segments::Vector{PolylineSegment} # array-like collection of segments
    function Polyline(points)
        segments = Vector{PolylineSegment}() # initialize segments
        N = length(points)
        @assert N ≥ 2 # must have at least 2 points
        for i = 1:(N-1)
            seg = StandardSegment(points[i], points[i+1]) 
            push!(segments, seg)
        end
        new(segments) # segments is an array of the points, tan, and norm of each segment
    end
    function Polyline(points...)
        Polyline(points)
    end
end

function get_polyline(map, my_location, target_segment)
    route = get_route(map, my_location, target_segment)
    points = [get_center(route[i], map, 80) for r =1:length(route)]
    Polyline(points)
end

"""
compute the signed distance from POINT to POLYLINE. 

Note that point has a positive signed distance if it is in the same side of the polyline as the normal vectors point.
"""
# compute signed part of signed distance
function signDist(a, b)
    dot_prod = dot(a, b)
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 1.0 
    end
end

# negative if to the right of normal vector, positive if to the left

# use binary search method to find alpha to minimize load time

# compute signed distance from standard
function signed_distance_standard(seg::StandardSegment, q) 
    # alpha is an interpolating value - find the alpha value that results in the minimum distance

    # dist should be the minimum 
    alpha0 = 0.0
    alpha1 = 1.0
    dist0 = norm(alpha0*seg.p1 + (1.0 - alpha0)*seg.p2 - q)
    dist1 = norm(alpha1*seg.p1 + (1.0 - alpha1)*seg.p2 - q)
    
    while abs(alpha1 - alpha0) > 0.00000001 || abs(dist0-dist1) > 0.0000000001
        alpha = (alpha0+alpha1)/2.0
        new_point = alpha*seg.p1 + (1.0 - alpha)*seg.p2
        diff = new_point - q
        dist = norm(alpha*seg.p1 + (1.0 - alpha)*seg.p2 - q)
        
        if abs(dist1) < abs(dist0)
            dist0 = dist
            alpha0 = alpha
        elseif abs(dist1) > abs(dist0)
            dist1 = dist
            alpha1 = alpha
        else
            break
        end
    end

    return signDist(seg.normal, q - seg.p1)*dist0

end

# loop through computations to find the minimum a
function signed_distance(polyline::Polyline, point)
    N = length(polyline.segments)

    dist_min = starting_dist; # find min starting w starting ray

    for i = 2:(N -1) # do you start at the second point?
        dist = signed_distance_standard(polyline.segments[i], point)
        if abs(dist) < abs(dist_min)
            dist_min = dist
        end
    end

    if abs(term_dist) < abs(dist_min)
        dist_min = term_dist
    end
    return dist_min

    return 0.0
end

# -- CONTROL -- #

function control(state_ch, control_ch, stop_ch, path, L)
    pure_pursuit(state_ch, control_ch, stop_ch, path, L)
    # or try dummy_controller(...) if you like
end

# signed distance function
function signOfDot(a, b)
    dot_prod = dot(a, b)
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 0.0
    end
end

# should car steer left or right?
# calculate the normal vector to the vehicle's direction vector.
# rotate 90 degrees counterclockwise
# sign of Dot tells you if you're to the left or right
# For a 2D vector [x, y], the normal vector is [-y,x]
function left_right(v_direction, target_direction)
    v_normal = [-v_direction[2], v_direction[1]]
    return signOfDot(v_normal, target_direction)
end

function pure_pursuit(state_ch, control_ch, stop_ch, path, L; ls = 1.445) # ls = lookahead time
    N = length(path.segments)

    while true
        sleep(0.01)
        fetch(stop_ch) && return # check if this task should end
        x = fetch(state_ch) # get latest simulation state
        p1, p2, θ, v = x # unpack state variables
        
# ------- my control law implementation ---
        vehicle_pos = [p1, p2]
        len0 = v * ls # velocity times look ahead time
        min_diff = len0 # difference between desired and durr distance - like alpha
        best_i = 0 # next best target on path
        v_direction = [cos(θ), sin(θ)] # direction vector of heading based on its orientation angle θ - like delta

        # for all the standard segments
        # find the segment that is the minimum distance away
        for i = 2 : N - 1
            target = path.segments[i].p1 # try current path segment as target
            sign = signOfDot(v_direction, target - vehicle_pos) 
            if sign > 0 # target pos must be ahead of vehicle
                l = norm(target - vehicle_pos) 
                diff = abs(l - len0) #euc distance -> absolute distance
                if diff < min_diff 
                    min_diff = diff
                    best_i = i
                end
            end
        end
        
        if best_i > 0
            target = path.segments[best_i].p1
            # cosine of the angle between the vehicle's direction and the target direction
            cos_alpha = dot(v_direction, target - vehicle_pos) / norm(target - vehicle_pos)
            # calculate the angle alpha using the arccosine function
            alpha = acos(cos_alpha)
            # calculate the sine of the angle alpha
            sin_alpha = sin(alpha)
            left_or_right = left_right(v_direction, target - vehicle_pos)
            # calculate the steering angle (delta) using a formula
            delta = atan(2.0 * L * sin_alpha * left_or_right, v * ls)
            u = [delta, 0.0]
            println("line 88")
        else
            println("line 90")
            u = zeros(2) # No valid target found
        end
# ------- my control law implementation ---

        take!(control_ch)  # Clear old control
        put!(control_ch, u)  # Put new control on the channel
    end
end

# --- NEW CODE ENDS ----- #

function decision_making(localization_state_channel, 
        perception_state_channel, 
        target_segment_channel,
        shutdown_channel,
        map, 
        socket)
    my_location = [0,0]
    target_segment = 80
    Polyline = get_polyline(map, my_location, target_segment)
    while true

        # implement pure pursuit controller here
        fetch(shutdown_channel) && break
        target_segment = fetch(target_segment_channel)
        #println("target_segment=$target_segment")
        if target_segment > 0
            #println("target_segment=$target_segment")
            latest_localization_state = fetch(localization_state_channel)
            println("dm func state: $latest_localization_state")
            #latest_perception_state = fetch(perception_state_channel)
            # TO-DO: change these functions to call the code written
            steering_angle = 0.0
            target_vel = 1.0
            cmd = (steering_angle, target_vel, true)
            currTime = Dates.format(now(), "HH:MM:SS.s")
            serialize(socket, cmd)
        end
        sleep(0.5)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444; use_gt=true)
    println("connect client")
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    println("initialize channels")
    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)
    target_segment_channel = Channel{Int}(1)
    shutdown_channel = Channel{Bool}(1)
    put!(shutdown_channel, false)

    println("initialize target map segment and ego vehicle")
    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    put!(target_segment_channel, target_map_segment)

    println("start error moniter")
    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket) # recieve package of measurements
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment # tells you the id of the next loading zone
        old_target_segment = fetch(target_segment_channel)
        if target_map_segment ≠ old_target_segment
            take!(target_segment_channel)
            put!(target_segment_channel, target_map_segment)
        end
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

    println("get measurement and status")

    # unpacking and processing the gt OR OTHER channel
    if use_gt
        @async process_gt(gt_channel,
                      shutdown_channel,
                      localization_state_channel,
                      perception_state_channel)
    else
        @async localize(gps_channel, 
                    imu_channel, 
                    localization_state_channel, 
                    shutdown_channel)

        @async perception(cam_channel, 
                      localization_state_channel, 
                      perception_state_channel, 
                      shutdown_channel)
    end

    println("call decision making function")

    @async decision_making(localization_state_channel, 
                           perception_state_channel, 
                           target_segment_channel, 
                           shutdown_channel,
                           map_segments, 
                           socket)
end

function shutdown_listener(shutdown_channel)
    info_string = 
        "***************
      CLIENT COMMANDS
      ***************
            -Make sure focus is on this terminal window. Then:
            -Press 'q' to shutdown threads. 
    "
    @info info_string
    while true
        sleep(0.1)
        key = get_c()

        if key == 'q'
            # terminate threads
            take!(shutdown_channel)
            put!(shutdown_channel, true)
            break
        end
    end
end
