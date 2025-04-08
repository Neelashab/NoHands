import Statistics # used to average measurements

# constants
BOUNDING_BOX = SVector(13.2, 5.7, 5.3)  

struct MyLocalizationType
    time::Float64
    position::SVector{3, Float64} # position of center of vehicle
    quaternion::SVector{4, Float64} # quaternion [w, x, y, z]
    velocity::SVector{3, Float64} # [vx, vy, vz]
    angular_velocity::SVector{3, Float64} # [wx, wy, wz]
end

# update to include detailed state
# i.e. vector of vehicle state (bounding boxes, velocity, heading, acceleration, etc.)
struct MyPerceptionType
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end

function heading_to_quaternion(yaw::Float64)
    # Assuming no roll/pitch, convert yaw → quaternion [w, x, y, z]
    # Axis of rotation is [0, 0, 1] for yaw
    half_yaw = yaw / 2
    return SVector(cos(half_yaw), 0.0, 0.0, sin(half_yaw))  # [w, x, y, z]
end

# NOTE - this function is also defined in VehicleSim
# Extract yaw from quaternion [w, x, y, z]
function extract_yaw_from_quaternion(q::SVector{4, Float64})
    atan(2(q[1]*q[4] + q[2]*q[3]), 1 - 2*(q[3]^2 + q[4]^2))
end


function localize(gps_channel, imu_channel, localization_state_channel)

    # WARM UP LOOP 
    gps_buffer = []
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
     init_quat = heading_to_quaternion(init_yaw)

     # Initialize state vector
     state = @SVector [
        init_x, init_y, 0.0, # position
        init_quat[1], init_quat[2], init_quat[3], init_quat[4], # quaternion
        0.0, 0.0, 0.0, # positional velocity
        0.0, 0.0, 0.0 # angular velocity
    ]

    Σ = 0.1 * I(13) # initial covariance matrix
    
    Q = 0.01 * I(13) # TODO - customize process noise for each segment of vector
    R_gps = Diagonal([1.0, 1.0, 0.1]) # GPS noise
    R_imu = Diagonal([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]) # IMU noise

    last_time = time()

    # ENTER MAIN LOOP
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

         # Use the most recent measurement from GPS and IMU
         latest_gps = isempty(fresh_gps_meas) ? nothing : last(fresh_gps_meas)
         latest_imu = isempty(fresh_imu_meas) ? nothing : last(fresh_imu_meas)
 
         # Compute dt
         now = time()
         dt = now - last_time 
         last_time = now


        # PREDICT: Where should we be given the previous state and how we expect the car to move?
        # TODO define motion model that accepts localization state and computes:
            # predicted state = f(previous state, current controls, dt)
            # predicted covariance = F (jacobian of current state) * previous covariance * F' + Q
            # * you can reuse the h in the measurement model in measurements.jl, check to see what the state of x is

            # jacobian of current state 
            F = Jac_x_f(state, dt)

            # where did EKF predict it would be given its previous state? 
            predicted_ref = f(state, dt)

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
                # actual measurement
                z_gps = @SVector [latest_gps.lat, latest_gps.long, latest_gps.heading]

                # actual measurement prediction 
                z_gps_pred = h_gps(predicted_ref)

                # residual 
                y_gps = z_gps - z_gps_pred
    
                H_gps = Jac_h_gps(predicted_ref)
                S_gps = H_gps * Σ * H_gps' + R_gps
                K_gps = Σ * H_gps' * inv(S_gps)
                
                # updated state factoring in state prediction with Kalman gain 
                predicted_state = predicted_ref + K * y_gps

                # updated covariance
                Σ = (I - K_gps * H_gps) * Σ
            end

            if latest_imu !== nothing
                # same as GPS
                z_imu = vcat(latest_imu.linear_vel, latest_imu.angular_vel)
                z_imu_pred = h_imu(predicted_ref)
                y_imu = z_imu - z_imu_pred
    
                H_imu = Jac_h_imu(predicted_ref)
                S_imu = H_imu * Σ * H_imu' + R_imu
                K_imu = Σ * H_imu' * inv(S_imu)
    
                predicted_state = predicted_ref + K_imu * y_imu
                Σ = (I - K_imu * H_imu) * Σ
            end
    
        # Publish relevant localization state
        localization_state = MyLocalizationType(
            now,
            state[1:3], # positions
            state[4:7], # quaternion
            state[8:10], # linear velocity
            state[11:13] # angular velocity
        )

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


struct TrackedObject
    id::Int
    time::Float64
    pos::SVector{3, Float64}  # x, y, z
    orientation::SVector{4, Float64}
    vel::SVector{3, Float64}  # vx, vy, vz
    angular_velocity::SVector{3, Float64}
    P::SMatrix{13,13,Float64} # covariance matrix
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

    vehicle_size = SVector(13.2, 5.7, 5.3) # retrieved from measurements.jl

    # program
    while true

        # set up stuff

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
    
        # skip if no measurements available
        if isempty(fresh_cam_meas) || isempty(fresh_gt_meas)
            sleep(0.01)
            continue
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

        # Get transformation from vehicle body to world
        T_world_from_body = get_body_transform(ego_quaternion, ego_position)

        # Combined transformation from camera to world
        T_world_from_cam = multiply_transforms(T_world_from_body, T_body_from_cam)

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


            world_point = T_world_from_cam * point_from_cam

            pos = world_point[1:3]


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
        # only keep tracks that have been updated recently
        current_tracks = TrackedObject[]
        for track in tracks
            if t - track.time < 2.0  # Keep tracks updated in last 2 seconds
                push!(current_tracks, track)
            end
        end
        tracks = current_tracks
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
    Q = Diagonal([                    # process noise..maybe?
                  0.1, 0.1, 0.1,               # position noise
                  0.01, 0.01, 0.01, 0.01,      # orientation noise
                  0.5, 0.5, 0.5,               # velocity noise
                  0.1, 0.1, 0.1                # angular velocity noise
                  ])  

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
   
  

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

    
end
 
 


struct MyPerceptionType
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end

function gt_state(gt)
    MyLocalizationType(gt.vehicle_id,gt.position,gt.orientation,gt.velocity,gt.angular_velocity,gt.size)
end

function gt_perception(gt)
    MyPerceptionType(gt.vehicle_id,gt.position,gt.orientation,gt.velocity,gt.angular_velocity,gt.size)
end

abstract type PolylineSegment end

function perp(x)
    [-x[2], x[1]]
end
# GOOOOOOOOD
struct StandardSegment <: PolylineSegment
    p1::SVector{2, Float64}
    p2::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
    road::Int # road ID
    part::Int # what part of your road is it on? - not curved or long, always one
    function StandardSegment(p1, p2, road, part)
        tangent = p2 - p1
        tangent ./= norm(tangent)
        normal = perp(tangent)
        new(p1, p2, tangent, normal, road, part)
    end
end

struct Polyline
    segments::Vector{PolylineSegment}
    function Polyline(points, roads, parts)
        segments = Vector{PolylineSegment}()
        N = length(points)
        @assert N ≥ 2
        for i = 1:N-1
            seg = StandardSegment(points[i], points[i+1],roads[i],parts[i])
            push!(segments, seg)
        end
        new(segments)
    end
    function Polyline(points...)
        Polyline(points)
    end
end
 

function signOfDot0(a, b)
    dot_prod = dot(a, b)
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 0.0 #???
    end
end

 

function left_right(v_direction, target_direction)
    v_normal = [-v_direction[2], v_direction[1]]
    return signOfDot0(v_normal, target_direction)
end

 

function signOfDot1(a, b)
    #println("in signOfDot1 a=$a, b=$b")
    dot_prod = dot(a, b)
    #println("in signOfDot1 dot_prod=$dot_prod")
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 1.0 #???
    end
end

 
function signed_distance_standard(seg::StandardSegment, q)
    alpha0 = 0.0
    alpha1 = 1.0
    dist0 = norm(alpha0*seg.p1 + (1.0 - alpha0)*seg.p2 - q)
    dist1 = norm(alpha1*seg.p1 + (1.0 - alpha1)*seg.p2 - q)
    while abs(alpha1 - alpha0) > 0.000001 || abs(dist0-dist1) > 0.000001
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
    # println("q=$q")
    # println("dist0=$dist0")
    dist0 < 0.01 ? 0.0 : signOfDot1(seg.normal, q - seg.p1)*dist0
end

 
"""
compute the signed distance from POINT to POLYLINE. 
Note that point has a positive signed distance if it is in the same side of the polyline as the normal vectors point.
"""
function signed_distance(polyline::Polyline, point, index_start, index_end)
    #print(",debug2")
    N = length(polyline.segments)
    dis_min = Inf
    if index_end > N || index_end < index_start || index_start < 1
        return dis_min
    end
    #print(",debug3")
    for i = index_start:index_end
        #println("i=$i, dis_min=$dis_min")
        dist = signed_distance_standard(polyline.segments[i], point)
        #println("i=$i, dist=$dist, dis_min=$dis_min")
        if abs(dist) < dis_min
            dis_min = abs(dist)
        end
    end
    #println("end of signed_distance, dist_min=$dist_min")
    return dis_min
end

#GOOF
function process_gt(
        gt_channel,
        shutdown_channel,
        localization_state_channel,
        perception_state_channel,
        ego_vehicle_id_channel)

    localization_initialized = false
    perception_initialized = false
    while true
        try
            fetch(shutdown_channel) && break
            found_this_vehicle = false
            found_other_vehicle = false
            ego_vehicle_id = fetch(ego_vehicle_id_channel)
            if ego_vehicle_id > 0
                fresh_gt_meas = []
                meas = fetch(gt_channel)
                while meas.time > 0 && length(fresh_gt_meas)<10
                    take!(gt_channel)
                    push!(fresh_gt_meas, meas)
                    meas = fetch(gt_channel)
                end

                new_localization_state_from_gt = MyLocalizationType(
                    0,zeros(3),zeros(4),zeros(3),zeros(3),zeros(3)
                )
                new_perception_state_from_gt = []
                gt_count = length(fresh_gt_meas)
                for i=1:gt_count
                    if fresh_gt_meas[i].vehicle_id==ego_vehicle_id
                        new_localization_state_from_gt = gt_state(fresh_gt_meas[i])
                        found_this_vehicle = true
                    else
                        push!(new_perception_state_from_gt,gt_perception(fresh_gt_meas[i]))
                        found_other_vehicle = true
                    end
                end
                if found_this_vehicle
                    if localization_initialized
                        take!(localization_state_channel)
                    end
                    put!(localization_state_channel, new_localization_state_from_gt)
                    localization_initialized = true
                end
                if found_other_vehicle
                    if perception_initialized
                        take!(perception_state_channel)
                    end
                    put!(perception_state_channel, new_perception_state_from_gt)
                    perception_initialized = true
                end
            end            
        catch
            println("exception in process_gt")
        end
        sleep(0.05)
    end
end

 
function is_in_seg(pos, seg)
    is_loading_zone = length(seg.lane_types) > 1 && seg.lane_types[2] == loading_zone
    i = is_loading_zone ? 3 : 2
    A = seg.lane_boundaries[1].pt_a
    B = seg.lane_boundaries[1].pt_b
    C = seg.lane_boundaries[i].pt_a
    D = seg.lane_boundaries[i].pt_b
    min_x = min(A[1], B[1], C[1], D[1])
    max_x = max(A[1], B[1], C[1], D[1])
    min_y = min(A[2], B[2], C[2], D[2])
    max_y = max(A[2], B[2], C[2], D[2])
    return min_x <= pos[1] <= max_x && min_y <= pos[2] <= max_y
end

 
function get_center(seg_id, map_segments, loading_id)
    seg = map_segments[seg_id]
    i = seg_id == loading_id ? 2 : 1
    A = seg.lane_boundaries[i].pt_a
    B = seg.lane_boundaries[i].pt_b
    C = seg.lane_boundaries[i+1].pt_a
    D = seg.lane_boundaries[i+1].pt_b
    return MVector((A + B + C + D)/4)
end
 
function get_loading_center(loading_id, map_segments)
    return get_center(loading_id, map_segments, loading_id)
end
 
function get_pos_seg_id(map_segments, pos)
    seg_id = 0
    for (id, seg) in map_segments
        if is_in_seg(pos, seg)
            seg_id = id
            break
        end
    end
    return seg_id
end
 
function get_path(parents, t)
    u = t
    path = [u]
    while parents[u] != 0
        u = parents[u]
        pushfirst!(path, u)
    end
    return path
end

 
"""
Use Disjtrkas shortest path algorithm in order to calculate the best route
"""

function get_route(map_segments, start_position, target_id)
    println("start get route function")
    start_id = get_pos_seg_id(map_segments, start_position)
    
    if start_id == 0
        @warn "Vehicle not found in any segment. Using closest segment."
        # Fallback: find closest segment
        min_dist = Inf
        for (id, seg) in map_segments
            center = get_center(id, map_segments, 0)
            dist = norm(center - start_position)
            if dist < min_dist
                min_dist = dist
                start_id = id
            end
        end
    end

    println("this is the start ID: $start_id")

    node1 = Int[]
    node2 = Int[]
    dists = Float64[]

    for (parent_id, parent_seg) in map_segments
        parent_center = get_center(parent_id, map_segments, 0)
        no_child = length(parent_seg.children)
        for j=1:no_child
            child_id = parent_seg.children[j]
            child_center = get_center(child_id, map_segments, 0)
            dist = norm(parent_center - child_center)
            push!(node1, parent_id)
            push!(node2, child_id)
            push!(dists, dist)
        end
    end

    println("finished looping through map segments")

    no_node = max(maximum(node1), maximum(node2))
    no_arc = Base.length(node1)

    graph = DiGraph(no_node)
    for i=1:no_arc
        add_edge!(graph, node1[i], node2[i])
    end

    distmx = Inf*ones(no_node, no_node)
    for i in 1:no_arc
        distmx[node1[i], node2[i]] = dists[i]
    end

    println("call dijkstra's shortest path")
    state = dijkstra_shortest_paths(graph, start_id, distmx)
    println("state: $state")
    return get_path(state.parents, target_id)
end

# function log_route(route, roads, parts, points)
#     log_file = open("decision_making_route.txt", "a")
#     currTime = Dates.format(now(), "HH:MM:SS.s")
#     println(log_file, currTime)
#     println(log_file, "route=$route")
#     println(log_file, "roads=$roads")
#     println(log_file, "parts=$parts")
#     println(log_file, "points=$points")
#     close(log_file)
# end

 
function get_first_point(seg)
    A = seg.lane_boundaries[1].pt_a
    C = seg.lane_boundaries[2].pt_a
    return MVector((A + C)/2)
end

 
"""
add mid point in two cases:
1. curved lane
convert mid point on chord to mid point on arc
Assuming only 90° turns for now
center calculation is copied code from map.jl
2. long lane
"""
function get_middle_point(seg)
    # io = open("get_middle_point.txt", "a")
    # println(io, "seg=$seg")
    A = seg.lane_boundaries[1].pt_a
    B = seg.lane_boundaries[1].pt_b
    C = seg.lane_boundaries[2].pt_a
    D = seg.lane_boundaries[2].pt_b
    # println(io, "A=$A")
    # println(io, "B=$B")
    # println(io, "C=$C")
    # println(io, "D=$D")
    # convert road into 2 points - start and end, in the center
    pt_a = (A+C)/2
    pt_b = (B+D)/2
    # println(io, "pt_a=$pt_a")
    # println(io, "pt_b=$pt_b")
    pt_m = (pt_a+pt_b)/2
    # println(io, "pt_m=$pt_m")
    delta = pt_b - pt_a
    dist = norm(pt_b-pt_a)
    # println(io, "dist=$dist")
    curvature1 = seg.lane_boundaries[1].curvature
    curvature2 = seg.lane_boundaries[2].curvature
    curved1 = !isapprox(curvature1, 0.0; atol=1e-6)
    curved2 = !isapprox(curvature2, 0.0; atol=1e-6)

    add_mid_point = false
    # should calculate both radii from 1/curvature (left and right, larger and smaller)
    # only care abt size here, not nessicarily l-r
    if curved1 && curved2
        rad1 = 1.0 / abs(curvature1)
        rad2 = 1.0 / abs(curvature2)
        rad = (rad1+rad2)/2
        # println(io, "rad=$rad")
        # calculate if the center is left or right
        # center calculation copied from map.jl
        left = curvature1 > 0
        if left
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [0, delta[2]]
            else
                center = pt_a + [delta[1], 0]
            end
        else
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [delta[1], 0]
            else
                center = pt_a + [0, delta[2]]
            end
        end
        # println(io, "center=$center")
        #convert mid point on chord to mid point on arc
        delta_to_center = pt_m - center
        # println(io, "delta_to_center=$delta_to_center")
        direction_from_center = delta_to_center/norm(delta_to_center)
        # println(io, "direction_from_center=$direction_from_center")
        # unit vector * radius is the additional vector to arc
        vector_from_center = rad*direction_from_center
        # println(io, "vector_from_center=$vector_from_center")
        pt_m = center + vector_from_center
        add_mid_point = true
    # also add midpoint if the road is long
    # TO DO: test if this helps or not 
    elseif dist > 79.9
        add_mid_point = true
    end
    # println(io, "add_mid_point=$add_mid_point, pt_m=$pt_m")
    # close(io)
    add_mid_point, pt_m
end

function get_polyline(map_segments, start_position, target_segment)
    println("get route")
    route = get_route(map_segments, start_position, target_segment) # get the route, map, start point, target from server
    println("route=$route")
    points = [start_position]
    roads = [route[1]]
    parts = [1] # for roads that you need to split into 3 points/ 2 parts - curved road has middle points 
    route_count = length(route)
    for r = 2:route_count # start from second point because first point is yourself
        seg = map_segments[route[r]]
        if r == route_count # if it is the target, it'll be a loading zone
            push!(points, get_loading_center(route[r], map_segments))
            push!(roads, route[r])
            push!(parts, 1)
        else # other road segments, push every first point
            push!(points, get_first_point(seg))
            push!(roads, route[r])
            push!(parts, 1)
        end
         # circular and long roads have a mid point
        add_mid_point, mid_point = get_middle_point(seg)
        if add_mid_point
            push!(points, mid_point)
            push!(roads, route[r])
            push!(parts, 2)
        end
    end
    #log_route(route, roads, parts, points)
    return Polyline(points, roads, parts)
end

 
function target_velocity(current_velocity, 
    distance_to_target,
    steering_angle,
    angular_velocity,
    veh_wid, 
    poly_count, 
    best_next, 
    signed_dist; 
    speed_limit=4.0)

    # Calculate absolute lateral error
    abs_dist = abs(signed_dist)

    # Base acceleration depends on how far from center line
    # More acceleration when close to center, less when far away
    base_acceleration = abs_dist < 3.0 ? 0.5 : 0.2

    # Starting point for target velocity
    target_vel = current_velocity + base_acceleration

    # Slow down if too far from center line
    if abs_dist > veh_wid
    target_vel = current_velocity / 2
    end

    # Ensure minimum speed
    target_vel = max(target_vel, 0.5)

    # Calculate combined steering effect (both steering angle and angular velocity)
    angular_effect = abs(angular_velocity) + abs(steering_angle)

    # Reduce speed limit based on angular effect
    # More steering/rotation = lower speed limit
    if angular_effect > π/2
    adjusted_limit = 1.0  # Hard limit for sharp turns
    else
    #  Linear interpolation: more angular effect = lower speed
    adjusted_limit = 1.0 + (speed_limit - 1.0) * (1 - 2 * angular_effect / π)
    end

    # Special handling for start of route where we need careful maneuvering
    if best_next < 5 && angular_effect > 0.001
    adjusted_limit = min(adjusted_limit, 1.5)
    end

    # Cap velocity at adjusted limit
    target_vel = min(target_vel, adjusted_limit)

    # Slow down as we approach the target destination
    poly_count_down = poly_count - best_next

    # When approaching end of route (last 2 segments)
    if poly_count_down < 2 && target_vel > poly_count_down
    target_vel = poly_count_down + 1.5  # Gradual slowdown
    end

    # When at final destination and close to target
    if poly_count_down < 1 && distance_to_target < veh_wid
    target_vel = 0.0  # Stop vehicle
    end

    return target_vel
end

 
function decision_making(localization_state_channel, 
        perception_state_channel, 
        target_segment_channel,
        shutdown_channel,
        map_segments, 
        socket)
    
    println("set lookahead time")
    ls = 2.0
    last_target_segment = 0
    # log for debugging
    #log_file = open("decision_making_log.txt", "a")
    #currTime = Dates.format(now(), "HH:MM:SS.s")
    #println(log_file, currTime)
    println("start decision making function")


    dummy_points = [[-91.66666666666667, -80.0], 
        [-91.66666666666667, 0.0], 
        [-80.0, 11.666666666666668], 
        [0.0, 11.666666666666668], 
        [21.666666666666668, 33.33333333333333], 
        [31.666666666666668, 73.33333333333333]
    ]
    dummy_roads = [1,2,3,4,5,6]
    dummy_parts = [1,1,1,1,1,1]
    poly = Polyline(dummy_points, dummy_roads, dummy_parts) # dummy polyline

    poly_count = 0
    poly_leaving = 0 # front wheel touch the end of this line
    best_next = 0
    max_signed_dist = 0.0
    signed_dist = 0.0
    target_location = [0.0,0.0]
    println("initalize while loop")
    while true
        fetch(shutdown_channel) && break
        println("fetch target segment")
        target_segment = fetch(target_segment_channel)
        println("got target segment = $target_segment")
        if target_segment > 0
            println("fetch latest localization state")
            latest_localization_state = fetch(localization_state_channel)
            pos = latest_localization_state.position
            println("position from localization: $pos")
            veh_pos = pos[1:2]
            if target_segment!=last_target_segment
                #currTime = Dates.format(now(), "HH:MM:SS.s")
                #println(log_file, currTime)
                println("new target_segment= $target_segment")
                target_location = get_center(target_segment, map_segments, target_segment)
                println("target_location=$target_location")
                poly = get_polyline(map_segments, veh_pos, target_segment)
                println("poly=$poly")
                poly_count = length(poly.segments)
                biggest_dist_to_poly = 0.0
                poly_leaving = 0
                best_next = 0
                max_signed_dist = 0.0
                signed_dist = 0.0
                last_target_segment = target_segment
            end
            # messages from server/localization
            ori = latest_localization_state.orientation
            vel = latest_localization_state.velocity
            a_vel = latest_localization_state.angular_velocity
            size = latest_localization_state.size
            # convert orientation to rotation matrix
            # Rot_3D is Rotation Matrix in 3D
            # When vehicle rotates on 2D with θ,
            # Rot_3D = [cos(θ)  -sin(θ)  0;
            #           sin(θ)   cos(θ)  0;
            #               0         0  1]
            Rot_3D = Rot_from_quat(ori)
            veh_vel = vel[1:2] # only want the first two (x, y)
            veh_dir = [Rot_3D[1,1],Rot_3D[2,1]] #cos(θ), sin(θ)
            veh_len = size[1] #vehicle Length
            veh_wid = size[2] #vehicle width
            rear_wl = veh_pos - 0.5 * veh_len * veh_dir # rear wheel (L) decides the movement
            distance_to_target = norm(target_location - veh_pos) # just for reference
            curr_vel = norm(veh_vel)
            print("tgt=$target_segment")
            # calculate the steering angle
            steering_angle = 0.0
            if curr_vel > 0.00001 # help handle error
                len0 = curr_vel * ls # how far you are expected to move given the calculation
                min_diff = Inf # looking for smallest number
                three_after = poly_leaving + 3 # check 3 next points on the route
                three_after = three_after > poly_count ? poly_count : three_after # might be less than 3 at end of task
                best_next = 0
                #println("poly_leaving=$poly_leaving")
                #println("three_after=$three_after")
                for i = poly_leaving+1 : three_after
                    #println("i=$i")
                    # this p2 is the same point as p1 of next poly segment
                    # we cannot use p1, because vehicle starts from p1 of first poly segment
                    try_point = poly.segments[i].p2#here p2 is the same point as p1 of next poly segment
                    try_dist = norm(try_point - rear_wl)
                    if try_dist < veh_len #front wheel touched poly line seg
                        poly_leaving = i
                        continue #too close
                    end
                    sign = signOfDot0(veh_dir, try_point - rear_wl)
                    # make sure you're moving forward and its not behind you
                    if sign > 0
                        l = norm(try_point - rear_wl)
                        diff = abs(l - len0)
                        if diff < min_diff
                            min_diff = diff
                            best_next = i
                        end
                    end
                end #for i = poly_leaving+1 : poly_count
                #println("best_next=$best_next")
                # if best next is = 0, that means you did not find a good point, so you should go to the next polyline segment
                best_next = best_next > 0 ? best_next : poly_leaving+1
                # if calculation for best_next is greater than number of polylines, you go to the max polyline count
                best_next = best_next > poly_count ? poly_count : best_next
                poly_next_seg = poly.segments[best_next]
                #println("poly_next_seg=$poly_next_seg")
                #---these lines are for print display---#
                next_road = poly_next_seg.road
                next_part = poly_next_seg.part
                print(",poly=$poly_count")
                if poly_leaving > 0 && poly_leaving <= best_next
                    poly_leaving_seg = poly.segments[poly_leaving]
                    #println("poly_leaving_seg=$poly_leaving_seg")
                    leaving_road = poly_leaving_seg.road
                    leaving_part = poly_leaving_seg.part
                    print(",lv=$leaving_road($leaving_part),to=$next_road($next_part)")
                    #print(",debug1")
                    signed_dist = signed_distance(poly, veh_pos, poly_leaving, best_next)
                    #print(",debugn")
                    if abs(signed_dist) > abs(max_signed_dist)
                        max_signed_dist = signed_dist
                        #println(log_file, "max_signed_dist=$max_signed_dist between lv=$leaving_road($leaving_part),to=$next_road($next_part)")
                    end
                else
                    print(",lv=0(0),to=$next_road($next_part)")
                end
                print(",s_d=$signed_dist, max_s_d=$max_signed_dist")
                #---these lines are for print display---#
                # control law calculation
                next_point = poly_next_seg.p2
                distance_to_node = norm(next_point - rear_wl)
                cos_alpha = dot(veh_dir, next_point - rear_wl)/norm(next_point-rear_wl)
                cos_alpha = round(cos_alpha, digits=3) # three decimal place
                alpha = acos(cos_alpha)
                sin_alpha = sin(alpha)
                left_or_right = left_right(veh_dir, next_point - rear_wl)
                # 0.75 is for smoothing to help with error
                steering_angle = 0.75 * atan(2.0*veh_len*sin_alpha*left_or_right, curr_vel*ls)
            end #if curr_vel > 0.0
            #latest_perception_state = fetch(perception_state_channel)            
            target_vel = target_velocity(curr_vel, distance_to_target, steering_angle, a_vel[3], veh_wid, poly_count, best_next, signed_dist)
            cmd = (steering_angle, target_vel, true)
            steering_degree = round(steering_angle * 180 / 3.14, digits=3)
            println(", str=$steering_degree, v=$curr_vel")
            serialize(socket, cmd)
        end #if target_segment > 0
        sleep(0.05)
    end#while true
    #close(log_file)
end#function def

 
function my_client(host::IPAddr=IPv4(0), port=4444; use_gt=true)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)
    target_segment_channel = Channel{Int}(1)
    ego_vehicle_id_channel = Channel{Int}(1)
    shutdown_channel = Channel{Bool}(1)
    put!(shutdown_channel, false)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    put!(target_segment_channel, target_map_segment)
    put!(ego_vehicle_id_channel, ego_vehicle_id)
    #println("before errormonitor(@async while true")
    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                #println("bytesavailble")
                measurement_msg = deserialize(socket)
                received = true
            else
                #println("no more bytesavailble")
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        old_target_segment = fetch(target_segment_channel)
        if target_map_segment ≠ old_target_segment
            take!(target_segment_channel)
            put!(target_segment_channel, target_map_segment)
        end
        ego_vehicle_id = measurement_msg.vehicle_id
        old_ego_vehicle_id = fetch(ego_vehicle_id_channel)
        if ego_vehicle_id ≠ old_ego_vehicle_id
            take!(ego_vehicle_id_channel)
            put!(ego_vehicle_id_channel, ego_vehicle_id)
        end

        for meas in measurement_msg.measurements
            #println("for meas in measurement_msg.measurements")
            if meas isa GPSMeasurement
                #println("meas isa GPSMeasurement")
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                #println("meas isa IMUMeasurement")
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                #println("meas isa CameraMeasurement")
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                #println("meas isa GroundTruthMeasurement")
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)
    
    if use_gt
        @async process_gt(gt_channel,
                      shutdown_channel,
                      localization_state_channel,
                      perception_state_channel, 
                      ego_vehicle_id_channel)
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

################################################################################
#ALTERNATE ROUTING IMPLEMENTATION (bef i saw kath routing stuff) idk if it works
################################################################################
#function build_graph(map_segments)
#    graph = Dict{Int, Vector{Tuple{Int, Float64}}}()
#    for segment in map_segments
#        u = segment.start_id
#        v = segment.end_id
#        weight = norm(segment.p2 - segment.p1)  # Euclidean distance

#        if !haskey(graph, u)
#            graph[u] = []
#        end
#        push!(graph[u], (v, weight))

#        # If roads are bidirectional, add the reverse edge
#        if !haskey(graph, v)
#            graph[v] = []
#        end
#        push!(graph[v], (u, weight))
#    end
#    return graph
#end

#function dijkstra(graph::Dict{Int, Vector{Tuple{Int, Float64}}}, start_node::Int)
#    dist = Dict(n => Inf for n in keys(graph))
#    prev = Dict{Int, Union{Nothing, Int}}(n => nothing for n in keys(graph))
#    dist[start_node] = 0.0

#    q = PriorityQueue(dist, Base.Order.Forward)

#    while !isempty(q)
#        u = dequeue!(q)

#        for (v, weight) in graph[u]
#            alt = dist[u] + weight
#            if alt < dist[v]
#                dist[v] = alt
#                prev[v] = u
#                q[v] = alt
#            end
#        end
#    end

#    return dist, prev
#end

#function reconstruct_path(prev::Dict{Int, Union{Nothing, Int}}, target::Int)
#    path = []
#    while target !== nothing
#        pushfirst!(path, target)
#        target = prev[target]
#    end
#    return path
#end

#function decision_making(localization_state_channel, 
#    perception_state_channel, 
#    map_segments, 
#    target_road_segment_id, 
#    socket)

#    graph = build_graph(map_segments)
#    route = []
#    route_idx = 1

#    while true
#        latest_localization_state = fetch(localization_state_channel)
#        latest_perception_state = fetch(perception_state_channel)
 
#        current_segment_id = get_current_segment(latest_localization_state, map_segments)

#        # Re-plan if the route is empty or the target has changed
#        if isempty(route) || route[end] != target_road_segment_id
#            dist, prev = dijkstra(graph, current_segment_id)
#            route = reconstruct_path(prev, target_road_segment_id)
#            route_idx = 1
#        end

#        if route_idx <= length(route)
#            next_segment = route[route_idx]
#            # Implement logic to drive towards `next_segment`
#            # Increment route_idx when the segment is reached
#        end

#        steering_angle = 0.0  # Placeholder
#        target_vel = 0.0      # Placeholder
#        cmd = (steering_angle, target_vel, true)
#        serialize(socket, cmd)
#    end
#end

#function get_current_segment(localization_state, map_segments)
#    vehicle_position = localization_state.position
#    for segment in map_segments
#        if is_point_near_segment(vehicle_position, segment)
#            return segment.start_id  # or another identifier for the segment
#        end
#    end
#    return nothing  # or handle the case where no segment is found
#end

#function is_point_near_segment(point::SVector{2, Float64}, segment; threshold=1.0)
#    p = point
#    a = segment.p1
#    b = segment.p2
#    ab = b - a
#    ap = p - a
#    t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0)  # Projection scalar onto segment
#    closest = a + t * ab  # Closest point on the segment
#    distance = norm(p - closest)
#    return distance < threshold
#end

