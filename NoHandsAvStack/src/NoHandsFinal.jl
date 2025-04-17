using Graphs
using Rotations
using StaticArrays, LinearAlgebra, Statistics
using Dates

#include("measurements.jl")

# constants
BOUNDING_BOX = SVector(13.2, 5.7, 5.3)

""" STRUCTURES """

struct MyLocalizationType
    time::Float64
    position::SVector{3,Float64} # position of center of vehicle
    orientation::SVector{4,Float64} # quaternion [w, x, y, z]
    velocity::SVector{3,Float64} # [vx, vy, vz]
    angular_velocity::SVector{3,Float64} # [wx, wy, wz]
end

struct TrackedObject
    id::Int
    time::Float64
    pos::SVector{3,Float64}  # x, y, z
    orientation::SVector{4,Float64}
    vel::SVector{3,Float64}  # vx, vy, vz
    angular_velocity::SVector{3,Float64}
    P::SMatrix{13,13,Float64} # covariance matrix
end

struct MyPerceptionType
    time::Float64
    next_id::Int
    tracked_objs::Vector{TrackedObject}
end

""" GROUND TRUTH SUPPORT """
function gt_state(gt)
    MyLocalizationType(time(), gt.position, gt.orientation, gt.velocity, gt.angular_velocity)
end

function gt_perception(gt)
    MyPerceptionType(time(), 1, Vector{TrackedObject[]})
end


""" LOCALIZATION """

function heading_to_quaternion(yaw::Float64)
    half_yaw = yaw / 2
    return SVector(cos(half_yaw), 0.0, 0.0, sin(half_yaw))  # [w, x, y, z]
end

function extract_yaw_from_quaternion(q::Vector{Float64})
    VehicleSim.extract_yaw_from_quaternion(SVector{4,Float64}(q))
end


# IMU measurement model - predicts IMU measurement given current state
function h_imu(x)
    # transform from body frame to IMU frame w/ rotation matrix and offset
    T_body_imu = VehicleSim.get_imu_transform()
    R = T_body_imu[1:3, 1:3]
    p = T_body_imu[1:3, end]

    # extract velocity and angular velocity from state and transfomr to IMU frame
    v_body = x[8:10]
    w_body = x[11:13]

    w_imu = R * w_body
    v_imu = R * v_body + cross(p, w_imu)

    return [v_imu; w_imu]
end


# Determines how sensitive IMU measurements are to changes in state
function Jac_h_imu(x; δ=1e-6)
    H = zeros(6, 13)
    h0 = h_imu(x)
    for i in 1:13
        dx = zeros(SVector{13,Float64})
        dx = setindex(dx, δ, i)
        H[:, i] = (h_imu(x + dx) - h0) / δ
    end
    return H
end

function localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
    println("Entering localize")
    gps_buffer = VehicleSim.GPSMeasurement[]

    while length(gps_buffer) < 5
        if isready(gps_channel)
            msg = take!(gps_channel)
            push!(gps_buffer, msg)
        else
            sleep(0.01)  # don't spin too fast if channel is empty
        end
    end

    # Estimate initial x, y, heading from GPS
    init_x = mean([m.lat for m in gps_buffer])
    init_y = mean([m.long for m in gps_buffer])
    init_yaw = mean([m.heading for m in gps_buffer])
    init_quat = heading_to_quaternion(init_yaw)

    # Initialize state vector
    state = @SVector [
        init_x, init_y, 2.6455622444987412, # position
        init_quat[1], init_quat[2], init_quat[3], init_quat[4], # quaternion
        0.0, 0.0, 0.0, # positional velocity
        0.0, 0.0, 0.0 # angular velocity
    ]


    Σ = 0.1 * I(13) # initial covariance matrix

    Q = Diagonal([
        0.01,   # x
        0.01,   # y
        0.0,    # z (hardcoded)
        1.0,    # qw
        1.0,    # qx
        1.0,    # qy
        1.0,    # qz
        0.05,   # vx
        0.05,   # vy
        0.05,   # vz (hardcoded)
        0.1,    # wx
        0.1,    # wy
        0.1     # wz
    ])
    
    
    R_gps = Diagonal([1.0, 1.0, 0.0005]) # GPS noise
    R_imu = Diagonal([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]) # IMU noise

    last_time = gps_buffer[end].time  # use timestamp from last GPS warmup


    # MAIN LOOP
    while true
        if isready(shutdown_channel) && take!(shutdown_channel)
            println("shutting down localize")
            break
        end

        fresh_gps_meas = []
        while isready(gps_channel)
            push!(fresh_gps_meas, take!(gps_channel))
        end

        fresh_imu_meas = []
        while isready(imu_channel)
            push!(fresh_imu_meas, take!(imu_channel))
        end

        if isempty(fresh_gps_meas) && isempty(fresh_imu_meas)
            sleep(0.01)
            continue
        end

        # Use the most recent measurement from GPS and IMU
        latest_gps = isempty(fresh_gps_meas) ? nothing : last(fresh_gps_meas)
        latest_imu = isempty(fresh_imu_meas) ? nothing : last(fresh_imu_meas)

        # use most recent timestamp between GPS and IMU
        now = maximum([
            latest_gps !== nothing ? latest_gps.time : -Inf,
            latest_imu !== nothing ? latest_imu.time : -Inf
        ])

        # computer dt
        dt = now - last_time
        last_time = now

        # PREDICT: Where should we be given the previous state and how we expect the car to move?
        # TODO define motion model that accepts localization state and computes:
        # predicted state = f(previous state, current controls, dt)
        # predicted covariance = F (jacobian of current state) * previous covariance * F' + Q
        # * you can reuse the h in the measurement model in measurements.jl, check to see what the state of x is

        # jacobian of current state 
        F = VehicleSim.Jac_x_f(state, dt)

        # where did EKF predict it would be given its previous state? 
        predicted_ref = VehicleSim.f(state, dt)

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

        predicted_state = predicted_ref

        if latest_gps !== nothing
            # actual measurement
            z_gps = @SVector [latest_gps.lat, latest_gps.long, latest_gps.heading]

            # actual measurement prediction
            z_gps_pred = VehicleSim.h_gps(predicted_state)

            # residual
            y_gps = z_gps - z_gps_pred

            H_gps = VehicleSim.Jac_h_gps(predicted_state)
            S_gps = H_gps * Σ * H_gps' + R_gps
            K_gps = Σ * H_gps' * inv(S_gps)

            # updated state factoring in state prediction with Kalman gain 
            predicted_state = predicted_state + K_gps * y_gps

            # updated covariance
            Σ = (I - K_gps * H_gps) * Σ
        end

        state = predicted_state

        if latest_imu !== nothing
            # same as GPS
            z_imu = vcat(latest_imu.linear_vel, latest_imu.angular_vel)
            z_imu_pred = h_imu(predicted_state)
            y_imu = z_imu - z_imu_pred

            H_imu = Jac_h_imu(predicted_state)
            S_imu = H_imu * Σ * H_imu' + R_imu
            K_imu = Σ * H_imu' * inv(S_imu)

            predicted_state = predicted_state + K_imu * y_imu
            Σ = (I - K_imu * H_imu) * Σ
        end

        state = predicted_state
        # Normalize quaternion and rebuild full state
        q_norm = normalize(state[4:7])
        state = SVector{13,Float64}(
            predicted_state[1:2]...,             # x, y
            2.6455622444987412,                  # hardcoded z
            q_norm...,                           # normalized quaternion
            predicted_state[8:13]...             # velocities
        )


        # Publish relevant localization state
        localization_state = MyLocalizationType(
            now,
            state[1:3],
            state[4:7],
            state[8:10],
            state[11:13]
        )

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end

        put!(localization_state_channel, localization_state)
    end
end


"""PERCEPTION"""


function perception(cam_meas_channel, gt_channel, perception_state_channel, shutdown_channel)

    println("Perception function started!")

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
                #println("Received camera message")
                meas = take!(cam_meas_channel)
                push!(fresh_cam_meas, meas)
            end
    
            # this is to be replaced with localization
            fresh_gt_meas = []
            while isready(gt_channel)
                #println("Received GT message")
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
                #println("No bounding boxes yet")
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
            #println("T_world_camrot: ", T_world_camrot)
            #println("\n\n")

    
            focal_len = latest_cam.focal_length
            px_len = latest_cam.pixel_length
            img_w = latest_cam.image_width + 1
            img_h = latest_cam.image_height + 1
            t = latest_cam.time
            vehicle_size = SVector(13.2, 5.7, 5.3)
    
            for box in latest_cam.bounding_boxes
                #println("hello?.")
                u = (box[1] + box[3]) / 2
                v = (box[2] + box[4]) / 2
                x = (u - img_w / 2) * px_len
                y = (v - img_h / 2) * px_len

                uu = VehicleSim.convert_to_pixel(img_w, px_len, x)

          

                #pitch = atan2(-forward[2], norm(forward[[1, 3]]))
                

                # find center of bounding box 
                # so the camera is on the top of the vehicle
                # TODO: find exact adjacent_leg
                    # will be how above the base/center is
                # TODO: find angle camera is pointing alpha 
                    # angle between the camera's optical axis and the direction to the center of the bounding box
                # hypotnuse/depth = adjacent_leg / cos(alpha) 

                # research -- 
                    #  t_cam_to_body = [1.35, 1.7, 2.4]
                    #  t_cam_to_body[2] = -1.7
                # so for both cmaeras its 2.4 meters above the base 
                
            
                pitch = 0.02 - atan(y/focal_len) # 0.02 is the built in oiriginal tilt, add more andle 
                # angle R_cam_to_body = RotY(0.02)
                # alpha = 0.02 radians 

                #R = T_world_camrot[1:3, 1:3]
                #rad = atan((-R[3,1])/(sqrt(R[3,2]^2 + R[3,3]^2)))
                #(pi/2 - 0.02)
                #pitch_fixed = (pi/2 - pitch) 
                
               
                max_angle = deg2rad(80.0)
                pitch_fixed = clamp(π/2 - pitch, -max_angle, max_angle)
                z = 2.4 / cos(pitch_fixed)
            

                # we know t_cam rot is basically the camera is respect to the world 
                # if i drew a straight line from camera to ob until when does it hit a height of 2.5
                # until the height of that staight line 

                #@info "uu is $uu, u is $u"
                
                #z = focal_len
                #depth = 7 0

                #xx = x * z / focal_len
                #yy = y * z / focal_len
                
                point_from_cam = SVector{4, Float64}(x, y, z, 1.0)

                pos = T_world_camrot * point_from_cam

                #@info "pos is $pos"
                #@info "gt is $ego_position"
                #@info "gt orientation is $ego_quaternion"

                pos[3] = vehicle_size[3] / 2
                #println("world position (before EFK): ", pos)
    
                
                num_bounding_boxes = length(latest_cam.bounding_boxes)
                num_tracks = length(tracks)


                if num_tracks == 0 && num_bounding_boxes != 0
                    while length(tracks) < num_bounding_boxes
                        new_track = initialize_track(ego_position, ego_quaternion, SVector{3, Float64}(pos[1:3]), next_id, t)
                        push!(tracks, new_track)
                        next_id += 1
                    end

                else
                    matched = false

                    if num_bounding_boxes == num_tracks
                        matched = true
                    end

                    if matched == true
                        for(i, track) in enumerate(tracks)
                            Δt = t - track.time
                            updated_track = ekf(track, SVector{3, Float64}(pos[1:3]), Δt)
                            tracks[i] = updated_track
                        end
                    else

                        for (i, track) in enumerate(tracks)
                            dist = norm(track.pos[1:2] - pos[1:2])
                            println("dist: ", dist)
                            if dist < 30
                            
                                Δt = t - track.time
                                updated_track = ekf(track, SVector{3, Float64}(pos[1:3]), Δt)
                                tracks[i] = updated_track
                        
                                matched = true
                                break

                            else
                                new_track = TrackedObject
                                new_track = initialize_track(ego_position, ego_quaternion, SVector{3, Float64}(pos[1:3]), next_id, t)
                                push!(tracks, new_track)
                                next_id += 1

                            end
                        end
                    end
                end
    
            perception_state = MyPerceptionType(t, next_id, tracks)
           
            put!(perception_state_channel, perception_state)
    
            sleep(0.05)
        end
    end
    catch e
        println("ERROR in perception: ", e)
        println(sprint(showerror, e))
    end
    println("Perception finished.")
    
end


function initialize_track(ego_pos::SVector{3, Float64}, ego_quat::SVector{4, Float64}, obj_pos::SVector{3, Float64}, next_id::Int, t::Float64)
    
    lane_width = 10.0       # based on the city_map defintions in map.jl

    ego_yaw = VehicleSim.extract_yaw_from_quaternion(ego_quat)
    ego_heading = SVector(cos(ego_yaw), sin(ego_yaw))

    # compute how sideways teh object is from ego
    delta = obj_pos[1:2] - ego_pos[1:2]
    lateral_offset = abs(det(hcat(ego_heading, delta)) / norm(ego_heading))

    # decide orientation of the object
    if lateral_offset < lane_width / 2
        # same lane = same heaidng
        initial_orientation = ego_quat
    elseif lateral_offset < lane_width * 1.5
        # opposite lane = flipped heading
        flipped_yaw = ego_yaw + π
        initial_orientation = SVector(cos(flipped_yaw/2), 0.0, 0.0, sin(flipped_yaw/2))
    else
        seg_id = get_pos_seg_id(map_segments, obj_pos[1:2])
        if haskey(map_segments, seg_id)
            seg = map_segments[seg_id]
            A = seg.lane_boundaries[1].pt_a
            B = seg.lane_boundaries[1].pt_b
            direction = B - A
            direction /= norm(direction)
            yaw = atan(direction[2], direction[1])
            #@warn "Valid Yaw calculated: $seg_id"
            initial_orientation = heading_to_quaternion(yaw)
        else
            # fall back is to asusme worst case the car is heading straight towards you 
            vec_xy = ego_pos[1:2] - obj_pos[1:2]
            yaw = atan(vec_xy[2], vec_xy[1])

            initial_orientation = heading_to_quaternion(yaw)  #  fallback orientation
        end

    end

    initial_velocity = SVector(0.0, 0.0, 0.0)
    initial_angular_velocity = SVector(0.0, 0.0, 0.0)

    cov_diag = [
            2.0, 2.0, 1.0, 
            2.0, 2.0, 2.0, 2.0,
            10.0, 10.0, 0.5,
            10.0, 10.0, 0.5
        ]
    initial_covariance = Diagonal(SVector{13, Float64}(cov_diag))

    new_track = TrackedObject(
        next_id, t, obj_pos,
        initial_orientation,
        initial_velocity,
        initial_angular_velocity,
        initial_covariance
    )

    return new_track

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
    R = Diagonal(@SVector [1.75, 3.0, 1.0])
    # we trust x and y but not z 
 
    # TODO take data and calculate what noise is 
    Q = I(13) * 5.5
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




function process_gt(
    gt_channel,
    shutdown_channel,
    localization_state_channel,
    perception_state_channel,
    ego_vehicle_id_channel)
    while true
        fetch(shutdown_channel) && break
        found_this_vehicle = false
        found_other_vehicle = false
        ego_vehicle_id = fetch(ego_vehicle_id_channel)
        if ego_vehicle_id > 0
            fresh_gt_meas = [] 
            meas = fetch(gt_channel) # get messages from gt
            while meas.time > 0 && length(fresh_gt_meas)<10 # if you get a meaningful message 
                take!(gt_channel)
                push!(fresh_gt_meas, meas)
                meas = fetch(gt_channel)
            end

            new_localization_state_from_gt = MyLocalizationType(
                0,zeros(3),zeros(4),zeros(3),zeros(3),zeros(3)
            )
            new_perception_list_from_gt = Vector{MyPerceptionType}(undef, 0)# it's Vector{MyPerceptionType} of all other vehicles
            gt_count = length(fresh_gt_meas)
            for i=1:gt_count
                if fresh_gt_meas[i].vehicle_id==ego_vehicle_id
                    new_localization_state_from_gt = gt_state(fresh_gt_meas[i])
                    found_this_vehicle = true
                else
                    one_perception = gt_perception(fresh_gt_meas[i])
                    push!(new_perception_list_from_gt, one_perception)
                    found_other_vehicle = true
                end
            end

            if found_this_vehicle
                if length(localization_state_channel.data)>=1
                    take!(localization_state_channel)
                end
                put!(localization_state_channel, new_localization_state_from_gt)
            end
            if found_other_vehicle
                if length(perception_state_channel.data)>=1
                    take!(perception_state_channel)
                end
                put!(perception_state_channel, new_perception_list_from_gt)
            end
        end            

        sleep(0.05)
    end
end



""" SIGNED DISTANCE FROM POINT TO POLYLINE """

abstract type PolylineSegment end

function perp(x)
    [-x[2], x[1]]
end

struct StandardSegment <: PolylineSegment
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
    tangent::SVector{2,Float64}
    normal::SVector{2,Float64}
    road::Int
    part::Int
    stop::Int
    function StandardSegment(p1, p2, road, part, stop)
        tangent = p2 - p1
        tangent ./= norm(tangent)
        normal = perp(tangent)
        new(p1, p2, tangent, normal, road, part, stop)
    end
end


struct Polyline
    segments::Vector{PolylineSegment}
    function Polyline(points, roads, parts, stops)
        segments = Vector{PolylineSegment}()
        N = length(points)
        @assert N ≥ 2
        for i = 1:N-1
            seg = StandardSegment(points[i], points[i+1], roads[i], parts[i], stops[i])
            push!(segments, seg)
        end
        new(segments)
    end
    function Polyline(points...) # var args
        Polyline(points)
    end

    # default constructor 
    function Polyline()
        segments = Vector{PolylineSegment}()
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

function signOfDot1(a, b)
    #println("in signOfDot1 a=$a, b=$b")
    dot_prod = dot(a, b)
    #println("in signOfDot1 dot_prod=$dot_prod")
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 1.0 
    end
end

function left_right(v_direction, target_direction)
    v_normal = [-v_direction[2], v_direction[1]]
    return signOfDot0(v_normal, target_direction)
end

function signed_distance_standard(seg::StandardSegment, q)
    alpha0 = 0.0
    alpha1 = 1.0
    dist0 = norm(alpha0 * seg.p1 + (1.0 - alpha0) * seg.p2 - q)
    dist1 = norm(alpha1 * seg.p1 + (1.0 - alpha1) * seg.p2 - q)
    while abs(alpha1 - alpha0) > 0.000001 || abs(dist0 - dist1) > 0.000001
        alpha = (alpha0 + alpha1) / 2.0
        new_point = alpha * seg.p1 + (1.0 - alpha) * seg.p2
        diff = new_point - q
        dist = norm(alpha * seg.p1 + (1.0 - alpha) * seg.p2 - q)
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
    dist0 < 0.01 ? 0.0 : signOfDot1(seg.normal, q - seg.p1) * dist0
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
    for i = 1:N
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


function is_in_seg(pos, seg)
    is_loading_zone = length(seg.lane_types) > 1 && seg.lane_types[2] == VehicleSim.loading_zone
    i = is_loading_zone ? 3 : 2
    A = seg.lane_boundaries[1].pt_a
    B = seg.lane_boundaries[1].pt_b
    C = seg.lane_boundaries[i].pt_a
    D = seg.lane_boundaries[i].pt_b
    min_x = min(A[1], B[1], C[1], D[1])
    max_x = max(A[1], B[1], C[1], D[1])
    min_y = min(A[2], B[2], C[2], D[2])
    max_y = max(A[2], B[2], C[2], D[2])
    min_x <= pos[1] <= max_x && min_y <= pos[2] <= max_y
end

function get_center(seg_id, map_segments, loading_id)
    seg = map_segments[seg_id]
    i = seg_id == loading_id ? 2 : 1
    A = seg.lane_boundaries[i].pt_a
    B = seg.lane_boundaries[i].pt_b
    C = seg.lane_boundaries[i+1].pt_a
    D = seg.lane_boundaries[i+1].pt_b
    MVector((A + B + C + D) / 4)
end

function get_loading_center(loading_id, map_segments)
    get_center(loading_id, map_segments, loading_id)
end

function get_pos_seg_id(map_segments, pos)
    seg_id = 0
    for (id, seg) in map_segments
        if is_in_seg(pos, seg)
            seg_id = id
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

function get_route(map_segments, start_position, target_id)
    start_id = get_pos_seg_id(map_segments, start_position)

    node1 = []
    node2 = []
    dists = []

    for (parent_id, parent_seg) in map_segments
        parent_center = get_center(parent_id, map_segments, 0)
        no_child = length(parent_seg.children)
        for j = 1:no_child
            child_id = parent_seg.children[j]
            child_center = get_center(child_id, map_segments, 0)
            dist = norm(parent_center - child_center)
            push!(node1, parent_id)
            push!(node2, child_id)
            push!(dists, dist)
        end
    end

    no_node = max(maximum(node1), maximum(node2))
    no_arc = Base.length(node1)

    graph = DiGraph(no_node)
    for i = 1:no_arc
        add_edge!(graph, node1[i], node2[i])
    end

    distmx = Inf * ones(no_node, no_node)
    for i in 1:no_arc
        distmx[node1[i], node2[i]] = dists[i]
    end

    println("Implement dijkstra's")
    state = dijkstra_shortest_paths(graph, start_id, distmx)
    path = get_path(state.parents, target_id)
end

function log_route(route, roads, parts, stops, points)
    log_file = open("decision_making_route.txt", "a")
    currTime = Dates.format(now(), "HH:MM:SS.s")
    println(log_file, currTime)
    println(log_file, "route=$route")
    println(log_file, "roads=$roads")
    println(log_file, "parts=$parts")
    println(log_file, "stops=$stops")
    println(log_file, "points=$points")
    close(log_file)
end

function get_first_point(seg)
    A = seg.lane_boundaries[1].pt_a
    C = seg.lane_boundaries[2].pt_a
    MVector((A + C) / 2)
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
    pt_a = (A + C) / 2
    pt_b = (B + D) / 2
    # println(io, "pt_a=$pt_a")
    # println(io, "pt_b=$pt_b")
    pt_m = (pt_a + pt_b) / 2
    # println(io, "pt_m=$pt_m")
    delta = pt_b - pt_a
    dist = norm(pt_b - pt_a)
    # println(io, "dist=$dist")
    curvature1 = seg.lane_boundaries[1].curvature
    curvature2 = seg.lane_boundaries[2].curvature
    curved1 = !isapprox(curvature1, 0.0; atol=1e-6)
    curved2 = !isapprox(curvature2, 0.0; atol=1e-6)

    add_mid_point = false
    if curved1 && curved2
        rad1 = 1.0 / abs(curvature1)
        rad2 = 1.0 / abs(curvature2)
        rad = (rad1 + rad2) / 2
        # println(io, "rad=$rad")
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
        direction_from_center = delta_to_center / norm(delta_to_center)
        # println(io, "direction_from_center=$direction_from_center")
        vector_from_center = rad * direction_from_center
        # println(io, "vector_from_center=$vector_from_center")
        pt_m = center + vector_from_center
        add_mid_point = true
    elseif dist > 79.9
        add_mid_point = true
    end
    # println(io, "add_mid_point=$add_mid_point, pt_m=$pt_m")
    # close(io)
    add_mid_point, pt_m
end

function get_polyline(map_segments, start_position, target_segment)
    println("get new route")
    route = get_route(map_segments, start_position, target_segment)
    println("route=$route")
    points = [start_position]
    roads = [route[1]]
    parts = [1] # curve road has middle points
    stops = [0]
    route_count = length(route)
    for r = 2:route_count
        seg = map_segments[route[r]]
        if r == route_count
            push!(points, get_loading_center(route[r], map_segments))
            push!(roads, route[r])
            push!(parts, 1)
        else
            push!(points, get_first_point(seg))
            push!(roads, route[r])
            push!(parts, 1)
        end

        add_mid_point, mid_point = get_middle_point(seg)
        if add_mid_point
            push!(points, mid_point)
            push!(roads, route[r])
            push!(parts, 2)

            push!(stops, 0)
        end

        push!(stops, has_stop_sign(seg) ? 1 : 0)

    end

    log_route(route, roads, parts, stops, points)
    poly = Polyline(points, roads, parts, stops)
    return poly
end

function has_stop_sign(seg)
    yes_stop_sign = false
    for i = 1:length(seg.lane_types)
        if seg.lane_types[i] == VehicleSim.stop_sign
            yes_stop_sign = true
        end
    end
    return yes_stop_sign
end

function target_velocity(
    veh_pos,
    avoid_collision_speed,
    current_velocity,
    distance_to_target,
    found_stop_sign,
    distance_to_stop_sign,
    steering_angle,
    angular_velocity,
    veh_wid,
    poly_count,
    best_next,
    signed_dist,
    perception_state_channel;
    speed_limit=7)

    println("ENTERED TARGET VELOCITY")

    target_vel = current_velocity + 0.5
    angular_effect = abs(angular_velocity) + abs(steering_angle)
    adjusted_limit = angular_effect > pi / 2 ? 1.0 : (1.0 + (speed_limit - 1.0) * (1 - 2 * angular_effect / pi))
    target_vel = target_vel > adjusted_limit ? adjusted_limit : target_vel

    #slow down when vehicle approaches the target
    poly_count_down = poly_count - best_next
    target_vel = poly_count_down < 2 && target_vel > poly_count_down ? (poly_count_down + 1.5) : target_vel
    target_vel = poly_count_down < 1 && distance_to_target < veh_wid ? 0 : target_vel

    # slow to zero when vehicle approaches stop sign
    if found_stop_sign
        dist = distance_to_stop_sign - 3
        @info("found stop sign is true, target vel: $target_vel, dist from stop sign: $dist")
        target_vel = min(target_vel, dist)
    else
        @info("stop sign not found")
    end

    target_vel = avoid_collision_speed < target_vel ? avoid_collision_speed : target_vel

    target_vel = target_vel < 0 ? 0 : target_vel
    return target_vel

end

#our model to avoid collision:
#my vehicle has a potential collision cone
#the cone angle is 30 degree (pi/6)
#the cone center is my vehicle center
# \-------------/
#  \----30-----/
#   \-degree--/
#    \-------/
#     \-----/
#      \|^|/
#       |+|
#when other vehicle is within my potential collision cone
#the distance is the key to stop/slow myself
#
function avoid_collision(localization_state_channel, 
    perception_state_channel,
    avoid_collision_channel,
    shutdown_channel)
    
    #vehicle_size = SVector(13.2, 5.7, 5.3)

    dt = 0.05    
    time_step = 1 # Int won't cause overflow. Steps in 4 hour = 4*60*60/dt = 288000

    deadlock_time_step = 0
    intersections = [[16.67, 16.67], [16.67,130.0], [-96.67, 16.67]] # centerpoint of the 3 intersections

    cos_half_angle = 0.965
    safe_distance = 24 

    println("starting collision thread")
    while true
        fetch(shutdown_channel) && break
        avoid_collision_speed = 10
        latest_localization_state = fetch(localization_state_channel)
        println("localization info in avoid collision: $latest_localization_state")


		# Rot_3D is Rotation Matrix in 3D
		# When vehicle rotates on 2D with θ,
		# Rot_3D = [cos(θ)  -sin(θ)  0;
		#           sin(θ)   cos(θ)  0;
		#               0         0  1]

		#Rot_3D = Rot_from_quat(latest_localization_state.ori)

        q = QuatRotation(latest_localization_state.orientation)
        Rot_3D = Matrix(q)

		veh_dir = [Rot_3D[1,1],Rot_3D[2,1]] #cos(θ), sin(θ)

        infront = 1

        me_to_intersection = Inf
        inter_idx = 0
        for i = 1:3
            dist = norm(intersections[i] - latest_localization_state.position[1:2])
            if dist < me_to_intersection
                me_to_intersection = dist
                inter_idx = i
            end
        end

        if me_to_intersection < 24
            cos_half_angle = 0.707 #90 degree cone
            safe_distance = 18
        else
            cos_half_angle = 0.965 #30 degree cone
            safe_distance = 24
        end

        safe_distance = deadlock_time_step > 24 ? 18 : safe_distance # set to closer than 30 at intersection and during deadlocks

		new_perception_list = fetch(perception_state_channel)
        println("perception info in avoid collision: $new_perception_list")
        count = length(new_perception_list)
        min_other_to_me = Inf
        min_other_to_inter = Inf

        if count >0
            for i=1:count
                one_perception = new_perception_list[i]
                other_to_inter = norm(one_perception.position[1:2]-intersections[inter_idx])

                displacement = one_perception.position[1:2] - latest_localization_state.position[1:2]
                distance = norm(displacement)
                @info("distance from other car = $distance")
                #infront is projection of a unit vector on my vehicle orientation
                # is the other car within 30 degrees of mine
                infront = dot(displacement, veh_dir)/distance
                #within cone means cos(15degree)=0.965
                #if the cosine is bigger, that means you are inside the cone
                if infront > cos_half_angle && distance < min_other_to_me
                    min_other_to_me = distance
                    min_other_to_inter = other_to_inter
                end
            end
        end

        
        min_dist = round(min_other_to_me,digits=3)

        println("minimum distance: $min_dist")

        avoid_collision_speed = min_other_to_me-(safe_distance * infront) #L=13.2
        if me_to_intersection < 24
            avoid_collision_speed = avoid_collision_speed > 2 ? 2 : avoid_collision_speed
            if min_other_to_inter < me_to_intersection
                avoid_collision_speed = avoid_collision_speed > 0.5 ? 0.5 : avoid_collision_speed
            end
        else
            avoid_collision_speed = avoid_collision_speed > 10 ? 10 : avoid_collision_speed
        end

        if avoid_collision_speed < 0.01
            deadlock_time_step = deadlock_time_step + 1
        else
            deadlock_time_step = 0
        end

        saved_speed = fetch(avoid_collision_channel)

        if abs(saved_speed - avoid_collision_speed)>0.05
            take!(avoid_collision_channel)
            put!(avoid_collision_channel, avoid_collision_speed)
        end

        sleep(dt)
        time_step=time_step+1
    end
end


function decision_making(localization_state_channel,
    perception_state_channel,
    target_segment_channel,
    avoid_collision_channel,
    shutdown_channel,
    map_segments,
    socket)
    ls = 2.0
    last_target_segment = 0
    log_file = open("decision_making_log.txt", "a")
    currTime = Dates.format(now(), "HH:MM:SS.s")
    println(log_file, currTime)

    poly = Polyline() #dummy polyline
    poly_count = 0
    poly_leaving = 0 # front wheel touch the end of this line
    best_next = 0
    max_signed_dist = 0.0
    signed_dist = 0.0
    target_location = [0.0, 0.0]

    # heuristic flags
    found_stop_sign = false
    stop_sign_location = [0.0, 0.0]

    while true
        fetch(shutdown_channel) && break
        if isready(target_segment_channel)
            target_segment = fetch(target_segment_channel)
        else
            target_segment = 0
        end
        println("target_segment = $target_segment")
        println("")
        if target_segment > 0
            avoid_collision_speed = fetch(avoid_collision_channel)
            @info("avoid_collision_speed = $avoid_collision_speed")

            latest_localization_state = fetch(localization_state_channel)

            vel = latest_localization_state.velocity

            veh_vel = vel[1:2]

            curr_vel = norm(veh_vel)

            pos = latest_localization_state.position
            veh_pos = pos[1:2]
            if target_segment != last_target_segment
                currTime = Dates.format(now(), "HH:MM:SS.s")
                println(log_file, currTime)
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
            ori = latest_localization_state.orientation
            vel = latest_localization_state.velocity
            a_vel = latest_localization_state.angular_velocity
            size = BOUNDING_BOX
            # Rot_3D is Rotation Matrix in 3D
            # When vehicle rotates on 2D with θ,
            # Rot_3D = [cos(θ)  -sin(θ)  0;
            #           sin(θ)   cos(θ)  0;
            #               0         0  1]
            q = QuatRotation(ori)
            Rot_3D = Matrix(q)
            veh_vel = vel[1:2]
            veh_dir = [Rot_3D[1, 1], Rot_3D[2, 1]] #cos(θ), sin(θ)
            veh_len = size[1] #vehicle Length
            veh_wid = size[2] #vehicle width
            rear_wl = veh_pos - 0.5 * veh_len * veh_dir
            front_end = veh_pos + 0.5 * veh_len * veh_dir
            distance_to_target = norm(target_location - veh_pos)
            distance_to_stop_sign = norm(stop_sign_location - front_end)

            curr_vel = norm(veh_vel)
            print("tgt=$target_segment")
            steering_angle = 0.0

            if found_stop_sign == true && curr_vel < 0.08
                found_stop_sign = false
            end

            if curr_vel > 0.0001
                len0 = curr_vel * ls
                min_diff = distance_to_target
                three_after = poly_leaving + 3
                three_after = three_after > poly_count ? poly_count : three_after
                best_next = 0
                #println("poly_leaving=$poly_leaving")
                #println("three_after=$three_after")
                for i = poly_leaving+1:three_after
                    #println("i=$i")
                    # this p2 is the same point as p1 of next poly segment
                    # we cannot use p1, because vehicle starts from p1 of first poly segment
                    try_point = poly.segments[i].p2 #here p2 is the same point as p1 of next poly segment

                    if poly.segments[i].stop == 1
                        found_stop_sign = true
                        @info("polyline coordinate = $try_point, i=$i")
                        stop_sign_location = try_point
                        #distance_to_stop_sign = norm(stop_sign_location-front_end) # need to recalc distance
                    end

                    try_dist = norm(try_point - rear_wl)
                    if try_dist < veh_len #front wheel touched poly line seg
                        poly_leaving = i
                        continue #too close
                    end
                    sign = signOfDot0(veh_dir, try_point - rear_wl)
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
                best_next = best_next > 0 ? best_next : poly_leaving + 1
                best_next = best_next > poly_count ? poly_count : best_next
                poly_next_seg = poly.segments[best_next]
                #println("poly_next_seg=$poly_next_seg")
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
                        println(log_file, "max_signed_dist=$max_signed_dist between lv=$leaving_road($leaving_part),to=$next_road($next_part)")
                    end
                else
                    print(",lv=0(0),to=$next_road($next_part)")
                end
                print(",s_d=$signed_dist, max_s_d=$max_signed_dist")
                next_point = poly_next_seg.p2
                distance_to_node = norm(next_point - rear_wl)
                cos_alpha = dot(veh_dir, next_point - rear_wl) / norm(next_point - rear_wl)
                cos_alpha = round(cos_alpha, digits=3) # three decimal place
                alpha = acos(cos_alpha)
                sin_alpha = sin(alpha)
                left_or_right = left_right(veh_dir, next_point - rear_wl)
                steering_angle = 0.75 * atan(2.0 * veh_len * sin_alpha * left_or_right, curr_vel * ls)
            end #if curr_vel > 0.0
            #latest_perception_state = fetch(perception_state_channel)  

            println()
            println("\nCURRENT VELOCITY: $curr_vel\n")
            target_vel = target_velocity(veh_pos, avoid_collision_speed, curr_vel, distance_to_target, found_stop_sign, distance_to_stop_sign, steering_angle, a_vel[3], veh_wid, poly_count, best_next, signed_dist, perception_state_channel)
            println()

            cmd = (steering_angle, target_vel, true)
            steering_degree = round(steering_angle * 180 / 3.14, digits=3)
            println(", str=$steering_degree, v=$curr_vel")
            serialize(socket, cmd)
        end #if target_segment > 0
        sleep(0.05)
    end#while true
    close(log_file)
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444; use_gt=false)

    # connect to simulation server & load map
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.city_map()

    msg = deserialize(socket) # Visualization info
    @info msg

    # initialize sensor channels
    gps_channel = Channel{VehicleSim.GPSMeasurement}(32)
    imu_channel = Channel{VehicleSim.IMUMeasurement}(32)
    cam_channel = Channel{VehicleSim.CameraMeasurement}(32)
    gt_channel = Channel{VehicleSim.GroundTruthMeasurement}(32)

    # initialize localization and perception channels
    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    # shared state channels
    target_segment_channel = Channel{Int}(1)
    ego_vehicle_id_channel = Channel{Int}(1)
    avoid_collision_channel = Channel{Float64}(1)
    put!(avoid_collision_channel, 10) #speed limit 

    # kill switch
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

        # populate map segment channel
        !received && continue
        target_map_segment = measurement_msg.target_segment
        old_target_segment = fetch(target_segment_channel)
        if target_map_segment ≠ old_target_segment
            take!(target_segment_channel)
            put!(target_segment_channel, target_map_segment)
        end

        # populate ego vehicle id channel no repeats
        ego_vehicle_id = measurement_msg.vehicle_id
        old_ego_vehicle_id = fetch(ego_vehicle_id_channel)
        if ego_vehicle_id ≠ old_ego_vehicle_id
            take!(ego_vehicle_id_channel)
            put!(ego_vehicle_id_channel, ego_vehicle_id)
        end

        # populate GPS, IMU, Camera Channels appropriately
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

    # choose to use ground truth or localization
    if use_gt
        errormonitor(@async process_gt(gt_channel,
            shutdown_channel,
            localization_state_channel,
            perception_state_channel,
            ego_vehicle_id_channel))
    else
        errormonitor(@async localize(gps_channel,
            imu_channel,
            localization_state_channel,
            shutdown_channel))

        errormonitor(@async perception(cam_channel,
            localization_state_channel,
            perception_state_channel,
            shutdown_channel))
    end

    errormonitor(@async avoid_collision(localization_state_channel,
        perception_state_channel,
        avoid_collision_channel,
        shutdown_channel
    ))
    println("ENTERED CLIENT DECISION MAKING LOOP")
    errormonitor(@async decision_making(localization_state_channel,
        perception_state_channel,
        target_segment_channel,
        avoid_collision_channel,
        shutdown_channel,
        map_segments,
        socket))
end

function shutdown_listener(shutdown_channel)
    info_string = "***************
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