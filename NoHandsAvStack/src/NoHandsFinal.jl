using Graphs
using Rotations
using StaticArrays, LinearAlgebra, Statistics


#include("measurements.jl")

# constants
BOUNDING_BOX = SVector(13.2, 5.7, 5.3)

""" STRUCTURES """

struct MyLocalizationType
    time::Float64
    position::SVector{3,Float64} # position of center of vehicle
    quaternion::SVector{4,Float64} # quaternion [w, x, y, z]
    velocity::SVector{3,Float64} # [vx, vy, vz]
    angular_velocity::SVector{3,Float64} # [wx, wy, wz]
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
        init_x, init_y, 0.0, # position
        init_quat[1], init_quat[2], init_quat[3], init_quat[4], # quaternion
        0.0, 0.0, 0.0, # positional velocity
        0.0, 0.0, 0.0 # angular velocity
    ]


    Σ = 0.1 * I(13) # initial covariance matrix

    Q = Diagonal([
    0.01,   # x
    0.01,   # y
    0.0,    # z (hardcoded)
    0.2,    # qw
    0.2,    # qx
    0.2,    # qy
    0.2,    # qz
    0.05,   # vx
    0.05,   # vy
    0.05,   # vz
    0.1,    # wx
    0.1,    # wy
    0.1     # wz
])


    R_gps = Diagonal([1.0, 1.0, 0.1]) # GPS noise
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

        # Publish relevant localization state
        localization_state = MyLocalizationType(
            now,
            [state[1], state[2], 2.6455622444987412], # hardcode Z value as 2.5
            normalize(state[4:7]),
            [state[8], state[9], 0.0],
            state[11:13]
        )

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end

        put!(localization_state_channel, localization_state)
    end
end


"""PERCEPTION"""


function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)

    next_id = 1
    tracks = TrackedObject[]

    try
        while true

            # CHANNELS
            # collecting information from the channels
            fresh_cam_meas = []
            fresh_localization_meas = []
    
            if isready(shutdown_channel) && take!(shutdown_channel)
                break
            end

            while isready(cam_meas_channel)
                #println("Received camera message")
                meas = take!(cam_meas_channel)
                push!(fresh_cam_meas, meas)
            end
     
            # this is to be replaced with localization
            while isready(localization_state_channel)
                #println("Received GT message")
                meas = take!(localization_state_channel)
                push!(fresh_localization_meas, meas)
            end
    
            if isempty(fresh_cam_meas) || isempty(fresh_localization_meas)
                if !isopen(cam_meas_channel) && !isopen(localization_state_channel)
                    break
                end
                # take a break ; no new measurements
                sleep(0.05)
                continue
            end
    
            latest_cam = last(fresh_cam_meas)
            latest_localization = last(fresh_localization_meas)

           
            if isempty(latest_cam.bounding_boxes)
                # no bouding boxes yet, take a break
                sleep(0.05)
                continue
            end

            # UNPACK INFROMATION FROM CHANNELS
    
            # unpack GT state (will become localization)
            ego_position = latest_localization.position
            ego_quaternion = latest_localization.quaternion
    
            # create necessary transformation matrices
            camera_id = latest_cam.camera_id
            T_body_from_cam = VehicleSim.get_cam_transform(camera_id)
            T_cam_camrot = VehicleSim.get_rotated_camera_transform()
            T_body_camrot = VehicleSim.multiply_transforms(T_body_from_cam, T_cam_camrot)
            T_world_body = VehicleSim.get_body_transform(ego_quaternion, ego_position)
            T_world_camrot = VehicleSim.multiply_transforms(T_world_body, T_body_camrot)
            
            # unpack camera state
            focal_len = latest_cam.focal_length
            px_len = latest_cam.pixel_length
            img_w = latest_cam.image_width + 1
            img_h = latest_cam.image_height + 1
            t = latest_cam.time
            vehicle_size = SVector(13.2, 5.7, 5.3)
    
            # for each bounding box (represnting other vehciles)
            # create a location estimation from camera information
            for box in latest_cam.bounding_boxes

                u = (box[1] + box[3]) / 2
                v = (box[2] + box[4]) / 2
                x = (u - img_w / 2) * px_len
                y = (v - img_h / 2) * px_len

            
                pitch = 0.02 - atan(y/focal_len) # 0.02 is the built in oiriginal tilt, add more angle 
                pitch_fixed = (pi/2 - pitch) 
                z = (2.4)/cos(pitch_fixed) # both camera 1 & 2 have a height of 2.4 above the base/center
            
                
                point_from_cam = SVector{4, Float64}(x, y, z, 1.0)
                pos = T_world_camrot * point_from_cam
                pos[3] = vehicle_size[3] / 2
    
                
                matched = false
                for (i, track) in enumerate(tracks)
                    dist = norm(track.pos[1:2] - pos[1:2])

                    # if its within 15 meters, its porbably the same vehicle
                    if dist < 15
                       
                        Δt = t - track.time
                        updated_track = ekf(track, SVector{3, Float64}(pos[1:3]), Δt)
                        tracks[i] = updated_track
                        matched = true
                        break

                    end
                end
    
                # if no exsisting track was matched, create a new one
                if !matched
    
                    initial_orientation = track_orientation_estimate(
                        ego_position, ego_quaternion, SVector{3, Float64}(pos[1:3]))
    
                    initial_velocity = SVector(0.0, 0.0, 0.0)
                    initial_angular_velocity = SVector(0.0, 0.0, 0.0)
    
                    cov_diag = [
                        5.0, 50.0, 1.0,         # y-pos tends to be our most uncertain 
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
    println("Perception finished.")
    
end


function track_orientation_estimate(ego_pos::SVector{3, Float64}, ego_quat::SVector{4, Float64}, obj_pos::SVector{3, Float64})

    #TODO: amina has to make a quick update to this

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
    x = vcat(track.pos,                     # 1:3 position
             track.orientation,             # 4:7 orientation quaternion
             track.vel,                     # 8:10 velocity
             track.angular_velocity)        # 11:13 angular velocity

    # setting up necessary noise matrices
    R = Diagonal(@SVector [1.0, 15.0, 1.0])     # we trust x and z but not y 
    
 
   
    Q = I(13) * 10 
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
