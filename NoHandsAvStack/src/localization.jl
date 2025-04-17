using Graphs
using Rotations
using StaticArrays, LinearAlgebra, Statistics


#include("measurements.jl")

# constants
BOUNDING_BOX = SVector(13.2, 5.7, 5.3)

struct MyLocalizationType
    time::Float64
    position::SVector{3,Float64} # position of center of vehicle
    quaternion::SVector{4,Float64} # quaternion [w, x, y, z]
    velocity::SVector{3,Float64} # [vx, vy, vz]
    angular_velocity::SVector{3,Float64} # [wx, wy, wz]
end

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
    0.0,    # qw
    0.0,    # qx
    0.0,    # qy
    0.0,    # qz
    0.05,   # vx
    0.05,   # vy
    0.05,   # vz (hardcoded)
    0.1,    # wx
    0.1,    # wy
    0.1     # wz
])


    R_gps = Diagonal([1.0, 1.0, 0.01]) # GPS noise
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
        state = SVector{13, Float64}(
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
