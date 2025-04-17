using JLD2
using StaticArrays, LinearAlgebra, Statistics
using Graphs
using Rotations

using CSV, DataFrames

# Code to write test data to CSV

output_dir = "./localization_error_logs"
isdir(output_dir) || mkdir(output_dir)

metrics = [
    "x_pos", "y_pos", "z_pos", 
    "yaw", 
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz", 
    "wx", "wy", "wz"
]
metric_buffers = Dict(m => DataFrame(time=Float64[], ground_truth=Float64[], estimated=Float64[]) for m in metrics)



# Load localization and vehicle modules
push!(LOAD_PATH, "/Users/neelashabhattacharjee/av/VehicleSim/src")
using .VehicleSim
include("../localization.jl")

##### READING FILES & ORGANIZING DATA

# Load messages
filename = "/Users/neelashabhattacharjee/av/NoHands/NoHandsAvStack/src/tests/full_track_buff.jld2"
messages = jldopen(filename, "r") do file
    read(file, "msg_buf")
end

println("Total messages: ", length(messages))

# Separate relevant measurements
gps_measurements = VehicleSim.GPSMeasurement[]
imu_measurements = VehicleSim.IMUMeasurement[]
gt_measurements = VehicleSim.GroundTruthMeasurement[]

for msg in messages
    for m in msg.measurements
        if m isa VehicleSim.GPSMeasurement
            push!(gps_measurements, m)
        elseif m isa VehicleSim.IMUMeasurement
            push!(imu_measurements, m)
        elseif m isa VehicleSim.GroundTruthMeasurement
            push!(gt_measurements, m)
        end
    end
end

##### GENERATING LOCALIZATION RESULTS

# Create channels
gps_channel = Channel{VehicleSim.GPSMeasurement}(0)
imu_channel = Channel{VehicleSim.IMUMeasurement}(0)
localization_state_channel = Channel{MyLocalizationType}(0)
shutdown_channel = Channel{Bool}(1)
put!(shutdown_channel, false)  # default value

# Launch localization loop
@async try
    localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
catch e
    println("Error in localize: ", e)
    display(stacktrace(catch_backtrace()))
end

# Feed GPS and IMU data asynchronously
@async begin
    for gps in gps_measurements
        put!(gps_channel, gps)
        sleep(0.01)
    end
    close(gps_channel)
end

@async begin
    for imu in imu_measurements
        put!(imu_channel, imu)
        sleep(0.01)
    end
    close(imu_channel)
end

# Collect localization results
localization_log = MyLocalizationType[]

@async begin
    println("Listening for localization output...")
    while true
        if isready(shutdown_channel) && take!(shutdown_channel)
            println("Shutting down localization listener")
            break
        end
        if isready(localization_state_channel)
            state = take!(localization_state_channel)
            push!(localization_log, state)
        else
            sleep(0.01)
        end
    end
end

# Allow localization loop to run, then shutdown
sleep(30)
put!(shutdown_channel, true)
sleep(2)

##### COMPARE RESULTS TO GROUND TRUTH

println("\n--- COMPARING RESULTS ---")
println("Localization states collected: ", length(localization_log))
println("Ground truth messages: ", length(gt_measurements))

println("🧪 Last localization timestamp: ", localization_log[end].time)
println("🧪 Last ground truth timestamp: ", gt_measurements[end].time)

function quaternion_error_deg(q1::SVector{4, Float64}, q2::SVector{4, Float64})
    dot_product = abs(dot(q1, q2))
    return 2 * acos(clamp(dot_product, -1.0, 1.0)) * (180 / π)
end

function angle_error(a, b)
    d = a - b
    return atan(sin(d), cos(d))  
end

global total_errors = 0

for state in localization_log
    time = state.time
    curr_time_range = (time - 0.01, time + 0.01)

    matching_gt = filter(m -> curr_time_range[1] <= m.time <= curr_time_range[2], gt_measurements)
    if isempty(matching_gt)
        continue
    end

    gt = matching_gt[1]

    pos_error = norm(state.position - gt.position)
    vel_error = norm(state.velocity - gt.velocity)
    ang_error = norm(state.angular_velocity - gt.angular_velocity)

    # Compute yaw error
    yaw_est = VehicleSim.extract_yaw_from_quaternion(state.quaternion)
    yaw_gt = VehicleSim.extract_yaw_from_quaternion(gt.orientation)
    yaw_error_rad = abs(angle_error(yaw_est, yaw_gt))
    yaw_error_deg = rad2deg(yaw_error_rad)

    # log values for plotting
    push!(metric_buffers["x_pos"], (state.time, gt.position[1], state.position[1]))
    push!(metric_buffers["y_pos"], (state.time, gt.position[2], state.position[2]))
    push!(metric_buffers["z_pos"], (state.time, gt.position[3], state.position[3]))

    yaw_est = VehicleSim.extract_yaw_from_quaternion(state.quaternion)
    yaw_gt = VehicleSim.extract_yaw_from_quaternion(gt.orientation)
    push!(metric_buffers["yaw"], (state.time, yaw_gt, yaw_est))

    push!(metric_buffers["qw"], (state.time, gt.orientation[1], state.quaternion[1]))
    push!(metric_buffers["qx"], (state.time, gt.orientation[2], state.quaternion[2]))
    push!(metric_buffers["qy"], (state.time, gt.orientation[3], state.quaternion[3]))
    push!(metric_buffers["qz"], (state.time, gt.orientation[4], state.quaternion[4]))


    push!(metric_buffers["vx"], (state.time, gt.velocity[1], state.velocity[1]))
    push!(metric_buffers["vy"], (state.time, gt.velocity[2], state.velocity[2]))
    push!(metric_buffers["vz"], (state.time, gt.velocity[3], state.velocity[3]))

    push!(metric_buffers["wx"], (state.time, gt.angular_velocity[1], state.angular_velocity[1]))
    push!(metric_buffers["wy"], (state.time, gt.angular_velocity[2], state.angular_velocity[2]))
    push!(metric_buffers["wz"], (state.time, gt.angular_velocity[3], state.angular_velocity[3]))


    #= if total_errors > 10
        println("⚠️  Too many errors detected, stopping further checks.")
        break
    end

    if (pos_error > 100 || vel_error > 100 || ang_error > 100|| yaw_error_deg > 70) 
        global total_errors = total_errors + 1
        println("⚠️  High error detected at t=$(round(state.time, digits=2))")
        println()

        print("------- POSITION ERROR -------\n")
        println("Ground Truth Pos: ", gt.position)
        println("Estimated Post: ", state.position)
        println("Pos Error: ", pos_error)
        println()

        print("------- VELOCITY ERROR -------\n")
        println("Ground Truth Vel: ", gt.velocity)
        println("Estimated Vel: ", state.velocity)
        println("Vel Error: ", vel_error)
        println()

        print("------- ANGULAR VELOCITY ERROR -------\n")
        println("Ground Truth Ang Vel: ", gt.angular_velocity)
        println("Estimated Ang Vel: ", state.angular_velocity)
        println("Ang Vel Error: ", ang_error)
        println()


        print("------- YAW ERROR -------\n")
        println("Ground Truth Yaw: ", yaw_gt)
        println("Estimated Yaw: ", yaw_est)
        println("Yaw Error (Radians): ", yaw_error_rad)
        println("Yaw Error (Degrees): ", yaw_error_deg)
        println()
    end =#
end

for (metric, df) in metric_buffers
    # compute average absolute error
    errors = abs.(df.ground_truth .- df.estimated)
    avg_error = mean(errors)

    # save to CSV (with only data points, no avg row)
    csv_path = joinpath(output_dir, "$(metric)_comparison.csv")
    CSV.write(csv_path, df)

    # write average error to separate .txt file
    error_path = joinpath(output_dir, "$(metric)_avg_error.txt")
    open(error_path, "w") do io
        write(io, "Average error for $metric: $avg_error\n")
    end
end


#println("\n-TOTAL ERRORS: ",total_errors, "\n")  



