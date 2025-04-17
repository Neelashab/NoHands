include("../src/example_project.jl")
using JLD2
using StaticArrays
using Graphs
using Rotations
using Statistics
using DataFrames
using CSV

"""
The first test for the perception module focuses on a moving vehicle moving towards ego (sationary).

To replicate this scenario and evaluate whether the NoHands perception function operates correctly, we will 
feed it data saved by the logger during the event. The test will parse the recorded data to extract 
camera and ground truth measurements for both vehicles.

Ground Truth Set Up:
(timestamp, vehicle_id, 
[position], [orientation/quaternion], [linear_velocity], [angular_velocity], [vehicle_size])

Camera Set Up:
(timestamp, camera_id, focal_length, pixel_length, image_width, image_height, [bounding_boxes])

"""

##### READING FILES & ORGANIZING DATA

filename = joinpath(@__DIR__, "driving_side_by_side.jld2")
jldopen(filename, "r") do file
buf = file["msg_buf"]
#println(buf[1])
#println(buf[2])

gt_moving_car = GroundTruthMeasurement[]
ego_cam = CameraMeasurement[]
gt_ego_stationary = GroundTruthMeasurement[]

for message in buf
    for m in message.measurements
        if m isa GroundTruthMeasurement
            if m.vehicle_id == 1
                push!(gt_ego_stationary, m)
            elseif m.vehicle_id == 2
                push!(gt_moving_car, m)
            end
        elseif m isa CameraMeasurement
            push!(ego_cam, m)  
        end
    end
end

#println(length(ego_cam))
#println(ego_cam[100])
#println(ego_cam[100].bounding_boxes)

##### GENERATING CHANNELS & PERCEPTION RESULTS
cam_channel = Channel{CameraMeasurement}(256)
gt_channel = Channel{GroundTruthMeasurement}(256)
perception_state_channel = Channel{MyPerceptionType}(100)

shutdown_channel = Channel{Bool}(1)
put!(shutdown_channel, false) 

# launch perception function
perception_log = MyPerceptionType[]

@async perception(cam_channel, gt_channel, perception_state_channel, shutdown_channel)

@async begin
    for m in ego_cam
        put!(cam_channel, m)
        sleep(0.01)  # simulate real-time data feed
    end
    close(cam_channel)  # signal that no more camera data will be sent
end

@async begin
    for m in gt_ego_stationary
        put!(gt_channel, m)
        sleep(0.01)  # simulate real-time data feed
    end
    close(gt_channel)  # signal that no more ground truth data will be sent
end


@async begin
    println("Listening for perception output...")
    while true
        if isready(shutdown_channel) && take!(shutdown_channel)
            println("Shutting down perception listener")
            break
        end
        if isready(perception_state_channel)
            result = take!(perception_state_channel)
            #println("Moving vehcile position [Estimate]: ", result.tracked_objs[1].pos)
            #println("Moving vehcile velocity [Estimate]: ", result.tracked_objs[1].vel)
            push!(perception_log, deepcopy(result))
        else
            sleep(0.1)
        end
    end
end

sleep(15)  
# Now send the shutdown signal to the perception listener
put!(shutdown_channel, true)
# Wait for listener to flush all perception outputs
sleep(2)  # Give some time for the shutdown to happen

println("\n--- COMPARING RESULTS ---")

time_gt = []
x_gt = []
y_gt = []
vel_gt_magnitude = []
time_est = []
x_est = []
y_est = []
vel_est_magnitude = []
heading_est = []
heading_gt = []
angular_vel_gt = []
angular_vel_est = []



pos_errors = []
vel_errors = []
orientation_errors = []
angular_velocity_errors = []

for state in perception_log

    # Choose a window to match perception to GT (e.g., ±0.25 sec)
    time = state.time
    curr_time_range = (time - 0.0001, time + 0.0001)

    # Match against GT of the stationary vehicle
    matching_gt = filter(m -> curr_time_range[1] <= m.time <= curr_time_range[2], gt_moving_car)

    if isempty(matching_gt) || isempty(state.tracked_objs)
        continue
    end


    # Use the first GT match and first tracked object for this simple test
    m = matching_gt[1]
    obj = state.tracked_objs[1]

    est_pos = obj.pos
    est_vel = obj.vel
    est_orientation = obj.orientation
    est_angular_velocity = obj.angular_velocity

    actual_pos = m.position
    actual_vel = m.velocity
    actual_orientation = m.orientation
    actual_angular_velocity = m.angular_velocity

    push!(time_gt, m.time)
    push!(x_gt, actual_pos[1])
    push!(y_gt, actual_pos[2])
    push!(time_est, state.time)
    push!(x_est, est_pos[1])
    push!(y_est, est_pos[2])
    push!(vel_gt_magnitude, norm(actual_vel))
    push!(vel_est_magnitude, norm(est_vel))
    push!(heading_gt, VehicleSim.extract_yaw_from_quaternion(actual_orientation))
    push!(heading_est, VehicleSim.extract_yaw_from_quaternion(est_orientation))
    push!(angular_vel_gt, norm(actual_angular_velocity))
    push!(angular_vel_est, norm(est_angular_velocity))


    # Calculate errors
    pos_error = norm(est_pos - actual_pos)
    vel_error = norm(est_vel - actual_vel)
    orientation_error = mod(VehicleSim.extract_yaw_from_quaternion(est_orientation) - VehicleSim.extract_yaw_from_quaternion(actual_orientation) + π, 2π) - π
    angular_velocity_error = norm(est_angular_velocity - actual_angular_velocity)
    
    push!(pos_errors, pos_error)
    push!(vel_errors, vel_error)
    push!(orientation_errors, orientation_error)
    push!(angular_velocity_errors, angular_velocity_error)

    println("Perception Time = ", time)
    println("  GT Pos:      ", actual_pos)
    println("  Est Pos:      ", est_pos)
    println("  GT Vel:      ", actual_vel)
    println("  Est Vel:      ", est_vel)
   #println("  Pos Error:    ", pos_error)
    #println("  Vel Error:    ", vel_error)
   # println("  Orientation error:", round(orientation_error, digits=3))
    #println("  Angular vel error:", round(angular_velocity_error, digits=3))
    println()
end

df_results = DataFrame(
    time_gt_data = time_gt,
    x_gt_data = x_gt,
    y_gt_data = y_gt,
    time_est_data = time_est,
    x_est_data = x_est,
    y_est_data = y_est,
    vel_gt_data = vel_gt_magnitude,
    vel_est_data = vel_est_magnitude,
    heading_gt_data = heading_gt,
    heading_est_data = heading_est,
    angular_vel_gt_data = angular_vel_gt,
    angular_vel_est_data = angular_vel_est
)

CSV.write("perception_results03.csv", df_results)

println("##### TEST 02 RESULTS")

println("\nHow accurate was the perecption function in identifying the stationary vehicle?")
println("\nResult Stats")
println("   Number of perception results: ", length(perception_log))
println("   Average position error: ", mean(pos_errors))
println("   Median velocity error: ", median(vel_errors))
println("   Median orientation error: ", median(orientation_errors))
println("   Median angular velocity error: ", median(angular_velocity_errors))
println("   Done comparing results")

end

