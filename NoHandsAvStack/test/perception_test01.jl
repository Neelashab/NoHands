using Infiltrator
include("../src/example_project.jl")
using JLD2
using StaticArrays
using Graphs
using Rotations
using Statistics



"""
The first test for the perception module focuses on the ego moving towards a sationary vehicle.

To replicate this scenario and evaluate whether the NoHands perception function operates correctly, we will 
feed it data saved by the logger during the event. The test will parse the recorded data to extract 
camera and ground truth measurements for both vehicles.

Moving Vehicle (EGO):
* We are testing whether the perception function can recognize the presence and location of the 
    stationary vehicle.
* The ego vehicle will be fed its own ground truth measurements for localization.
* It will also receive its own camera measurements.
* By combining the ego’s ground truth and camera data, we can evaluate whether the 
    perception function correctly places the stationary vehicle on the map.
* We will use the stationary vehicle’s ground truth as the reference to assess the accuracy 
    of the ego vehicle's perception.
* Essentially, we are testing whether the perception function, given perfect localization and camera 
    measurements, can accurately perceive the location of the stationary vehicle on the map.


Ground Truth Set Up:
(timestamp, vehicle_id, 
[position], [orientation/quaternion], [linear_velocity], [angular_velocity], [vehicle_size])

Camera Set Up:
(timestamp, camera_id, focal_length, pixel_length, image_width, image_height, [bounding_boxes])

"""

##### READING FILES & ORGANIZING DATA

filename = joinpath(@__DIR__, "01ego_is_moving_towards_stationary_vehicle.jld2")
jldopen(filename, "r") do file
buf = file["msg_buf"]

gt_moving = GroundTruthMeasurement[]
cam_moving = CameraMeasurement[]
gt_stationary = GroundTruthMeasurement[]


# separate measurements by vehicle and type
for message in buf
    for m in message.measurements
        if m isa GroundTruthMeasurement
            if m.vehicle_id == 1
                push!(gt_moving, m)
            elseif m.vehicle_id == 2
                push!(gt_stationary, m)
            end
        elseif m isa CameraMeasurement
            push!(cam_moving, m)  
        end
    end
end


##### GENERATING PERCEPTION RESULTS


cam_channel = Channel{CameraMeasurement}(32)
gt_channel = Channel{GroundTruthMeasurement}(32)
perception_state_channel = Channel{MyPerceptionType}(80)

shutdown_channel = Channel{Bool}(1)
put!(shutdown_channel, false) 

# launch perception function
perception_log = MyPerceptionType[]

# setting up channels
@async perception(cam_channel, gt_channel, perception_state_channel, shutdown_channel)

@async begin
    for m in cam_moving
        put!(cam_channel, m)
        sleep(0.01)  # simulate real-time data feed
    end
    close(cam_channel)  # signal that no more camera data will be sent
end

@async begin
    for m in gt_moving
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
            push!(perception_log, deepcopy(result))
        else
            sleep(0.1)
        end
    end
end


sleep(5)  # Adjust as needed

# Now send the shutdown signal to the perception listener
put!(shutdown_channel, true)

# Wait for listener to flush all perception outputs
sleep(2)  # Give some time for the shutdown to happen

println("\n--- COMPARING RESULTS ---")

#= println("First perception timestamp: ", perception_log[1].time)
println("First ground truth timestamp: ", gt_stationary[1].time)
println("First perception pos: ", perception_log[1].tracked_objs[1].pos)
println("Last perception pos: ", perception_log[40].tracked_objs[1].pos) =#

pos_errors = []
vel_errors = []
orientation_errors = []
angular_velocity_errors = []

for state in perception_log

    # Choose a window to match perception to GT (e.g., ±0.25 sec)
    time = state.time
    curr_time_range = (time - 0.0001, time + 0.0001)

    # Match against GT of the stationary vehicle
    matching_gt = filter(m -> curr_time_range[1] <= m.time <= curr_time_range[2], gt_stationary)

    if isempty(matching_gt) || isempty(state.tracked_objs)
        continue
    end

    # Use the first GT match and first tracked object for this simple test
    m = matching_gt[1]
    obj = state.tracked_objs[1]

    # Calculate errors
    pos_error = norm(obj.pos - m.position)
    vel_error = norm(obj.vel - m.velocity)
    orientation_error = mod(VehicleSim.extract_yaw_from_quaternion(obj.orientation) - VehicleSim.extract_yaw_from_quaternion(m.orientation) + π, 2π) - π
    angular_velocity_error = norm(obj.angular_velocity - m.angular_velocity)
    
    push!(pos_errors, pos_error)
    push!(vel_errors, vel_error)
    push!(orientation_errors, orientation_error)
    push!(angular_velocity_errors, angular_velocity_error)

    println("Perception Time = ", time)
    println("  GT Pos:      ", m.position)
    println("  Est Pos:      ", obj.pos)
    println("  Pos Error:    ", pos_error)
    println("  Vel Error:    ", vel_error)
    println("  Orientation error:", round(orientation_error, digits=3))
    println("  Angular vel error:", round(angular_velocity_error, digits=3))
    println()
end
println("##### TEST 01 RESULTS")

println("\nHow accurate was the perecption function in identifying the stationary vehicle?")
println("   Stationary Vehicle's true position: ", gt_stationary[1].position)
println("   First position estimate: ", perception_log[1].tracked_objs[1].pos)
println("   Final position estimate: ", perception_log[end].tracked_objs[1].pos)

println("\nResult Stats")
println("   Average position error: ", mean(pos_errors))
println("   Median velocity error: ", median(vel_errors))
println("   Median orientation error: ", median(orientation_errors))
println("   Median angular velocity error: ", median(angular_velocity_errors))
println("   Done comparing results")

end
