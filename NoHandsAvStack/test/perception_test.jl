#using NoHandsAvStack
#using VehicleSim
include("../src/example_project.jl")
using JLD2
using StaticArrays
using Graphs
using Rotations
using Statistics




"""
The first test for the perception module focuses on the following scenario: two vehicles are present—one 
stationary and one moving toward the stationary vehicle. The moving vehicle is the ego vehicle in this case, 
but since ground truth data includes all objects in the environment, we also have information about the 
stationary vehicle.

To replicate this scenario and evaluate whether the NoHands perception function operates correctly, we will 
feed it data saved by the logger during the event. The test will parse the recorded data to extract 
camera and ground truth measurements for both vehicles.

Moving Vehicle (EGO since it recorded the measurements):
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

Stationary Vehicle:
* We are testing whether the vehicle can detect that another vehicle is approaching it from behind.
* The stationary vehicle will be fed its own ground truth measurements for localization.
* It will also receive its own camera measurements.
* By analyzing these inputs, we can evaluate whether the perception function accurately estimates the approaching (moving) vehicle’s location and velocity on the map.
* We will use the moving vehicle’s ground truth data to assess the perception accuracy from the stationary vehicle’s perspective.

Need to:
### collect ground truth of vehicle 1
### collect ground truth of vehicle 2
### collect camera measurements of vehicle 1
### collect camera measurements of vehicle 2

### create "channels" to feed to perception function

### compare GTs to perecption function results 


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
cam_time_test = Float64[]
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


# this part works
#print(cam_time_test)
#print(cam_moving)
#print(gt_stationary)


##### GENERATING PERCEPTION RESULTS

# creating channels

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
        sleep(0.01)  # Simulate real-time data feed
    end
    close(cam_channel)  # Signal that no more camera data will be sent
end

@async begin
    for m in gt_moving
        put!(gt_channel, m)
        sleep(0.01)  # Simulate real-time data feed
    end
    close(gt_channel)  # Signal that no more ground truth data will be sent
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


#sleep(10)  # Adjust this duration as needed
#println("Is perception_state_channel ready? ", isready(perception_state_channel))
#take!(shutdown_channel)        # Remove the old value
#put!(shutdown_channel, true)   # Send shutdown signal
#print("done")

sleep(5)  # Adjust as needed

# Now send the shutdown signal to the perception listener
put!(shutdown_channel, true)

# Wait for listener to flush all perception outputs
sleep(2)  # Give some time for the shutdown to happen

println("\n--- COMPARING RESULTS ---")
println("Number of perception states: ", length(perception_log))
println("Number of GT stationary measurements: ", length(gt_stationary))
println("Some prelimnary notes:")
println("Number of perception states: ", length(perception_log))
println("Number of GT stationary measurements: ", length(gt_stationary))
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

    est_pos = obj.pos
    est_vel = obj.vel
    est_orientation = obj.orientation
    est_angular_velocity = obj.angular_velocity

    actual_pos = m.position
    actual_vel = m.velocity
    actual_orientation = m.orientation
    actual_angular_velocity = m.angular_velocity

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
    #println("  GT Pos:      ", actual_pos)
    #println("  Est Pos:      ", est_pos)
    println("  Pos Error:    ", pos_error)
    println("  Vel Error:    ", vel_error)
    println("  Orientation error:", round(orientation_error, digits=3))
    println("  Angular vel error:", round(angular_velocity_error, digits=3))
    println()
end
println("##### Summary")
println("Average position error: ", mean(pos_errors))
println("Final position error: ", pos_errors[end])
println("Median velocity error: ", median(vel_errors))
println("Median orientation error: ", median(orientation_errors))
println("Median angular velocity error: ", median(angular_velocity_errors))
println("Done comparing results")

end
