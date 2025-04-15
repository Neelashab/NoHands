include("../src/example_project.jl")
using JLD2
using StaticArrays
using Graphs
using Rotations
using Statistics

"""
The first test for the perception module focuses on a ego moving vehcile with two stationary cars.

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
filename = joinpath(@__DIR__, "message_buff.jld2")
jldopen(filename, "r") do file
buf = file["msg_buf"]


ego_car_gt = GroundTruthMeasurement[]
ego_cam = CameraMeasurement[]
stationay1_gt = GroundTruthMeasurement[]
stationay2_gt = GroundTruthMeasurement[]

tester = 20

for message in buf
    for m in message.measurements
        if m isa GroundTruthMeasurement
            

            #println("Vehicle ID: ", m.vehicle_id)
            #println(m)
            #println("\n\n")

            if m.vehicle_id == 3
                push!(ego_car_gt, m)
            elseif m.vehicle_id == 2
                push!(stationay2_gt, m)
            elseif m.vehicle_id == 1
                push!(stationay1_gt, m)
            end
        elseif m isa CameraMeasurement
            push!(ego_cam, m)  
        end
    end

end

println("Number of camera measurements: ", length(ego_cam))
println("Number of ego ground truth measurements: ", length(ego_car_gt))
println("Number of stationary vehicle 1 measurements: ", length(stationay1_gt))
println("Number of stationary vehicle 2 measurements: ", length(stationay2_gt))

##### GENERATING PERCEPTION RESULTS
cam_channel = Channel{CameraMeasurement}(128)
gt_channel = Channel{GroundTruthMeasurement}(128)
perception_state_channel = Channel{MyPerceptionType}(256)

perception_log = []

shutdown_channel = Channel{Bool}(1)
put!(shutdown_channel, false) 

@async perception(cam_channel, gt_channel, perception_state_channel, shutdown_channel)

@async begin
    for m in ego_cam
        put!(cam_channel, m)
        sleep(0.01)  # simulate real-time data feed
    end
    close(cam_channel)  # signal that no more camera data will be sent
end

@async begin
    for m in ego_car_gt
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
            println("Current State measures this many bounding boxes:", length(result.tracked_objs))
            #print("\n\n")
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



perception1 = []
perception2 = []

testing = perception_log[end]
for obj in testing.tracked_objs
    println("ID: ", obj.id)
    println("Position: ", obj.pos)
    println("Velocity: ", obj.vel)
    println("Bounding Box: ", obj.bounding_box)
    println("Orientation: ", obj.orientation)
    println("Size: ", obj.size)
    println("Camera ID: ", obj.camera_id)
    println("\n")
end
println("bounding boxes:", testing.tracked_objs)

for state in perception_log
    if state.tracked_objs[1].id == 1
        push!(perception1, state)
    
    elseif state.tracked_objs[1].id == 2
        push!(perception2, state)    

    end 
end

println("##### TEST 03 RESULTS")

println("\nHow accurate was the perecption function in identifying the stationary vehicles?")
println("   Stationary Vehicle1's true position: ", stationay1_gt[1].position)
println("   Stationary Vehicle2's true position: ", stationay2_gt[1].position)
println("   First position estimate of vehicle a: ", perception2[1].tracked_objs[1].pos)
println("   Final position estimate of vehicle a: ", perception2[end].tracked_objs[1].pos)
println("   First position estimate of vehicle b: ", perception2[1].tracked_objs[1].pos)
println("   Final position estimate of vehcile b: ", perception2[end].tracked_objs[1].pos)


end



    
       

