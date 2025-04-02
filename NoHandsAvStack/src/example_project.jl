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

function get_polyline(map, my_location, target_segment)
    route = get_route(map, my_location, target_segment)
    points = [get_center(route[i], map, 80) for r =1:length(route)]
    Polyline(points)
end

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
            #latest_perception_state = fetch(perception_state_channel)
            # figure out what to do ... setup motion planning problem etc
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
