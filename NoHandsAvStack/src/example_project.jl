using Graphs
using Rotations

struct MyLocalizationType
    vehicle_id::Int
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    size::SVector{3, Float64} # length, width, height of 3d bounding box centered at (position/orientation)
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

struct StandardSegment <: PolylineSegment
    p1::SVector{2, Float64}
    p2::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
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
            seg = StandardSegment(points[i], points[i+1],roads[i],parts[i], stops[i])
            push!(segments, seg)
        end
        new(segments)
    end
    function Polyline(points...)
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

function localize(
        gps_channel, 
        imu_channel, 
        localization_state_channel, 
        shutdown_channel)
    # Set up algorithm / initialize variables
    while true
        sleep(0.001)

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
        sleep(0.001)
        
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
    min_x <= pos[1] <= max_x && min_y <= pos[2] <= max_y
end

function get_center(seg_id, map_segments, loading_id)
    seg = map_segments[seg_id]
    i = seg_id == loading_id ? 2 : 1
    A = seg.lane_boundaries[i].pt_a
    B = seg.lane_boundaries[i].pt_b
    C = seg.lane_boundaries[i+1].pt_a
    D = seg.lane_boundaries[i+1].pt_b
    MVector((A + B + C + D)/4)
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
        for j=1:no_child
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
    for i=1:no_arc
        add_edge!(graph, node1[i], node2[i])
    end

    distmx = Inf*ones(no_node, no_node)
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
    MVector((A + C)/2)
end

"""
add mid point in two cases:
1. curved lane
convert mid point on chord to mid point on arc
Assuming only 90° turns for now
center calculation is copied code from map.jl
2. long lane
"""
function get_middel_point(seg)
    # io = open("get_middel_point.txt", "a")
    # println(io, "seg=$seg")
    A = seg.lane_boundaries[1].pt_a
    B = seg.lane_boundaries[1].pt_b
    C = seg.lane_boundaries[2].pt_a
    D = seg.lane_boundaries[2].pt_b
    # println(io, "A=$A")
    # println(io, "B=$B")
    # println(io, "C=$C")
    # println(io, "D=$D")
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
    if curved1 && curved2
        rad1 = 1.0 / abs(curvature1)
        rad2 = 1.0 / abs(curvature2)
        rad = (rad1+rad2)/2
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
        direction_from_center = delta_to_center/norm(delta_to_center)
        # println(io, "direction_from_center=$direction_from_center")
        vector_from_center = rad*direction_from_center
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

        add_mid_point, mid_point = get_middel_point(seg)
        if add_mid_point
            push!(points, mid_point)
            push!(roads, route[r])
            push!(parts, 2)
            
            push!(stops, 0)
        end

        push!(stops, has_stop_sign(seg))

    end

    log_route(route, roads, parts, stops, points)
    poly = Polyline(points, roads, stops, parts)
    return poly
end

function has_stop_sign(seg)
    for i=1:length(seg.lane_types)
        if seg.lane_types[i] == stop_sign
            return 1
        end
    end

    return 0
end

function target_velocity(current_velocity, 
        distance_to_target,
        found_stop_sign, 
        distance_to_stop_sign,
        steering_angle,
        angular_velocity,
        veh_wid, 
        poly_count, 
        best_next, signed_dist; speed_limit=4)
    # abs_dist = abs(signed_dist)
    # #try to increase speed if it's not off track
    # increase = abs_dist < 3.0 ? 0.5 : 0.2
    # target_vel = abs_dist < 5.0 ? current_velocity + increase : current_velocity
    # target_vel = abs_dist > veh_wid ? target_vel / 2 : target_vel
    # target_vel = target_vel < 0.5 ? 0.5 : target_vel
    # #adjust speed limit by the angular velocity and steering angle
    # angular_effect = abs(angular_velocity)+abs(steering_angle)
    # adjusted_limit = angular_effect > pi/2 ? 1.0 : (1.0+(speed_limit-1.0) * (1-2*angular_effect/pi))
    # #adjust speed limit in the beginning 
    # adjusted_limit = (best_next < 5 && angular_effect > 0.001) ? 1.5 : adjusted_limit
    # target_vel = target_vel > adjusted_limit ? adjusted_limit : target_vel
    
    target_vel = current_velocity + 0.5
    angular_effect = abs(angular_velocity)+abs(steering_angle)
    adjusted_limit = angular_effect > pi/2 ? 1.0 : (1.0+(speed_limit-1.0) * (1-2*angular_effect/pi))
    target_vel = target_vel > adjusted_limit ? adjusted_limit : target_vel
    
    #slow down when vehicle approaches the target
    poly_count_down = poly_count - best_next
    target_vel = poly_count_down < 2 && target_vel > poly_count_down ? (poly_count_down+1.5) : target_vel
    target_vel = poly_count_down < 1 && distance_to_target < veh_wid ? 0 : target_vel

    # slow to zero when vehicle approaches stop sign
    if found_stop_sign
        target_vel = min(target_vel, distance_to_stop_sign - 3)
    end 

    target_vel = target_vel < 0 ? 0 : target_vel

end

function decision_making(localization_state_channel, 
        perception_state_channel, 
        target_segment_channel,
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
    target_location = [0.0,0.0]

    # heuristic flags
    found_stop_sign = false
    stop_sign_location = [0.0,0.0]

    while true
        fetch(shutdown_channel) && break
        target_segment = fetch(target_segment_channel)
        if target_segment > 0
            latest_localization_state = fetch(localization_state_channel)
            pos = latest_localization_state.position
            veh_pos = pos[1:2]
            if target_segment!=last_target_segment
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
            size = latest_localization_state.size
            # Rot_3D is Rotation Matrix in 3D
            # When vehicle rotates on 2D with θ,
            # Rot_3D = [cos(θ)  -sin(θ)  0;
            #           sin(θ)   cos(θ)  0;
            #               0         0  1]
            q = QuatRotation(ori)
            Rot_3D = Matrix(q)
            veh_vel = vel[1:2]
            veh_dir = [Rot_3D[1,1],Rot_3D[2,1]] #cos(θ), sin(θ)
            veh_len = size[1] #vehicle Length
            veh_wid = size[2] #vehicle width
            rear_wl = veh_pos - 0.5 * veh_len * veh_dir 
            front_end = veh_pos + 0.5 * veh_len * veh_dir 
            distance_to_target = norm(target_location-veh_pos)
            distance_to_stop_sign = norm(stop_sign_location-front_end)

            curr_vel = norm(veh_vel)
            print("tgt=$target_segment")
            steering_angle = 0.0
            if curr_vel > 0.00001
                len0 = curr_vel * ls
                min_diff = distance_to_target
                three_after = poly_leaving + 3
                three_after = three_after > poly_count ? poly_count : three_after
                best_next = 0
                #println("poly_leaving=$poly_leaving")
                #println("three_after=$three_after")
                for i = poly_leaving+1 : three_after
                    #println("i=$i")
                    # this p2 is the same point as p1 of next poly segment
                    # we cannot use p1, because vehicle starts from p1 of first poly segment
                    try_point = poly.segments[i].p2 #here p2 is the same point as p1 of next poly segment
                    
                    if poly.segments[i].stop == 1
                        found_stop_sign = true
                        stop_sign_location = try_point
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
                best_next = best_next > 0 ? best_next : poly_leaving+1
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
                cos_alpha = dot(veh_dir, next_point - rear_wl)/norm(next_point-rear_wl)
                cos_alpha = round(cos_alpha, digits=3) # three decimal place
                alpha = acos(cos_alpha)
                sin_alpha = sin(alpha)
                left_or_right = left_right(veh_dir, next_point - rear_wl)
                steering_angle = 0.75 * atan(2.0*veh_len*sin_alpha*left_or_right, curr_vel*ls)
            end #if curr_vel > 0.0
            #latest_perception_state = fetch(perception_state_channel)  
            if found_stop_sign == true && curr_vel == 0
                found_stop_sign = false
            end

            target_vel = target_velocity(curr_vel, distance_to_target, found_stop_sign, distance_to_stop_sign, steering_angle, a_vel[3],    veh_wid, poly_count, best_next, signed_dist)


            cmd = (steering_angle, target_vel, true)
            steering_degree = round(steering_angle * 180 / 3.14, digits=3)
            println(", str=$steering_degree, v=$curr_vel")
            serialize(socket, cmd)
        end #if target_segment > 0
        sleep(0.05)
    end#while true
    close(log_file)
end#function def

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


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

    errormonitor(@async decision_making(localization_state_channel, 
                           perception_state_channel, 
                           target_segment_channel, 
                           shutdown_channel,
                           map_segments, 
                           socket))
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
