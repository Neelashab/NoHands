"""
Decision making is the function where the car will decide whether to start, move forward, stop, or change direction (left, right). 

We will recieve a target location from the target_road_segment_id. We should define a polyline and connect the line, creating a trajectory. The polyline may have some straight lines and some curves. 

Define the bandwidth ??

This trajectory will power the pure pursuit controller. We will attempt to match the current path of the vehicle to the given path with minimal error using the control law.

We will define a timestep/look ahead distance that will move the car forward X steps at a time. You will fetch the information at each timestep to make a decision about how to move forward.

The car will recieve information from latest_localization_state and latest_perception_state. latest_perception_state will help us understand where the car is in given input of the map. 

How to I change the trajectory from a polyline to a combination of straight and curved lines?

Avoid obstacles using heuristics. 
See stop sign = stop.
See stationary obstacle = stop
See other vehicles = stop

Do I have to account for merging lanes? (halfspace representation)

For the heuristics, how do I implement them?
What exactly is given by localization_state_channel?
What exactly is given by perception_state_channel?

"""
# --- construct polyline --- #

# standard segment has a starting and an ending point that are both finite
# TO DO: add curvature attribute
struct StandardSegment <: PolylineSegment
    p1::SVector{2, Float64}
    p2::SVector{2, Float64}
    tangent::SVector{2, Float64}
    normal::SVector{2, Float64}
    function StandardSegment(p1, p2)
        tangent = p2 - p1
        tangent ./= norm(tangent)
        normal = perp(tangent)
        new(p1, p2, tangent, normal) 
    end
end

struct Polyline
    segments::Vector{PolylineSegment} # array-like collection of segments
    function Polyline(points)
        segments = Vector{PolylineSegment}() # initialize segments
        N = length(points)
        @assert N ≥ 2 # must have at least 2 points
        for i = 1:(N-1)
            seg = StandardSegment(points[i], points[i+1]) 
            push!(segments, seg)
        end
        push!(segments, terminal_ray)
        new(segments) # segments is an array of the points, tan, and norm of each segment
    end
    function Polyline(points...)
        Polyline(points)
    end
end

"""
TODO compute the signed distance from POINT to POLYLINE. 

Note that point has a positive signed distance if it is in the same side of the polyline as the normal vectors point.
"""
# compute signed part of signed distance
function signDist(a, b)
    dot_prod = dot(a, b)
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 1.0 
    end
end

# negative if to the right of normal vector, positive if to the left

# use binary search method to find alpha to minimize load time

# compute signed distance from starting
function signed_distance_initial(seg::InitialRay, q) 
    # alpha is an interpolating value - find the alpha value that results in the minimum distance

    # dist should be the minimum 
    alpha0 = 0.0
    alpha1 = norm(seg.point - q)
    dist0 = norm(seg.point - alpha0*seg.tangent - q)
    dist1 = norm(seg.point - alpha1*seg.tangent - q)

    while abs(alpha1 - alpha0) > 0.00000001 || abs(dist0-dist1) > 0.0000000001
        alpha = (alpha0+alpha1)/2.0
        dist = norm(seg.point - alpha*seg.tangent - q)
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

    return signDist(seg.normal, q - seg.point)*dist0

end

# compute signed distance from terminal

function signed_distance_terminal(seg::TerminalRay, q) 
    # alpha is an interpolating value - find the alpha value that results in the minimum distance

    # dist should be the minimum 
    alpha0 = 0.0
    alpha1 = norm(seg.point - q)
    dist0 = norm(seg.point + alpha0*seg.tangent - q)
    dist1 = norm(seg.point + alpha1*seg.tangent - q)

    while abs(alpha1 - alpha0) > 0.00000001 || abs(dist0-dist1) > 0.0000000001
        alpha = (alpha0+alpha1)/2.0
        dist = norm(seg.point + alpha*seg.tangent - q)
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

    return signDist(seg.normal, q - seg.point)*dist0

end

# compute signed distance from standard
function signed_distance_standard(seg::StandardSegment, q) 
    # alpha is an interpolating value - find the alpha value that results in the minimum distance

    # dist should be the minimum 
    alpha0 = 0.0
    alpha1 = 1.0
    dist0 = norm(alpha0*seg.p1 + (1.0 - alpha0)*seg.p2 - q)
    dist1 = norm(alpha1*seg.p1 + (1.0 - alpha1)*seg.p2 - q)
    
    while abs(alpha1 - alpha0) > 0.00000001 || abs(dist0-dist1) > 0.0000000001
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

    return signDist(seg.normal, q - seg.p1)*dist0

end

# loop through computations to find the minimum a
function signed_distance(polyline::Polyline, point)
    N = length(polyline.segments)
    starting_dist = signed_distance_initial(polyline.segments[1], point)

    dist_min = starting_dist; # find min starting w starting ray

    for i = 2:(N -1) # do you start at the second point?
        dist = signed_distance_standard(polyline.segments[i], point)
        if abs(dist) < abs(dist_min)
            dist_min = dist
        end
    end

    term_dist = signed_distance_terminal(polyline.segments[N], point)
    if abs(term_dist) < abs(dist_min)
        dist_min = term_dist
    end
    return dist_min

    return 0.0
end

# --- implement pure pursuit controller --- #

function control(state_ch, control_ch, stop_ch, path, L)
    pure_pursuit(state_ch, control_ch, stop_ch, path, L)
end

# signed distance function
function signOfDot(a, b)
    dot_prod = dot(a, b)
    if dot_prod > 0
        return 1.0
    elseif dot_prod < 0
        return -1.0
    else
        return 0.0
    end
end

# should car steer left or right?
# calculate the normal vector to the vehicle's direction vector.
# rotate 90 degrees counterclockwise
# sign of Dot tells you if you're to the left or right
# For a 2D vector [x, y], the normal vector is [-y,x]
function left_right(v_direction, target_direction)
    v_normal = [-v_direction[2], v_direction[1]]
    return signOfDot(v_normal, target_direction)
end

function pure_pursuit(state_ch, control_ch, stop_ch, path, L; ls = 1.445) # ls = lookahead time
    N = length(path.segments)

    while true
        sleep(0.01)
        fetch(stop_ch) && return # check if this task should end
        x = fetch(state_ch) # get latest simulation state
        p1, p2, θ, v = x # unpack state variables
        
# ------- my control law implementation ---
        vehicle_pos = [p1, p2]
        len0 = v * ls # velocity times look ahead time
        min_diff = len0 # difference between desired and durr distance - like alpha
        best_i = 0 # next best target on path
        v_direction = [cos(θ), sin(θ)] # direction vector of heading based on its orientation angle θ - like delta

        # for all the standard segments
        # find the segment that is the minimum distance away
        for i = 2 : N - 1
            target = path.segments[i].p1 # try current path segment as target
            sign = signOfDot(v_direction, target - vehicle_pos) 
            if sign > 0 # target pos must be ahead of vehicle
                l = norm(target - vehicle_pos) 
                diff = abs(l - len0) #euc distance -> absolute distance
                if diff < min_diff 
                    min_diff = diff
                    best_i = i
                end
            end
        end
        
        if best_i > 0
            target = path.segments[best_i].p1
            # cosine of the angle between the vehicle's direction and the target direction
            cos_alpha = dot(v_direction, target - vehicle_pos) / norm(target - vehicle_pos)
            # calculate the angle alpha using the arccosine function
            alpha = acos(cos_alpha)
            # calculate the sine of the angle alpha
            sin_alpha = sin(alpha)
            left_or_right = left_right(v_direction, target - vehicle_pos)
            # calculate the steering angle (delta) using a formula
            delta = atan(2.0 * L * sin_alpha * left_or_right, v * ls)
            u = [delta, 0.0]
            println("line 88")
        else
            println("line 90")
            u = zeros(2) # No valid target found
        end
# ------- my control law implementation ---

        take!(control_ch)  # Clear old control
        put!(control_ch, u)  # Put new control on the channel
    end
end

# --- decision making function --- #

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
		# TO-DO: implement trajectory and pure pursuit controller
        steering_angle = 0.0
        target_vel = 0.0
        cmd = (steering_angle, target_vel, true)
        serialize(socket, cmd)

		# TO-DO: add heuristics
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end