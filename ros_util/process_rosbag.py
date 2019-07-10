#processes bagfile and runs optimization on each separated trajectory
#Schmittle
#romesc

import argparse
import rosbag
import math
import numpy as np
import tf.transformations
import librhc.utils as utils
import rhctensor
import librhc.types as types
import matplotlib.pyplot as plt
import os

CROP_SQUARE_SZ = 200


def quaternion_to_angle(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw
def separate_trajectories(bag,separate):
    '''
    Separates trjactories by the correction timestamps
    '''
    bag = rosbag.Bag(bag)
    trajectories = []
    curr_traj = []
    position_set = False
    target_set = False 
    for topic, msg, t in bag.read_messages(topics=['/feedback','/sim_car_pose/pose','/vesc/low_level/ackermann_cmd_mux/input/teleop','/target']):
        if topic == "/feedback" and separate:
            trajectories.append(curr_traj)
            curr_traj = []
        if topic == "/target":
            target = [msg.pose.position.x,msg.pose.position.y,0]
            target_set = True
        if topic == "/sim_car_pose/pose" and not position_set:
            position = msg.pose.position
            orientation = msg.pose.orientation
            curr_pose = (position.x,position.y,quaternion_to_angle(orientation))
            position_set = True
        if topic == '/vesc/low_level/ackermann_cmd_mux/input/teleop':
            drive =  msg.drive
            curr_u = (drive.steering_angle,drive.speed)
            if position_set and target_set:
                curr_traj.append([curr_pose,curr_u,target])
                position_set = False
    trajectories.append(curr_traj)
    bag.close()
    return trajectories

def get_map(bag):
    '''
    Gets map from bag and converts to a MapData type
    '''
    bag = rosbag.Bag(bag)
    for _, msg, _ in bag.read_messages(topics=['/map']):
        raw_map = msg
        break
    bag.close()
    map_data = types.MapData(
            name = "map",
            resolution = raw_map.info.resolution,
            origin_x = raw_map.info.origin.position.x,
            origin_y = raw_map.info.origin.position.y,
            orientation_angle = quaternion_to_angle(raw_map.info.origin.orientation),
            width = raw_map.info.width,
            height = raw_map.info.height,
            get_map_data = raw_map.data
    )
    return map_data

def objective_fn(w,mpc,traj,dtype):
    mpc.cost.dist_w = w[0]
    mpc.cost.obs_dist_w = w[1]
    mpc.cost.cost2go_w = w[2]
    cost = 0
    for p in traj:
        u_i = mpc.step(dtype(p[0]))
        if u_i is not None:
            cost+= np.linalg.norm(u_i - dtype(p[1]))
    return cost

def viz_map(map_img):
    plt.imshow(map_img, cmap='gray', vmin=0, vmax=255, origin='lower')
    plt.show()

def pixelMapInRobotFrame(map_data, pose_worldFrame, crop_square=200):
    import cv2
    pose_mapFrame = np.asarray([pose_worldFrame]) # this variable WILL get converted to mapFrame after next function call
    utils.world2mapnp(map_data, pose_mapFrame) #this is a bad function signature which does inplace assignment of pose_mapFrame (rather than explicitly returns a new object)...

    map_img = np.array(map_data._get_map_data)
    map_img = map_img.reshape([map_data.height, map_data.width]) #shape map into 2D image
    
    # Get all working parameters
    (h, w) = map_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)           
    (x, y, th) = pose_mapFrame[0]
    x, y = int(x), int(y)
    th_r_deg = 90 - np.rad2deg(th)
    marg = crop_square // 2

    ###### Convert to grayscale image
    # Note this is changing occupancy grid -> image which is white where freespace is and black where wall or unknown is
    
    #convert 'freespace'=0 to value 255 (white)
    #convert 'unknown'=-1 and 'wall'=100 to value 0 (black)
    map_img[map_img == 0] = 255 
    map_img[map_img == -1] = 0 
    map_img[map_img == 100] = 0 

    ##################################3
    
    #convert to float32 for OpenCV
    map_img = map_img.astype(np.uint8)

    # Translation
    # TODO: need to verify that padding occurs during this process
    dx = x - cX
    dy = y - cY
    M_trans = np.float32([[1,0,-dx],[0,1,-dy]])
    map_img_trans = cv2.warpAffine(map_img, M_trans,(w,h))

    # Rotation
    M_rot = cv2.getRotationMatrix2D((cX, cY), -th_r_deg ,1)
    map_img_rot = cv2.warpAffine(map_img_trans, M_rot,(w,h), flags=cv2.INTER_NEAREST)

    # Cropping
    # Check the crop region is valid
    assert(x > marg and x < w-marg and y > marg and y < h-marg), "Crop region too large for the map margins! Try padding your map or reducing crop region."
    map_img_crop = map_img_rot[cY-marg:cY+marg , cX-marg:cX+marg] 

    #return map_img_crop, th_r_deg #use this line for visualizing with arrow embedded in map
    map_img_crop = map_img_crop.astype(np.uint8)

    del map_img; del map_img_rot; del map_img_trans;
    return map_img_crop

def main(args):
    rel_dir = '../../MuSHR_corrFB_trajectories_dataset/centered_straight/'
    print("Reading in rosbag from '" + args.infile + "'...")
    map_data = get_map(args.infile)

    # Separate trajectories out of rosbag
    state_control_target_trajs = separate_trajectories(args.infile, True)

    for traj_idx, traj in enumerate(state_control_target_trajs):
        occGrid_fname = rel_dir + 'occGrid_robotFrameCrops_traj'+'{:02d}'.format(traj_idx)+'.npy'
        state_control_target_fname = rel_dir + 'state_control_target_traj'+'{:02d}'.format(traj_idx)+'.npy'
        #check if this trajectory has already been processed. Note this will not be valid if the bagfile changes
        if not os.path.isfile(occGrid_fname):
            sct_traj_np = np.array(state_control_target_trajs)
            # Generate our image crops for each waypoint in trajectory
            map_img_crops = np.empty((len(traj),CROP_SQUARE_SZ, CROP_SQUARE_SZ), dtype='uint8')
            for t_num in range(len(traj)):
                map_img_crops[t_num, :, :] = pixelMapInRobotFrame(map_data, traj[t_num][0], CROP_SQUARE_SZ)
            #map_img_crops = np.concatenate([pixelMapInRobotFrame(map_data, traj[0], CROP_SQUARE_SZ) for traj in trajs])


            # Save the image crops  to external file
            np.save(occGrid_fname, map_img_crops)
            np.save(state_control_target_fname, sct_traj_np)

    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input rosbag to be processed")
    main(parser.parse_args())

    #logger = logger.StdLog()
    #raw_params = {"T":15,"K":62,"xy_threshold":1.5}
    #params = parameters.DictParams(raw_params)
    #dtype = rhctensor.float_tensor()
    #model = model.Kinematics(params,logger,dtype)
    #trajgen = trajgen.Dispersion(params,logger,dtype,model) #Dispersion library, there is also the fan library  'tl'
    #value_fn = value.SimpleKNN(params,logger,dtype,map_data)
    #world_rep = worldrep.Simple(params,logger,dtype,map_data)
    #cost_fn = cost.Waypoints(params,logger,dtype,map_data,world_rep,value_fn)

    #mpc = librhc.MPC(params,logger,dtype,model,trajgen,cost_fn)

    #goal = dtype(trajs[2][30][0])
    #mpc.set_goal(goal)
    #w = np.array([1.0,5.0,1.0])
    #for traj in trajs:
    #    res = minimize(objective_fn,w,args=(mpc,traj,dtype),method='nelder-mead',options={'xtol': 1e-3, 'disp': True})
    #    print res.x




