#processes bagfile and runs optimization on each separated trajectory
#Schmittle
#romesc

import argparse
import rosbag
import math
import numpy as np
import tf.transformations
import librhc.utils as utils
#import parameters
import rhctensor
#import librhc.cost as cost
#import librhc.model as model
#import librhc.trajgen as trajgen
#import librhc.value as value
#import librhc.worldrep as worldrep
import librhc.types as types
#import logger
#from scipy.optimize import minimize


def quaternion_to_angle(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw

def separate_trajectories(bag):
    '''
    Separates trjactories by the correction timestamps
    '''
    bag = rosbag.Bag(bag)
    trajectories = []
    curr_traj = []
    position_set = False
    for topic, msg, t in bag.read_messages(topics=['/feedback','/sim_car_pose/pose','/vesc/low_level/ackermann_cmd_mux/input/teleop']):
        if topic == "/feedback":
            trajectories.append(curr_traj)
            curr_traj = []
        if topic == "/sim_car_pose/pose" and not position_set:
            position = msg.pose.position
            orientation = msg.pose.orientation
            curr_pose = (position.x,position.y,quaternion_to_angle(orientation))
            position_set = True
        if topic == '/vesc/low_level/ackermann_cmd_mux/input/teleop':
            drive =  msg.drive
            curr_u = (drive.steering_angle,drive.speed)
            if position_set:
                curr_traj.append([curr_pose,curr_u])
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

def viz_map(map_img):
    import matplotlib.pyplot as plt
    plt.imshow(map_img, cmap='gray_r', vmin=0, vmax=255)
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

def pixelMapInRobotFrame(map_data, pose, crop_square=200):
    import torchvision
    pose_mapFrame = np.asarray([pose])
    utils.world2mapnp(map_data, pose_mapFrame) #this is a bad function which does inplace assignment (rather than explicitly returns a new object)...

    map_img = np.array(map_data._get_map_data)
    #convert 'unknown'=-1 and 'wall'=100 to value 1; keep freespace=0
    map_img[map_img == -1] = 255
    map_img[map_img == 100] = 255 

    #shape into 2D image
    map_img = map_img.reshape([map_data.height,map_data.width])
    
    #convert to uint8 (just in case)
    #map_img = map_img.astype(np.uint8)

    #crop map according to pose for now just [x,y] crop, TODO:rotation
    x_r = int(pose_mapFrame[0][0])
    y_r = int(pose_mapFrame[0][1])
    marg = int(math.floor(crop_square/2))

    #Check the crop region is valid
    assert(x_r > marg and x_r < map_data.width-marg and y_r > marg and y_r < map_data.height-marg), "Crop region too large for the map margins! Try padding your map or reducing crop region."

    map_img_crop = map_img[y_r-marg:y_r+marg , x_r-marg:x_r+marg]

    return map_img_crop

def main(args):
    print("Reading in rosbag from '" + args.infile + "'...")
    map_data = get_map(args.infile)

    trajs = separate_trajectories(args.infile)
    map_img_crop = pixelMapInRobotFrame(map_data, trajs[0][0][0], 400)

    viz_map(map_img_crop)

    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input rosbag to be processed")
    main(parser.parse_args())


    #logger = logger.StdLog()
    #trajs = separate_trajectories('test.bag')
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




