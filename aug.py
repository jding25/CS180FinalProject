import cv2
import skimage as ski
import skvideo as skv
from skvideo import io as skvio
import matplotlib.pyplot as plt
import json
import numpy as np
import pyrender
import trimesh

def save_points(frame, amount):
    plt.imshow(frame)
    data = plt.ginput(amount, -1)
    with open('data.json', 'w') as f:
        json.dump(data, f)

def load_points(file):
    file = open(file)
    data = json.load(file)
    return data

def create_points(frame, points):
    for p in points:
        frame = cv2.circle(frame, (round(p[0]),round(p[1])), radius=5, color=(0,0,153), thickness=-1)
    return frame

def create_rectangles(frame, points, size):
    for p in points:
        frame = cv2.rectangle(frame, (round(p[0]) - (size//2),round(p[1])- (size//2)), (round(p[0]) + (size//2),round(p[1]) + (size//2)),color=(0,102,204),thickness=2)
    return frame

def generate_bboxes(points, size=8):
    boxes = []
    for p in points:
        boxes.append([round(p[0] - (size//2)),round(p[1] - (size//2)),size,size])
    return np.array(boxes)

def cam_calib(point2D, point3D):
    point2D = np.array(point2D)
    point3D = np.array(point3D)
    point3D = np.hstack((point3D,np.ones((point3D.shape[0],1))))
    #print(point3D.shape)

    val1 = -1*point2D[0][0]*point3D[0][0]
    val2 = -1*point2D[0][0]*point3D[0][1]
    val3 = -1*point2D[0][0]*point3D[0][2]
    val4 = -1*point2D[0][1]*point3D[0][0]
    val5 = -1*point2D[0][1]*point3D[0][1]
    val6 = -1*point2D[0][1]*point3D[0][2]
    temp1 = np.array([point3D[0][0],point3D[0][1],point3D[0][2],point3D[0][3], 0,0,0,0, val1,val2,val3])
    temp2 = np.array([0,0,0,0,point3D[0][0],point3D[0][1],point3D[0][2],point3D[0][3], val4,val5,val6])
    A = np.vstack((temp1, temp2))

    for i in range(1,point2D.shape[0]):
        val1 = -1*point2D[i][0]*point3D[i][0]
        val2 = -1*point2D[i][0]*point3D[i][1]
        val3 = -1*point2D[i][0]*point3D[i][2]
        val4 = -1*point2D[i][1]*point3D[i][0]
        val5 = -1*point2D[i][1]*point3D[i][1]
        val6 = -1*point2D[i][1]*point3D[i][2]

        temp1 = np.array([point3D[i][0],point3D[i][1],point3D[i][2],point3D[i][3], 0,0,0,0, val1,val2,val3])
        temp2 = np.array([0,0,0,0,point3D[i][0],point3D[i][1],point3D[i][2],point3D[i][3], val4,val5,val6])
        next = np.vstack((temp1, temp2))
        A = np.vstack((A, next))

    
    calibration = np.linalg.lstsq(A, point2D.reshape((point2D.shape[0]*2,1)))
    B = np.vstack((calibration[0], np.array([1])))
    return B.reshape(3, 4)

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def draw_axis(frame, points3D, calibration):
    impts = np.transpose(np.matmul(calibration, np.transpose(points3D)))
    frame = cv2.line(frame,(round(impts[0][0]/impts[0][2]), round(impts[0][1]/impts[0][2])), (round(impts[1][0]/impts[1][2]), round(impts[1][1]/impts[1][2])),color=(0,0,255), thickness=3) #x_axis
    frame = cv2.line(frame,(round(impts[0][0]/impts[0][2]), round(impts[0][1]/impts[0][2])), (round(impts[2][0]/impts[2][2]), round(impts[2][1]/impts[2][2])),color=(255,0,0), thickness=3) #y_axis
    frame = cv2.line(frame,(round(impts[0][0]/impts[0][2]), round(impts[0][1]/impts[0][2])), (round(impts[3][0]/impts[3][2]), round(impts[3][1]/impts[3][2])),color=(0,255,0), thickness=3) #z_axis
    return frame


def generate_cube(video_data, calibration_list, pos=(0,0,0), cube_size=1):
    i = 0
    for f in video_data:
        calibration = calibration_list[i]
        expected3D = np.array([[0*cube_size+pos[0],0*cube_size+pos[1],0*cube_size+pos[2],1],
                            [1*cube_size+pos[0],0*cube_size+pos[1],0*cube_size+pos[2],1],
                            [1*cube_size+pos[0],1*cube_size+pos[1],0*cube_size+pos[2],1],
                            [0*cube_size+pos[0],1*cube_size+pos[1],0*cube_size+pos[2],1],
                            [0*cube_size+pos[0],0*cube_size+pos[1],1*cube_size+pos[2],1],
                            [1*cube_size+pos[0],0*cube_size+pos[1],1*cube_size+pos[2],1],
                            [1*cube_size+pos[0],1*cube_size+pos[1],1*cube_size+pos[2],1],
                            [0*cube_size+pos[0],1*cube_size+pos[1],1*cube_size+pos[2],1]])
        projected2D = np.transpose(np.matmul(calibration, np.transpose(expected3D)))
        finalPoints = []
        for x in projected2D:
                x[0] = x[0]/x[2]
                x[1] = x[1]/x[2]
                finalPoints.append((x[0], x[1]))

        draw(f, np.array(finalPoints))
        i += 1

def get_calibration_list(points2D, points3D, trackerFunction=cv2.legacy.TrackerMOSSE(), box_size=60, generate_points=False, generate_rect=False):
    r = generate_bboxes(points2D,box_size)
    trackers = []
    for roi in r:
        tracker = cv2.legacy.TrackerMOSSE().create()
        tracker.init(first_frame, roi)
        trackers.append(tracker)
    calibration_list = []
    for f in video_data:
        i = 0
        dots = []
        new3D = points_3d.copy()
        for t in trackers:
            found, bbox = t.update(f)
            if found:
                dots.append([bbox[0], bbox[1]])
            else:
                new3D.remove(new3D[i])
            i +=1
        dots = np.add(dots, box_size//2)
        calibration_list.append(cam_calib(dots, new3D))
        if generate_rect:
            create_rectangles(f, dots, box_size)
        if generate_points:
            create_points(f, dots)
    return calibration_list

#def generate_cheese(first_frame,matrix, r, t):
#    rotation_matrix, _ = cv2.Rodrigues(r)
#
#    # Create a 4x4 transformation matrix (camera position matrix)
#    camera_position_matrix = np.eye(4)
#    camera_position_matrix[:3, :3] = rotation_matrix
#    t = t.flatten()
#    camera_position_matrix[:3, 3] = t.T
#    cheese_trimesh = trimesh.load('cheese.OBJ')
#    mesh = pyrender.Mesh.from_trimesh(cheese_trimesh)
#    camera = pyrender.IntrinsicsCamera(matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2])
#    scene = pyrender.Scene()
#    scene.add(camera, pose=camera_position_matrix)
#    scene.add(mesh)
#    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, 
#                                innerConeAngle=np.pi/16.0,
#                                outerConeAngle=np.pi/6.0)
#    scene.add(light, pose=camera_position_matrix)
#   pyrender.Viewer(scene, use_raymond_lighting=True)
    #r = pyrender.OffscreenRenderer(1920, 1080)
    #color, depth = r.render(scene)
    #plt.figure()
    #plt.imshow(color)
    #plt.show()
    #pyrender.Viewer(scene, use_raymond_lighting=True)

#video_data = skvio.vread('aug.mp4')   #load video frames into memory
#first_frame = video_data[0]           #get first frame
#plt.imsave('first.jpg', first_frame)
#save_points(first_frame, 49)          #use this to create points and save points from the first frame of the video(inputs: first frame, number of points) -> (outputs: data.json)
#points_2d = load_points('data.json')   #use this to load points from a json file
#points_3d = []
#for z in range(2):
#    for x in range(7):
#        points_3d.append([np.float32(x),np.float32(0),np.float32(z)])
#for y in range(5):
#    for x in range(7):        
#        points_3d.append([np.float32(x),np.float32(y),np.float32(2)])  #Makes 3D coord system


#calibration_list = get_calibration_list(points_2d, points_3d, generate_points=True, generate_rect=False) #Creates list of calibration matrices and can generate bounding boxes/points
#generate_cube(video_data,calibration_list(0,0,2)) #Use to make cube in scene at a point
#generate_cube(video_data, calibration_list, (0,1,2), 2)
#calib = calibration_list[0]
#pnts = [[0,0,0,1],
#        [2,0,0,1],
#        [0,2,0,1],
#        [0,0,2,1]]
#im = draw_axis(first_frame, pnts, calib)
#plt.imsave('axis.jpg', im)

#generate_cube(video_data, calibration_list,(0,0,2))
#print(calibration_list[0])
#cameraMatrixInit = np.array([[8666.67, 0, first_frame.shape[1]/2],
#                             [0, 8666.67, first_frame.shape[0]/2],
#                             [0, 0, 1]], dtype=np.float32)
#print(cameraMatrixInit.shape)

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array([points_3d], np.float32),np.array([points_2d], np.float32),first_frame.shape[:2], cameraMatrixInit, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
#print(rvecs)
#print(tvecs)

#generate_cheese(first_frame,mtx, np.array(rvecs), np.array(tvecs))

#generate_cube(video_data, calibration_list, (0,0,2))

#skvio.vwrite('out.mp4',video_data) #writes output video
#plt.show()