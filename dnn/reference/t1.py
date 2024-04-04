import argparse

import cv2
import numpy as np
import torch
import time
import json
import os
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)  
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth, video_info, label_index, label_name, annotation_name, output_dir):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    frameIndex = 0    #json file
    data_json = {}    #json file
    data_action = []    #json file
    track_num = 5
    Pose.last_id = -1
    delay = 1
    for img in image_provider:
        frame_skeleton = {}
        frame_skeleton["frame_index"] = frameIndex + 1
        have_skeleton = False
        time_start = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            # if len(pose_entries[n]) == 0:
            if len(pose_entries[n]) < 6:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][19])
            current_poses.append(pose)

        time_end = time.time()
        print('time cost', time_end-time_start, 's')
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        for pose in current_poses:
            pose.draw(img)

        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.namedWindow('Lightweight Human Pose Estimation Python Demo', 0)
        cv2.resizeWindow('Lightweight Human Pose Estimation Python Demo', 1080, 720)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        cv2.waitKey(1)

        if len(current_poses) > 0:
            data_skeleton = [{"pose": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                "score": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}] * track_num
            # data_skeleton = [{}] * track_num
            data_one_skeleton = {}
            for pose in current_poses:
                data_pose = []
                data_score = []
                for kpt_id in range(0, num_keypoints):    #18 keypoint
                    if pose.keypoints[kpt_id][0] != -1 and pose.keypoints[kpt_id][1] != -1:
                        data_pose.append(float(format(pose.keypoints[kpt_id][0]/img.shape[1], '.3f')))
                        data_pose.append(float(format(pose.keypoints[kpt_id][1]/img.shape[0], '.3f')))
                        data_score.append(1.0)
                    else:
                        data_pose.append(0.0)
                        data_pose.append(0.0)
                        data_score.append(0.0)

                data_one_skeleton["pose"] = data_pose
                data_one_skeleton["score"] = data_score
                if pose.id < track_num:
                    data_skeleton[pose.id] = data_one_skeleton

            frame_skeleton["skeleton"] = data_skeleton
            have_skeleton = True
        else:
            frame_skeleton["skeleton"] = []

        #generate skeleton for stgcn
        data_action.append(frame_skeleton)
        # increase frame index
        frameIndex += 1

    data_json["data"] = data_action
    data_json["label"] = str(file.split('.')[0].split('_')[0])
    data_json["label_index"] = label_index

    video_info[str(file.split('.')[0])] = {
        "has_skeleton": have_skeleton,
        "label": str(file.split('.')[0].split('_')[0]),
        "label_index": label_index
    }

    label_index += 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, file.split('.')[0] + ".json"), 'w') as f:
        json.dump(data_json, f)
    with open(annotation_name, 'w') as f1:
        f1.write(json.dumps(video_info, ensure_ascii=False, indent=1))
    return video_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint_iter_370000.pth', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # 这里设置需要处理的视频路径
    parser.add_argument('--videos', type=str, default='videos/', help='path to video dir')
    parser.add_argument('--output_dir', type=str, default='output/skeletons/', help='path to output result')
    parser.add_argument('--annotation_name', type=str, default='output/kinetics_train_label.json', help='Path to save output as json file. If nothing is given, the output will cant be saved.')
    # 若GPU不能用需切换CPU
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.videos == '':
        raise ValueError('Either --videos has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    files = os.listdir(args.videos)
    video_info = {}
    for file in files:
        if not os.path.isdir(file):
            video_path = os.path.join(args.videos, file)

        # 根据文件名确定label_index
        label_name = file.split('_')[0]
        label_list = ['dancing', 'standing', 'walking', 'laying']
        label_index = label_list.index(label_name)

        frame_provider = VideoReader(video_path)
        video_info = run_demo(net, frame_provider, args.height_size, args.cpu, args.track,
                        args.smooth, video_info, label_index, label_name, args.annotation_name, args.output_dir)

