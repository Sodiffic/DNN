import os.path
import pickle
from tqdm import tqdm
import numpy as np

"""用于将ntu-120数据集（pkl）转换成npy"""
num_person_out = 2
num_person_in = 5
num_joint = 17
max_frame = 300
info_channel = 7
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # 用于完整打印
# 假设pickle文件的路径是'your_dataset.pkl'
pickle_file_path = '../dnn/ntu120_hrnet.pkl'
save_path = 'F:/ntu_120/'
# 使用pickle模块打开并加载文件
with open(pickle_file_path, 'rb') as f:
    dataset_dict = pickle.load(f)

# 现在dataset_dict包含pickle文件中的annotation内容
annotation_dict = dataset_dict['annotations']

for item in tqdm(annotation_dict, total=len(annotation_dict)):
    """
    frame_dir (str): The identifier of the corresponding video.
    total_frames (int): The number of frames in this video.
    img_shape (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
    original_shape (tuple[int]): Same as img_shape.
    label (int): The action label.
    keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as total_frames); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
    keypoint_score (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.
    """
    total_frames = item['total_frames']
    keypoint_score = item['keypoint_score']
    img_shape = item['img_shape']
    keypoint = item['keypoint']
    label = item['label']
    file_name = item['frame_dir']
    save_file_path = save_path + '/' + ('train' if file_name in dataset_dict['split']['xset_train'] else 'val') +'/'
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    data_numpy = np.zeros((total_frames, num_person_in, num_joint, info_channel))  # [frame, people, joint, channel]
    # 帧索引，关节索引
    frame_indices = np.arange(total_frames)[:, None, None]  # 形状为 (frames, 1, 1, 1)
    joint_indices = np.arange(num_joint)[None, None, :]  # 形状为 (1, 1, joints, 1)
    data_numpy[..., 0] = frame_indices  # 将帧数索引赋值给第0个通道
    # print(data_numpy.shape)
    data_numpy[..., 1] = joint_indices  # 将关节索引赋值给第1个通道
    # print(data_numpy.shape)

    # 关节置信度赋值
    # 判断逻辑：如果人数维度keypoint_score.shape[0]小于num_person_in，那么keypoint_score填充成相应维度后赋值；如果keypoint_score人数超过，则
    #data_numpy[..., 4] = keypoint_score[:num_person_in, :, :]
    # x,y归一化
    height = img_shape[0]
    weight = img_shape[1]
    devidor = height if (height - weight) > 0 else weight
    # 关节归一化坐标赋值
    for m in range(keypoint.shape[0]):  # 遍历人数
        # data_numpy[..., 4] = keypoint_score
        if m >= num_person_in:
            break
        # 等于1人的情况先不考虑……
        for t in range(keypoint.shape[1]):  # 遍历帧数
            data_numpy[t, m, :, 2] = keypoint[m, t, :, 0] / devidor  # x坐标赋值给第3个通道（索引为2）
            data_numpy[t, m, :, 3] = keypoint[m, t, :, 1] / devidor  # y
            data_numpy[t, m, :, 4] = keypoint_score[m, t, :]         # score

            # 质心归一化
            x_coords = np.mean(data_numpy[t, m, :, 2])
            y_coords = np.mean(data_numpy[t, m, :, 3])  # 第3通道是y坐标
            data_numpy[t, m, :, 5] = x_coords
            data_numpy[t, m, :, 6] = y_coords

            # 关节索引归一化，帧索引归一化
            data_numpy[t, m, :, 1] /= num_joint

    # 根据置信度对五个人排序
    if keypoint.shape[0] > num_person_out:
        sort_index = (-data_numpy[:, :, :, 3].sum(axis=(0, 2))).argsort()
        data_numpy = np.moveaxis(data_numpy[:, sort_index, :, :], 1, 0).transpose((1, 0, 2, 3))
    data_numpy = data_numpy[:, 0:num_person_out, :, :]
    # 取300帧
    if keypoint.shape[1] < max_frame:
        # mode='wrap' 意味着它会循环使用原始数组中的值来填充
        padding_width = [(max_frame - keypoint.shape[1], 0)] + [(0, 0)] * (data_numpy.ndim - 1)
        data_numpy = np.pad(data_numpy, padding_width, mode='wrap')
    # 如果帧数大于300，则截断到300帧
    else:
        data_numpy = data_numpy[:max_frame, :, :, :]
    max_frame_indices = np.arange(max_frame)[:, None, None]
    data_numpy[:, :, :, 0] = max_frame_indices / max_frame

    # print(data_numpy.shape)
    # print(data_numpy)

    np.save(save_file_path+str(label)+'_'+file_name, data_numpy)

    #     # # 计算质心并赋值给data数组的第6和第7通道
    # # for t in range(T):  # 遍历帧数
    # #     for n in range(N):  # 遍历人数
    # #         # 提取当前帧当前人的所有关节的x和y坐标
    # #         x_coords = data[t, n, :, 1]  # 第2通道是x坐标
    # #         y_coords = data[t, n, :, 2]  # 第3通道是y坐标
    # #
    # #         # 计算质心坐标
    # #         centroid_x = np.mean(x_coords)
    # #         centroid_y = np.mean(y_coords)
    # #
    # #         # 将质心坐标赋值给data数组的第6和第7通道
    # #         # 注意：索引从0开始，所以第6通道是索引5，第7通道是索引6
    # #         data[t, n, :, 5] = centroid_x
    # #         data[t, n, :, 6] = centroid_y
    #
    #
    #
    # gap = height - weight if (height - weight) > 0 else weight - height
    # for frame_index in enumerate(total_frames):
    #     for person_index in enumerate(num_person_in):
    #         if height > weight:  #
    #             data_numpy[frame_index, person_index,:,0] = (gap + keypoint[person_index, frame_index, :,0])/height
    #             data_numpy[frame_index, person_index, :, 1]=keypoint[person_index, frame_index, :,1]/height
    #         else:
    #             data_numpy[frame_index, person_index, :, 0] = keypoint[person_index, frame_index, :, 0] / weight
    #             data_numpy[frame_index, person_index, :, 1] = (gap + keypoint[person_index, frame_index, :,1])/weight
    #
    #

# 处理split字段
# split_dict = dataset_dict['split']
# for split_name, video_ids in tqdm(split_dict.items(), total=len(split_dict)):
#     print(f"Split name: {split_name}")
#     print(f"Video IDs: {video_ids}")
    # 这里您可以根据需要对video_ids进行进一步处理，比如读取视频文件等

"""
label = dir
if item.label in dataset_dict['split']['train']  #?
train_list.append(item)
"""
