import os
import numpy as np
from utils.utils import get_front_frames, get_path_front_images

def get_mapping_front_frames(mapping_result):
    mapping_front_frames = []
    indexes = np.where(mapping_result == 1)[0]
    front_frame = [indexes[0]]
    for a, b in zip(indexes, indexes[1:]):
        if b == a + 1:
            front_frame.append(b)
        else:
            mapping_front_frames.append(front_frame)
            front_frame = [b]
    mapping_front_frames.append(front_frame)
    return mapping_front_frames


def get_front_frames_accuracy(mapping_front_frames, front_frames):
    front_frames_accuracy = np.zeros(len(front_frames))
    for i, front_frame in enumerate(front_frames):
        mapping_statistic = np.array([
            len(np.intersect1d(front_frame, mapping_front_frame)) / max(len(front_frame), len(mapping_front_frame))
            for mapping_front_frame in mapping_front_frames
        ], dtype='float')
        front_frames_accuracy[i] += (mapping_statistic.max() > 0.5)
    return front_frames_accuracy



def get_cv_accuracy(path_folders, data_name, mapping):
    global_accuracy = []
    for i, path_folder in enumerate(path_folders):
        front_frames = get_front_frames(path_folder, data_name)
        if len(front_frames) == 0:
            continue

        path_front_images = get_path_front_images(path_folder)
        mapping_result = mapping(path_front_images)

        np.save(os.path.join(path_folder, "mapping_result"), mapping_result)

        mapping_front_frames = get_mapping_front_frames(mapping_result)

        front_frames_accuracy = get_front_frames_accuracy(mapping_front_frames, front_frames)

        global_accuracy.append(front_frames_accuracy.mean())

        #print(path_folder, front_frames_accuracy)

    return np.mean(global_accuracy)


def get_rnn_accuracy(net, dataset):
    net.eval()
    device = next(net.parameters()).device

    global_accuracy = []
    for i, path_folder in enumerate(dataset.path_folders):
        front_frames = dataset.get_front_frames(path_folder)
        if len(front_frames) == 0:
            continue

        images, labels = dataset[i]
        mapping_result = np.argmax(net(images.to(device)).detach().to('cpu'), axis=2)
        mapping_result = np.array((mapping_result == 2)[0], dtype=np.uint8)

        np.save(os.path.join(path_folder, "lstm_mapping_result"), mapping_result)

        mapping_front_frames = get_mapping_front_frames(mapping_result)

        front_frames_accuracy = get_front_frames_accuracy(mapping_front_frames, front_frames)

        global_accuracy.append(front_frames_accuracy.mean())

        #print(path_folder, front_frames_accuracy)

    return np.mean(global_accuracy)