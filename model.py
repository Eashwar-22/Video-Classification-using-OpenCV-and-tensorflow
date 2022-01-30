import tensorflow as tf
import numpy as np
import cv2
import os
import tempfile


IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ['BaseballPitch', 'Biking', 'Diving', 'HighJump', 'HorseRace']

convlstm_model = tf.keras.models.load_model('models/convlstm_model___Date_Time_2022_01_11__14_28_36___Loss_0.8412893414497375___Accuracy_0.8171428442001343.h5')
lrcn_model = tf.keras.models.load_model('models/LRCN_model___Date_Time_2022_01_11__14_36_23___Loss_0.3567538559436798___Accuracy_0.9085714221000671.h5')


def predict_single_action_1(video_file):
    '''
    This function will perform single action recognition prediction on a video using either of the models.
    Args:
    video_file:  The video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 20
    CLASSES_LIST = ['BaseballPitch', 'Biking', 'Diving', 'HighJump', 'HorseRace']

    # Initialize the VideoCapture object to read from the video file.
    video_reader = video_file

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
        # print("Frame shape ",frames_list.shape)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    conf = str(round(predicted_labels_probabilities[predicted_label] * 100,2))+"%"
    action = predicted_class_name

    # Release the VideoCapture object.
    video_reader.release()

    return action,conf
def predict_single_action_2(video_file):
    '''
    This function will perform single action recognition prediction on a video using either of the models.
    Args:
    video_file:  The video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 20
    CLASSES_LIST = ['BaseballPitch', 'Biking', 'Diving', 'HighJump', 'HorseRace']

    # Initialize the VideoCapture object to read from the video file.
    video_reader = video_file

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
        # print("Frame shape ",frames_list.shape)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = lrcn_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    conf = str(round(predicted_labels_probabilities[predicted_label] * 100,2))+"%"
    action = predicted_class_name

    # Release the VideoCapture object.
    video_reader.release()

    return action,conf


# Passing a sample video for prediction
# print(os.getcwd())
# input_video_file_path = "sample_videos/v_BaseballPitch_g02_c01.avi"
#
# print("ConvLSTM Model")
# predict_single_action(convlstm_model,input_video_file_path, SEQUENCE_LENGTH)
#
# print("LRCN Model")
# predict_single_action(lrcn_model,input_video_file_path, SEQUENCE_LENGTH)

