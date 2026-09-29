import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time

model_path = '/Users/muzimpofu/Downloads/gesture-synth/hand_landmarker.task'
gesture_model_path = '/Users/muzimpofu/Downloads/gesture-synth/gesture_recognizer.task'


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult


def draw_landmarks_on_image(frame_rgb, result):
    hand_landmarks_list = result.hand_landmarks
    annotated_image = np.copy(frame_rgb)
    

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        for landmark in hand_landmarks:
            converted_landmark = landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z)

            hand_landmarks_proto.landmark.append(converted_landmark)
                
        


        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )
        

    return annotated_image


latest_annotated_image = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_image

    # print('hand landmarker result: {}'.format(result))

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

    latest_annotated_image = annotated_image


def gesture_print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))


gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_print_result)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path= model_path,
        delegate=BaseOptions.Delegate.CPU
        ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result)


with HandLandmarker.create_from_options(options) as landmarker:
    with GestureRecognizer.create_from_options(gesture_options) as recognizer:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        else:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame (stream end?) Exiting...")
                    break

                flip = cv.flip(frame, 1)
                frame_rgb = cv.cvtColor(flip, cv.COLOR_BGR2RGB)
                frame_timestamp_ms = int(time.monotonic() * 1000)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,)
                
                            
                landmarker.detect_async(mp_image, frame_timestamp_ms)

                recognizer.recognize_async(mp_image, frame_timestamp_ms)

                if latest_annotated_image is not None:
                    new_image = cv.cvtColor(latest_annotated_image, cv.COLOR_RGB2BGR)
                    cv.imshow('frame', new_image)
                else:
                    cv.imshow('frame', flip)
                

        
            
                
                if cv.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()






