import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pygame
import time

pygame.mixer.init()

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


notes = {
    "A": pygame.mixer.Sound("Piano Notes/A5 Note.wav"),
    # "B": pygame.mixer.Sound("Piano_Notes/B5_Note.wav"),
    # "C": pygame.mixer.Sound("Piano_Notes/C5_Note.wav"),
    # "D": pygame.mixer.Sound("Piano_Notes/D5_Note.wav"),
    # "E": pygame.mixer.Sound("Piano_Notes/E5_Note.wav"),
    # "F": pygame.mixer.Sound("Piano_Notes/F5_Note.wav"),
    # "G": pygame.mixer.Sound("Piano_Notes/G5_Note.wav"),


}

note_playing = {
     "A": False,
     "B": False,
     "C": False,
     "D": False,
     "E": False,
     "F": False,
     "G": False
     
}

finger_bent = {
     "thumb": False,
     "index": False,
     "middle": False,
     "ring": False,
     "pinky": False
}

# def hand(result, handedness,hand_landmarks ):
#     handedness_list = result.handedness

#     if handedness is 'Right':
#         index_tip = hand_landmarks[8]
#         index_crease_two = hand_landmarks[6]
#         if index_tip.y > index_crease_two.y:
#             print("finger is bent")

release_tolerance = 0.05
bend_tolerance = 0.05

def is_finger_bent(tip, knuckle, was_bent ):

    if was_bent:
        return tip.y > knuckle.y - release_tolerance
    else:
        return tip.y > knuckle.y - bend_tolerance

def is_thumb_bent(tip, knuckle, was_bent):

    if was_bent:
        return tip.x < knuckle.x + release_tolerance

    else:
        return tip.x < knuckle.x + bend_tolerance


def draw_landmarks_on_image(frame_rgb, result):
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness
    annotated_image = np.copy(frame_rgb)
    

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx][0].category_name
        finger_position = None
        

        # print(f"Hand {idx}: {handedness}")

        if handedness == 'Right':
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]

            thumb_knuckle = hand_landmarks[1]
            index_knuckle = hand_landmarks[5]
            middle_knuckle = hand_landmarks[9]
            ring_knuckle = hand_landmarks[13]
            pinky_knuckle = hand_landmarks[17]

            # Index finger
            finger_bent["index"] = is_finger_bent(
                index_tip,
                index_knuckle,
                finger_bent["index"]
            )

            finger_bent["middle"] = is_finger_bent(
                middle_tip,
                middle_knuckle,
                finger_bent["middle"]
            )

            finger_bent["ring"] = is_finger_bent(
                ring_tip,
                ring_knuckle,
                finger_bent["ring"]
            )

            finger_bent["pinky"] = is_finger_bent(
                pinky_tip,
                pinky_knuckle,
                finger_bent["pinky"]
            )

            finger_bent["thumb"] = is_thumb_bent(
                thumb_tip,
                middle_knuckle,
                finger_bent["thumb"]
            )

            # print(finger_bent)



            if (
                finger_bent["thumb"]
                and not finger_bent["index"]
                and finger_bent["middle"]
                and finger_bent["ring"]
                and finger_bent["pinky"]
            ):
                finger_position = "one"

            elif (
                finger_bent["thumb"]
                and not finger_bent["index"]
                and not finger_bent["middle"]
                and finger_bent["ring"]
                and finger_bent["pinky"]
            ):
                finger_position = "two"

            elif (
                finger_bent["thumb"]
                and not finger_bent["index"]
                and not finger_bent["middle"]
                and not finger_bent["ring"]
                and finger_bent["pinky"]
            ):
                finger_position = "three"

            elif (
                finger_bent["thumb"]
                and not finger_bent["index"]
                and not finger_bent["middle"]
                and not finger_bent["ring"]
                and not finger_bent["pinky"]
            ):
                finger_position = "four"

            elif (
                not finger_bent["thumb"]
                and not finger_bent["index"]
                and not finger_bent["middle"]
                and not finger_bent["ring"]
                and not finger_bent["pinky"]
            ):
                finger_position = "five"

            elif (
                finger_bent["thumb"]
                and not finger_bent["index"]
                and finger_bent["middle"]
                and finger_bent["ring"]
                and not finger_bent["pinky"]
            ):
                finger_position = "six (rock n roll)"

            elif (
                not finger_bent["thumb"]
                and not finger_bent["index"]
                and finger_bent["middle"]
                and finger_bent["ring"]
                and not finger_bent["pinky"]
            ):
                finger_position = "seven (spider-Man)"

            print(finger_position)

                    

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
    # print(result.handedness)

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

    latest_annotated_image = annotated_image


# def gesture_print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     if result.gestures:
#         print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))


# gesture_options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path=gesture_model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=gesture_print_result)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path= model_path,
        delegate=BaseOptions.Delegate.CPU
        ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result)


with HandLandmarker.create_from_options(options) as landmarker:
    # with GestureRecognizer.create_from_options(gesture_options) as recognizer:
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

                # recognizer.recognize_async(mp_image, frame_timestamp_ms)

                if latest_annotated_image is not None:
                    new_image = cv.cvtColor(latest_annotated_image, cv.COLOR_RGB2BGR)
                    cv.imshow('frame', new_image)
                else:
                    cv.imshow('frame', flip)
                

        
            
                
                if cv.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()






