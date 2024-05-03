from flask import Flask, Response, jsonify, render_template, request
import base64 
from flask_cors import CORS
import cv2
import mediapipe as mp
import math,time

import numpy as np
app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
video = None
start_time=0
best_time=0
def initialize_camera():
    global video
    video = cv2.VideoCapture(0)
    
    if not video.isOpened():
        return False
    return True

def release_camera():
    global video
    if video is not None:
        video.release()

def calculate_angle(a, b, c):
    radians = math.atan2(c.y-b.y, c.x-b.x) - math.atan2(a.y-b.y, a.x-b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def start_timer(pose_correct):
    global start_time , best_time

    if pose_correct == True:
        if start_time == 0:
            start_time = time.time()

        current_time = time.time()
        seconds = int(current_time - start_time)
        if seconds>best_time:
            best_time=seconds
    else:
        start_time = 0
        seconds = 0

    return seconds,best_time
        
def correct_warrior_pose(results):
    result={}
    if results.pose_landmarks:
        for i in range(33):
            result[mp_pose.PoseLandmark(i).name] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        angle_right_arm=round( calculate_angle(result['RIGHT_HIP'],result['RIGHT_SHOULDER'],result['RIGHT_ELBOW']),2)
        angle_left_arm=round(calculate_angle(result['LEFT_HIP'],result['LEFT_SHOULDER'],result['LEFT_ELBOW']),2)
        angle_right_hand=round(calculate_angle(result['RIGHT_SHOULDER'],result['RIGHT_ELBOW'],result['RIGHT_WRIST']),2)
        angle_left_hand=round(calculate_angle(result['LEFT_SHOULDER'],result['LEFT_ELBOW'],result['LEFT_WRIST']),2)
        angle_right_leg=round(calculate_angle(result['RIGHT_HIP'],result['RIGHT_KNEE'],result['RIGHT_ANKLE']),2)
        angle_left_leg=round(calculate_angle(result['LEFT_HIP'],result['LEFT_KNEE'],result['LEFT_ANKLE']),2)
        angle_right_hip=round(calculate_angle(result['LEFT_HIP'],result['RIGHT_HIP'],result['RIGHT_KNEE']),2)
        angle_left_hip=round(calculate_angle(result['RIGHT_HIP'],result['LEFT_HIP'],result['LEFT_KNEE']),2)

        

        if not (angle_left_hand>=165 and angle_left_hand<=200 and angle_left_arm>=70 and angle_left_arm<=120):
            right_hand=False
        else:
            right_hand=True
        
        if not (angle_right_hand>=165 and angle_right_hand<=200 and angle_right_arm>=70 and angle_right_arm<=120):
            left_hand=False
        else:
            left_hand=True

        if not (angle_right_hip>=110 and angle_right_hip<=145 and angle_right_leg<=150):
            left_leg=False
        else:
            left_leg=True
        
        if not(angle_left_hip>=110 and angle_left_leg>=160 and angle_left_leg<=190):
            right_leg=False
        else:
            right_leg=True

        return right_hand,left_hand,right_leg,left_leg
    else:
        return False

def correct_goddess_pose(results):
    result={}
    if results.pose_landmarks:
        for i in range(33):
            result[mp_pose.PoseLandmark(i).name] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        angle_right_leg=round(calculate_angle(result['RIGHT_HIP'],result['RIGHT_KNEE'],result['RIGHT_ANKLE']),2)
        angle_left_leg=round(calculate_angle(result['LEFT_HIP'],result['LEFT_KNEE'],result['LEFT_ANKLE']),2)
        angle_right_hand=round(calculate_angle(result['RIGHT_SHOULDER'],result['RIGHT_ELBOW'],result['RIGHT_WRIST']),2)
        angle_left_hand=round(calculate_angle(result['LEFT_SHOULDER'],result['LEFT_ELBOW'],result['LEFT_WRIST']),2)

        if not (angle_right_leg<=150 and angle_right_leg>=90):
            left_leg=False
        else:
            left_leg=True
        if not ( angle_left_leg<=160 and angle_left_leg>=90):
            right_leg=False
        else:
            right_leg=True

        if not (angle_right_hand<=100 and angle_right_hand>=80):
            left_hand=False
        else:
            left_hand=True
        
        if not ( angle_left_hand<=100 and angle_left_hand>=80):
            right_hand=False
        else:
            right_hand=True

        # if(angle_right_leg<=150 and angle_right_leg>=90 and angle_left_leg<=160 and angle_left_leg>=90 and \
        #    angle_right_hand<=100 and angle_right_hand>=80 and angle_left_hand<=100 and angle_left_hand>=80):
        #     text='correct'
        #     value=True
        
        return right_hand,left_hand,right_leg,left_leg
    else:
        return False

def correct_mountain_pose(results):
    # Initialize a list to store the detected landmarks.
    result={}
    text=''
    value=False
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        for i in range(33):
            result[mp_pose.PoseLandmark(i).name] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        angle_right_arm=round( calculate_angle(result['RIGHT_HIP'],result['RIGHT_SHOULDER'],result['RIGHT_ELBOW']),2)
        angle_left_arm=round(calculate_angle(result['LEFT_HIP'],result['LEFT_SHOULDER'],result['LEFT_ELBOW']),2)
        angle_right_hand=round(calculate_angle(result['RIGHT_SHOULDER'],result['RIGHT_ELBOW'],result['RIGHT_WRIST']),2)
        angle_left_hand=round(calculate_angle(result['LEFT_SHOULDER'],result['LEFT_ELBOW'],result['LEFT_WRIST']),2)
        angle_right_leg=round(calculate_angle(result['RIGHT_HIP'],result['RIGHT_KNEE'],result['RIGHT_ANKLE']),2)
        angle_left_leg=round(calculate_angle(result['LEFT_HIP'],result['LEFT_KNEE'],result['LEFT_ANKLE']),2)

        if not (angle_left_arm<=190 and angle_left_arm>=170 and angle_left_hand<=190 and angle_left_hand>=170 ):
            right_hand=False
        else:
            right_hand=True

        if not (angle_right_arm<=190 and angle_right_arm>=170 and angle_right_hand<=190 and angle_right_hand>=170):
            left_hand=False
        else:
            left_hand=True
        
        if not (angle_right_leg<=190 and angle_right_leg>=170):
            left_leg=False
        else:
            left_leg=True
        if not ( angle_left_leg<=190 and angle_left_leg>=170):
            right_leg=False
        else:
            right_leg=True

        return right_hand,left_hand,right_leg,left_leg
    else:
        return False

def correct_lunge_pose(results):
    # Initialize a list to store the detected landmarks.
    result={}
    text=''
    value=False
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        for i in range(33):
            result[mp_pose.PoseLandmark(i).name] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        angle_right_leg=round(calculate_angle(result['RIGHT_HIP'],result['RIGHT_KNEE'],result['RIGHT_ANKLE']),2)
        angle_left_leg=round(calculate_angle(result['LEFT_HIP'],result['LEFT_KNEE'],result['LEFT_ANKLE']),2)
        angle_right_hand=round(calculate_angle(result['RIGHT_SHOULDER'],result['RIGHT_ELBOW'],result['RIGHT_WRIST']),2)
        angle_left_hand=round(calculate_angle(result['LEFT_SHOULDER'],result['LEFT_ELBOW'],result['LEFT_WRIST']),2)
        
        if not (angle_right_leg>=80 and angle_right_leg<=120):
            left_leg=False
        else:
            left_leg=True

        if not(angle_left_leg<=150 and angle_left_leg>=70):
            right_leg=False
        else:
            right_leg=True

        if not (angle_right_hand<=190 and angle_right_hand>=170):
            left_hand=False
        else:
            left_hand=True
        
        if not (angle_left_hand<=190 and angle_left_hand>=170):
            right_hand=False
        else:
            right_hand=True
        
        # if (angle_right_leg>=80 and angle_right_leg<=120 and angle_left_leg<=150 and angle_left_leg>=70):
        #     text="correct"
        #     value=True

        return right_hand,left_hand,right_leg,left_leg
    else:
        return False

def correct_tree_pose(results):
    # Initialize a list to store the detected landmarks.
    result={}
    text=''
    value=False
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        for i in range(33):
            result[mp_pose.PoseLandmark(i).name] = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        angle_right_hand = round(calculate_angle(result['RIGHT_SHOULDER'],result['RIGHT_ELBOW'],result['RIGHT_WRIST']),2)
        angle_left_hand= round(calculate_angle(result['LEFT_SHOULDER'],result['LEFT_ELBOW'],result['LEFT_WRIST']),2)
        angle_right_leg = round(calculate_angle(result['RIGHT_HIP'],result['RIGHT_KNEE'],result['RIGHT_ANKLE']),2)
        angle_left_leg= round(calculate_angle(result['LEFT_HIP'],result['LEFT_KNEE'],result['LEFT_ANKLE']),2)
        
        if not (angle_right_hand<=50):
            left_hand=False
        else:
            left_hand=True

        if not ( angle_left_hand<=50):
            right_hand=False
        else:
            right_hand=True

        if not (angle_right_leg>=170):
            left_leg=False
        else:
            left_leg=True

        if not ( angle_left_leg<=50):
            right_leg=False
        else:
            right_leg=True
        
        # if angle_right_hand<=50 and angle_left_hand<=50 and angle_right_leg>=170 and angle_left_leg<=50:
        #     text="correct"
        #     value=True

        return right_hand,left_hand,right_leg,left_leg
    else:
        return False
          
# Define correction functions for other yoga poses similarly
    

def define_color(body_part):
    correct_color = (200, 255, 0)
    incorrect_color = (0, 0, 0)
    if body_part:
        font_color=correct_color
    else:
        font_color=incorrect_color
    return font_color
    
def posture_card(right_hand,left_hand,right_leg,left_leg,image_path,frame, pose_correct):
    def draw_rectangle(board_height_1,board_width_1,board_pos_x_1,board_pos_y_1,board_color_1,board_thickness_1,board_radius_1):
        # Draw the outer rectangle
        cv2.rectangle(frame, (board_pos_x_1 + board_radius_1, board_pos_y_1),
            (board_pos_x_1 + board_width_1 - board_radius_1, board_pos_y_1 + board_height_1),
            board_color_1, board_thickness_1)
        cv2.rectangle(frame, (board_pos_x_1, board_pos_y_1 + board_radius_1),
            (board_pos_x_1 + board_width_1, board_pos_y_1 + board_height_1 - board_radius_1),
            board_color_1, board_thickness_1)
        # Draw the circles at the corners to create rounded effect
        cv2.circle(frame, (board_pos_x_1 + board_radius_1, board_pos_y_1 + board_radius_1-1), board_radius_1, board_color_1, -1)
        cv2.circle(frame, (board_pos_x_1 + board_width_1 - board_radius_1, board_pos_y_1 + board_radius_1-1), board_radius_1, board_color_1, -1)
        cv2.circle(frame, (board_pos_x_1 + board_radius_1, board_pos_y_1 + board_height_1 - board_radius_1 + 1), board_radius_1, board_color_1, -1)
        cv2.circle(frame, (board_pos_x_1 + board_width_1 - board_radius_1, board_pos_y_1 + board_height_1 - board_radius_1 +1), board_radius_1, board_color_1, -1)

    # Define parameters for the rounded rectangle
    board_height = 350
    board_width = 180
    board_pos_x = 10
    board_pos_y = 20
    board_color = (255,150,150)
    board_thickness = cv2.FILLED
    board_radius = 20

    draw_rectangle(board_height,board_width,board_pos_x,board_pos_y,board_color,board_thickness,board_radius)
            
    font_scale = 0.4
    font_color=(0,0,0)
    font_thickness = 1
    cv2.putText(frame, "POSTURE BOARD", (30,45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),1)
                            
    image = cv2.imread(image_path)
    # Resize the image
    new_width = 130
    new_height = 140 
    resized_image = cv2.resize(image, (new_width, new_height))
    image_height, image_width, _ = resized_image.shape
    position_x = 30  # Adjust as needed
    position_y = 55  # Adjust as needed
    frame[position_y:position_y+image_height, position_x:position_x+image_width] = resized_image
                            
    # Define parameters for the rounded rectangle
    board_height_1 = 20
    board_width_1 = 100
    board_pos_x_1 = 82
    board_pos_y_1 = 205
    board_color_1 = (255,255,255)
    board_thickness_1 = cv2.FILLED
    board_radius_1 = 10
                
    # right hand
    draw_rectangle(board_height_1,board_width_1,board_pos_x_1,board_pos_y_1,board_color_1,board_thickness_1,board_radius_1)
    cv2.putText(frame, "Right arm", (15,218), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    cv2.putText(frame,".Correct" if right_hand else "Incorrect" ,(103,218),cv2.FONT_HERSHEY_SIMPLEX,font_scale,define_color(right_hand),font_thickness)
                
    # left hand
    draw_rectangle(board_height_1,board_width_1,board_pos_x_1,board_pos_y_1+25,board_color_1,board_thickness_1,board_radius_1)
    cv2.putText(frame, "Left arm", (15,243), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    cv2.putText(frame,".Correct" if left_hand else "Incorrect" ,(103,243),cv2.FONT_HERSHEY_SIMPLEX,font_scale,define_color(left_hand),font_thickness)
                
    #right leg
    draw_rectangle(board_height_1,board_width_1,board_pos_x_1,board_pos_y_1+50,board_color_1,board_thickness_1,board_radius_1)
    cv2.putText(frame, "Right leg", (15,268), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    cv2.putText(frame,".Correct" if right_leg else "Incorrect" ,(103,268),cv2.FONT_HERSHEY_SIMPLEX,font_scale,define_color(right_leg),font_thickness)
                
    #left leg
    draw_rectangle(board_height_1,board_width_1,board_pos_x_1,board_pos_y_1+75,board_color_1,board_thickness_1,board_radius_1)
    cv2.putText(frame, "Left leg", (15,293), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    cv2.putText(frame,".Correct" if left_leg else "Incorrect" ,(103,293),cv2.FONT_HERSHEY_SIMPLEX,font_scale,define_color(left_leg),font_thickness)

    seconds,best_time=start_timer(pose_correct)
    draw_rectangle(board_height_1,board_width_1+60,board_pos_x_1-62,board_pos_y_1+115,board_color_1,board_thickness_1,board_radius_1)
    cv2.putText(frame,f"Timer: {seconds:.2f}s",(board_pos_x_1-25,board_pos_y_1+129),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1) 

def trigger_model(model_name):
    global video

    cv2.namedWindow(model_name, cv2.WINDOW_NORMAL) 
    cv2.setWindowProperty(model_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    cv2.setWindowProperty(model_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)  # Disable autosize
                            
    while True:
        ret, frame = video.read()
        if not ret:
            return {'error': 'Failed to read frame from camera'}
        
        frame = cv2.flip(frame, 1)  
        results = pose_video.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if model_name == 'warrior_pose': # warrior pose
                pose_correct=False
                right_hand,left_hand,right_leg,left_leg = correct_warrior_pose(results)
                pose_correct = True if(right_hand and left_hand and right_leg and left_leg) else False
                posture_card(right_hand,left_hand,right_leg,left_leg,'./warrior.jpg',frame,pose_correct)
                
            elif model_name == 'goddess_pose': 
                right_hand,left_hand,right_leg,left_leg = correct_goddess_pose(results)
                pose_correct = True if(right_hand and left_hand and right_leg and left_leg) else False
                posture_card(right_hand,left_hand,right_leg,left_leg,'./goddess.jpg',frame,pose_correct)

            elif model_name == 'mountain_pose': #mountain pose
                right_hand,left_hand,right_leg,left_leg = correct_mountain_pose(results)
                pose_correct = True if(right_hand and left_hand and right_leg and left_leg) else False
                posture_card(right_hand,left_hand,right_leg,left_leg,'./tadasana.jpg',frame,pose_correct)

            elif model_name =='lunge_pose':  #lunge pose
                right_hand,left_hand,right_leg,left_leg = correct_lunge_pose(results)
                pose_correct = True if(right_hand and left_hand and right_leg and left_leg) else False
                posture_card(right_hand,left_hand,right_leg,left_leg,'./lunge.jpg',frame,pose_correct)

            elif model_name == 'tree_pose':  #tree pose
                right_hand,left_hand,right_leg,left_leg = correct_tree_pose(results)
                pose_correct = True if(right_hand and left_hand and right_leg and left_leg) else False
                posture_card(right_hand,left_hand,right_leg,left_leg,'./tree.jpg',frame,pose_correct)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        cv2.imshow(model_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    release_camera()
    cv2.destroyAllWindows() 
    return {'message': 'Model finished execution','best_time':best_time}

@app.route('/run-model', methods=['POST'])
def run_model_route():
    global video
    if not video or not video.isOpened():
        video = cv2.VideoCapture(0)

    data = request.json
    model_name = data.get('modelName')
    
    # Determine which model to run based on model_name
    if model_name == 'warrior_pose':
        result = trigger_model(model_name)
        return jsonify(result)

    elif model_name == 'goddess_pose':
        result = trigger_model(model_name)
        return jsonify(result)

    elif model_name == 'mountain_pose':
        result = trigger_model(model_name)
        return jsonify(result)

    elif model_name == 'lunge_pose':
        result = trigger_model(model_name)
        return jsonify(result)

    elif model_name == 'tree_pose':
        result = trigger_model(model_name)
        return jsonify(result)

    return jsonify({'message': 'Model executed successfully'})

if __name__ == '__main__':
    app.run(debug=True)
