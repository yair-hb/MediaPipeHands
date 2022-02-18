import  cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands = 2,
    min_detection_confidence =0.5) as hands:

    imagen = cv2.imread('imagen_01.jpg')
    height, width, _ = imagen.shape

    imagen = cv2.flip(imagen,1)
    imagenRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    results = hands.process(imagenRGB)
    

    print ('handedness: ', results.multi_handedness)
    print ('hand landmarks', results.multi_hand_landmarks)


if results.multi_hand_landmarks is not None:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            imagen, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color= (255,255,0),thickness=4,circle_radius=5),
            mp_drawing.DrawingSpec(color= (255,0,255),thickness=4)
        )
    
    imagen = cv2.flip(imagen,1)
cv2.imshow('MediaPipe Hands', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()



