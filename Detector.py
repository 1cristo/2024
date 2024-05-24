#Se ejecuta primero
import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.python.solutions.holistic import Holistic, FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks

# Funciones auxiliares
def create_folder(path):
    """Crea una carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def configurar_resolucion(camara):
    """Configura la resolución de la cámara."""
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_keypoints(image, results):
    """Dibuja los keypoints en la imagen."""
    if results.face_landmarks:
        draw_landmarks(
            image,
            results.face_landmarks,
            FACEMESH_CONTOURS,
            DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        draw_landmarks(
            image,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        draw_landmarks(
            image,
            results.left_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        draw_landmarks(
            image,
            results.right_hand_landmarks,
            HAND_CONNECTIONS,
            DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

def save_frames(frames, output_folder):
    """Guarda los frames en la carpeta especificada."""
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

def extract_keypoints(results):
    """Extrae los keypoints de los resultados de Mediapipe."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def there_hand(results):
    """Verifica si hay manos detectadas en los resultados."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def mediapipe_detection(image, model):
    """Realiza la detección usando Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def capture_samples(path, margin_frame=2, min_cant_frames=5):
    """Captura muestras de video y guarda los frames en la carpeta especificada."""
    create_folder(path)
    
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image, results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2)
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-margin_frame]
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1
                
                frames = []
                count_frame = 0
                cv2.putText(image, 'Listo para capturar...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 100), 2)
                
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_dir = 'sign_language_dataset'
    word_name = "hola"
    word_path = os.path.join(dataset_dir, word_name)
    capture_samples(word_path)
