import cv2
import os
import shutil
import mediapipe as mp

# Pfade anpassen
usb_stick_path = 'E:/ayse fotos (2).zip/iCloud Photos'  # Pfad zum USB-Stick
output_path = 'E:/ayse fotos (2).zip/iCloud Photos/sorted'  # Pfad für die sortierten Dateien
reference_image_path = 'D:\9B2C1CB2-052D-44F7-ACC5-7C1FD96722BF.jpeg'  # Pfad zum Referenzbild des Kindes

# Ordner erstellen
child_folder = os.path.join(output_path, 'child')
no_child_folder = os.path.join(output_path, 'no_child')
videos_folder = os.path.join(output_path, 'videos')
failed_folder = os.path.join(output_path, 'failed')

os.makedirs(child_folder, exist_ok=True)
os.makedirs(no_child_folder, exist_ok=True)
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(failed_folder, exist_ok=True)

# Gesichtserkennung initialisieren
mp_face_detection = mp.solutions.face_detection
mp_face_recognition = mp.solutions.face_mesh

# Funktion zur Erkennung des Gesichts des Kindes
def load_reference_face(reference_image_path):
    """
    Diese Funktion lädt das Referenzbild des Kindes und gibt die Gesichtserkennungsergebnisse zurück.
    
    Parameter:
    reference_image_path: Pfad zum Referenzbild des Kindes.
    
    Rückgabe:
    Die Gesichtserkennungsergebnisse des Referenzbildes.
    """
    image = cv2.imread(reference_image_path)
    if image is None:
        raise ValueError(f"Konnte das Referenzbild nicht laden: {reference_image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_recognition.FaceMesh() as face_mesh:
        results = face_mesh.process(image_rgb)
        return results

def is_child_face(image, reference_face_results):
    """
    Diese Funktion überprüft, ob im Bild das Gesicht des Kindes vorhanden ist.
    Dafür wird Mediapipe's Gesichtserkennung verwendet und mit dem Referenzgesicht verglichen.
    
    Parameter:
    image: Das zu überprüfende Bild.
    reference_face_results: Gesichtserkennungsergebnisse des Referenzbildes.
    
    Rückgabe:
    True, wenn das Gesicht des Kindes erkannt wird, sonst False.
    """
    if image is None:
        return False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_recognition.FaceMesh() as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                return True
    return False

def process_files(root_path, reference_face_results):
    """
    Diese Funktion durchsucht alle Dateien im angegebenen Pfad,
    identifiziert Videos und Bilder, und sortiert sie entsprechend.
    
    Parameter:
    root_path: Der Pfad, der durchsucht werden soll.
    reference_face_results: Gesichtserkennungsergebnisse des Referenzbildes.
    """
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    # Videos verschieben
                    shutil.move(file_path, os.path.join(videos_folder, file))
                elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    # Bilder verarbeiten
                    image = cv2.imread(file_path)
                    if image is None:
                        shutil.move(file_path, os.path.join(failed_folder, file))
                        continue
                    if is_child_face(image, reference_face_results):
                        # Bild des Kindes
                        shutil.move(file_path, os.path.join(child_folder, file))
                    else:
                        # Bild ohne Kind
                        shutil.move(file_path, os.path.join(no_child_folder, file))
            except Exception as e:
                print(f"Fehler bei der Verarbeitung der Datei {file_path}: {e}")
                shutil.move(file_path, os.path.join(failed_folder, file))

# Referenzgesicht des Kindes laden
reference_face_results = load_reference_face(reference_image_path)

# Dateien auf dem USB-Stick verarbeiten
process_files(usb_stick_path, reference_face_results)