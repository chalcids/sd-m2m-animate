import launch

if not launch.is_installed("cv2"):
    print('Installing requirements for mov2mov Animate')
    launch.run_pip("install opencv-python", "requirements for opencv")

if not launch.is_installed('ffmpeg'):
        print('Installing requirements for mov2mov Animate')
        launch.run_pip("install ffmpeg", "requirements for ffmpeg")