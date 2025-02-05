python -m venv cam

cam\Scripts\activate.bat

python cam.py

deactivate

pip freeze > requirements.txt

pip install -r requirements.txt


pip install pillow psutil opencv-python
