# facial-emotion-recognition
A simple facial emotion recognition app

## Install required python modules
Check required libraries in requirements.txt
Install using pip

```zsh
pip install --user --requirement requirements.txt
```

## Run the app

To run the emotion detection from a real time web cam:
```zsh
# From the root directory of the project, run
python src/app.py
```

To run the emotion detection with a given image file:
```zsh
# From the root directory of the project, run
python src/imageclf.py
```

To exit the app, click the window showing your real time image,
make sure it is activated, and then press `q` from the keyboard.
Please be patient, for the web cam windows, it takes a few seconds
for the app to shutdown.

