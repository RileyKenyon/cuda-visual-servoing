gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=120/1, format=NV12'! nvvidconv flip-method=2 ! 'video/x-raw,width=960,height=616'! nvvidconv ! nvegltransform ! nveglglessink -e

