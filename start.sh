docker run --gpus=all -v "$(pwd)":/home/robotdataset/project -it \
--gpus=all \
--env DISPLAY=:0  --env QT_X11_NO_MITSHM=1 \
--env XDG_RUNTIME_DIR=/root/1000 --env XAUTHORITY=/root/.Xauthority \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--network=host --privileged -v /dev/bus/usb:/dev/bus/usb \
--name robotdataset-ros2 ar0s/robotdataset:ros2 bash
