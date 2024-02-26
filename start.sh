docker run --gpus=all -v "$(pwd)":/home/teleop/project -it \
--gpus=all \
--env DISPLAY=$DISPLAY  --env QT_X11_NO_MITSHM=1 \
--env XDG_RUNTIME_DIR=/root/1000 --env XAUTHORITY=/root/.Xauthority \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--network=host --privileged -v /dev/bus/usb:/dev/bus/usb \
--name panda-teleop ar0s/panda-teleop bash
