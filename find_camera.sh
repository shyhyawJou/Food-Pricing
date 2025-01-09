declare -a VIDEO_DEV=(`v4l2-ctl --list-devices | grep mtk-v4l2-camera -A 3 | grep video | tr -d "\n"`)
printf "Preview Node\t= ${VIDEO_DEV[0]}\nVideo Node\t= ${VIDEO_DEV[1]}\nCapture Node\t= ${VIDEO_DEV[2]}\n"
