#!/bin/bash

#
# This script downloads and unpacks the ibug 300W dataset 
# into the directory it's located in.
#

if [[ $(dirname "$0") != "." ]]; then
       echo "This script must be run from the directory it's located in."
       exit 1
fi

wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
tar -xvf ibug_300W_large_face_landmark_dataset.tar.gz
rm ibug_300W_large_face_landmark_dataset.tar.gz
mv ibug_300W_large_face_landmark_dataset/* .
exit 0
