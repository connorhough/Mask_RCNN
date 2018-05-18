
#!/bin/bash

mkdir ../coco

declare -a backbones=("http://images.cocodataset.org/zips/train2014.zip" "http://images.cocodataset.org/zips/val2014.zip" "http://images.cocodataset.org/zips/test2014.zip" "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "http://images.cocodataset.org/annotations/image_info_test2014.zip" "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip" "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip" )


for i in "${backbones[@]}"
do
    wget -P ../coco/ $i
    wait
done


