cnn:
	PYTHONPATH=./ python src/CNN/CNN.py

2conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_2conv.py

4conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_4conv.py

3conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_3conv.py

train:
	make 2conv
	make 3conv
	make 4conv
	

sliding:
	PYTHONPATH=./ python src/object_detection.py

sign_detection:
	PYTHONPATH=./ python src/sign_detection.py

phan_doan_mau:
	PYTHONPATH=./ python src/phan_doan_mau.py

generate_images:
	PYTHONPATH=./ python src/utils/crop_image.py

pickle_dataset:
	PYTHONPATH=./src python src/utils/pickle_dataset.py