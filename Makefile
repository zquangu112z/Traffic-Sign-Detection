cnn:
	PYTHONPATH=./ python src/CNN/CNN.py

3cnn_4conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_4conv.py

3cnn_3conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_3conv.py

train:
	make 3cnn_3conv
	make 3cnn_4conv
	


3cnn_2conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_2conv.py

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