cnn:
	PYTHONPATH=./ python src/CNN/CNN.py

2conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_2conv.py

test_gpu:
	PYTHONPATH=./ python src/CNN/test_gpu.py

4conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_4conv.py

3conv:
	PYTHONPATH=./ python src/CNN/CNN_3channels_3conv.py

train:
	make 2conv
	make 3conv
	make 4conv

gen_sample:
	PYTHONPATH=./ python src/utils/gen_sample.py

	

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