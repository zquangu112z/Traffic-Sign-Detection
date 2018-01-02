cnn:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/CNN/CNN.py

3cnn:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/CNN/CNN_3channels_3conv_eval.py

sliding:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/object_detection.py

sign_detection:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/sign_detection.py

phan_doan_mau:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/phan_doan_mau.py

generate_images:
	PYTHONPATH="/home/zquangu112z/Fiisoft/practice-opencv" python src/utils/crop_image.py
