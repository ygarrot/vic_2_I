# .ONESHELL:

all:
	cd src;\
		python3 ./main.py

unzip:
	for file in ./resources/*; do \
		tar -xvf $$file -C ./resources; \
	done

download:
	wget -O "resources/Plant_leaf_diseases_dataset_with_augmentation.zip" "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/b4e3a32f-c0bd-4060-81e9-6144231f2520/file_downloaded"
	wget -O "resources/Plant_leaf_diseases_dataset_without_augmentation.zip" "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
