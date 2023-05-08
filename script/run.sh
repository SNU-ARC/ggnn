#!/bin/sh

rm -rf result
mkdir result
cd result
mkdir ggnn_whole_batch
mkdir ggnn_single_batch
mkdir whole
mkdir single
cd ..

for BATCH_OPTION in "whole true" "single false"
do
	set -- $BATCH_OPTION
	PRINT_STYLE=$1
	IS_BATCHING=$2

	for TUPLE in "sift1M 128 1000000 light" "gist1M 960 1000000 heavy" "crawl 300 1989995 heavy" "msong 420 994185 light" "glove-100 100 1183514 light" "deep1M 256 1000000 light"
	do
        	for TOP_K in 1 10
        	do
                	set -- $TUPLE
                	DATASET=$1
                	VEC_DIM=$2
                	NUM_VEC=$3
			OPTION=$4

                	sed -i "85s/.*/const int D = ${VEC_DIM};/" src/${OPTION}.cu
                	sed -i "103s/.*/const int KQuery = ${TOP_K};/" src/${OPTION}.cu
			sed -i "91s/.*/batching = ${IS_BATCHING};/" include/ggnn/cuda_knn_ggnn.cuh

                	rm -rf build_local
               		mkdir build_local
                	cd build_local
                	cmake ..
                	make
                	cd ..

                	DATASET_LOWER=$(echo ${DATASET} | tr "[:upper:]" "[:lower:]")
                	./build_local/${OPTION} --base_filename ./datasets/${DATASET}/${DATASET_LOWER}_base.fvecs --query_filename ./datasets/${DATASET}/${DATASET_LOWER}_query.fvecs --groundtruth_filename ./datasets/${DATASET}/${DATASET_LOWER}_groundtruth.ivecs --gpu_id="0" &> ./result/${PRINT_STYLE}_${DATASET}_top${TOP_K}.txt
        	done
	done
done


