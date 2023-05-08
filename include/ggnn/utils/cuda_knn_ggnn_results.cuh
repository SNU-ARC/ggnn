/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch
//

#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_RESULTS_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_RESULTS_CUH_

#include <algorithm>
#include <limits>
#include <string>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/query/cuda_knn_ggnn_query.cuh"

template <DistanceMeasure measure, typename KeyT, typename ValueT, typename BaseT, typename BAddrT,
	 int KQuery>
	 struct GGNNResults {
		 KeyT* h_sorted_ids_gpu;
		 ValueT* h_sorted_dists_gpu;

		 KeyT* h_sorted_ids;

		 const Dataset<KeyT, BaseT, BAddrT>* dataset;

		 const int num_gpus;
		 const int num_iterations;

		 const int num_results_per_gpu;
		 const int num_results;

		 GGNNResults(const Dataset<KeyT, BaseT, BAddrT>* dataset,
				 const int num_gpus = 1, const int num_iterations = 1)
			 : dataset{dataset},
			 num_gpus{num_gpus},
			 num_iterations{num_iterations},
			 num_results_per_gpu{dataset->N_query * KQuery * num_iterations},
			 num_results{num_results_per_gpu * num_gpus} {
				 CHECK_CUDA(cudaMallocHost(&h_sorted_ids_gpu, num_results * sizeof(KeyT),
							 cudaHostAllocPortable));
				 CHECK_CUDA(cudaMallocHost(&h_sorted_dists_gpu, num_results * sizeof(ValueT),
							 cudaHostAllocPortable));

				 h_sorted_ids = (KeyT*)malloc(dataset->N_query * KQuery * sizeof(KeyT));
			 }

		 ~GGNNResults() {
			 cudaFreeHost(h_sorted_ids_gpu);
			 cudaFreeHost(h_sorted_dists_gpu);
			 free(h_sorted_ids);
		 }

		 GGNNResults(const GGNNResults&) = delete;
		 GGNNResults(GGNNResults&&) = delete;
		 GGNNResults& operator=(const GGNNResults&) = delete;
		 GGNNResults& operator=(GGNNResults&&) = delete;

		 void loadAsync(const GGNNQuery<KeyT, ValueT, BaseT>& ggnn_query,
				 const int gpu_index, const cudaStream_t stream, float MULT) {
			/*
			if (MULT != 1) {
				cudaFreeHost(h_sorted_ids_gpu);
				cudaFreeHost(h_sorted_dists_gpu);
				free(h_sorted_ids);

				CHECK_CUDA(cudaMallocHost(&h_sorted_ids_gpu, dataset->N_query * MULT * KQuery * sizeof(KeyT),
                                                         cudaHostAllocPortable));
                                CHECK_CUDA(cudaMallocHost(&h_sorted_dists_gpu, dataset->N_query * MULT * KQuery * sizeof(ValueT),
                                                         cudaHostAllocPortable));
                                h_sorted_ids = (KeyT*)malloc(dataset->N_query * MULT * KQuery * sizeof(KeyT));
			}
			*/
			 
		 	CHECK_CUDA(cudaMemcpyAsync(h_sorted_ids_gpu,
							ggnn_query.d_query_result_ids_sorted,
							dataset->N_query * MULT * KQuery * sizeof(KeyT), cudaMemcpyDeviceToHost, stream));
		 	CHECK_CUDA(cudaMemcpyAsync(h_sorted_dists_gpu,
							ggnn_query.d_query_result_dists_sorted,
							dataset->N_query * MULT * KQuery * sizeof(ValueT), cudaMemcpyDeviceToHost, stream));
		 }

		 void merge(float MULT) {
			 // If there is only one gpu, do just copy over the results to the expected memory.
			 if(num_gpus == 1) {
				 if(num_iterations == 1){
					 std::copy_n(h_sorted_ids_gpu, (int)(dataset->N_query * MULT * KQuery), h_sorted_ids);
				 } else {
					 for (int n = 0; n <  dataset->N_query; n++) {
						 std::copy_n(h_sorted_ids_gpu + n * KQuery * num_iterations, KQuery, h_sorted_ids + n * KQuery);
					 }     
				 }
				 return;
			 }

			 const int N_partition = dataset->N_base / num_gpus;
			 const int stride = KQuery * num_iterations;

			 auto start = std::chrono::steady_clock::now();

			 auto mergeResultPart = [&](int begin, int end) -> void {
				 struct KeyDistPartition {
					 KeyT key;
					 ValueT dist;
					 int partition;

					 KeyDistPartition(KeyT key, ValueT dist, int partition)
						 : key(key), dist(dist), partition(partition) {}
				 };
				 auto compare_heap = [](const KeyDistPartition& a,
						 const KeyDistPartition& b) -> bool {
					 return a.dist >= b.dist;
				 };

				 const int num_parts = num_gpus;
				 std::vector<int> part_offsets(num_parts, 1);

				 std::vector<KeyDistPartition> heap;
				 heap.reserve(num_parts);
				 for (int n = begin; n < end; ++n) {
					 heap.clear();
					 std::fill(part_offsets.begin(), part_offsets.end(), 1);
					 // fill heap with min per partition
					 for (int part_id = 0; part_id < num_parts; ++part_id) {
						 const size_t pos = (part_id * dataset->N_query + n) * stride;
						 heap.emplace_back(h_sorted_ids_gpu[pos], h_sorted_dists_gpu[pos],
								 part_id);
					 }
					 std::make_heap(heap.begin(), heap.end(), compare_heap);
					 // pop min and insert from popped partition until full
					 // we can safely assume not to run out of bounds within each partition
					 for (int k = 0; k < KQuery; ++k) {
						 const KeyDistPartition top = heap.front();
						 h_sorted_ids[n * KQuery + k] = top.partition * N_partition + top.key;
						 if (k == KQuery - 1) break;

						 std::pop_heap(heap.begin(), heap.end(), compare_heap);
						 heap.pop_back();
						 const size_t pos = (top.partition * dataset->N_query + n) * stride +
							 part_offsets[top.partition];
						 ++part_offsets[top.partition];
						 heap.emplace_back(h_sorted_ids_gpu[pos], h_sorted_dists_gpu[pos],
								 top.partition);
						 std::push_heap(heap.begin(), heap.end(), compare_heap);
					 }
				 }
			 };

			 std::vector<std::thread> mergeThreads;

			 int num_threads = std::min(dataset->N_query,
					 int(std::thread::hardware_concurrency()));
			 int elements_per_bin = (dataset->N_query+num_threads-1)/num_threads;
			 mergeThreads.reserve(num_threads);
			 for (int i=0; i<num_threads; ++i) {
				 mergeThreads.emplace_back(mergeResultPart, i*elements_per_bin,
						 std::min(dataset->N_query, (i+1)*elements_per_bin));
			 }
			 for (auto&& t : mergeThreads) {
				 t.join();
			 }

			 auto end = std::chrono::steady_clock::now();
			 auto cpu_ms =
				 std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			 VLOG(0) << "[CPU] partial merge completed. " << cpu_ms.count() << " ms.";
		 }

		 void evaluateResults(int i, float MULT) {
			 static int COUNT;
			 static int CORRECT;

			 if (i == 0) {
			 	COUNT = 0;
				CORRECT = 0;
			 }
	
			 int BATCH_SIZE = dataset->N_query * MULT;
			 for (int n = 0; n < BATCH_SIZE; n++) {
				 int cKQuery = 0;

				 for (int k_result = 0; k_result < KQuery; k_result++) {
					 KeyT q = h_sorted_ids[n * KQuery + k_result];

					 for (int k_gt = 0; k_gt < KQuery; k_gt++) {
						 KeyT gt = dataset->gt[(i * BATCH_SIZE + n) * dataset->K_gt + k_gt];
		
						 if (q == gt) {
							++cKQuery;
							break;
						 }
					 }
				 }
				 CORRECT += cKQuery;
				 COUNT += KQuery;
			 }

			 if (COUNT == KQuery * dataset->N_query)
			 	LOG(INFO) << "r@" << KQuery << " : " << CORRECT << " out of " << COUNT << "\n";
		 }
	 };

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_GGNN_RESULTS_CUH_
