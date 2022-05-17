#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>

#include "tscns.h"
#include "omp.h"
#include "tbb/parallel_sort.h"
#include "flags.h"
#include "utils.h"

#include "../core/lipp.h"

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class Benchmark {
    LIPP<KEY_TYPE, PAYLOAD_TYPE> index;

    enum Operation {
        READ = 0, INSERT, DELETE, SCAN, UPDATE
    };

    // parameters
    double read_ratio = 1;
    double insert_ratio = 0;
    size_t operations_num;
    long long int table_size = -1;
    size_t init_table_size;
    double init_table_ratio;
    size_t thread_num = 1;
    std::string keys_file_path;
    std::string keys_file_type;
    std::string sample_distribution;
    bool latency_sample = false;
    double latency_sample_ratio = 0.01;
    std::string output_path;
    size_t random_seed;

    std::vector <KEY_TYPE> init_keys;
    KEY_TYPE *keys;
    std::pair <KEY_TYPE, PAYLOAD_TYPE> *init_key_values;
    std::vector <std::pair<Operation, KEY_TYPE>> operations;
    std::mt19937 gen;

    struct Stat {
        std::vector<double> latency;
        uint64_t throughput = 0;
        long long memory_consumption = 0;
    } stat;

    struct alignas(CACHELINE_SIZE)
    ThreadParam {
        std::vector<std::pair<uint64_t, uint64_t>> latency;
    };
    typedef ThreadParam param_t;

public:
    Benchmark() {}

    void load_keys() {
        // Read keys from file
        // COUT_THIS("Loading keys from file.");

        if (table_size > 0) keys = new KEY_TYPE[table_size];


        if (keys_file_type == "binary") {
            table_size = load_binary_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else if (keys_file_type == "text") {
            table_size = load_text_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else {
            COUT_THIS("Could not open key file, please check the path of key file.");
            exit(0);
        }
        tbb::parallel_sort(keys, keys + table_size);
        std::shuffle(keys, keys + table_size, gen);

        init_table_size = init_table_ratio * table_size;

        init_keys.resize(init_table_size);
#pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < init_table_size; ++i) {
            init_keys[i] = (keys[i]);
        }
        tbb::parallel_sort(init_keys.begin(), init_keys.end());

        init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];
#pragma omp parallel for num_threads(thread_num)
        for (long unsigned int i = 0; i < init_keys.size(); i++) {
            init_key_values[i].first = init_keys[i];
            init_key_values[i].second = init_keys[i];
        }

        // COUT_THIS("Bulk loading.");
        index.bulk_load(init_key_values, init_keys.size());
    }

    inline void parse_args(int argc, char **argv) {
        auto flags = parse_flags(argc, argv);
        keys_file_path = get_required(flags, "keys_file");
        keys_file_type = get_with_default(flags, "keys_file_type", "binary");
        read_ratio = stod(get_required(flags, "read"));
        insert_ratio = stod(get_with_default(flags, "insert", "0"));
        operations_num = stoi(get_with_default(flags, "operations_num", "800000000"));
        table_size = stoi(get_with_default(flags, "table_size", "-1"));
        init_table_ratio = stod(get_with_default(flags, "init_table_ratio", "0.5"));
        init_table_size = -1;
        sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
        latency_sample = get_boolean_flag(flags, "latency_sample");
        latency_sample_ratio = stod(get_with_default(flags, "latency_sample_ratio", "0.01"));
        output_path = get_with_default(flags, "output_path", "./result");
        random_seed = stoul(get_with_default(flags, "seed", "1866"));
        thread_num = stoi(get_with_default(flags, "thread_num", "1"));
        gen.seed(random_seed);

        double ratio_sum = read_ratio + insert_ratio;
        INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);  // avoid precision lost
        INVARIANT(sample_distribution == "zipf" || sample_distribution == "uniform");
    }

    void generate_operations() {
        // COUT_THIS("Generating operations.");
        operations.reserve(operations_num);
        KEY_TYPE *sample_ptr = nullptr;
        if (sample_distribution == "uniform") {
            sample_ptr = get_search_keys(&init_keys[0], init_table_size, operations_num, &random_seed);
        } else if (sample_distribution == "zipf") {
            sample_ptr = get_search_keys_zipf(&init_keys[0], init_table_size, operations_num, &random_seed);
        }

        std::uniform_real_distribution<> ratio_dis(0, 1);
        size_t sample_counter = 0, insert_counter = init_table_size;

        for (size_t i = 0; i < operations_num; ++i) {
            auto prob = ratio_dis(gen);
            if (prob < read_ratio) {
                operations.push_back(std::pair<Operation, KEY_TYPE>(READ, sample_ptr[sample_counter++]));
            } else if (prob < read_ratio + insert_ratio) {
                if (insert_counter >= table_size) {
                    operations_num = i;
                    break;
                }
                operations.push_back(std::pair<Operation, KEY_TYPE>(INSERT, keys[insert_counter++]));
            }
        }

        delete[] sample_ptr;
    }

    void run() {
        std::thread *thread_array = new std::thread[thread_num];
        param_t params[thread_num];
        TSCNS tn;
        tn.init();
        // printf("Begin running\n");
        auto start_time = tn.rdtsc();
        auto end_time = tn.rdtsc();
//        System::profile("perf.data", [&]() {
#pragma omp parallel num_threads(thread_num)
        {
            // thread specifier
            auto thread_id = omp_get_thread_num();
            // Latency Sample Variable
            int latency_sample_interval = operations_num / (operations_num * latency_sample_ratio);
            auto latency_sample_start_time = tn.rdtsc();
            auto latency_sample_end_time = tn.rdtsc();
            param_t &thread_param = params[thread_id];
            thread_param.latency.reserve(operations_num / latency_sample_interval);
            // waiting all thread ready
#pragma omp barrier
#pragma omp master
            start_time = tn.rdtsc();
// running benchmark
#pragma omp for schedule(dynamic, 10000)
            for (auto i = 0; i < operations_num; i++) {
                auto op = operations[i].first;
                auto key = operations[i].second;

                if (latency_sample && i % latency_sample_interval == 0)
                    latency_sample_start_time = tn.rdtsc();

                if (op == READ) {  // get
                    PAYLOAD_TYPE val = index.at(key, false);
                    // if(val != key) {
                    //     printf("read failed, Key %lu, val %llu\n",key, val);
                    //     exit(1);
                    // }
                } else if (op == INSERT) {  // insert
                    index.insert(key, key);
                }

                if (latency_sample && i % latency_sample_interval == 0) {
                    latency_sample_end_time = tn.rdtsc();
                    thread_param.latency.push_back(std::make_pair(latency_sample_start_time, latency_sample_end_time));
                }
            } // omp for loop
#pragma omp master
            end_time = tn.rdtsc();
        } // all thread join here

//        });
        auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
        // printf("Finish running\n");

        // gather thread local variable
        for (auto &p: params) {
            if (latency_sample) {
                for (auto e : p.latency) {
                    stat.latency.push_back(tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
                }
            }
        }
        // calculate throughput
        stat.throughput = static_cast<uint64_t>(operations_num / (diff/(double) 1000000000));
        print_stat();

        delete[] thread_array;
    }

    void print_stat(bool header = false) {
        double avg_latency = 0;
        double latency_variance = 0;
        if (latency_sample) {
            for (auto t : stat.latency) {
                avg_latency += t;
                latency_variance += (t - avg_latency) * (t - avg_latency);
            }
            avg_latency /= stat.latency.size();
            latency_variance /= stat.latency.size();
            std::sort(stat.latency.begin(), stat.latency.end());
        }

        printf("Thread: %zu\tThroughput: %lu\n", thread_num, stat.throughput);

        if (!file_exists(output_path)) {
            std::ofstream ofile;
            ofile.open(output_path, std::ios::app);
            ofile << "key_path" << ",";
            ofile << "throughput" << ",";
            ofile << "thread_num" << std::endl;
        }

        std::ofstream ofile;
        ofile.open(output_path, std::ios::app);
        ofile << keys_file_path << ",";
        ofile << stat.throughput << ",";
        ofile << thread_num << std::endl;
        ofile.close();
    }
};

int main(int argc, char **argv) {
    Benchmark <uint64_t, uint64_t> bench;
    bench.parse_args(argc, argv);
    bench.load_keys();
    bench.generate_operations();
    bench.run();
}
