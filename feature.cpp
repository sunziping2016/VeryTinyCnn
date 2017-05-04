#include <iostream>
#include <fstream>
#include <chrono>
#include <iterator>
#include <cstdlib>
#include <iomanip>
#include "CImg.h"
#include "threadpool.h"
#include "layers/conv2d.h"
#include "layers/relu.h"
#include "layers/maxpool2d.h"
#include "layers/linear.h"
#include "layers/reshape.h"
#include "layers/bias.h"

struct program_options {
    const char *alexnet, *pca, *output;
    bool binary, verbose;
    std::size_t threads_num, batch_size;
    std::vector<const char *> files;
};

program_options parse_args(int argc, const char *argv[]);
void print_options(const program_options &options);
std::shared_ptr<tnn::layer<> > load_alexnet(const char *filename);
std::shared_ptr<tnn::layer<> > load_pca(const char *filename);
template <typename Iterator>
tnn::tensor<> load_sample(Iterator first, Iterator last, tnn::thread_pool &threads);
tnn::tensor<> load_raw_features(const char *filename);
void save_result(std::ostream &out, const tnn::tensor<> &result, bool binary);

template <typename Rep, typename Period>
std::ostream &operator << (std::ostream &out, const std::chrono::duration<Rep, Period> &duration);


int main(int argc, const char *argv[])
{
    std::chrono::high_resolution_clock::time_point begin, end, forward_begin, total_begin = std::chrono::high_resolution_clock::now();

    program_options options = parse_args(argc, argv);
    if (options.verbose)
        print_options(options);

    tnn::thread_pool threads(options.threads_num);

    std::shared_ptr<tnn::layer<> > alexnet, pca;
    if (options.alexnet) {
        begin = std::chrono::high_resolution_clock::now();
        alexnet = load_alexnet(options.alexnet);
        end = std::chrono::high_resolution_clock::now();
        if (options.verbose)
            std::cout << "Alexnet loaded.\t" << (end - begin) << "\n";
    }
    if (options.pca) {
        begin = std::chrono::high_resolution_clock::now();
        pca = load_pca(options.pca);
        end = std::chrono::high_resolution_clock::now();
        if (options.verbose)
            std::cout << "PCA loaded.\t" << (end - begin) << "\n";
    }
    if (options.verbose)
        std::cout << std::endl;

    forward_begin = std::chrono::high_resolution_clock::now();
    std::ofstream out;
    if (options.output) {
        if (options.binary)
            out.open(options.output, std::ios::out | std::ios::binary);
        else
            out.open(options.output, std::ios::out);
        if (!out) {
            std::cerr << "feature: failed to open output file \"" << options.output << "\"" << std::endl;
            std::exit(1);
        }
    }
    if (options.alexnet) {
        if (options.verbose)
            std::cout << "Forward finished:" << std::endl;
        for (std::size_t i = 0; i < options.files.size(); i += options.batch_size) {
            begin = std::chrono::high_resolution_clock::now();
            std::vector<const char *>::iterator first = options.files.begin() + i, last = first + options.batch_size;
            if (last > options.files.end())
                last = options.files.end();
            tnn::tensor<> sample = load_sample(first, last, threads);
            sample = alexnet->forward(std::move(sample), threads);
            if (options.pca)
                sample = pca->forward(std::move(sample), threads);
            if (options.output)
                save_result(out, sample, options.binary);
            else
                save_result(std::cout, sample, options.binary);
            end = std::chrono::high_resolution_clock::now();
            if (options.verbose)
                std::cout << "  "  << std::setw(4) << (i / options.batch_size + 1)
                          << " (" << std::setw(6) << std::setprecision(2) << std::fixed << (100.0 * (i + sample.shape(0)) / options.files.size()) << "%)\t"
                          << (end - begin) << std::endl;
        }
    } else {
        tnn::tensor<> sample = load_raw_features(options.files.front());
        sample = pca->forward(std::move(sample), threads);
        if (options.output)
            save_result(out, sample, options.binary);
        else
            save_result(std::cout, sample, options.binary);
    }
    if (options.output)
        out.close();

    end = std::chrono::high_resolution_clock::now();
    if (options.verbose) {
        std::cout << "Forward finished.\t" << (end - forward_begin) << "\n" << std::endl;
        std::cout << "All finished.\t" << (end - total_begin) << "\n" << std::endl;
    }

    return 0;
}

const char *help_str = ""
        "Usage: feature [DATA_OPTIONS]... [OPTION]... FILE...\n"
        "Data options:\n"
        "  -a, --alexnet=FILE        binary Alexnet data\n"
        "  -p, --pca=FILE            binary PCA data\n"
        "Options:\n"
        "  -a, --alexnet=FILE        binary Alexnet data\n"
        "  -p, --pca=FILE            binary PCA data\n"
        "  -t, --threads=NUM         create NUM worker threads\n"
        "  -s, --batch=NUM           set forward batch size\n"
        "  -o, --output=FILE         set output file\n"
        "  -b, --binary              set output mode to binary\n"
        "  -v, --verbose             enable verbose mode\n"
        "  -h, --help                print this help message\n"
        "Forward flow:\n"
        "                Alexnet        PCA\n"
        "             X ---------> Y ---------> Z\n"
        "\n"
        "At least one data option should be present to run this program. And forward\n"
        "flow is changed according to data options. Batch size and extra files is\n"
        "ignored in \"Y -> Z\" mode.\n"
;

program_options parse_args(int argc, const char *argv[]) {
    program_options options {
            nullptr, nullptr, nullptr,
            false, false,
            std::thread::hardware_concurrency(), std::thread::hardware_concurrency(),
            {}
    };
    const char *temp_str, *char_p;
    int temp_int;

    for (int i = 1; i < argc; ++i) {
        int sh = 1;
        if (!std::strcmp(argv[i], "-a") || (!std::strncmp(argv[i], "--alexnet=", 10) && sh--)) {
            if (sh) {
                if (++i == argc) {
                    std::cerr << "feature: requires path to Alexnet data after \"-a\"" << std::endl;
                    std::exit(1);
                }
                options.alexnet = argv[i];
            } else
                options.alexnet = argv[i] + 10;
        } else if (!std::strcmp(argv[i], "-p") || (!std::strncmp(argv[i], "--pca=", 6) && sh--)) {
            if (sh) {
                if (++i == argc) {
                    std::cerr << "feature: requires path to PCA data after \"-p\"" << std::endl;
                    std::exit(1);
                }
                options.pca = argv[i];
            } else
                options.pca = argv[i] + 6;
        } else if (!std::strcmp(argv[i], "-o") || (!std::strncmp(argv[i], "--output=", 9) && sh--)) {
            if (sh) {
                if (++i == argc) {
                    std::cerr << "feature: requires path to output file after \"-o\"" << std::endl;
                    std::exit(1);
                }
                options.output = argv[i];
            } else
                options.output = argv[i] + 9;
        } else if (!std::strcmp(argv[i], "-t") || (!std::strncmp(argv[i], "--threads=", 10) && sh--)) {
            if (sh) {
                if (++i == argc) {
                    std::cerr << "feature: requires number of threads after \"-t\"" << std::endl;
                    std::exit(1);
                }
                temp_str = argv[i];
            } else
                temp_str = argv[i] + 10;
            for (char_p = temp_str; *char_p && *char_p >= '0' && *char_p <= '9'; ++char_p);
            if (!*temp_str || *char_p || (temp_int = std::atoi(temp_str)) < 1) {
                std::cerr << "feature: invalid number of threads" << std::endl;
                std::exit(1);
            }
            options.threads_num = temp_int;
        } else if (!std::strcmp(argv[i], "-s") || (!std::strncmp(argv[i], "--batch=", 8) && sh--)) {
            if (sh) {
                if (++i == argc) {
                    std::cerr << "feature: requires number of batch size after \"-t\"" << std::endl;
                    std::exit(1);
                }
                temp_str = argv[i];
            } else
                temp_str = argv[i] + 8;
            for (char_p = temp_str; *char_p && *char_p >= '0' && *char_p <= '9'; ++char_p);
            if (!*temp_str || *char_p || (temp_int = std::atoi(temp_str)) < 1) {
                std::cerr << "feature: invalid number of batch size" << std::endl;
                std::exit(1);
            }
            options.batch_size = temp_int;
        } else if (!std::strcmp(argv[i], "-b") || !std::strcmp(argv[i], "--binary")) {
            options.binary = true;
        } else if (!std::strcmp(argv[i], "-v") || !std::strcmp(argv[i], "--verbose")) {
            options.verbose = true;
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::cout << help_str << std::endl;
            std::exit(0);
        } else if (argv[i][0] == '-') {
            std::cerr << "feature: unrecognized option \"" << argv[i] << "\"" << std::endl;
            std::exit(1);
        } else
            options.files.push_back(argv[i]);
    }
    if (!options.alexnet && !options.pca) {
        std::cerr << "feature: requires at least one data option" << std::endl;
        exit(1);
    }
    if (options.files.empty()) {
        std::cerr << "feature: requires at least one input file" << std::endl;
        std::exit(1);
    }
    return options;
}

void print_options(const program_options &options) {
    std::cout << "Options:\n";
    if (options.alexnet && options.pca)
        std::cout << "  Forward flow:       X -> Y -> Z\n";
    else if (options.alexnet)
        std::cout << "  Forward flow:       X -> Y\n";
    else if (options.pca)
        std::cout << "  Forward flow:       Y -> Z\n";
    if (options.alexnet)
        std::cout << "  Alexnet data:       \"" << options.alexnet << "\"\n";
    if (options.pca)
        std::cout << "  PCA data:           \"" << options.pca << "\"\n";
    if (options.output)
        std::cout << "  Output file:        \"" << options.output << "\"\n";
    else
        std::cout << "  Output file:        stdout\n";
    if (options.binary)
        std::cout << "  Output mode:        binary\n";
    else
        std::cout << "  Output mode:        text\n";
    if (options.alexnet) {
        std::cout << "  Files num:          " << options.files.size() << "\n";
        std::cout << "  Batch size:         " << options.batch_size <<"\n";
    }
    std::cout << "  Threads num:        " << options.threads_num <<"\n";
    std::cout << "  AVX2 enabled:       " << std::boolalpha << AVX_ENABLED << "\n";
    std::cout << std::endl;
}

std::shared_ptr<tnn::layer<> > load_alexnet(const char *filename) {
    std::ifstream in(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!in) {
        std::cerr << "feature: failed to open Alexnet data file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    if (in.tellg() != 228015360) {
        std::cerr << "feature: invalid size of Alexnet data file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    in.seekg(0);
    std::shared_ptr<tnn::layer<> > alexnet = std::make_shared<tnn::layers<> >(std::initializer_list<std::shared_ptr<tnn::layer<> > >({
            std::make_shared<tnn::conv2d<> >(3, 64, 11, 4, 2),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::maxpool2d<> >(3, 2),
            std::make_shared<tnn::conv2d<> >(64, 192, 5, 1, 2),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::maxpool2d<> >(3, 2),
            std::make_shared<tnn::conv2d<> >(192, 384, 3, 1, 1),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::conv2d<> >(384, 256, 3, 1, 1),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::conv2d<> >(256, 256, 3, 1, 1),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::maxpool2d<> >(3, 2),
            std::make_shared<tnn::reshape<> >(std::initializer_list<size_t>({256 * 6 * 6})),
            std::make_shared<tnn::linear<> >(256 * 6 * 6, 4096),
            std::make_shared<tnn::relu<> >(),
            std::make_shared<tnn::linear<> >(4096, 4096),
            std::make_shared<tnn::relu<> >()
    }));
    alexnet->load(in);
    in.close();
    return alexnet;
}

std::shared_ptr<tnn::layer<> > load_pca(const char *filename) {
    std::ifstream in(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!in) {
        std::cerr << "feature: failed to open PCA data file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    std::size_t features = in.tellg() / sizeof(float) / 4096 - 1;
    if (!features || (std::size_t) in.tellg() != sizeof(float) * 4096 * (features + 1)) {
        std::cerr << "feature: invalid size of PCA data file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    in.seekg(0);
    std::shared_ptr<tnn::layer<> > pca = std::make_shared<tnn::layers<> >(std::initializer_list<std::shared_ptr<tnn::layer<> > >({
            std::make_shared<tnn::bias<> >(4096),
            std::make_shared<tnn::linear<> >(4096, features, false)
    }));
    pca->load(in);
    in.close();
    return pca;
}

template <typename Iterator>
tnn::tensor<> load_sample(Iterator first, Iterator last, tnn::thread_pool &threads) {
    const static float mean[] = {0.485, 0.456, 0.406}, std[] = {0.229, 0.224, 0.225};
    std::size_t batch_size = std::distance(first, last);
    tnn::tensor<> sample{batch_size, 3, 224, 224};

    std::vector<std::future<void> > sync;
    sync.reserve(threads.get_thread_num());
    std::size_t start = 0;
    double step = (double) batch_size / threads.get_thread_num();
    for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
        std::size_t end = (int) (step * (i + 1) + 0.5);
        if (start != end)
            sync.emplace_back(threads.enqueue([&sample](Iterator iter, std::size_t s, std::size_t e) {
                for (; s < e; ++iter, ++s) {
                    cimg_library::CImg<> image(1, 1, 3, 1);
                    try {
                        image.load(*iter);
                    } catch (const cimg_library::CImgIOException &error) {
                        std::cerr << "feature: " << error.what() << std::endl;
                    }
                    image /= 255;
                    std::size_t h = 256, w = 256;
                    if (image.height() > image.width())
                        h = image.height() / image.width() * 256;
                    else
                        w = image.width() / image.height() * 256;
                    image.resize(w, h, 1, 3, 3);
                    size_t js = (size_t) ((w - 224) / 2.0 + 0.5), is = (size_t) ((h - 224) / 2.0 + 0.5);
                    for (size_t k = 0; k < 3; ++k)
                        for (size_t i = 0; i < 224; ++i)
                            for (size_t j = 0; j < 224; ++j)
                                sample.at(s, k, i, j) = (image(js + j, is + i, 0, k) - mean[k]) / std[k];
                }
            }, first, start, end));
        std::advance(first, end - start);
        start = end;
    }
    for (std::size_t i = 0; i < sync.size(); ++i)
        sync[i].get();
    return sample;
}

tnn::tensor<> load_raw_features(const char *filename) {
    std::ifstream in(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!in) {
        std::cerr << "feature: failed to open raw feature file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    std::size_t n = in.tellg() / sizeof(float) / 4096;
    if (!n || (std::size_t) in.tellg() != sizeof(float) * 4096 * n) {
        std::cerr << "feature: invalid size of raw feature file \"" << filename << "\"" << std::endl;
        std::exit(1);
    }
    in.seekg(0);
    tnn::tensor<> sample{n, 4096};
    sample.load(in);
    in.close();
    return sample;
}

template <typename Rep, typename Period>
std::ostream &operator << (std::ostream &out, const std::chrono::duration<Rep, Period> &duration) {
    double nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    std::size_t precision = out.precision(5);
    if (nanoseconds > 1e9)
        out << nanoseconds / 1e9 << "s";
    else if (nanoseconds > 1e6)
        out << nanoseconds / 1e6 << "ms";
    else if (nanoseconds > 1e3)
        out << nanoseconds / 1e3 << "us";
    else
        out << nanoseconds << "ns";
    out.precision(precision);
    return out;
}

void save_result(std::ostream &out, const tnn::tensor<> &result, bool binary) {
    assert(result.ndim() == 2);
    if (binary)
        result.save(out);
    else {
        for (std::size_t i = 0; i < result.shape(0); ++i) {
            out << result.at(i, 0);
            for (std::size_t j = 1; j < result.shape(1); ++j)
                out << " " << result.at(i, j);
            out << "\n";
        }
    }
}
