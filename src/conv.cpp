#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "ops.h"
#include "tensor.h"

#if __has_include(<png.h>)
#include <png.h>
#define CONV_HAS_LIBPNG 1
#else
#define CONV_HAS_LIBPNG 0
#endif

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> pixels; // grayscale in [0, 255]
};

std::vector<float> kernel_from_name(const std::string& name, int& ksize);

static std::string to_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static bool has_extension(const std::string& path, const std::string& ext) {
    std::string lower_path = to_lower_copy(path);
    std::string lower_ext = to_lower_copy(ext);
    if (lower_path.size() < lower_ext.size()) {
        return false;
    }
    return lower_path.compare(lower_path.size() - lower_ext.size(), lower_ext.size(), lower_ext) == 0;
}

static void skip_comments(std::istream& in) {
    while (true) {
        in >> std::ws;
        if (in.peek() == '#') {
            std::string line;
            std::getline(in, line);
            continue;
        }
        break;
    }
}

Image load_pgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    std::string magic;
    in >> magic;
    if (magic != "P2" && magic != "P5") {
        throw std::runtime_error("Unsupported image format. Use PGM P2 or P5.");
    }

    skip_comments(in);
    int w = 0, h = 0;
    in >> w >> h;
    if (w <= 0 || h <= 0) {
        throw std::runtime_error("Invalid image dimensions.");
    }

    skip_comments(in);
    int max_val = 0;
    in >> max_val;
    if (max_val <= 0 || max_val > 65535) {
        throw std::runtime_error("Invalid PGM max value.");
    }

    in.get();

    Image img;
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<size_t>(w) * h);

    if (magic == "P2") {
        for (int i = 0; i < w * h; ++i) {
            int v = 0;
            in >> v;
            if (!in) {
                throw std::runtime_error("Malformed P2 PGM data.");
            }
            img.pixels[i] = static_cast<float>(v) * (255.0f / static_cast<float>(max_val));
        }
    } else {
        if (max_val <= 255) {
            std::vector<unsigned char> data(static_cast<size_t>(w) * h);
            in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!in) {
                throw std::runtime_error("Malformed P5 PGM data.");
            }
            for (size_t i = 0; i < data.size(); ++i) {
                img.pixels[i] = static_cast<float>(data[i]) * (255.0f / static_cast<float>(max_val));
            }
        } else {
            for (int i = 0; i < w * h; ++i) {
                unsigned char hi = 0;
                unsigned char lo = 0;
                in.read(reinterpret_cast<char*>(&hi), 1);
                in.read(reinterpret_cast<char*>(&lo), 1);
                if (!in) {
                    throw std::runtime_error("Malformed 16-bit P5 PGM data.");
                }
                int v = (static_cast<int>(hi) << 8) | static_cast<int>(lo);
                img.pixels[i] = static_cast<float>(v) * (255.0f / static_cast<float>(max_val));
            }
        }
    }

    return img;
}

void save_pgm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    std::vector<unsigned char> data(static_cast<size_t>(img.width) * img.height);

    for (size_t i = 0; i < data.size(); ++i) {
        float v = std::clamp(img.pixels[i], 0.0f, 255.0f);
        data[i] = static_cast<unsigned char>(std::lround(v));
    }

    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

Image load_png(const std::string& path) {
#if CONV_HAS_LIBPNG
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Cannot open input PNG file: " + path);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        throw std::runtime_error("png_create_read_struct failed");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        throw std::runtime_error("Error while reading PNG");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = static_cast<int>(png_get_image_width(png, info));
    int height = static_cast<int>(png_get_image_height(png, info));
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }
    if ((color_type & PNG_COLOR_MASK_ALPHA) == 0) {
        png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
    }

    png_read_update_info(png, info);

    size_t rowbytes = png_get_rowbytes(png, info);
    std::vector<unsigned char> raw(static_cast<size_t>(height) * rowbytes);
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        rows[static_cast<size_t>(y)] = raw.data() + static_cast<size_t>(y) * rowbytes;
    }

    png_read_image(png, rows.data());
    png_read_end(png, nullptr);

    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    Image out;
    out.width = width;
    out.height = height;
    out.pixels.resize(static_cast<size_t>(width) * height);

    for (int y = 0; y < height; ++y) {
        const unsigned char* row = rows[static_cast<size_t>(y)];
        for (int x = 0; x < width; ++x) {
            const unsigned char r = row[4 * x + 0];
            const unsigned char g = row[4 * x + 1];
            const unsigned char b = row[4 * x + 2];
            out.pixels[static_cast<size_t>(y) * width + x] =
                0.299f * static_cast<float>(r) +
                0.587f * static_cast<float>(g) +
                0.114f * static_cast<float>(b);
        }
    }

    return out;
#else
    (void)path;
    throw std::runtime_error("PNG support is not available. Install libpng-dev to enable it.");
#endif
}

void save_png(const std::string& path, const Image& img) {
#if CONV_HAS_LIBPNG
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Cannot open output PNG file: " + path);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        throw std::runtime_error("png_create_write_struct failed");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        throw std::runtime_error("Error while writing PNG");
    }

    png_init_io(png, fp);
    png_set_IHDR(
        png,
        info,
        static_cast<png_uint_32>(img.width),
        static_cast<png_uint_32>(img.height),
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    std::vector<unsigned char> data(static_cast<size_t>(img.width) * img.height);
    for (size_t i = 0; i < data.size(); ++i) {
        float v = std::clamp(img.pixels[i], 0.0f, 255.0f);
        data[i] = static_cast<unsigned char>(std::lround(v));
    }

    std::vector<png_bytep> rows(static_cast<size_t>(img.height));
    for (int y = 0; y < img.height; ++y) {
        rows[static_cast<size_t>(y)] = data.data() + static_cast<size_t>(y) * img.width;
    }

    png_write_image(png, rows.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
#else
    (void)path;
    (void)img;
    throw std::runtime_error("PNG support is not available. Install libpng-dev to enable it.");
#endif
}

static inline int idx(int x, int y, int width) {
    return y * width + x;
}

Image apply_kernel_with_ops(const Image& input, const std::vector<float>& kernel, int ksize, int stride, int padding) {
    if (ksize <= 0 || (ksize % 2) == 0) {
        throw std::runtime_error("Kernel size must be a positive odd number.");
    }
    if (static_cast<int>(kernel.size()) != ksize * ksize) {
        throw std::runtime_error("Kernel data size does not match kernel dimensions.");
    }

    auto x = std::make_shared<Tensor>(
        std::vector<int>{1, 1, input.height, input.width},
        input.pixels
    );

    auto w = std::make_shared<Tensor>(
        std::vector<int>{1, 1, ksize, ksize},
        kernel
    );

    auto b = Tensor::zeros({1});

    auto y = conv2d(x, w,stride, padding);

    if (y->getDimension() != 4 || y->getShape()[0] != 1 || y->getShape()[1] != 1) {
        throw std::runtime_error("Unexpected output shape from ops::conv2d");
    }

    Image out;
    out.height = y->getShape()[2];
    out.width = y->getShape()[3];
    out.pixels.assign(y->getData(), y->getData() + y->getSize());
    return out;
}

Image load_image_auto(const std::string& path) {
    if (has_extension(path, ".png")) {
        return load_png(path);
    }
    return load_pgm(path);
}

void save_image_auto(const std::string& path, const Image& img) {
    if (has_extension(path, ".png")) {
        save_png(path, img);
    } else {
        save_pgm(path, img);
    }
}

float mean_abs_diff(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size()) {
        throw std::runtime_error("mean_abs_diff requires images with the same shape");
    }

    float acc = 0.0f;
    for (size_t i = 0; i < a.pixels.size(); ++i) {
        acc += std::fabs(a.pixels[i] - b.pixels[i]);
    }
    return acc / static_cast<float>(a.pixels.size());
}

bool run_real_image_test(const std::string& input_path, const std::string& output_prefix) {
    Image in = load_image_auto(input_path);

    const std::vector<std::string> test_kernels = {
        "identity", "gaussian", "sharpen", "edge", "sobelx", "sobely"
    };

    std::unordered_map<std::string, Image> outputs;
    std::cout << "[test] loaded image: " << in.width << "x" << in.height << "\n";

    for (const std::string& kname : test_kernels) {
        int ksize = 0;
        std::vector<float> kernel = kernel_from_name(kname, ksize);
        Image out = apply_kernel_with_ops(in, kernel, ksize, 1, ksize / 2);
        outputs[kname] = out;

        std::string out_path = output_prefix + "_" + kname + ".png";
        save_image_auto(out_path, out);
        std::cout << "[test] wrote: " << out_path << "\n";
    }

    float identity_mad = mean_abs_diff(in, outputs["identity"]);
    float edge_mad = mean_abs_diff(in, outputs["edge"]);

    bool ok = true;

    if (identity_mad > 1e-3f) {
        std::cout << "[test][fail] identity kernel changed the image, MAD=" << identity_mad << "\n";
        ok = false;
    } else {
        std::cout << "[test][pass] identity check, MAD=" << identity_mad << "\n";
    }

    if (edge_mad < 0.5f) {
        std::cout << "[test][fail] edge kernel barely changed the image, MAD=" << edge_mad << "\n";
        ok = false;
    } else {
        std::cout << "[test][pass] edge check, MAD=" << edge_mad << "\n";
    }

    return ok;
}

std::vector<float> kernel_from_name(const std::string& name, int& ksize) {
    static const std::unordered_map<std::string, std::vector<float>> kernels = {
        {"identity", {0, 0, 0, 0, 1, 0, 0, 0, 0}},
        {"blur", {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9}},
        {"gaussian", {1.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 4.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 1.0f / 16}},
        {"sharpen", {0, -1, 0, -1, 5, -1, 0, -1, 0}},
        {"edge", {-1, -1, -1, -1, 8, -1, -1, -1, -1}},
        {"emboss", {-2, -1, 0, -1, 1, 1, 0, 1, 2}},
        {"sobelx", {-1, 0, 1, -2, 0, 2, -1, 0, 1}},
        {"sobely", {-1, -2, -1, 0, 0, 0, 1, 2, 1}},
    };

    auto it = kernels.find(name);
    if (it == kernels.end()) {
        throw std::runtime_error("Unknown kernel: " + name);
    }

    ksize = 3;
    return it->second;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--test-real") {
        try {
            std::string input_path = argv[2];
            std::string output_prefix = argv[3];
            bool ok = run_real_image_test(input_path, output_prefix);
            if (ok) {
                std::cout << "[test] all checks passed\n";
                return 0;
            }
            std::cout << "[test] checks failed\n";
            return 2;
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            return 1;
        }
    }

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.(png|pgm)> <output.(png|pgm)> <kernel_name>\n";
        std::cerr << "Test mode: " << argv[0] << " --test-real <input.(png|pgm)> <output_prefix>\n";
        std::cerr << "Kernels: identity, blur, gaussian, sharpen, edge, emboss, sobelx, sobely\n";
        return 1;
    }

    try {
        std::string input_path = argv[1];
        std::string output_path = argv[2];
        std::string kernel_name = argv[3];

        for (char& c : kernel_name) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        int ksize = 0;
        std::vector<float> kernel = kernel_from_name(kernel_name, ksize);

        Image in = load_image_auto(input_path);

        Image out = apply_kernel_with_ops(in, kernel, ksize, 1, ksize / 2);

        save_image_auto(output_path, out);

        std::cout << "Done. Output saved to: " << output_path << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
