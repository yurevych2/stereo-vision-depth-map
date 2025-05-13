// Additional functions and enhancements for
// - Refactored image loading
// - Left-Right Consistency Check
// - Multi-scale Matching
// - Adaptive Cost Aggregation
// - Bilateral Filtering (simple approximation)
// - Disparity Confidence Map

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <filesystem>
#include <string>
#include <omp.h>
#include <cmath>
#include <algorithm>

typedef std::vector<std::vector<int>> Image;

bool load_image_as_grayscale(const std::string& filename, Image& grayscale, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "stbi_load failed for: " << filename << "\n";
        return false;
    }
    grayscale.resize(height, std::vector<int>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            int r = data[idx + 0];
            int g = (channels > 1) ? data[idx + 1] : r;
            int b = (channels > 2) ? data[idx + 2] : r;
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            grayscale[y][x] = gray;
        }
    }
    stbi_image_free(data);
    return true;
}

bool write_pgm(const std::string& filename, const Image& image) {
    int height = image.size();
    int width = image[0].size();
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file << "P5\n" << width << " " << height << "\n255\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char val = static_cast<unsigned char>(image[y][x]);
            file.write(reinterpret_cast<char*>(&val), 1);
        }
    }
    return true;
}

bool load_image_pair(const std::string& left_path, const std::string& right_path, Image& left, Image& right, int& width, int& height) {
    int w1, h1, w2, h2;
    if (!load_image_as_grayscale(left_path, left, w1, h1)) return false;
    if (!load_image_as_grayscale(right_path, right, w2, h2)) return false;
    if (w1 != w2 || h1 != h2) {
        std::cerr << "Image sizes do not match.\n";
        return false;
    }
    width = w1;
    height = h1;
    return true;
}

double bilateral_weight(int dx, int dy, int I1, int I2, double sigma_space, double sigma_intensity) {
    double gs = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space));
    double gi = std::exp(-(I1 - I2) * (I1 - I2) / (2.0 * sigma_intensity * sigma_intensity));
    return gs * gi;
}

Image apply_bilateral_filter(const Image& input, int kernel_size = 5, double sigma_space = 2.0, double sigma_intensity = 20.0) {
    int height = input.size();
    int width = input[0].size();
    int half_k = kernel_size / 2;
    Image output = input;
#pragma omp parallel for collapse(2)
    for (int y = half_k; y < height - half_k; ++y) {
        for (int x = half_k; x < width - half_k; ++x) {
            double sum = 0.0;
            double norm = 0.0;
            for (int dy = -half_k; dy <= half_k; ++dy) {
                for (int dx = -half_k; dx <= half_k; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;
                    double w = bilateral_weight(dx, dy, input[y][x], input[ny][nx], sigma_space, sigma_intensity);
                    sum += w * input[ny][nx];
                    norm += w;
                }
            }
            output[y][x] = static_cast<int>(sum / norm);
        }
    }
    return output;
}

Image downscale(const Image& img) {
    int h = img.size(), w = img[0].size();
    Image small(h / 2, std::vector<int>(w / 2));
    for (int y = 0; y < h - 1; y += 2) {
        for (int x = 0; x < w - 1; x += 2) {
            small[y / 2][x / 2] = (img[y][x] + img[y][x + 1] + img[y + 1][x] + img[y + 1][x + 1]) / 4;
        }
    }
    return small;
}

Image upscale(const Image& img) {
    int h = img.size(), w = img[0].size();
    Image large(h * 2, std::vector<int>(w * 2));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    large[y * 2 + dy][x * 2 + dx] = img[y][x];
    return large;
}

Image apply_lr_consistency(const Image& disp_left, const Image& disp_right, int threshold = 1) {
    int height = disp_left.size();
    int width = disp_left[0].size();
    Image output = disp_left;
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int d = disp_left[y][x];
            int xr = x - static_cast<int>(d * 64.0 / 255.0);
            if (xr >= 0 && xr < width) {
                int d_r = disp_right[y][xr];
                if (std::abs(d - d_r) > threshold)
                    output[y][x] = 0;
            } else {
                output[y][x] = 0;
            }
        }
    }
    return output;
}

Image compute_confidence_map(const Image& left, const Image& right, int window_size, int max_disparity, bool sad) {
    int height = left.size();
    int width = left[0].size();
    int half_window = window_size / 2;
    Image confidence_map(height, std::vector<int>(width, 0));
#pragma omp parallel for collapse(2)
    for (int y = half_window; y < height - half_window; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            std::vector<int> costs;
            for (int d = 0; d <= std::min(max_disparity, x - half_window); ++d) {
                int cost = 0;
                for (int dy = -half_window; dy <= half_window; ++dy) {
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        int lp = left[y + dy][x + dx];
                        int rp = right[y + dy][x + dx - d];
                        cost += sad ? std::abs(lp - rp) : (lp - rp) * (lp - rp);
                    }
                }
                costs.push_back(cost);
            }
            if (costs.size() > 1) {
                std::nth_element(costs.begin(), costs.begin() + 1, costs.end());
                int best = costs[0], second = costs[1];
                int conf = std::clamp(static_cast<int>(255.0 * (second - best) / (second + 1)), 0, 255);
                confidence_map[y][x] = conf;
            }
        }
    }
    return confidence_map;
}

int main() {
    omp_set_num_threads(4);
    int width, height;
    Image left, right;
    if (!load_image_pair("img/left_1.png", "img/right_1.png", left, right, width, height)) return 1;

    Image left_filtered = apply_bilateral_filter(left);
    Image right_filtered = apply_bilateral_filter(right);

    Image small_left = downscale(left_filtered);
    Image small_right = downscale(right_filtered);

    Image disparity_small = compute_disparity(small_left, small_right, 7, 64, true);
    Image disparity_full = upscale(disparity_small);

    Image confidence = compute_confidence_map(left_filtered, right_filtered, 7, 64, true);

    if (!write_pgm("output_disparity.pgm", disparity_full)) return 1;
    if (!write_pgm("output_confidence.pgm", confidence)) return 1;

    std::cout << "Saved disparity and confidence maps.\n";
    return 0;
}
