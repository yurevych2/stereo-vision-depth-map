#define STB_IMAGE_IMPLEMENTATION
// image loading/decoding from file/memory: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

typedef vector<vector<int>> Image;

// Load PNG (or any format) with stb_image
bool load_image_as_grayscale(const string& filename, Image& grayscale, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) return false;

    grayscale.resize(height, vector<int>(width));

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

// Save grayscale as PGM
bool write_pgm(const string& filename, const Image& image) {
    int height = image.size();
    int width = image[0].size();

    ofstream file(filename, ios::binary);
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

// SAD-based disparity map
Image compute_disparity_SAD(const Image& left, const Image& right, int window_size, int max_disparity) {
    int height = left.size();
    int width = left[0].size();
    int half_window = window_size / 2;

    Image disparity_map(height, vector<int>(width, 0));

    for (int y = half_window; y < height - half_window; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            int best_disparity = 0;
            int min_sad = std::numeric_limits<int>::max();

            int max_d = min(max_disparity, x - half_window);

            for (int d = 0; d <= max_d; ++d) {
                int sad = 0;

                for (int dy = -half_window; dy <= half_window; ++dy) {
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        int left_pixel = left[y + dy][x + dx];
                        int right_pixel = right[y + dy][x + dx - d];
                        sad += abs(left_pixel - right_pixel);
                    }
                }

                if (sad < min_sad) {
                    min_sad = sad;
                    best_disparity = d;
                }
            }

            disparity_map[y][x] = static_cast<int>(255.0 * best_disparity / max_disparity);
        }
    }

    return disparity_map;
}

int main() {
    Image left_gray, right_gray;
    int width1, height1, width2, height2;

    if (!load_image_as_grayscale("left.png", left_gray, width1, height1)) {
        cerr << "Failed to load left.png\n";
        return 1;
    }

    if (!load_image_as_grayscale("right.png", right_gray, width2, height2)) {
        cerr << "Failed to load right.png\n";
        return 1;
    }

    if (width1 != width2 || height1 != height2) {
        cerr << "Image sizes do not match.\n";
        return 1;
    }

    cout << "Computing disparity map...\n";
    int window_size = 9;
    int max_disparity = 64;
    Image disparity = compute_disparity_SAD(left_gray, right_gray, window_size, max_disparity);

    if (!write_pgm("disparity.pgm", disparity)) {
        cerr << "Failed to write disparity.pgm\n";
        return 1;
    }

    cout << "Saved: disparity.pgm\n";
    return 0;
}
