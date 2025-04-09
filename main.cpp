#define STB_IMAGE_IMPLEMENTATION
// image loading/decoding from file/memory: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
#include "lib/stb_image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <filesystem>
#include <string>
#include <omp.h>

typedef std::vector<std::vector<int>> Image;

//! Load PNG (or any format) with stb_image.
//! It converts the image to a grayscale 2D array.
//! It stores the image dimensions in width and height.
//! It returns true on success, false on failure.
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
// Standard parameters:
//    int *x                 -- outputs image width in pixels
//    int *y                 -- outputs image height in pixels
//    int *channels_in_file  -- outputs # of image components in image file
//    int desired_channels   -- if non-zero, # of image components requested in result
bool load_image_as_grayscale(const std::string& filename, Image& grayscale, int& width, int& height) {
    // Tells how many color channels the image has.
    int channels;
    // Reads the image into a 1D byte array of pixel data and updates its parameters.
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    // If the image fails to load, data will be nullptr.
    if (!data) {
        std::cerr << "stbi_load failed for: " << filename << "\n";
        return false;
    }

    // It must be resized to match the image dimensions (height × width)
    // before assignment values to grayscale[y][x].
    grayscale.resize(height, std::vector<int>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // The index in the flat data[] array for the current pixel.
            int idx = (y * width + x) * channels; // Multiply by channels because each pixel has 1 (grayscale), 3 (RGB), or 4 (RGBA) bytes.
            int r = data[idx + 0];
            int g = (channels > 1) ? data[idx + 1] : r; // If the image has only 1 channel (grayscale), reuse r for g and b.
            int b = (channels > 2) ? data[idx + 2] : r;
            // Converts RGB to grayscale using the luminosity method.
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            grayscale[y][x] = gray;
        }
    }

    // Frees the memory that stbi_load() allocated.
    stbi_image_free(data);
    return true;
}

//! Save grayscale as PGM (Portable Gray Map) - grayscale 2D images.
//! Returns true on success, false on failure.
// Standard parameters:
//   std::string& filename: name of the output file (e.g., "disparity.pgm")
//   Image& image: a 2D vector of grayscale values (Image = std::vector<std::vector<int>>)
bool write_pgm(const std::string& filename, const Image& image) {
    int height = image.size();
    int width = image[0].size();

    // Open file for binary writing. Returns false if fails
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write PGM header.
    // P5 - PGM binary format; width height - Image dimensions; 255 - Max pixel value (8-bit);
    file << "P5\n" << width << " " << height << "\n255\n";

    // Iterates over every pixel in the image.
    // Converts the pixel value (0–255) to unsigned char.
    // Writes it as 1 byte to the file using.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char val = static_cast<unsigned char>(image[y][x]);
            file.write(reinterpret_cast<char*>(&val), 1);
        }
    }
    return true;
}

//! SAD-based disparity map
// Using SAD if sad = 1, ssd otherwise
Image compute_disparity(const Image& left, const Image& right, int window_size, int max_disparity, bool sad) {
    int height = left.size();
    int width = left[0].size();
    int half_window = window_size / 2;

    // Output image.
    Image disparity_map(height, std::vector<int>(width, 0));

    // Iterate over valid pixels. Skip borders.
    #pragma omp parallel for default(none) shared(left, right, disparity_map, width, height, half_window, max_disparity, sad, window_size)
    for (int y = half_window; y < height - half_window; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            int best_disparity = 0; // Stores the shift with the lowest SAD.
            int min_sad_or_ssd = std::numeric_limits<int>::max(); // Highest possible value.

            int max_d = std::min(max_disparity, x - half_window); // Limit max disparity for boundary safety.

            // Try all disparities for the current pixel.
            for (int d = 0; d <= max_d; ++d) {
                int sad_or_ssd = 0;

                // Compute SAD for current disparity d.
                for (int dy = -half_window; dy <= half_window; ++dy) {
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        int left_pixel = left[y + dy][x + dx];
                        int right_pixel = right[y + dy][x + dx - d];
                        sad_or_ssd += sad ? abs(left_pixel - right_pixel) : (left_pixel-right_pixel) * (left_pixel-right_pixel);
                    }
                }

                if (sad_or_ssd < min_sad_or_ssd) {
                    min_sad_or_ssd = sad_or_ssd;
                    best_disparity = d;
                }
            }

            // Normalize and store the min disparity.
            disparity_map[y][x] = static_cast<int>(255.0 * best_disparity / max_disparity);
        }
    }

    return disparity_map;
}

int example_usage() {
    int max_disparity = 64;
    int window_size = 7;
    for (int sad = 0; sad <= 1; sad++) {
        std::cout << "Working directory: " << std::filesystem::current_path() << "\n";

        Image left_gray, right_gray;
        int width1, height1, width2, height2;

        if (!load_image_as_grayscale("img\\left_1.png", left_gray, width1, height1)) {
            std::cerr << "Failed to load left_1.png\n";
            return 1;
        }

        if (!load_image_as_grayscale("img\\right_1.png", right_gray, width2, height2)) {
            std::cerr << "Failed to load right_1.png\n";
            return 1;
        }

        if (width1 != width2 || height1 != height2) {
            std::cerr << "Image sizes do not match.\n";
            return 1;
        }

        std::cout << "Computing disparity map with "
                  << (sad ? "SAD" : "SSD") << ", window_size = " << window_size
                  << ", max_disparity = " << max_disparity << "...\n";

        Image disparity = compute_disparity(left_gray, right_gray, window_size, max_disparity, sad);

        std::string filename = "disparity_" +
                               std::string(sad ? "sad" : "ssd") +
                               "_ws" + std::to_string(window_size) +
                               "_md" + std::to_string(max_disparity) + ".pgm";

        if (!write_pgm(filename, disparity)) {
            std::cerr << "Failed to write " << filename << "\n";
            return 1;
        }

        std::cout << "Saved: " << filename << "\n";
    }

    return 0;
}

void benchmark() {
    int max_disparity = 64;
    int window_size = 7;

    std::chrono::duration<double> min_sec(5.0); // There is no slower algorithm, so it is equal to +inf
    std::chrono::duration<double> max_sec(0.0); // There is no faster algorithm, so it is equal to -inf

    std::vector<std::pair<std::string, std::string>> image_pairs = {
            {"img\\left_1.png", "img\\right_1.png"},
            {"img\\left_2.png", "img\\right_2.png"}
    };


    for (int sad = 0; sad <= 1; sad++) {
        for (const auto& [left_path, right_path] : image_pairs) {
            std::cout << "=== Benchmarking " << (sad ? "SSD" : "SAD") << " on " << left_path << " and " << right_path << " ===\n";
            for (int i = 0; i <= 100; i++) {
                Image left_gray, right_gray;
                int width1, height1, width2, height2;

                if (!load_image_as_grayscale(left_path, left_gray, width1, height1)) {
                    std::cerr << "Failed to load " << left_path << "\n";
                    continue;
                }

                if (!load_image_as_grayscale(right_path, right_gray, width2, height2)) {
                    std::cerr << "Failed to load " << right_path << "\n";
                    continue;
                }

                if (width1 != width2 || height1 != height2) {
                    std::cerr << "Image sizes do not match.\n";
                    continue;
                }

                auto start = std::chrono::high_resolution_clock::now();

                Image disparity = compute_disparity(left_gray, right_gray, window_size, max_disparity, sad);

                auto end = std::chrono::high_resolution_clock::now();
                min_sec = (end - start < min_sec) ? end - start : min_sec;
                max_sec = (end - start > max_sec) ? end - start : max_sec;
            }
        }
        std::cout << "Min computation time with " << (sad ? "SAD" : "SSD") << ", window_size = " << window_size << ", max_disparity = " << max_disparity << ": " << min_sec.count() << " seconds\n";
        std::cout << "Max computation time with " <<  ": " << max_sec.count() << " seconds\n\n";
    }
}

int main() {
    omp_set_num_threads(4);
    benchmark();

    return 0;
}
