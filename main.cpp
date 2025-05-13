#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
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
    // Tells how many color channels the image has
    int channels;
    // Reads the image into a 1D byte array of pixel data and updates its parameters
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    // If the image fails to load, data will be nullptr
    if (!data) {
        std::cerr << "stbi_load failed for: " << filename << "\n";
        return false;
    }

    // It must be resized to match the image dimensions (height × width)
    // before assignment values to grayscale[y][x]
    grayscale.resize(height, std::vector<int>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // The index in the flat data[] array for the current pixel
            int idx = (y * width + x) * channels; // Multiply by channels because each pixel has 1 (grayscale), 3 (RGB), or 4 (RGBA) bytes
            int r = data[idx + 0];
            int g = (channels > 1) ? data[idx + 1] : r; // If the image has only 1 channel (grayscale), reuse r for g and b
            int b = (channels > 2) ? data[idx + 2] : r;
            // Converts RGB to grayscale using the luminosity method
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            grayscale[y][x] = gray;
        }
    }

    // Frees the memory that stbi_load() allocated
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
            auto val = static_cast<unsigned char>(image[y][x]);
            file.write(reinterpret_cast<char*>(&val), 1);
        }
    }
    return true;
}


// For the latest compute_disparity()
void normalize_image(Image& img) {
    int max_val = 0;
    for (const auto& row : img)
        for (int val : row)
            max_val = std::max(max_val, val);
    if (max_val == 0) return;
    for (auto& row : img)
        for (int& val : row)
            val = (val * 255) / max_val;
}

// For the latest compute_disparity()
double spatial_weight(int dx, int dy, double sigma_space) {
    return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space));
}

// For the latest compute_disparity()
Image compute_confidence_map(const Image& left, const Image& right, int window_size, int max_disparity, bool sad) {
    int height = left.size(), width = left[0].size();
    int half_window = window_size / 2;
    Image confidence(height, std::vector<int>(width, 0));

    for (int y = half_window; y < height - half_window; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            std::vector<int> costs;
            for (int d = 0; d <= std::min(max_disparity, x - half_window); ++d) {
                int cost = 0;
                for (int dy = -half_window; dy <= half_window; ++dy)
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        int l = left[y + dy][x + dx];
                        int r = right[y + dy][x + dx - d];
                        cost += sad ? std::abs(l - r) : (l - r) * (l - r);
                    }
                costs.push_back(cost);
            }
            if (costs.size() > 1) {
                std::nth_element(costs.begin(), costs.begin() + 1, costs.end());
                int conf = static_cast<int>(255.0 * (costs[1] - costs[0]) / (costs[1] + 1));
                confidence[y][x] = std::clamp(conf, 0, 255);
            }
        }
    }
    return confidence;
}

// For the latest compute_disparity()
Image apply_median_filter(const Image& img) {
    int h = img.size(), w = img[0].size();
    Image out = img;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            std::vector<int> vals;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    vals.push_back(img[y + dy][x + dx]);
            std::nth_element(vals.begin(), vals.begin() + 4, vals.end());
            out[y][x] = vals[4];
        }
    }
    return out;
}

// For the latest compute_disparity()
Image joint_bilateral_filter(const Image& disparity, const Image& guidance, int kernel_size = 5, double sigma_spatial = 3.0, double sigma_intensity = 15.0) {
    int height = disparity.size();
    int width = disparity[0].size();
    int half_k = kernel_size / 2;

    Image output = disparity;

    for (int y = half_k; y < height - half_k; ++y) {
        for (int x = half_k; x < width - half_k; ++x) {
            double sum = 0.0;
            double norm = 0.0;
            int g_center = guidance[y][x];

            for (int dy = -half_k; dy <= half_k; ++dy) {
                for (int dx = -half_k; dx <= half_k; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;

                    int g_neighbor = guidance[ny][nx];
                    int d_neighbor = disparity[ny][nx];

                    double w_spatial = std::exp(-(dx * dx + dy * dy) / (2 * sigma_spatial * sigma_spatial));
                    double w_intensity = std::exp(-(g_neighbor - g_center) * (g_neighbor - g_center) / (2 * sigma_intensity * sigma_intensity));

                    double weight = w_spatial * w_intensity;

                    sum += weight * d_neighbor;
                    norm += weight;
                }
            }

            output[y][x] = static_cast<int>(sum / (norm + 1e-5));
        }
    }

    return output;
}

// For the latest compute_disparity()
Image fill_occlusions(const Image& disp_left, const Image& disp_right) {
    int height = disp_left.size();
    int width = disp_left[0].size();
    Image filled = disp_left;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (disp_left[y][x] == 0) {
                // Estimate corresponding pixel in right view
                for (int d = 1; d < 64; ++d) {
                    int xr = x - d;
                    if (xr >= 0 && disp_right[y][xr] > 0) {
                        filled[y][x] = disp_right[y][xr];
                        break;
                    }
                }
            }
        }
    }

    return filled;
}

//! SAD/SSD-based disparity map
// Using SAD if sad = 1, ssd otherwise
Image compute_disparity(const Image& left, const Image& right, int window_size, int max_disparity, bool sad) {
    int height = left.size();
    int width = left[0].size();
    int half_window = window_size / 2;

    Image disparity_map(height, std::vector<int>(width, 0));

#pragma omp parallel for default(none) shared(left, right, disparity_map, width, height, half_window, max_disparity, sad, window_size)
    for (int y = half_window; y < height - half_window; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            int best_disparity = 0;
            int min_sad_or_ssd = std::numeric_limits<int>::max();

            int max_d = std::min(max_disparity, x - half_window);

            // Try all disparities for the current pixel
            for (int d = 0; d <= max_d; ++d) {
                int sad_or_ssd = 0;

                // SAD/SSD for current disparity d
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

            // Normalize and store the min disparity
            disparity_map[y][x] = static_cast<int>(255.0 * best_disparity / max_disparity);
        }
    }

    return disparity_map;
}

// The latest compute_disparity()
//Image compute_disparity(const Image& left, const Image& right, int window_size, int max_disparity, bool sad, double sigma_space) {
//    int height = left.size(), width = left[0].size();
//    int half_window = window_size / 2;
//    Image disparity(height, std::vector<int>(width, 0));
//
//    for (int y = half_window; y < height - half_window; ++y) {
//        for (int x = half_window; x < width - half_window; ++x) {
//            int best_d = 0;
//            int min_cost = std::numeric_limits<int>::max();
//            std::vector<int> cost_at_d(max_disparity + 1, 0);
//
//            for (int d = 0; d <= std::min(max_disparity, x - half_window); ++d) {
//                double sum = 0.0, weight_sum = 0.0;
//
//                for (int dy = -half_window; dy <= half_window; ++dy)
//                    for (int dx = -half_window; dx <= half_window; ++dx) {
//                        int l = left[y + dy][x + dx];
//                        int r = right[y + dy][x + dx - d];
//                        double w = spatial_weight(dx, dy, sigma_space);
//                        int diff = sad ? std::abs(l - r) : (l - r) * (l - r);
//                        sum += w * diff;
//                        weight_sum += w;
//                    }
//
//                int cost = static_cast<int>(sum / (weight_sum + 1e-5));
//                cost_at_d[d] = cost;
//
//                if (cost < min_cost) {
//                    min_cost = cost;
//                    best_d = d;
//                }
//            }
//
//            float subpixel_d = static_cast<float>(best_d);
//            if (best_d > 0 && best_d < max_disparity) {
//                int c1 = cost_at_d[best_d - 1];
//                int c0 = cost_at_d[best_d];
//                int c2 = cost_at_d[best_d + 1];
//                int denom = 2 * (c1 - 2 * c0 + c2);
//                if (denom != 0) {
//                    float offset = static_cast<float>(c1 - c2) / denom;
//                    subpixel_d = best_d + offset;
//                }
//            }
//
//            int disp_val = static_cast<int>(255.0f * subpixel_d / max_disparity);
//            disparity[y][x] = std::clamp(disp_val, 0, 255);
//        }
//    }
//
//    return disparity;
//}

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

    std::vector<std::pair<std::string, std::string>> image_pairs = {
            {"img\\left_1.png", "img\\right_1.png"},
            {"img\\left_2.png", "img\\right_2.png"}
    };

    std::vector<int> window_sizes;
    for (int ws = 1; ws <= 31; ws += 2)
        window_sizes.push_back(ws);

    std::vector<double> sad_times, ssd_times;

    for (int sad = 0; sad <= 1; sad++) {
        std::vector<double>& result_vec = sad ? sad_times : ssd_times;
        for (int ws : window_sizes) {
            std::chrono::duration<double> min_time(1000.0); // start high
            for (const auto& [left_path, right_path] : image_pairs) {
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

                for (int i = 0; i < 100; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();

                    compute_disparity(left_gray, right_gray, ws, max_disparity, sad);

                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = end - start;
                    if (duration < min_time) min_time = duration;
                }
            }

            result_vec.push_back(min_time.count());
            std::cout << (sad ? "SAD" : "SSD")
                      << " | window_size = " << ws
                      << " | min_time = " << min_time.count() << " sec\n";
        }
    }

    std::cout << "\nwindow_sizes = [";
    for (size_t i = 0; i < window_sizes.size(); ++i)
        std::cout << window_sizes[i] << (i + 1 < window_sizes.size() ? ", " : "");
    std::cout << "]\n";

    std::cout << "sad_times = [";
    for (size_t i = 0; i < sad_times.size(); ++i)
        std::cout << sad_times[i] << (i + 1 < sad_times.size() ? ", " : "");
    std::cout << "]\n";

    std::cout << "ssd_times = [";
    for (size_t i = 0; i < ssd_times.size(); ++i)
        std::cout << ssd_times[i] << (i + 1 < ssd_times.size() ? ", " : "");
    std::cout << "]\n";
}

int main() {
    omp_set_num_threads(4);
    benchmark();

    return 0;
}

// For the latest compute_disparity()
//int main() {
//    int width, height;
//    Image left, right;
//
//    if (!load_image_as_grayscale("img/left_1.png", left, width, height) ||
//        !load_image_as_grayscale("img/right_1.png", right, width, height)) {
//        std::cerr << "Failed to load stereo pair.\n";
//        return 1;
//    }
//
//    // Just in case normalize input images to 0–255
//    normalize_image(left);
//    normalize_image(right);
//
//    const int window_size = 9;
//    const int max_disparity = 64;
//    const double sigma_space = 3.0;
//
//    // Compute disparity maps for both views
//    Image disparity_left = compute_disparity(left, right, window_size, max_disparity, true, sigma_space);
//    Image disparity_right = compute_disparity(right, left, window_size, max_disparity, true, sigma_space);
//
//    // Fill occluded/invalid pixels in left disparity using right disparity
//    Image disparity_filled = fill_occlusions(disparity_left, disparity_right);
//
//    // Refine disparity using joint bilateral filter (guided by left image)
//    Image filtered = joint_bilateral_filter(disparity_filled, left);
//
//    // Confidence for debugging
//    Image confidence = compute_confidence_map(left, right, window_size, max_disparity, true);
//
//    write_pgm("output_disparity_filled.pgm", filtered);
//    write_pgm("output_confidence_filled.pgm", confidence);
//
//    std::cout << "Disparity computed with occlusion filling and joint bilateral filtering.\n";
//    return 0;
//}