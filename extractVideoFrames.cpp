#include <iostream>
#include <filesystem>

#include "opencv2/opencv.hpp"

using namespace cv;
namespace fs = std::filesystem;

/*
 * Store each frame of each video under the specified path as jpg using the
 * naming scheme <video name>_<frame number>.jpg
 */
int main() {
	std::string path = "D:\\CSVP2019\\videos_20190930\\";
	std::string storePath = "D:\\CSVP2019\\extracted\\";
	std::string folders[] = {
		"test\\efilms",
		"test\\imc",
		"train\\efilms",
		"train\\imc",
		"val\\efilms",
		"val\\imc"
	};
	
	for (std::string folder : folders) {
		// Iterate through all videos
		for (const auto &entry : fs::directory_iterator(path + folder)) {
			std::cout << entry.path() << std::endl;

			// Open video
			VideoCapture cap(entry.path().string());

			// Iterate through all frames of the opened video
			while (cap.isOpened()) {
				int frameId = cap.get(1);
				Mat frame;
				bool ret = cap.read(frame);
				if (!ret) {
					break;
				}

				// store frame as jpg under the name <video name>_<frame number>.jpg
				std::string videoName = entry.path().filename().string();
				std::string imageName = videoName.substr(0, videoName.length() - 4) + "_" + std::to_string(frameId) + ".jpg";
				imwrite(storePath + folder + "\\" + imageName, frame);
			}

			cap.release();
		}
	}
}
