#pragma once
#include "ggl.h"
#include "stb_image_write.h"

template<typename T>
class ImageBuffer
{
public:
	ImageBuffer() = default;
	ImageBuffer(int w, int h) : mWidth(w), mHeight(h) { mPixels.resize(mWidth * mHeight * 4, (T)0); }
public:
	int getIndex(const int i, const int j) const
	{
		if (i < 0 || i >= mHeight || j < 0 || j >= mWidth) return -1;
		return (i * mWidth + j) * 4;
	}
	void setPixel(int i, int j, const vec3& val)
	{
		auto index = getIndex(i, j);
		if (index == -1) return;
		mPixels[index + 0] = val.x();
		mPixels[index + 1] = val.y();
		mPixels[index + 2] = val.z();
		mPixels[index + 3] = 1.0;
	}
	vec3 getPixel(const int i, const int j)
	{
		auto index = getIndex(i, j);
		vec3 val = { -1, -1, -1 };
		if (index == -1) return val;
		
		val.x() = mPixels[index + 0];
		val.y() = mPixels[index + 1];
		val.z() = mPixels[index + 2];
		return val;
	}
	int getWidth() const { return mWidth; }
	int getHeight() const { return mHeight; }
public:
	std::vector<T> mPixels;
protected:
	int mWidth; int mHeight;
	
};


template<typename Accessor>
inline void SetPixel(Accessor img_acc, const int w, const int h, const int i, const int j, const vec3& val)
{
	if (i < 0 || i >= h || j < 0 || j >= w) return;
	auto index = (i * w + j) * 4;

	img_acc[index + 0] = val.x();
	img_acc[index + 1] = val.y();
	img_acc[index + 2] = val.z();
	img_acc[index + 3] = 1.0;
}

template<typename Accessor>
inline void GetPixel(Accessor img_acc, const int w, const int h, const int i, const int j, vec3& val)
{
	if (i < 0 || i >= h || j < 0 || j >= w) return;
	auto index = (i * w + j) * 4;

	val.x() = img_acc[index + 0];
	val.y() = img_acc[index + 1];
	val.z() = img_acc[index + 2];
}

//template<typename Accessor>
//inline void SetPixel(Accessor img_acc, const int w, const int h, const int i, const int j, const vec3& val) {
//	if (i < 0 || i >= h || j < 0 || j >= w) return;
//	auto index = (i * w + j) * 4;
//
//	img_acc[index + 0] = val.x();
//	img_acc[index + 1] = val.y();
//	img_acc[index + 2] = val.z();
//	img_acc[index + 3] = 1.0;
//}

inline void SaveImage(std::vector<double>& img, int width, int height) 
{
	// Convert double image data to unsigned char (0-255) for stb_image_write
	std::vector<unsigned char> img_uchar(img.size());
	for (size_t i = 0; i < img.size(); ++i) {
		img_uchar[i] = static_cast<unsigned char>(std::clamp(img[i] * 255.0, 0.0, 255.0));
	}

	// Save the image using stb_image_write
	if (stbi_write_png("output.png", width, height, 4, img_uchar.data(), width * 4) == 0) {
		std::cerr << "Failed to write image to output.png" << std::endl;
	}
}