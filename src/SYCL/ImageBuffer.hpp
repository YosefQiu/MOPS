#pragma once
#include "ggl.h"
#include "Utils/stb_image_write.h"
#include "Utils/tinycolormap.hpp"
#include <cmath>

namespace MOPS
{

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
		vec3 getPixel(const int i, const int j) const
		{
			auto index = getIndex(i, j);
			vec3 val = { -1, -1, -1 };
			if (index == -1) return val;
			
			val.x() = mPixels[index + 0];
			val.y() = mPixels[index + 1];
			val.z() = mPixels[index + 2];
			return val;
		}
		std::vector<T> getChannel(int channel) const 
		{
			std::vector<T> channelData;
			if (channel < 0 || channel > 3) return channelData;
			channelData.reserve(mWidth * mHeight);
			for (int i = 0; i < mHeight; ++i) {
				for (int j = 0; j < mWidth; ++j) {
					int idx = getIndex(i, j);
					channelData.push_back(mPixels[idx + channel]);
				}
			}
			return channelData;
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


	template <typename T>
	inline bool SaveToPNG(const MOPS::ImageBuffer<T>& buffer, const std::string& filename, int channel = 3) 
	{
		const int w = buffer.getWidth();
		const int h = buffer.getHeight();

		std::vector<T> channelData = buffer.getChannel(channel);
		if (channelData.empty()) return false;

		// 1. Find min/max while skipping NaNs
		float minVal = std::numeric_limits<float>::max();
		float maxVal = std::numeric_limits<float>::lowest();

		for (const auto& v : channelData) {
			float fv = static_cast<float>(v);
			if (!sycl::isnan(fv)) {
				minVal = std::min(minVal, fv);
				maxVal = std::max(maxVal, fv);
			}
		}
		if (minVal >= maxVal) maxVal = minVal + 1e-5f;

		// 2. Map + colormap + handle NaNs
		std::vector<unsigned char> image_u8(w * h * 4);  // RGBA
		for (int i = 0; i < w * h; ++i) {
			float raw_val = static_cast<float>(channelData[i]);

			if (sycl::isnan(raw_val)) {
				// Assign transparent black for NaNs
				image_u8[4 * i + 0] = 0;
				image_u8[4 * i + 1] = 0;
				image_u8[4 * i + 2] = 0;
				image_u8[4 * i + 3] = 0;  // alpha = 0 (transparent)
			} else {
				float norm_val = (raw_val - minVal) / (maxVal - minVal);
				tinycolormap::Color c = tinycolormap::GetColor(norm_val, tinycolormap::ColormapType::Viridis);
				image_u8[4 * i + 0] = static_cast<unsigned char>(c.r() * 255.0f);
				image_u8[4 * i + 1] = static_cast<unsigned char>(c.g() * 255.0f);
				image_u8[4 * i + 2] = static_cast<unsigned char>(c.b() * 255.0f);
				image_u8[4 * i + 3] = 255;  // fully opaque
			}
		}

		// 3. Write PNG
		return stbi_write_png(filename.c_str(), w, h, 4, image_u8.data(), w * 4) != 0;
	}
}