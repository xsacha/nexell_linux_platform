/*
 * Copyright (C) Guangzhou FriendlyARM Computer Tech. Co., Ltd.
 * (http://www.friendlyarm.com)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#include <linux/v4l2-mediabus.h>
#include <linux/videodev2.h>
#include <linux/nxp_ion.h>

#include <ion.h>
#include <nxp-v4l2.h>

#include "opencv2/opencv.hpp"
#if CV_MAJOR_VERSION >= 3
#include "opencv2/utility.hpp"
#endif

#include <chrono>
#include "yuv2rgb.neon.h"

#define err(fmt, arg...)			\
	fprintf(stderr, "E/%.3d: " fmt, __LINE__, ## arg)

// --------------------------------------------------------
#ifndef ALIGN
#define ALIGN(x, a)		(((x) + (a) - 1) & ~((a) - 1))
#endif

static inline int get_size(int format, int num, int width, int height)
{
	int size;

	width = ALIGN(width, 32);
	height = ALIGN(height, 32);

	switch (format) {
		case V4L2_PIX_FMT_YUV420M:
		case V4L2_PIX_FMT_YUV422P:
			if (num == 0)
				size = width * height;
			else
				size = (width * height) >> 1;
			break;
		default:
			size = width * height * 2;
			break;
	}

	return size;
}

int alloc_buffers(int ion_fd, int count, struct nxp_vid_buffer *bufs,
		int width, int height, int format)
{
	struct nxp_vid_buffer *vb;
	int plane_num = 3;
	int i, j;
	int ret;

	int size[plane_num];
	for (j = 0; j < plane_num; j++)
		size[j] = get_size(format, j, width, height);

	for (i = 0; i < count; i++) {
		vb = &bufs[i];
		vb->plane_num = plane_num;
		for (j = 0; j < plane_num; j++) {
			ret = ion_alloc_fd(ion_fd, size[j], 0,
					ION_HEAP_NXP_CONTIG_MASK, 0, &vb->fds[j]);
			if (ret < 0) {
				err("failed to ion alloc %d\n", size[j]);
				return ret;
			}
			vb->virt[j] = (char *)mmap(NULL, size[j],
					PROT_READ | PROT_WRITE, MAP_SHARED, vb->fds[j], 0);
			if (!vb->virt[j]) {
				err("failed to mmap\n");
				return -1;
			}
			ret = ion_get_phys(ion_fd, vb->fds[j], &vb->phys[j]);
			if (ret < 0) {
				err("failed to get phys\n");
				return ret;
			}
			vb->sizes[j] = size[j];
		}
	}

	return 0;
}

// --------------------------------------------------------
#define MAX_BUFFER_COUNT	4
#define FMT_SENDOR		V4L2_MBUS_FMT_YUYV8_2X8

#define __V4L2_S(cmd)		\
	do {					\
		int ret = cmd;		\
		if (ret < 0) {		\
			fprintf(stderr, "%.3d: `%s' = %d\n", __LINE__, #cmd, ret);	\
			/*return ret;*/		\
		}					\
	} while (0)

/*
 * Note:
 *  - S5P6818: clipper0 + YUV422P
 */
static int init_preview(int width, int height, int format)
{
	__V4L2_S(v4l2_set_format(nxp_v4l2_clipper0, width, height, format));
	__V4L2_S(v4l2_set_crop_with_pad(nxp_v4l2_clipper0, 2, 0, 0, width, height));

	// Set to default format (800x600) at first
	if (width != 800)
		__V4L2_S(v4l2_set_format(nxp_v4l2_sensor0, 800, 600, FMT_SENDOR));
	__V4L2_S(v4l2_set_format(nxp_v4l2_sensor0, width, height, FMT_SENDOR));

	__V4L2_S(v4l2_reqbuf(nxp_v4l2_clipper0, MAX_BUFFER_COUNT));
	return 0;
}

static int do_preview(struct nxp_vid_buffer *bufs, int width, int height,
		int preview_frames, bool grey)
{
	struct nxp_vid_buffer *buf;
	int i;

	for (i = 0; i < MAX_BUFFER_COUNT; i++) {
		buf = &bufs[i];
		v4l2_qbuf(nxp_v4l2_clipper0, buf->plane_num, i, buf, -1, NULL);
	}

	__V4L2_S(v4l2_streamon(nxp_v4l2_clipper0));

	// fill all buffer for display
	int index = 0;
	for (i = 0; i < MAX_BUFFER_COUNT; i++) {
		buf = &bufs[index];
		v4l2_dqbuf(nxp_v4l2_clipper0, buf->plane_num, &index, NULL);
		v4l2_qbuf(nxp_v4l2_clipper0, buf->plane_num, index, buf, -1, NULL);
		index++;
	}

	int cap_index = 0;
	cv::namedWindow("preview",1);

	for (i = 0; i < preview_frames; i++) {
		buf = &bufs[cap_index];
		v4l2_dqbuf(nxp_v4l2_clipper0, buf->plane_num, &cap_index, NULL);

		cv::Mat out;
		if (grey)
		{
			out = cv::Mat(height, width, CV_8UC1, buf->virt[0], ALIGN(width, 32));
		}
		else
		{
			out = cv::Mat(height, width, CV_8UC4);
			yuv422_2_rgb8888_neon((uint8_t *)out.ptr<unsigned char>(0), (const uint8_t*)buf->virt[0], (const uint8_t*)buf->virt[1], (const uint8_t*)buf->virt[2], width, height, ALIGN(width, 32), width >> 1, width * 4);
		}
		cv::imshow("preview", out);
		cv::waitKey(1);

		v4l2_qbuf(nxp_v4l2_clipper0, buf->plane_num, cap_index, buf, -1, NULL);
	}
	
	__V4L2_S(v4l2_streamoff(nxp_v4l2_clipper0));
	return 0;
}

// --------------------------------------------------------
#define FMT_PREVIEW	V4L2_PIX_FMT_YUV422P

int main(int argc, char *argv[])
{
#if CV_MAJOR_VERSION >= 3
	const cv::String keys = 
		"{help h usage ? |       | print this message   }"
		"{width w        |800    | camera output width  }"
		"{height h       |600    | camera output height }"
		"{count n        |30     | camera output frames }"
		"{grey g         |false  | camera output grey   }";
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Nexell OpenCV Camera Test");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	bool grey = parser.get<bool>("grey");
	int width = parser.get<int>("width");
	int height = parser.get<int>("height");
	int count = parser.get<int>("count");
#else
	bool grey = false;
        int width = 800;
        int height = 600;
        int count = 100;
#endif

	int ion_fd = ion_open();
	
	if (ion_fd < 0) {
		err("failed to ion_open, errno = %d\n", errno);
		return -EINVAL;
	}

	struct V4l2UsageScheme s;
	memset(&s, 0, sizeof(s));
	// Only using the clipper v4l2 interface
	s.useDecimator0 = false;
	s.useClipper0 = true;
	s.useMlc0Video = false;
	int ret = v4l2_init(&s);

	if (ret < 0) {
		err("initialize V4L2 failed, %d\n", ret);
		close(ion_fd);
		return ret;
	}

	struct nxp_vid_buffer bufs[MAX_BUFFER_COUNT];
	ret = alloc_buffers(ion_fd, MAX_BUFFER_COUNT, bufs,
			width, height, FMT_PREVIEW);

	if (ret >= 0 && width > 0) {
		init_preview(width, height, FMT_PREVIEW);
		auto baseTime = std::chrono::high_resolution_clock::now();
		do_preview(bufs, width, height, count, grey);
		auto currentTime = std::chrono::high_resolution_clock::now();
		int millisecondsPassed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - baseTime).count();
		printf("Took %.2f seconds to show %i frames. %.2f FPS.\n", millisecondsPassed / 1000.0f, count, (float)count/(float)(millisecondsPassed/1000.0));
	}

	v4l2_exit();
	close(ion_fd);

	return ret;
}

