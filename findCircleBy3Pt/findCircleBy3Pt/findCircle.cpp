//���㶨Բ��������ͼƬ�е�Բ

#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
#pragma comment(lib,"opencv_core320d.lib")
#pragma comment(lib,"opencv_highgui320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")

using namespace cv;
using namespace std;


#define N 100010
using namespace std;
//struct node { double x, y; } Pt[N];
Point2d O;
double R;

double sqr(double x) { return x*x; }
double dis(Point2d x, Point2d y)
{
	return sqrt(sqr(x.x - y.x) + sqr(x.y - y.y));
}
bool incircle(Point2d x)
{
	if (dis(O, x) <= R) return true;
	return false;
}
Point2d solve(double a, double b, double c, double d, double e, double f)
{
	double y = (f*a - c*d) / (b*d - e*a);
	double x = (f*b - c*e) / (a*e - b*d);
	 
	return Point2d(x,y);
}


 
int main()
{
 
/*	Mat src_color = imread("20180928154225.bmp");//��ȡԭ��ɫͼ
	imshow("ԭͼ-��ɫ", src_color);

	//����һ����ͨ��ͼ������ֵȫΪ0������������任������Բ��������
	Mat dst(src_color.size(), src_color.type());
	dst = Scalar::all(0);

	Mat src_gray;//��ɫͼ��ת���ɻҶ�ͼ
	cvtColor(src_color, src_gray, COLOR_BGR2GRAY);
	imshow("ԭͼ-�Ҷ�", src_gray);
	imwrite("src_gray.png", src_gray);

	Mat bf;//�ԻҶ�ͼ�����˫���˲�
	bilateralFilter(src_gray, bf, kvalue, kvalue * 2, kvalue / 2);
	imshow("�Ҷ�˫���˲�����", bf);
	imwrite("src_bf.png", bf);

	vector<Vec3f> circles;//����һ�����������������Բ��Բ������Ͱ뾶
	HoughCircles(bf, circles, CV_HOUGH_GRADIENT, 1.5, 20, 130, 1, 0, 0);//����任���Բ


	for (size_t i = 0; i < circles.size(); i++)//�ѻ���任������Բ������
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(dst, center, 0, Scalar(0, 255, 0), -1, 8, 0);
		circle(dst, center, radius, Scalar(0, 0, 255), 1, 8, 0);

		cout << cvRound(circles[i][0]) << "\t" << cvRound(circles[i][1]) << "\t"
			<< cvRound(circles[i][2]) << endl;//�ڿ���̨���Բ������Ͱ뾶				
	}

	imshow("������ȡ", dst);
//	imwrite("dst.png", dst);
*/

	Mat image = imread("circle1.bmp", 1);
	namedWindow("sourceImage");
	
	Mat dstImage;
	//��ֵ�˲�
	medianBlur(image, dstImage, 3 * 3);
	namedWindow("dstImage");
	Mat dstImage1;

 
	 //��ֵ��
	Mat edge;
	//adaptiveThreshold(dstImage, dstImage1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, block_size, 12);
	threshold(dstImage, dstImage1, 35, 255, CV_THRESH_BINARY_INV);

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	
	dilate(dstImage1, dstImage1, element);//����
	erode(dstImage1, dstImage1, element);//��ʴ
 
	Canny(dstImage1, edge, 3, 9);
	std::vector<Vec4i> hierarchy;
	std::vector<std::vector<Point>> contours;
	findContours(edge, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE,Point(0,0));

	char string[25];
	std::vector<std::vector<Point>>::iterator it;  
	for (it = contours.begin() ; it + 1 != contours.end();)
	{
		int size1 = it->size();
		int size2 = (it + 1)->size();
		if (abs(size1 - size2) < 5)
			contours.erase(it + 1);
		else
			it++;
		 
	}


	int K = 0;
	for (K = 0,it = contours.begin(); it != contours.end(); it++)
	{
		int size = it->size();
		if (size > 1000)
			K++;
	}

	for (int n = 0; n < K; n++)
	{
		for (it = contours.begin(); it != contours.end(); )
		{
			int size1 = it->size();
			if (size1 > 1000)
			{
				contours.erase(it);
				break;
			}

			else it++;

		}
	}
	
///////���㶨Բ����Բ
	//�˵������һ����
	size_t num = contours.size();
	int i, j, k;	
	std::map<double,Point> circles;
		
	for (size_t m = 0; m < num; m++)
	{
		size_t pt_num = contours[m].size();
		R = 0;
		std::vector<Point> circle = contours[m];//ȡ��һ��Բ
		random_shuffle(circle.begin(), circle.end());//�������

		for (i = 0; i < pt_num; i++) //ȷ��ÿ��Բ�İ뾶����Բ������
		{ 

			if (!incircle(circle[i]))
			{
				O.x = circle[i].x; O.y = circle[i].y; R = 0;
				for (j = 0; j<i; j++)
					if (!incircle(circle[j]))
					{
						O.x = (circle[i].x + circle[j].x) / 2;
						O.y = (circle[i].y + circle[j].y) / 2;
						R = dis(O, circle[i]);
						for (k = 0; k<j; k++)
							if (!incircle(circle[k]))
							{
								O = solve(
									circle[i].x - circle[j].x, circle[i].y - circle[j].y, (sqr(circle[j].x) + sqr(circle[j].y) - sqr(circle[i].x) - sqr(circle[i].y)) / 2,
									circle[i].x - circle[k].x, circle[i].y - circle[k].y, (sqr(circle[k].x) + sqr(circle[k].y) - sqr(circle[i].x) - sqr(circle[i].y)) / 2
								);
								R = dis(circle[i], O);
							}
					}
			}
		}
		printf("%.10lf: %f %f\n", R, O.x, O.y);
		int count = 0;
		bool bCircle = true;
		for (std::vector<Point>::iterator iter = circle.begin(); iter != circle.end(); iter++)
		{
			double leng = sqrt((iter->x - O.x)*(iter->x - O.x) + (iter->y - O.y)*(iter->y - O.y));//x^2 + y^2 = r^2

			 
			if(leng - R < -1*R/5.0)
				count++;

			if (count > (pt_num / 5))
			{
				count = 0;
				bCircle = false;
				break;
			}

		}

		
		if (bCircle)
		{
			circles[R] = O;
		}
	}
 
	size_t length = circles.size();
	std::map<double, Point>::iterator iter;  i = 0;
	for (iter = circles.begin(); iter != circles.end(); iter++)
	{
	 
		if (iter->second.x < 0) break;
		cv::circle(image, iter->second, iter->first, Scalar(0, 255, 255), 2);
		
		cv::String str(itoa(i+1, string, 10));
		cv::putText(image, str, iter->second, FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 255));
		i++;
	}

	imshow("dstImage", edge);
	imshow("sourceImage", image);

	cv::waitKey();
}
 