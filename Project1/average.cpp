#include "average.h"

//벡터를 기능을 활용하여 사용자가 입력한 숫자들의 평균을 계산
double AverageCalculator::calculateAverage(const std::vector<double>& numbers){
	double sum = 0.0;
	for (double num : numbers) {
		sum += num;
	}
	return sum / numbers.size();
}