#include "average.h"

//���͸� ����� Ȱ���Ͽ� ����ڰ� �Է��� ���ڵ��� ����� ���
double AverageCalculator::calculateAverage(const std::vector<double>& numbers){
	double sum = 0.0;
	for (double num : numbers) {
		sum += num;
	}
	return sum / numbers.size();
}