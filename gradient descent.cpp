#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h> 

#define N 2			// Number of Inputs
#define X0 -1		// First Static Input
#define C 0.01		// Learning Rate

class Learner {
private:
	int i = 0, result;
	int x[N + 1];
	float w[N];
	double max;
	float c;
	float delta_weight[N + 1];
	char gate[4];
	FILE *fp = fopen("ErrorGraph.txt", "w");	// ErrorGraph.txt�� ����� ����
	int trials = 1;								// Learning Trials

public:
	Learner(int gate_choice, int x0, float c) {

		// for random value generate
		max = 32767;
		srand(time(0));

		// ����ڰ� And Ȥ�� and Ȥ�� and �� ���� ��� == gate choice ==1
		if (gate_choice == 1) {
			printf("Start And Gate Learning...\n");
			strcpy_s(gate, "and");
		}
		else {
			// ����ڰ� OR Ȥ�� Or Ȥ�� or �� ���� ��� == gate choice == 0
			printf("Start Or Gate Learning...\n");
			strcpy_s(gate, "or");
		}

		// file input
		for (int i = 0; i < N + 1; i++) {
			fprintf(fp, "w[%d]\t\t", i);
		}fprintf(fp, "Error Rate\t\n==================================================\n\n");

		this->c = c;		// learning rate initialize
		printf("===================initialize===============\n");

		x[0] = x0;	// first static input initialize 
		for (int i = 1; i < N + 1; i++) {
			printf("input integer data for  x[%d] : ", i);
			scanf("%d", &x[i]);
		}

		// input data �Է�
		printf("input data : ");
		for (int i = 0; i < N + 1; i++) {
			printf("%d ", x[i]);

		}
		printf("\n\n");
		// ������ Weight N+1�� ���� (���� : 1~5)
		for (i = 0; i < N + 1; i++) {
			w[i] = (rand() / max) * 5;
			printf("initial weight[%d] randomly selected : %f \n", i, w[i]);
		}
		// �ʱⰪ �����ֱ�
		printf("x[0] initialized as : %d\n", x[0]);
		printf("learning rate c : %f\n\n", c);
		printf("==============initialize Complete==========\n\n");
	}

	int cal_target(int * x) {
		printf("Calculating Target Data ... \n");
		int result = -1;

		if (!strcmp(gate, "and")) {
			// N���� input�� ���Ͽ� and gate ����
			result = x[1] && x[2];
			for (i = 3; i < N + 1; i++) {
				result = result && x[i];
			}
		}
		if (!strcmp(gate, "or")) {
			// N���� input�� ���Ͽ� or gate ����
			result = x[1] || x[2];
			for (i = 3; i < N + 1; i++) {
				result = result || x[i];
			}
		}

		return result;
	}

	// delta �� ���ϱ� : target data - Output data ( activation ��� )
	float get_delta(float output) {

		float target, delta;
		target = float(cal_target(x));

		printf("target : %f\n", target);
		delta = target - output;
		printf("delta : %f\n", delta);
		// file write
		// ���� �̷� �� �̷��� ǥ���ϴ� ���� ������ �ð� ȿ���� ���� 
		// activation function�� ��ġ�� ���� ä�� ����غ��Ҵ�.
		// �̷� �� : fprintf(fp, "Error : %f \n", (delta*delta));

		printf("\n");

		return delta;
	}

	float * get_delta_weight(float delta) {
		printf("Calculating Differential Of Weight ... \n");

		for (int i = 0; i < N + 1; i++) {
			// delata weight (delta_w) ���ϱ� : C*delta*input_data
			delta_weight[i] = c * delta*x[i];
		}
		printf("\n");

		for (int i = 0; i < N + 1; i++) {
			// print each delta_weight
			printf("delta_weight[%d] : %f\n", i, delta_weight[i]);
		}
		printf("\n");

		return delta_weight;
	}

	void judge(float * weight_delta) {
		int result = 0;
		int check = 1;	// �ϳ��� weight�� 0������ 0
						// ��� weight_delta�� 0�� ��� 1

		printf("Judging ... \n");
		// N���� delata weight�� ��� - => +�� 0�� �Ǿ��� : Learning ���� ��� print 

		for (int i = 0; i < N + 1; i++) {
			printf("previous weight : %f : \n", w[i]); // delta_weight�� ���ϱ� �� weight

		}
		printf("\n");
		for (int i = 0; i < N + 1; i++) {
			// N�� weight ���� : weight = weight + delta_w
			w[i] = w[i] + weight_delta[i];
		}

		for (int i = 0; i < N + 1; i++) {
			printf("final weight : %f : \n", w[i]); // delta_weight�� ���� �� weight

		}

		// ��� Learning�� �������� Check : ��� delta_weight�� 0.0�� �Ǿ����� Ȯ��
		for (int i = 0; i < N + 1; i++) {
			if (weight_delta[i] != 0.0) {
				check = 0;
				break;
			}

		}
		// if not all delta_weight is 0 > learn again
		if (check == 0) {
			printf(">>> Learning not over.. Keep Learning\n");
			printf("===================trial : %d ====================\n", trials);
			trials++;
			learn();
		}
		// if all delta_weight is 0 > learning over
		else {
			printf(">>> Learning All Over! all weight_delta is 0\n");
			printf("===================trial : %d ====================\n", trials);
			return;
		}
	}

	float get_output() {

		float net = 0.0, error;
		int output;
		printf("\nCalculating Output Data... \n", c);

		// output �� ���ϱ� : net = x[0]*w[0] + x[1]*w[1] + ... + x[N]*w[N]
		for (int i = 0; i < N + 1; i++) {
			net += x[i] * w[i];
		}

		// file write : weight��
		for (int i = 0; i < N + 1; i++) {
			fprintf(fp, "%f\t\t", w[i]);
		}

		// error rate���ϱ�
		error = (cal_target(x) - net);
		fprintf(fp, "%f\n", error);

		printf("net : %f\n", net);

		// activation function. Threshold�� 0�ӿ� ����
		if (net > 0) {
			output = 1;
		}
		else {
			output = 0;
		}
		printf("output : %d\n", output);

		printf("\n");

		return output;
	}


	int learn() {

		float output;
		float del;
		float * w_del;

		// output �� ���ϱ� : net = x[0]*w[0] + x[1]*w[1] + ... + x[N]*w[N]
		output = get_output();

		// delta �� ���ϱ� : target data - Output data ( activation ��� )
		del = get_delta(output);

		// delata weight (delta_w) ���ϱ� : C*delta*input_data
		w_del = get_delta_weight(del);

		// N�� weight ���� : weight = weight + delta_w
		judge(w_del);

		return 0;
	}
};

int main()
{

	char type[50];
	int gate_choice = -1;

	// � Gate�� �н���ų���� �Է¹޴´�.
	do {
		printf("���� input : %d����. (�ڵ� ���� ��ó���⸦ ���� ������ �÷� �Է¹��� �� �ֽ��ϴ�!)\n\n ", N);
		printf("Learning ��� : And Gate? or Or Gate? : ");

		scanf("%s", type);
		if (!strcmp(type, "AND") || !strcmp(type, "and") || !strcmp(type, "And")) {
			gate_choice = 1;
			break;
		}
		else if (!strcmp(type, "OR") || !strcmp(type, "Or") || !strcmp(type, "or")) {
			gate_choice = 0;
			break;
		}
		else {
			printf("�ٽ� �Է��ϼ���\n");
		}
	} while (1);

	// x0�� 1�� �ʱ�ȭ. C���� ���ڷ� �ʱ�ȭ
	Learner learner(gate_choice, X0, C);
	learner.learn();

	return 0;
}
