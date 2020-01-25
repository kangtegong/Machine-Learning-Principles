#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define THRESHOLD 0.5
#define TOLERANCE 0.0001
#define input 2
#define hidden 10
#define LEARNING_RATE 5		// 5로 상정하면 15000번 이내로 학습이 끝남
#define RAND_MAX 65535
#define bignum 100

class Learner {
private:
	double hidden_w[hidden][input + 1];	// hidden layer의 weight
	double output_w[hidden];		// output layer의 weight weight. 거꾸로 계산할 것이기 때문에 1차원 배열이면 충분. 

	float train_set_x[9][2] = { {0.0 , 0.0 }, {0.0 ,1.0 }, {1.0 ,0.0 }, {1.0 ,1.0 }, {0.5, 1.0 }, {1.0 ,0.5}, {0.0 ,0.5}, {0.5, 0.0}, {0.5,0.5} };
	float train_set_y[9][2] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
	double train_set[9][3] = { {0.0, 0.0, 0.0}, {0.0,1.0,0.0}, {1.0,0.0,0.0}, {1.0,1.0,0.0}, {0.5,1.0,0.0}, {1.0,0.5,0.0}, {0.0,0.5,0.0}, {0.5,0.0,0.0}, {0.5,0.5,1.0} };

	double hideen_node[hidden]; // hidden Layer의 노드 자체
	double output; // 최종 output
	double error = bignum; // 오류를 발생시키기 위한 조건.
	int i, j;
	int nodes; // 학습 데이터의 갯수
	int count = 0;

	FILE *fp = fopen("Weight_data.txt", "w");	// ErrorGraph.txt trial-errror 쌍 결과값 저장
	FILE *fp2 = fopen("Weight_vector.txt", "w");		// Weight.txt에 행렬 형식으로 가중치 결과값 저장
	FILE *fp3 = fopen("Error_Rate.txt", "w");		// Error_Rate.txt에 Error Graph 결과값 저장

public:
	Learner(int choice) {
		srand(RAND_MAX);

		init_hidden_w(hidden_w); // forward sweep을 위해 hidden layer의 weight을 모두 초기화해준다 
		init_output_w(output_w); // forward sweep을 위해 최종 output weight까지 모두 초기화해준다
		print(hidden_w, output_w); // 초기화된 hidden weight, output weight print

		nodes = 9;	// 학습할 데이터의 갯수
		printf("학습 데이터 개수 : %d\n", nodes);

		// 초기 세팅 완료 후 learning 수행
		learn();

	}

	void learn() {
		// learning 수행. TOLERANCE 보다 error가 클 동안 learning process 반복
		// 실질적인 learning은 여기에서 일어난다.
		while (error > TOLERANCE) {
			error = 0.0;

			// 학습 데이터에 대해 수행
			for (i = 0; i < nodes; ++i) {

				output = forward(hidden_w, output_w, hideen_node, train_set[i]); // forward sweep (그로 인해 얻은 값은 output)
				learn_output(output_w, hideen_node, train_set[i], output); // output weight 계산
				learn_hidden(hidden_w, output_w, hideen_node, train_set[i], output); //hidden weight 계산
				error += (output - train_set[i][input])*(output - train_set[i][input]); // output-target(input)의 제곱을 error로 삼는다
			}
			++count;
			printf("trials : %d \t error : %lf\n", count, error);
			fprintf(fp3, "trials : %d\t\t error : %lf\n", count, error);
		}

		// learning process 종료 후, learning 완료된 hidden 과 output weight 다시 print
		print(hidden_w, output_w);

		printf("\n\n");
		printf("======== < Final Output Report > =======");
		printf("\n\n");
		for (i = 0; i < nodes; ++i) {
			printf("%d번째 input에 대한 결과\n", i + 1);
			for (j = 0; j < input + 1; ++j) {
				printf("input data : %lf\n", train_set[i][j]);
			}
			output = forward(hidden_w, output_w, hideen_node, train_set[i]);
			printf("sigmoid output : %lf\n", output);

			// Threshold보다 크면 1 print, 작으면 0 print
			if (output > THRESHOLD) {
				printf("결과 : %lf\n\n", 1.0);
			}
			else {
				printf("결과 : %lf\n\n", 0.0);
			}
		}
		printf("=============================\n");

	}

	// sigmoid 함수
	double sigmoid(double u)
	{
		return 1.0 / (1.0 + exp(-u));
	}

	//  hidden layer의 weight 초기화 함수
	void init_hidden_w(double hidden_w[hidden][input + 1])
	{
		int i, j;
		for (i = 0; i < hidden; ++i) {
			for (j = 0; j < input + 1; ++j) {
				hidden_w[i][j] = double_random();

			}
		}
	}

	// output의 weight을 초기화하는 함수
	void init_output_w(double output_w[hidden + 1])
	{
		for (int i = 0; i < hidden + 1; ++i)
		{
			output_w[i] = double_random();
		}
	}

	// 실제 backpropagation algorithm을 시행하는 함수 (output weight 연산)
	void learn_output(double output_w[hidden + 1], double hideen_node[], double e[input + 1], double output)
	{
		int i;
		double delta;

		delta = (e[input] - output)*output*(1 - output);	// delta weight 연산

		fprintf(fp2, "(learning process) weight diff : \t");
		fprintf(fp2, "[");
		for (i = 0; i < hidden; ++i)

		{
			output_w[i] += LEARNING_RATE * hideen_node[i] * delta;	// output weight 갱신
		}

		output_w[i] += LEARNING_RATE * (-1.0)*delta;
		fprintf(fp, "(learning) output : %lf\n", output_w[i]);
		fprintf(fp2, "%lf, ", output_w[i]);
		//fprintf(fp2, "]\n");

	}

	// hidden layer 안의 노드 학습 함수
	void learn_hidden(double hidden_w[hidden][input + 1], double output_w[hidden + 1], double hideen_node[], double e[input + 1], double output)
	{
		int i, j;
		double dj;

		for (j = 0; j < hidden; ++j)
		{
			dj = hideen_node[j] * (1 - hideen_node[j]) * output_w[j] * (e[input] - output)*output*(1 - output);

			for (i = 0; i < input; ++i)
			{
				hidden_w[j][i] += LEARNING_RATE * e[i] * dj;
			}

			hidden_w[j][i] += LEARNING_RATE * (-1.0)*dj;
			fprintf(fp, "(learning) hidden : %lf\n", hidden_w[j][i]);
			fprintf(fp2, "%lf, ", hidden_w[j][i]);
		}
		fprintf(fp2, "]\n");
		fprintf(fp, "\n");

	}

	// 랜덤 숫자 발생기. 편의를 위해 RANDOM Range를 기존의 2배로 만든 것을 포함하였다
	double double_random(void)
	{
		double randnum;

		while ((randnum = (double)rand() / RAND_MAX) == 1.0);
		randnum = randnum * 2 - 1;

		return randnum;

	}

	// 현재까지의 weight를 출력해주는 함수
	void print(double hidden_w[hidden][input + 1], double output_w[hidden + 1])
	{
		int i, j;
		printf("\n==========< Weight Report >=========\n");

		for (i = 0; i < hidden; ++i) {

			fprintf(fp2, "weight of hidden layer\n");
			fprintf(fp2, "[");
			for (j = 0; j < input; j++) {
				printf("\n");
				printf("weight of hidden layer : ");
				printf("%lf ", hidden_w[i][j]);
				fprintf(fp, "hidden : %lf\n", hidden_w[i][j]);
				fprintf(fp2, "%lf, ", hidden_w[i][j]);
			}
			fprintf(fp2, "]\n");
		}
		printf("\n");

		fprintf(fp2, "weight of output layer\n");
		fprintf(fp2, "[");
		for (i = 0; i < hidden; ++i) {
			printf("\n");
			printf("%lf ", output_w[i]);
			fprintf(fp, "output : %lf\n", output_w[i]);
			fprintf(fp2, "%lf, ", output_w[i]);
		}
		printf("\n");
		fprintf(fp, "\n");
		fprintf(fp2, "]\n");
	}

	// forword sweep 함수. back prop 마지막에 final weight을 구할 때 한번 더 쓰인다. 
	double forward(double hidden_w[hidden][input + 1], double output_w[hidden + 1], double hideen_node[], double e[input + 1])
	{
		int i, j;
		double net;
		double output;

		for (i = 0; i < hidden; ++i) {
			net = 0;
			for (j = 0; j < input; ++j)
			{
				net += e[j] * hidden_w[i][j];		// net
			}
			net -= hidden_w[i][j];
			hideen_node[i] = sigmoid(net);	// sigmoid 결과

		}
		output = 0;
		for (i = 0; i < hidden; ++i) {
			output += hideen_node[i] * output_w[i];
		}
		output -= output_w[i];


		return sigmoid(output);

	}

};


int main() {
	int choice;
	choice = 1;

	Learner learner(1);
	return 0;
}