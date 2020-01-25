#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h> 
#include<math.h>

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
	FILE *fp = fopen("ErrorGraph.txt", "w");	// ErrorGraph.txt로 결과값 저장
	int trials = 1;								// Learning Trials
	int layer = 3;

public:
	Learner(int gate_choice, int x0, float c) {

		// for random value generate
		max = 32767;
		srand(time(0));

		// 사용자가 And 혹은 and 혹은 and 를 누른 경우 == gate choice ==1
		if (gate_choice == 1) {
			printf("Start And Gate Learning...\n");
			strcpy_s(gate, "and");
		}
		else {
			// 사용자가 OR 혹은 Or 혹은 or 를 누른 경우 == gate choice == 0
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

		printf("Learning Layer : %d\n\n", layer);

		// 임의의 Weight 2N+1개 생성 (범위 : 1~5) 하나는 편향 조정용
		for (i = 0; i < 2*N+1; i++) {
			w[i] = (rand() / max) * 5;
			printf("initial weight[%d] randomly selected : %f \n", i, w[i]);
		}


		// 초기값 보여주기
		printf("x[0] initialized as : %d\n", x[0]);
		printf("learning rate c : %f\n\n", c);
		printf("==============initialize Complete==========\n\n");
	}

	// sigmoid serves as avtivation function
	float sigmoid(float x) {
		float output;
		printf("sigmoid : %f\n", 1.0 / (1.0 + exp(-x)));
		
		output = 1.0 / (1.0 + exp(-x));
		return output;
	}


	float * forward_sweep() {

		float results[3];
		float net = 0.0, net2=0.0, error, output, output2;
		printf("\nCalculating Output Data... \n", c);

		net = x[0] * w[0] + x[1] * w[1] + x[2] * w[2];
		printf("net1 : %f\n", net);
		
		net2 = x[0] * w[0] + x[1] * w[3] + x[2] * w[4];
		printf("net2 : %f\n", net2);

		output = sigmoid(net);
		output2 = sigmoid(net2);

		printf("output1 : %f\n", output);
		printf("output2 : %f\n", output2);

		results[0] = keep_learn(output, output2)[0];
		results[1] = keep_learn(output, output2)[1];

		return results;
	}

	float * keep_learn(float output1, float output2)
	{
		float net = 0.0, net2 = 0.0, error, fin_output, fin_output2;
		float results[3];
		// weight random 생성 4개
		max = 32767;
		srand(time(0));
		printf("\n\n==============Second Layer==========\n\n");
		
		for (i = 0; i < 2 * N + 1; i++) {
			w[i] = (rand() / max) * 5;
			printf("initial weight[%d] randomly selected : %f \n", i, w[i]);
		}

		net = x[0] * w[0] + output1 * w[1] + output2 * w[2];
		printf("net1 : %f\n", net);

		net2 = x[0] * w[0] + output1 * w[3] + output2 * w[4];
		printf("net2 : %f\n", net2);

		fin_output = sigmoid(net);
		fin_output2 = sigmoid(net2);

		printf("output1 : %f\n", fin_output);
		printf("output2 : %f\n", fin_output2);

		fin_out(fin_output, fin_output2);
		
		results[0] = output1;
		results[1] = output2;

		return results;
	}

	float fin_out(float input1, float input2)
	{
		float net = 0.0, net2 = 0.0, error, fin_output, fin_output2, result;
		
		// weight random 생성 4개
		max = 32767;
		srand(time(0));
		printf("\n\n============== Last Layer==========\n\n");
		for (i = 0; i < N + 1; i++) {
			w[i] = (rand() / max) * 5;
			printf("initial weight[%d] randomly selected : %f \n", i, w[i]);
		}
		
		net = x[0] * w[0] + input1 * w[1] + input2 * w[2];
		result = sigmoid(net);
		printf("result = %f", result);

		return result;
	}


	// target 값 계산해주는 함수
	int cal_target(int * x) {
		printf("Calculating Target Data ... \n");
		int result = -1;

		if (!strcmp(gate, "and")) {
			// N개의 input에 대하여 and gate 연산
			result = x[1] && x[2];
			for (i = 3; i < N + 1; i++) {
				result = result && x[i];
			}
		}
		if (!strcmp(gate, "or")) {
			// N개의 input에 대하여 or gate 연산
			result = x[1] || x[2];
			for (i = 3; i < N + 1; i++) {
				result = result || x[i];
			}
		}

		return result;
	}
	void back_prop() {

		float output;
		float del;
		float * w_del;
		float results[3];

		printf("\n\n=========== backword propagation ===========\n");
		printf("in backword prop : %f\n", forward_sweep()[0]);
		printf("in backword prop : %f\n", forward_sweep()[1]);
	
		output = get_output();
		del = get_delta(output);
		w_del = get_delta_weight(del);
		judge(w_del);

	}

	float get_output() {

		float net = 0.0, error;
		float output;
		printf("\nCalculating Output Data... \n", c);

		// output 값 구하기 : net = x[0]*w[0] + x[1]*w[1] + ... + x[N]*w[N]
		for (int i = 0; i < N + 1; i++) {
			net += x[i] * w[i];
		}

		// file write : weight값
		for (int i = 0; i < N + 1; i++) {
			fprintf(fp, "%f\t\t", w[i]);
		}

		// error rate구하기
		error = (cal_target(x) - net);
		fprintf(fp, "%f\n", error);

		printf("net : %f\n", net);

		// activation function. Threshold가 0임에 착안
		output = sigmoid(net);
		printf("sigmoid output : %d\n", output);

		printf("\n");

		return output;
	}
	// delta 값 구하기 : target data - Output data ( activation 결과 )
	float get_delta(float output) {

		float target, delta;
		target = float(cal_target(x));

		printf("target : %f\n", target);
		delta = target - output;
		printf("delta : %f\n", delta);

		return delta;
	}

	float * get_delta_weight(float delta) {
		printf("Calculating Differential Of Weight ... \n");

		for (int i = 0; i < N + 1; i++) {
			// delata weight (delta_w) 구하기 : C*delta*input_data
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
		int check = 1;	// 하나라도 weight에 0있으면 0
						// 모든 weight_delta가 0일 경우 1

		printf("Judging ... \n");
		// N개의 delata weight가 모두 - => +로 0이 되었다 : Learning 종료 결과 print 

		for (int i = 0; i < N + 1; i++) {
			printf("previous weight : %f : \n", w[i]); // delta_weight를 더하기 전 weight

		}
		printf("\n");
		for (int i = 0; i < N + 1; i++) {
			// N개 weight 갱신 : weight = weight + delta_w
			w[i] = w[i] + weight_delta[i];
		}

		for (int i = 0; i < N + 1; i++) {
			printf("final weight : %f : \n", w[i]); // delta_weight을 더한 뒤 weight

		}

		// 계속 Learning을 지속할지 Check : 모든 delta_weight이 0.0이 되었는지 확인
		for (int i = 0; i < N + 1; i++) {
			if (weight_delta[i] != 0.0) {
				check = 0;
				break;
			}

		}
		// if not all delta_weight is 0 > learn again
		if (check == 0 && layer != 0) {
			printf(">>> Learning not over.. Keep Learning\n");
			printf("===================trial : %d ====================\n", trials);
			trials++;
			layer -= 1;
			learn();
		}
		// if all delta_weight is 0 > learning over
		else {
			printf(">>> Learning All Over! all weight_delta is below TOLERENCE \n");
			printf("===================trial : %d ====================\n", trials);
			return;
		}
	}


	int learn() {

		forward_sweep();
		back_prop();


		return 0;
	}
};

int main()
{

	char type[50];
	int gate_choice = -1;

	// 어떤 Gate를 학습시킬지를 입력받는다.
	do {
		printf("현재 input : %d차원. (코드 상의 전처리기를 통해 차원을 늘려 입력받을 수 있습니다!)\n\n", N);
		printf("Learning 대상 : And Gate? or Or Gate? : ");

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
			printf("다시 입력하세요\n");
		}
	} while (1);

	// x0는 1로 초기화. C값을 인자로 초기화
	Learner learner(gate_choice, X0, C);
	learner.learn();

	return 0;
}
