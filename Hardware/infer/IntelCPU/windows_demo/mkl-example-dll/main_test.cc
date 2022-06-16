#include <stdio.h>
#include <stdlib.h>

extern int test_cblas_dgemm(int N);

int main(int argc, char* argv[])
{
	int N;
    if(argc < 2)
	{
		printf("Enter matrix size N=");
		//please enter small number first to ensure that the 
		//multiplication is correct! and then you may enter 
		//a "reasonably" large number say like 500 or even 1000
		scanf("%d",&N);
	}
	else
	{
		N = atoi(argv[1]);
	}

    return test_cblas_dgemm(N);
}