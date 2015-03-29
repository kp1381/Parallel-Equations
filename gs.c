#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/

float *a; /* The coefficients */
float *x; /* The unknowns */
float *b; /* The constants */
float *diag_vec; /*diagonal vector of each row in a*/
float err; /* The absolute relative error */
int num = 0; /* number of unknowns */

/****** Function declarations */

void check_matrix(); /* Check whether the matrix will converge */
void get_input(); /* Read input from file */
int is_done();/*checks whether the difference in our calculated values is less than the required*/
/********************************/

/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

int is_done( float *new_x, int num){
    int i;
    for(i = 0; i < num; i++){
        if(((new_x[i] - x[i])/new_x[i]) > err){
            return 1;
        }
    }
    return 0;
}

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix(){
    int bigger = 0; /* Set to 1 if at least one diag element > sum  */
    int i, j; 
    float sum = 0;
    float aii = 0; 
    for(i = 0; i < num; i++){
        sum = 0;
        aii = fabs(a[i*(num + 1)]);
        for(j = 0; j < num; j++){
            if( j != i) sum += fabs(a[i*num + j]);
        }
        if( aii < sum){
            printf("The matrix will not convergen");
            exit(1);
        }
        if(aii > sum) bigger++;
    }
    if( !bigger ){
        printf("The matrix will not convergen");
        exit(1);
    }
}

/******************************************************/
/* Read input from file */
void get_input(char filename[]){
    FILE * fp;
    int i ,j;
    fp = fopen(filename, "r");
    if(!fp){
        printf("Cannot open file %sn", filename);
        exit(1);
    }
    fscanf(fp,"%d ",&num);
    fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
    a = (float*)malloc(num * num * sizeof(float));
    if(!a){
        printf("Cannot allocate a!n");
        exit(1);
    }
    x = (float *) malloc(num * sizeof(float));
    if(!x){
        printf("Cannot allocate x!n");
        exit(1);
    }
    b = (float *) malloc(num * sizeof(float));
    if( !b){
        printf("Cannot allocate b!n");
        exit(1);
    }
     /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
    diag_vec = (float *) malloc(num * sizeof(float));
    for(i = 0; i < num; i++){
        fscanf(fp,"%f ", &x[i]);
    }
    for(i = 0; i < num ; i++){
        for(j = 0; j < num; j++){
            fscanf(fp,"%f ",&a[i*num + j]);
            if(i == j){
                diag_vec[i] = a[i*num + j];
            }
        }
         /* reading the b element */
        fscanf(fp,"%f ",&b[i]);
    }
    fclose(fp);
}

int main(int argc, char *argv[]){
    int nit = 0; /* number of iterations */
    int i, j, k; 
    if( argc != 2) {
        printf("Usage: gsref filenamen");
        exit(1);
    }
    /* Read the input file and fill the global data structure above */
    get_input(argv[1]);
    /* Check for convergence condition */
    check_matrix();
    //Initialize MPI
    int comm_sz;
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //MPI variables
    int my_first_i = num/comm_sz;
    int remainder = num % comm_sz;
    int sendcounts[comm_sz], displs[comm_sz], recv[comm_sz],sendcountsA[comm_sz],displsA[comm_sz],recvA[comm_sz];
    int disp = 0;
    //create sendcounts,recv and displs for MPI_Scatterv and MPI_Allgatherv
    //sendcountsA is the same as sendcounts but every element is multiplied by num, because it is used to send a's
    for(i = 0; i < comm_sz; i++){
        if(i < remainder){
            sendcounts[i] = my_first_i + 1;
            } else {
            sendcounts[i] = my_first_i;
        }
        recv[i] = sendcounts[i];
        recvA[i]=recv[i]*num;
        sendcountsA[i]=sendcounts[i]*num;
        displs[i] = disp;
        displsA[i]=displs[i]*num;
        disp = disp + sendcounts[i];
    }
    //For each processes, allocate space for local(partial) x, b ,a
    float *local_x = (float *) malloc(sendcounts[my_rank] * sizeof(float));
    float *local_a = (float *) malloc(sendcountsA[my_rank] * sizeof(float));
    float *local_b = (float *) malloc(sendcounts[my_rank] * sizeof(float));
    float *current = (float *) malloc(num * sizeof(float));
    float *local_diagonal = (float *) malloc(sendcounts[my_rank] * sizeof(float));

	MPI_Scatterv(a, sendcountsA, displsA, MPI_FLOAT,
    local_a, recvA[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(b, sendcounts, displs, MPI_FLOAT,
    local_b, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
   
    MPI_Scatterv(diag_vec, sendcounts, displs, MPI_FLOAT,
    local_diagonal, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
   
    MPI_Scatterv(x, sendcounts, displs, MPI_FLOAT,
    local_x, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    //initialize current to x
    for(i = 0; i < num; i++){
        current[i] = x[i];
    }

    do{
        nit++;
        //set x to values of current
        for(i = 0; i < num; i++){
            x[i] = current[i];
        }
        //calculate local x's
        //each process calculate their local_x new values given currx
        for(i = 0; i < sendcounts[my_rank]; i++){
            int global_i = i;
            for(k = 0; k < my_rank; k++){
                global_i += sendcounts[k];
            }
            //initially set to the corresponding constant
            local_x[i] = local_b[i];
            //calculate up to global_i...
            for(j = 0; j < global_i; j++){
                local_x[i] = local_x[i] - local_a[(i*num)+ j] * x[j];
            }
            //continue after global_i...
            for(j = global_i + 1; j < num; j++){
                local_x[i] = local_x[i] - local_a[(i*num)+ j] * x[j];
            }
            //divide local_x[i] by its corresponding diagonal element in a
            local_x[i] = local_x[i]/local_diagonal[i];
        }
        //gather all local_x's from all processes and set to current
        MPI_Allgatherv(local_x, sendcounts[my_rank], MPI_FLOAT,
        current, recv, displs, MPI_FLOAT, MPI_COMM_WORLD);
        //check error, if greater than threshold repeat...
    }while(is_done(current, num));
    
    if( my_rank == 0){
        /* Writing to the stdout */
        /* Keep that same format */
        for(i = 0; i < num; i++){
            printf("%f\n", x[i]);
        }
        //print total number of iterations...
        printf("total number of iterations: %d\n", nit);
        free(x);
        free(a);
        free(b);
        free(diag_vec);
    }
    //dealocating memory
    free(local_x);
    free(local_a);
    free(local_b);
    free(current);
    free(local_diagonal);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
