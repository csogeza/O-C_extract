#include<stdio.h>
#include<math.h>
#include<stdlib.h>

double absd(double szam){
    double absvalue;
     if(szam<0){
         absvalue=-szam;
     }
     else{
         absvalue=szam;
     }
    return absvalue;
 }

 // THE PROGRAM IS SENSITIVE FOR THE ORDERING OF THE DATA: MEASUREMENTS MUST BE IN ASCENDING ORDER ACCORDING TO TIME, OR THE PROGRAM WILL NOT WORK PROPERLY

int main(int argc, char* argv[]){
    double refepoch,period,freq,date,mindate,maxdate,maxdelta,lowlimit;
    int mpoints,epoch,lowepoch;
    double* input;
    double* output;
    int* epocharr;

    FILE *file=fopen(argv[1],"r");
    refepoch=atof(argv[2]);                 // Julian date of reference epoch
    freq=atof(argv[3]);                     // frequency of the pulsation
    period=1.0/freq;
    mindate=atof(argv[4]);                  // minimum date, max the smallest Julian date-period
    maxdate=atof(argv[5]);                  // maximum date, at least period+the largest Julian date
    mpoints=atoi(argv[6]);                  // number of rows in file
    maxdelta=atof(argv[7]);                 // radii of confidence interval

    input=(double*)malloc(3*mpoints*sizeof(double));
    output=(double*)malloc(2*mpoints*sizeof(double));
    epocharr=(int*)malloc(1*mpoints*sizeof(int));

    int i,j,ind;
    for(i=0;i<3*mpoints;i++){
        fscanf(file,"%lf",&input[i]);
    }
    fclose(file);

    date=refepoch;
    epoch=0;
    while(date>mindate){
        date=date-period;
        epoch-=1;
    }
    lowlimit=date;
    lowepoch=epoch;
    printf("%lf\n",lowlimit);

    i=0;
    ind=0;
    for(j=0;j<mpoints;j++){
        date=lowlimit;
        epoch=lowepoch;
        while(date<maxdate){
            if(absd(input[3*i]-date)<maxdelta){
                output[2*ind]=input[3*i];
                epocharr[ind]=epoch;
                output[2*ind+1]=input[3*i]-date;
                i+=1;
                ind+=1;
                break;
            }
            if(date>input[3*i]){
                i+=1;
                break;
            }
            date=date+period;
            epoch+=1;
        }
    }

    file=fopen(argv[8],"w");            // output file

    for(i=0;i<ind;i++){
        fprintf(file,"%10f %d %10f %10f\n",output[2*i],epocharr[i],output[2*i+1],input[3*i+2]);
    }

return 0;
}
