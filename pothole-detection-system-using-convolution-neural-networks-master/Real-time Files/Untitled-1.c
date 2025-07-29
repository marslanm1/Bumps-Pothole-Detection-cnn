#include<stdio.h>
imt main()
{
    int marks;
    printf("Enter marks: ");
    scanf("%d" ;&marks);
    if(marks<30){
        printf("Fail");
    }
    else if(marks>=30 && marks<=100){
        printf("Pass");
    }
    else{printf("Wrong choice");}
    return 0;



}