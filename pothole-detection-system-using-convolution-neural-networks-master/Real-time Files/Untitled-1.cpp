#include <iostream>
using namespace std;
void hybrid();
void practical();
void theory();
int main()
{
	int choice;
	cout<<"Internals Calculator"<<endl;
	while(true)
	{
		cout<<"Select Course type: "<<endl;
		cout<<"1. Hybrid "<<endl;
		cout<<"2. Practical "<<endl;
		cout<<"3. Theory "<<endl;
		cout<<"4. Exit. "<<endl;
		cout<<"Enter your choice: "; cin>>choice;
		switch(choice)
		{
			case 1:
				hybrid();
				break;
			case 2:
				practical();
				break;
			case 3: 
				theory();
				break;
			case 4:
				exit(0);
		}
	}
	return 0;
}
void hybrid()
{
	float arr[10];
	float works = 0;
	float mst1, mst2, mst;
	float pfd, surp, ass, pmst;
	float att, finp, res;
	for(int i = 0; i<10; i++)
	{
		cout<<"Enter your marks in worksheet(out of 30) "<<(i+1)<<" :";
		cin>>arr[i];
		works += arr[i];
	}
	works /= 300;
	works *= 20;
	cout<<"Marks in MST 1(out of 20): "; cin>>mst1;
	cout<<"Marks in MST 2(out of 20): "; cin>>mst2;
	mst = (mst1+mst2)/4;
	cout<<"Portfolio or Discussion(out of 4): "; cin>>pfd;
	cout<<"Surprise test(out of 12): "; cin>>surp;
	surp /= 12; 
	surp *= 4;
	cout<<"Assignment(out of 10): "; cin>>ass;
	ass /= 10; 
	ass *= 6;
	cout<<"Practical MST(out of 10): "; cin>>pmst;
	pmst /= 10; 
	pmst *= 4;
	cout<<"Attendance(out of 2): "; cin>>att;
	cout<<"Final Practical(out of 40): "; cin>>finp;
	finp /= 2;
	res = works + mst + surp + ass + pmst + att + finp + pfd;
	cout<<"You got "<<res<<" out of 70 in internals."<<endl;
}
void theory()
{
	float mst, mst1, mst2, res, surp, ass, att, pfd;
	cout<<"Marks in MST 1(out of 20): "; cin>>mst1;
	cout<<"Marks in MST 2(out of 20): "; cin>>mst2;
	mst = (mst1+mst2)/2;
	cout<<"Portfolio or Discussion(out of 4): "; cin>>pfd;
	cout<<"Surprise test(out of 12): "; cin>>surp;
	surp /= 12; 
	surp *= 8;
	cout<<"Assignment(out of 10): "; cin>>ass;
	ass /= 10; 
	ass *= 6;
	cout<<"Attendance(out of 2): "; cin>>att;
	res = mst + pfd + surp + ass + att;
	cout<<"You got "<<res<<" out of 40 in internals. "<<endl;
	
}
void practical()
{
	float arr[10], works, pmst, finp, res;
	for(int i = 0; i<10; i++)
	{
		cout<<"Enter your marks in worksheet(out of 30) "<<(i+1)<<" :";
		cin>>arr[i];
		works += arr[i];
	}
	works /= 300;
	works *= 45;
	cout<<"Practical MST(out of 15): "; cin>>pmst;
	cout<<"Final Practicals(out of 40): "; cin>>finp;
	res = works + pmst + finp;
	cout<<"You got "<<res<<" out of 100. "<<endl;	
}