#include<iostream>
#include<vector>
#include<fstream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<string>
using namespace std;

class cnn
{
private:
	vector< vector<double> > ihweights; //input -> hidden weights
	vector< vector<double> > howeights; //hidden -> output weights
	double error; //sum of squares error
	vector<double> errors; //individual output errors
	vector<double> hiddenvals; //hidden values while running
	vector< vector< vector<double> > > filters; //filters for the convolutional layers of the network
	vector< vector< vector<double> > > featmaps; //feature maps
	double lr; //learning rate
	double patternerror; //error for each run of the network to be backpropogated
public:
	cnn(int inputs, int hidden, int out, int filtersize, int numfilters);
	void convolution();
	void pool();
	double runnet(vector< vector< vector<double> > > in, vector<double> exp, vector<double>& out);
	void backprop();
};
