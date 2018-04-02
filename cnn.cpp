#include"cnn.h"

//initialize weights filters and learning rate
//takes the number of input, hidden, and output nodes required and the filter size and number of filters
cnn::cnn(int inputs, int hidden, int out, int filtersize, int numfilters)
{
	vector<double> temp1;
	vector<double> temp2;
	vector< vector<double> > temp3;	

	lr = 0.004;

	//initalize all input -> hidden weights
	for(int i = 0; i < hidden; i++)
	{
		for(int j = 0; j < inputs; j++)
		{
			temp1.push_back((double(rand() / double(RAND_MAX)) - pow(inputs, -0.5)) / 2);
		}
		ihweights.push_back(temp1);
		temp1.clear();
	}

	//initalize all hidden -> output weights
	for(int i = 0; i < out; i++)
	{
		for(int j = 0; j < hidden; j++)
		{
			temp1.push_back((double(rand() / double(RAND_MAX)) - pow(inputs, -0.5)) / 2);
		}
		temp1.push_back((double(rand() / double(RAND_MAX)) - pow(inputs, -0.5)) / 2);
		howeights.push_back(temp1);
		temp1.clear();
	}

	//initalize all filters
	for(int i = 0; i < numfilters; i++)
	{
		for(int j = 0; j < filtersize; j++)
		{
			for(int k = 0; k < filtersize; k++)
			{
				temp2.push_back((double(rand() / double(RAND_MAX)) - pow(inputs, -0.5)) / 2);
			} 
			temp3.push_back(temp2);
			temp2.clear();
		}
		filters.push_back(temp3);
		temp3.clear();
	}

}

//takes all the filters and performs a dot product over all of the inputs stored in the feature map vector moving over the inputs by one stride 
//then it stores the result back into the feature map vector and run the pool function
void cnn::convolution()
{
	vector<double> temp1;
	vector< vector<double> > temp2;
	vector< vector< vector<double> > > temp3;
	double total = 0;
	int b = 0;
	int d = 0;

	for(int i = 0; i < featmaps.size(); i++) //for every feature map
	{
		for(int j = 0; j < filters.size(); j++) //for every filter
		{
			for(int k = 0; k < featmaps[i].size() - filters[j].size() + 1; k++) //move the filter down one stride
			{
				for(int q = 0; q < featmaps[i].size() - filters[j].size() + 1; q++) //move the filter to the right one stride
				{
					total = 0;
					b = 0;
					for(int a = k; a < filters[j].size() + k; a++) //area of the input vector to move down
					{
						d = 0;
						for(int c = q; c < filters[j].size() + q ; c++) // area of the input to move right
						{
							total += featmaps[i][a][c] * filters[j][b][d];
							d++;
						}
						b++;
					}
					//RElU function
					if(total > 0)
					{
						temp1.push_back(total);
					}
					else
					{
						temp1.push_back(0);
					}

				}
				temp2.push_back(temp1);
				temp1.clear();
			}
			temp3.push_back(temp2);
			temp2.clear();
		}
	}
	
	//clear the inputs and store the new feature map
	featmaps.clear();
        for(int i = 0; i < temp3.size(); i++)
        {
                featmaps.push_back(temp3[i]);
        }

	pool();
}

//takes all of the feature maps breaks them into quadrants and takes the maximum value from the quadrant and stores the results in the feature map vector
void cnn::pool()
{
	vector<double> temp1;
        vector< vector<double> > temp2;
        vector< vector< vector<double> > > temp3;
	double max = -1;

	for(int i = 0; i < featmaps.size(); i++) //for every feature map
	{
		for(int j = 0; j < featmaps[i].size(); j += featmaps[i].size() / 2) //select quadrant of the feature map vertically
		{
			for(int k = 0; k < featmaps[i].size(); k += featmaps[i].size() / 2) //select quadrant of the feature map horizontally
			{
				max = -1;
				for(int a = j; a < (featmaps[i].size() / 2) + j; a++) //horizontal movement space for the feature map to stay in the quadrant
				{
					for(int b = k; b < (featmaps[i].size() / 2) + k; b++) //vertical movement space for the feature map to stay in the quadrant
					{
						if(featmaps[i][a][b] > max)
						{
							max = featmaps[i][a][b];
						}
					}
				}
				temp1.push_back(max);
			}
			temp2.push_back(temp1);
			temp1.clear();
		}
		temp3.push_back(temp2);
		temp2.clear();
	}

	//clear the feature maps and replace them with the pools
	featmaps.clear();
        for(int i = 0; i < temp3.size(); i++)
        {
                featmaps.push_back(temp3[i]);
        }
}

//takes the input and expected outputs
//put the given input into the feature maps vector run the convolution function multiply all inputs by the input weights and all the hidden values by the output weights store the output in the out vector and return the overall error
double cnn::runnet(vector< vector< vector<double> > > in, vector<double> exp, vector<double>& out)
{
	double temp = 0;
	int s = 0;
	error = 0;
	featmaps.clear();	

	//put the input vector into the feature maps vector
	for(int i = 0; i < in.size(); i++)
	{
		featmaps.push_back(in[i]);
	}
	convolution();
	temp = 0;
	for(int i = 0; i < ihweights.size(); i++) //loop through the hidden nodes
	{
		s = 0;
		for(int a = 0; a < featmaps.size(); a++) //loop through the pools
		{
			for(int b = 0; b < featmaps[a].size(); b++)
			{
				for(int c = 0; c < featmaps[a][b].size(); c++)
				{
					temp += featmaps[a][b][c] * ihweights[i][s];
					s++;
				}
			}
		}
		hiddenvals.push_back(tanh(temp));
		temp = 0;
	}	

	for(int i = 0; i < howeights.size(); i++) //loop through the output nodes
	{
		for(int j = 0; j < howeights[i].size(); j++) //loop through the hidden nodes
		{
			temp += hiddenvals[j] * howeights[i][j];
		}
		temp += howeights[i][howeights[i].size() - 1];
		out.push_back(tanh(temp));
		temp = 0;
	}

	//store indiviual output errors and calculate the overall error
	for(int i = 0; i < out.size(); i++)
	{
		error += pow(out[i] - exp[i], 2);
		errors.push_back(out[i] - exp[i]);
	}
	error /= out.size();

	return error;
}

// go through the output to hidden weights and update and then go through the input to hidden and filter weights and subtract out the errors
void cnn::backprop()
{
	double temp;
	double temp2 = 0;
	double temp3 = 0;
	double temp4 = 0;
	double temp5 = 0;
	int s = 0;
	int t = 0;

	for(int i = 0; i < howeights.size(); i++) // for all the outputs
	{
		for(int j = 0; j < howeights[i].size() - 1; j++) //for all the hidden values
		{
			howeights[i][j] -= lr * errors[i] * hiddenvals[j]; //subtract out the error

			//normalize weights
			if(howeights[i][j] < -2)
			{
				howeights[i][j] = -2;
			}
			else if(howeights[i][j] > 2)
			{
				howeights[i][j] = 2;
			}
		}
		howeights[i][howeights[i].size() - 1] -= lr * errors[i]; //update the bias

		if(howeights[i][howeights[i].size() - 1] < -2)
             	{
        		howeights[i][howeights[i].size() - 1] = -2;
          	}
           	else if(howeights[i][howeights[i].size() - 1] > 2)
           	{
          	       howeights[i][howeights[i].size() - 1] = 2;
            	}
	}



	for(int i = 0; i < ihweights.size(); i++) //for all hidden nodes
	{
		s = 0;
		t = 0;
		temp2 = 0;
		for(int k = 0; k < howeights.size(); k++) //sum up all hidden to output weights for this hidden node
		{
			temp2 += howeights[k][i];
		}
		for(int a = 0; a < featmaps.size(); a++)
		{
			for(int b = 0; b < featmaps[a].size(); b++)
			{
				for(int c = 0; c < featmaps[a][b].size(); c++)
				{
					temp2 = 0;
					temp = 1 - (hiddenvals[i] * hiddenvals[i]);
					temp *= temp2 * error * lr;
					temp *= featmaps[a][b][c];
					ihweights[i][s] -= temp;
					temp3 += temp;
					temp4 += ihweights[i][s];
					s++;	
                			if(ihweights[i][s] < -2)
                			{
                        			ihweights[i][s] = -2;
                			}
                			else if(ihweights[i][s] > 2)
                			{
                       				ihweights[i][s] = 2;
                			}
				}
			}
			if(a % (featmaps.size() / filters.size()) == 0 && a != 0) //ensure that the correct filter is being updated
			{
				temp3 /= (featmaps.size() / filters.size() *4);
                      		temp3 *=  lr * temp3 * temp4;
                  		for(int x = 0; x < filters[t].size() ; x++)
                       		{
                        	  	for(int y = 0; y < filters[t][x].size(); y++)
                        	        {
                        	              	filters[t][x][y] -= temp3;
						if(filters[t][x][y] < -2)
						{
							filters[t][x][y] = -2;
						}
						else if(filters[t][x][y] > 2)
						{
							filters[t][x][y] = 2;
						}
                        	      	}
                    		}
                  		temp3 = 0;
				temp4 = 0;
				temp5 = 0;
                    		t++;
			}
		}
	}
	hiddenvals.clear();
	error = 0;
}
