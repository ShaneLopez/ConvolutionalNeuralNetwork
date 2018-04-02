#include"cnn.h"

int main()
{
	srand((unsigned)time(0));
	vector<double> input1;
	vector< vector<double> > input2;
	vector< vector < vector<double> > > input3;
	vector< vector< vector< vector <double> > > > input4;
	vector< vector< vector< vector < vector<double> > > > > input5;
	vector<double> exp;
	vector< vector<double> > exp1;
	vector<double> out;
	double temp;
	string file = "";
	int hcount = 0;
	int vcount = 0;
	ifstream fin;
	int c = 0;
	double mean;

	//take in and store all inputs into separate vectors
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 4; j++)
		{	
			file += to_string(i);
			file += to_string(j);
			file += ".txt";
			fin.open(file.data());

			while(!fin.eof() && c != 100)
			{
				if(hcount == 5)
				{
					hcount = 0;
					vcount++;
					input2.push_back(input1);
					input1.clear();
				}
				if(vcount == 5)
				{
					vcount = 0;
					input3.push_back(input2);
					input2.clear();
				}
		
				hcount++;
				c++;
				fin >> temp;
				temp /= 1000;
				input1.push_back(temp);
			}
			c = 0;
			input4.push_back(input3);
			input3.clear();
			fin.close();
			file = "";
		}
		input5.push_back(input4);
		input4.clear();

		//generate expected output
		for(int k = 0; k < 3; k++)
		{
			if(k == i)
			{
				exp.push_back(1);
			}
			else
			{
				exp.push_back(-1);
			}
		}
		exp1.push_back(exp);
		exp.clear();
	}
	hcount = 0;

	int filtercount = 2;
	int filtersize = 2;
	int insize = (4 * filtercount) * pow(filtersize, 2); 

	cnn net(insize, insize / 2, 3, filtersize, filtercount);

	int num = 0;
	int choice = 0;
	double avg = 1;
	double total = 0;

	//train the network until to a low enough average error
	do
	{
		num = rand() % 3;
		choice = rand() % 4;

		out.clear();
		temp = net.runnet(input5[num][choice], exp1[num], out);
		cout << "############################# " << num << choice << endl;
		for(int i = 0; i < out.size(); i++)
		{
			cout << out[i] << endl;
		}
		total += temp;
		hcount++;
		if(hcount == 10)
		{
			avg = total/10;
			total = 0;
			hcount = 0;
		}
		net.backprop();
	}while(avg > 0.005);

	for(int i = 0; i < exp1[0].size(); i++)
	{
		cout << exp1[0][i] << endl;
	}
	return 0;
}
