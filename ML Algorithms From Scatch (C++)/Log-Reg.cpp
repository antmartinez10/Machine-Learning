#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include<numeric>
#include <algorithm>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

vector<double> sigmoid(vector<vector<double>> matrix){
  // takes matrix as inpu and reutrns a vector of sigmoid values for each observation
  
  vector<double> result;
  for (int i = 0; i < matrix.size(); i++){
     result.push_back(1.0 / (1+exp(-(matrix[i][0] + matrix[i][1]))));
   
  }
  return result;
}

int main(int args, char** argv){
  ifstream inFS;
  string line, x_in, pclass_in, survived_in, sex_in, age_in;
  const int MAX_LEN = 1000;
  vector<int> pclass(0);
  vector<int> survived(0);
  inFS.open("data/titanic_project.csv");
  //cout << "Reading line 1" << endl;
  getline(inFS, line);
  //cout << "heading: " << line << endl;

  while(inFS.good()){
    getline(inFS, x_in, ',');
    getline(inFS, pclass_in, ',');
    getline(inFS,survived_in, ',');
    getline(inFS, sex_in, ',');
    getline(inFS, age_in, '\n');

    int pclass_int = stoi(pclass_in);
    int surivior_int = stoi(survived_in);

    pclass.push_back(pclass_int);
    survived.push_back(surivior_int);

  }
  inFS.close();
  vector<int> pclass_training(0);
  vector<int> survived_training(0);
  vector<int> pclass_test(0);
  vector<int> survived_test(0);
  for (int i = 0; i < 900; i++){
    pclass_training.push_back(pclass.at(i));
  }
  for (int i = 0; i < 900; i++){
    survived_training.push_back(survived.at(i));
  }
  for (int i = 900; i < pclass.size(); i++){
    pclass_test.push_back(pclass.at(i));
  }
  for (int i = 900; i < survived.size(); i++){
    survived_test.push_back(survived.at(i));
  }


  // sig function that takes matrix and returns vector of calues
  // weights: set equal to 1.
 
  auto start = high_resolution_clock::now();
  // initalize weights vector where w0 =1 and w1 =0
  vector<double> weights;
  weights.push_back(1);
  weights.push_back(1);

   // data matrix where col 1 is all 1 and will be mult by the intercept, col 2 is the predictor and will be mult by w1
   
  vector<vector<double>> data_matrix(900,vector<double>(2,1));
  for (int i = 0; i < 900; i++){
    data_matrix[i][1] = pclass_training[i];
  }
 

   vector<int> labels;
   for (int i = 0; i < 900; i++){
    labels.push_back(survived_training.at(i));
  }

  double learning_rate = .001;
  vector<double> error(900,0);
  vector<vector<double>> multiple(900,vector<double>(2,0));

  for (int j = 0; j < 500000; j++){
    
    for (int i = 0; i < data_matrix.size(); i++)
    {
        multiple[i][0] = data_matrix[i][0] * weights[0];
        multiple[i][1] = data_matrix[i][1] * weights[1];  
    }
     vector<double> prob_vector = sigmoid(multiple);
    
     for (int i=0; i < 900; i++){
       error[i] = labels[i] - prob_vector[i];
       weights[0]+=data_matrix[i][0] * error[i] * learning_rate;
       weights[1]+=data_matrix[i][1] * error[i] * learning_rate;
       
     }
  }
  auto stop = high_resolution_clock::now();

  vector<vector<double>> test_matrix(146,vector<double>(2,1));
  for (int i = 0; i < 146; i++){
    test_matrix[i][1] = pclass_test[i];
  }

vector<vector<double>> multiple2(146,vector<double>(2,0));
  for (int i = 0; i < test_matrix.size(); i++)
    {
        multiple2[i][0] = test_matrix[i][0] * weights[0];
        multiple2[i][1] = test_matrix[i][1] * weights[1];  
    }

vector<double> prob_vector_test = sigmoid(multiple2);
vector<int> pred_values(146,5);

for (int i = 0; i < 146; i++){
    if(prob_vector_test[i] >= 0.5){
      pred_values[i] = 1;
    }
    else{
      pred_values[i] = 0;
    }
  //cout << pred_values[i] << " " << survived_test[i] << endl;
  }

  //accuracy, sensitivity, specificity
  int correctCount = 0;
  int tp = 0;
  int fn = 0;
  int tn = 0; 
  int fp = 0;
  for(int i = 0; i < 146; i++){
    if(pred_values[i] == 0 && survived_test[i]==0){
        tn += 1;
        correctCount += 1;
      }
     else if(pred_values[i] == 1 && survived_test[i]==1){
        tp += 1;
        correctCount += 1;
      }
    else if(pred_values[i] == 0 && survived_test[i]==1){
        fn += 1;
      }
    else if(pred_values[i] == 1 && survived_test[i]==0){
        fp += 1;
      }
   
    }

  //cout << tp << " " << tn <<  " " << fp <<  " " << fn <<  " " << correctCount << endl;
  double accuracy = float(correctCount)/float(146);
  double spec = float(tp)/float(tp + fn);
  double sens = float(tn)/float(tn+fp);

  cout << "Weights: " << weights[0] << " " << weights[1] << endl;
  cout << "Accuracy: " << accuracy << endl;
  cout << "Sensitivity: " << sens << endl;
  cout << "Specificity: " << spec << endl;

  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "runtime: " << float(duration.count())/1000 << endl;

  return 0;
}


