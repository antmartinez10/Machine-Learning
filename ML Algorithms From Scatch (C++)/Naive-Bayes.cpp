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

// used to store priors
struct Priors {
   double perished_prior; 
   double survived_prior; 

   // will also store survived count and perished count since we will need for lilihood functions
   double survived_count;
   double perished_count;

}typedef Priors;
Priors calculate_priors(vector<int>training){
    // take in training date to calculate prior for each category
    double perished_count = 0;
    double survived_count = 0;
    double perished_prob = 0;
    double survived_prob = 0;
    Priors results;

    for (int i = 0; i < training.size(); i++){
        if (training[i] == 0){
            perished_count++;
        }
        else if (training[i] == 1){
            survived_count++;
        }
    }
    perished_prob = perished_count/training.size();
    survived_prob = survived_count/training.size();

    results.perished_prior = perished_prob;
    results.survived_prior = survived_prob;
    
    results.survived_count = survived_count;
    results.perished_count = perished_count;
    return results;
    
}

vector<vector<double>>calculate_pclass_likelihood(double survived, double perished, vector<int> pclass_train, vector<int> survived_tran){
    // returns a 3x2 matrix of lilkhood values for p(pclass|survived)
    vector<vector<double>> pclass_likelihood(2,vector<double>(3,0));
    int sv [2] = {0,1};
    int pc [3] = {1,2,3};
    int survived_p1 = 0;
    int survived_p2 = 0;
    int survived_p3 = 0;
    int perished_p1 = 0;
    int perised_p2 = 0;
    int perished_p3 = 0;

    for(int i = 0; i<= 900; i++){
        if(survived_tran[i] == 1){
            if(pclass_train[i] == 1)
                survived_p1++;
            if(pclass_train[i] == 2)
                survived_p2++;
            if(pclass_train[i] == 3)
                survived_p2++;
        }
        else if(survived_tran[i] == 0){
            if(pclass_train[i] == 1)
                perished_p1++;
            if(pclass_train[i] == 2)
                perised_p2++;
            if(pclass_train[i] == 3)
                perised_p2++;
        }
    }
    double numSurvivedPclass1 = double(survived_p1);
    double numSurvivedPclass2 = double(survived_p2);
    double numSurvivedPclass3 = double(survived_p2);
    double numDiedPclass1 = double(perished_p1);
    double numDiedPclass2 = double(perised_p2);
    double numDiedPclass3 = double(perised_p2);

    // survived pclass likelihood
    pclass_likelihood[1][0] = numSurvivedPclass1/survived;
    pclass_likelihood[1][1] = numSurvivedPclass2/survived;
    pclass_likelihood[1][2] = numSurvivedPclass3/survived;

    // perished pclass likelihood
    pclass_likelihood[0][0] = numDiedPclass1/perished;
    pclass_likelihood[0][1] = numDiedPclass2/perished;
    pclass_likelihood[0][2] = numDiedPclass3/perished;
 
    return pclass_likelihood;
}  

vector<vector<double>>calculate_sex_likelihood(double survived, double perished, vector<int> sex_train, vector<int> survived_train){
    vector<vector<double>> sex_likelihood(2,vector<double>(2,0));
    int survived_f = 0, survived_m =0, perished_f = 0, perished_m = 0;
    for(int i = 0; i<= 900; i++){
        if(survived_train[i] == 1){
            if(sex_train[i] == 0)
                survived_f++;
            if(sex_train[i] == 1)
                survived_m++;
        }
        else if(survived_train[i] == 0){
            if(sex_train[i] == 0)
                perished_f++;
            if(sex_train[i] == 1)
                perished_m++;
        }
    }
    double survived_female = double(survived_f);
    double survived_male = double(survived_m);
    double perished_female = double(perished_f);
    double perished_male = double(perished_m);

    sex_likelihood[1][0] = survived_female/survived;
    sex_likelihood[1][1] = survived_male/survived;
    sex_likelihood[0][0] = perished_female/perished;
    sex_likelihood[0][1] = perished_male/perished;

    return sex_likelihood;
}

double get_average(vector<int>ages){
    return double(accumulate(ages.begin(),ages.end(),0.0))/ages.size();
}

double get_var(vector<int>ages,double mean){
    double variance = 0;
    for(int i = 0; i < ages.size(); i++){
        variance += (double(ages[i]) - mean) * (double(ages[i]) - mean);
        }
    variance /= ages.size();
    return variance;
}

double calculate_age_likelihood(int age,double mean, double variance){
    double pi = 3.145; 
    double result = 0;
    result = 1 / sqrt(2 * pi * variance) * exp(-((double(age)-mean)*((double(age)-mean)))/(2 * variance));

    return result;
}

vector<double> calculate_raw_probabilities(vector<vector<double>>pclass_lh, vector<vector<double>>sex_lh,
double perished_prior, double survive_prior, int pclass, int sex, int age, vector<double>means, vector<double>variance){
    
    double num_s = 0;
    double pclass_lh_s = pclass_lh[1][pclass];
    double sex_lh_s = sex_lh[1][sex];
    double age_lh_s = calculate_age_likelihood(age,means[1],variance[1]);
    num_s = (pclass_lh_s* sex_lh_s * survive_prior * age_lh_s);

    float num_p = 0;
    double pclass_lh_p = pclass_lh[0][pclass];
    double sex_lh_p = sex_lh[0][sex];
    double age_lh_p = calculate_age_likelihood(age,means[0],variance[0]);
    num_p = (float(pclass_lh_p)* float(sex_lh_p) * float(perished_prior) * float(age_lh_p));
    
    double denominator = 0;
    denominator = num_p + num_s;
    double prob_survived = num_s/denominator;
    double prob_perished = num_p/denominator;
    vector<double> result{prob_survived, prob_perished};
    return result;
    
}
int main(){
    ifstream inFS;
    string line, x_in, pclass_in, survived_in, sex_in, age_in;
    vector<int> pclass(0);
    vector<int> survived(0);
    vector<int> sex(0);
    vector<int> age(0);

    // open file
    cout << "opening data" << endl;
    inFS.open("data/titanic_project.csv");
    if (!inFS.is_open()){
        cout << "could not open data" << endl;
        return 1; // means 1 error
        }
    cout << "Reading line 1" << endl << endl;
    getline(inFS, line);

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, x_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS,survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        int pclass_int = stoi(pclass_in);
        int sex_int = stoi(sex_in);
        int age_int = stoi(age_in);
        int surivior_int = stoi(survived_in);

        pclass.push_back(pclass_int);
        survived.push_back(surivior_int);
        sex.push_back(sex_int);
        age.push_back(age_int);
        //medv.at(numObservations) = stof(medv_in);
        numObservations++;
    }
    inFS.close();
    // end of reading in file

    // split test and training data. firt 900 is training, rest is test data
    vector<int> pclass_training(0);
    vector<int> survived_training(0);
    vector<int> sex_training(0);
    vector<vector<int>> age_training(900,vector<int>(2,0));

    // put first 900 observations into training vectors
    for (int i = 0; i < 900; i++){
        pclass_training.push_back(pclass.at(i));
        survived_training.push_back(survived.at(i));
        sex_training.push_back(sex.at(i));
        age_training[i][0] = age.at(i);
        age_training[i][1] = survived.at(i);
        }

    // create test vectors, should be 146 observations
    vector<int> pclass_test(0);
    vector<int> survived_test(0);
    vector<int> sex_test(0);
    vector<int> age_test(0);

    for (int i = 900; i < pclass.size(); i++){
        pclass_test.push_back(pclass.at(i));
        survived_test.push_back(survived.at(i));
        sex_test.push_back(sex.at(i));
        age_test.push_back(age.at(i));
        }
    /*
    steps
    1. calculate prior probabillity of survived/perisehd as percentage for each category
    2. calculate liklihoods for qualitative data (pclass,sex)
    3. calculate liklihood for quanatiative data (age)
    4. probability densitiy for quantitative data (age)
    5. calculate raw probabilities 
    */

    // Step 1.
    auto start = high_resolution_clock::now();
    auto results = calculate_priors(survived_training);
    cout << "A-priori probabilities" << endl;
    cout << "perisehd prior " << results.perished_prior << endl;
    cout << "survived prior: " << results.survived_prior << endl << endl;
    
    // Step 2.

    // matrix of 2x3
    double survived_count = results.survived_count;
    double perished_count = results.perished_count;
    vector<vector<double>> pclass_likelihoods = calculate_pclass_likelihood(survived_count, perished_count, pclass_training, survived_training);
    
    cout << "Conditional probabilities: " << endl;
    cout << "pclass" << endl;
    for(int i=0; i<pclass_likelihoods.size(); i++){
        for(int j=0; j<pclass_likelihoods[i].size(); j++)
            cout << pclass_likelihoods[i][j] << " ";
        cout<<endl;
    }
    cout << endl;

    // matrix of 2x2
    vector<vector<double>> sex_likelihoods = calculate_sex_likelihood(survived_count, perished_count, sex_training, survived_training);
    cout << "sex" << endl;
    for(int i=0; i<sex_likelihoods.size(); i++){
        cout << i;
        for(int j=0; j<sex_likelihoods[i].size(); j++)
            cout << sex_likelihoods[i][j] <<" ";
        cout<<endl;
    }
    cout << endl;

    
    // Step 3/4 liklihood for age
    // calc_age_mean_variance()
    // calculate_age_likelihood

    //get_average(age_training);
    vector<double> age_mean(2,0);
    vector<double> age_var(2,0);

    vector<int> age_survived(0);
    vector<int> age_perished(0);

    for (int i =0; i < age_training.size();i++){
        if (age_training[i][1] == 0){
            age_perished.push_back(age_training[i][0]);
        }
        else if (age_training[i][1] == 1){
            age_survived.push_back(age_training[i][0]);
        }
    }
    
    age_mean[0] = get_average(age_perished);
    age_mean[1] = get_average(age_survived);

    age_var[0] = get_var(age_perished,age_mean[0]);
    age_var[1] = get_var(age_survived,age_mean[1]);

    cout << "age" << endl;
    cout << age_mean[0] << " " << sqrt(age_var[0]) << endl;
    cout << age_mean[1] << " " << sqrt(age_var[1]) << endl << endl;
    
    
     
    // step 5 calc raw probabilities
    vector<vector<double>> raw_proababilities; 
    for (int i = 0; i < 146; i++){
        raw_proababilities.push_back(calculate_raw_probabilities(pclass_likelihoods, sex_likelihoods, results.perished_prior,results.survived_prior,pclass_test[i],sex_test[i],age_test[i],
        age_mean,age_var));
    }
        //cout << raw_proababilities[1][0] << " " << raw_proababilities[1][1] << endl;
    
    auto stop = high_resolution_clock::now();

    vector<int> predictions;

    for (int i = 0; i < raw_proababilities.size(); i++){
        if (raw_proababilities[i][0] >= 0.5){
            predictions.push_back(1);
        }
        else{
            predictions.push_back(0);
        }
    }
   
  int correctCount = 0;
  int tp = 0;
  int fn = 0;
  int tn = 0; 
  int fp = 0;
  for(int i = 0; i < 146; i++){
    if(predictions[i] == 0 && survived_test[i]==0){
        tn += 1;
        correctCount += 1;
      }
     else if(predictions[i] == 1 && survived_test[i]==1){
        tp += 1;
        correctCount += 1;
      }
    else if(predictions[i] == 0 && survived_test[i]==1){
        fn += 1;
      }
    else if(predictions[i] == 1 && survived_test[i]==0){
        fp += 1;
      }
   
    }

  //cout << tp << " " << tn <<  " " << fp <<  " " << fn <<  " " << correctCount << endl;
    double accuracy = double(correctCount)/double(146);
    double sens = double(tp)/double(tp + fn);
    double spec = double(tn)/double(tn+fp);
    cout << "Accuracy: " << accuracy << endl;
    cout << "Specificity: " << spec << endl;
    cout << "Sensitivity: " << sens << endl;

    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "runtime: " << float(duration.count())/1000000000 << endl;

    return 0;
}




