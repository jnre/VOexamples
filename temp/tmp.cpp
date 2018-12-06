#include <iostream>
#include <string>
using namespace std;

const string& myFunc(string& s,string b){
    s = "naughty";
    return s;
};    



int main(void){

    string happy = "baby";
    string sad = "popop";
    string angry = myFunc(happy,sad);
    cout<<"angry:" << angry <<endl;
    cout <<"happy: " <<happy <<endl;



    return 0;
}
