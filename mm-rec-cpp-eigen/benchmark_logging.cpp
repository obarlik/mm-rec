#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

int main() {
    const int N = 1000;
    
    // Test 1: Logging her step
    auto t1 = chrono::high_resolution_clock::now();
    for(int i=0; i<N; ++i) {
        cout << "Step " << i << " | Loss: 1.234 | LR: 0.001 | 755 tok/s" << endl;
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto dt1 = chrono::duration<double,milli>(t2-t1).count();
    
    // Test 2: Logging her 10 step
    auto t3 = chrono::high_resolution_clock::now();
    for(int i=0; i<N; ++i) {
        if(i % 10 == 0)
            cout << "Step " << i << " | Loss: 1.234 | LR: 0.001 | 755 tok/s" << endl;
    }
    auto t4 = chrono::high_resolution_clock::now();
    auto dt2 = chrono::duration<double,milli>(t4-t3).count();
    
    cout << "\n=== RESULTS ===" << endl;
    cout << "Every step: " << dt1 << " ms" << endl;
    cout << "Every 10th: " << dt2 << " ms" << endl;
    cout << "Speedup: " << (dt1/dt2) << "x" << endl;
    
    return 0;
}
