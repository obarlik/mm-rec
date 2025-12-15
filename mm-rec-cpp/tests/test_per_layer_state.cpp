#include "mm_rec/layers/memory_state.h"
#include <iostream>

using namespace mm_rec;

int main() {
    std::cout << "[TEST] Per-Layer State Isolation (Bug #2 check)" << std::endl;
    
    // Test that each layer has truly isolated state
    auto state = PerLayerMemoryState(3, 2, 64);
    
    // Modify layer 0
    auto layer0 = state.get_layer_state(0);
    layer0.fill_(1.0);
    state.update_layer_state(0, layer0);
    
    // Check layer 1 is still zero (not shared!)
    auto layer1 = state.get_layer_state(1);
    auto sum1 = layer1.sum().item<float>();
    
    if (sum1 == 0.0f) {
        std::cout << "✅ States are truly isolated (Bug #2 prevented)" << std::endl;
        return 0;
    } else {
        std::cerr << "❌ States are SHARED - Bug #2 present!" << std::endl;
        return 1;
    }
}
