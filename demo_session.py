
import os
import sys
import shutil
import torch
# Proper imports from the project structure
sys.path.append(os.getcwd())
from mm_rec.core.session_memory import SessionMemoryManager
from mm_rec.core.memory_state import MemoryState

def demo_session_management():
    print("🧠 Starting MM-Rec Session Management Demo...\n")
    
    # 1. Setup Environment
    storage_path = "./demo_sessions"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path) 
    
    # Initialize Manager
    manager = SessionMemoryManager(base_dir=storage_path)
    
    # Define Memory Config (Small for demo)
    mem_cfg = {
        "k_dim": 64, 
        "v_dim": 64, 
        "num_slots": 128
    }
    
    def create_state():
        return MemoryState(
            short_term_config=mem_cfg,
            long_term_config=mem_cfg
        )

    # --- SCENARIO 1: Alice ---
    print("👤 User A (Alice) connects...")
    session_id_a = "alice_001"
    
    # Create State & Simulate Learning
    print("   - Creating Alice's memory...")
    state_a = create_state()
    
    # Write specific pattern to verify persistence
    # Let's say Alice learns 'Blue' -> Pattern 1.1 at slot 0
    with torch.no_grad():
        state_a.short_term.k.data[0, :] = 1.1 
    
    # Save
    print("   - Saving Alice's session...")
    # Wrap in expert dict (usually 'text' and 'code' experts)
    memory_states = {"text": [state_a]} 
    manager.serialize_state(session_id_a, memory_states)
    print("   ✅ Access saved to disk.\n")
    
    
    # --- SCENARIO 2: Bob ---
    print("👤 User B (Bob) connects...")
    session_id_b = "bob_002"
    state_b = create_state()
    
    # Bob learns 'Red' -> Pattern 9.9
    with torch.no_grad():
        state_b.short_term.k.data[0, :] = 9.9
        
    print("   - Saving Bob's session...")
    manager.serialize_state(session_id_b, {"text": [state_b]})
    print("   ✅ Access saved to disk.\n")
    
    
    # --- SCENARIO 3: Alice Returns ---
    print("👤 User A (Alice) returns later...")
    
    # Load
    loaded_data = manager.load_state(
        session_id_a, 
        device=torch.device('cpu'), 
        expert_names=["text"]
    )
    
    # Verify
    if loaded_data:
        # Create empty template to load into (if load_state returned dict of tensors, but here it might return states)
        # Wait, load_state returns Dict[str, List[MemoryState]] ?
        # Checking implementation... _load_from_file returns loaded *state dicts* or objects?
        # Actually standard pytorch load returns state_dict usually.
        # But SessionMemoryManager likely manages reconstruction if it uses pickle/torch.load on objects
        # OR it loads state_dict and we need to load_state_dict.
        
        # Let's assume it returns dictionary of states based on serialization logic
        # Looking at session_memory.py imports: pickle.
        # It likely pickles the whole state or list of states.
        
        loaded_state_a = loaded_data["text"][0]
        
        # Inspect
        val = loaded_state_a.short_term.k.data[0, 0].item()
        print(f"   - Memory Value at Slot 0: {val:.1f}")
        
        if abs(val - 1.1) < 1e-5:
            print("   ✅ SUCCESS: Alice's memory persisted!")
        elif abs(val - 9.9) < 1e-5:
            print("   ❌ FAILURE: Data Leakage! (Found Bob's memory)")
        else:
            print(f"   ❌ FAILURE: Value mismatch (Got {val})")
            
    else:
        print("   ❌ FAILURE: Could not load session.")
        
    print("\n✨ Demo Complete.")

if __name__ == "__main__":
    demo_session_management()
