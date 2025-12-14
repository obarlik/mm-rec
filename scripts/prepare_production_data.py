
import json
import random
import os

def create_reasoning_example(id):
    """Creates a synthetic Chain-of-Thought (Reasoning) example."""
    # Simple Math Logic
    num1 = random.randint(10, 99)
    num2 = random.randint(2, 9)
    result = num1 * num2
    
    return {
        "text": f"User: Calculate {num1} * {num2}.\nModel: <THOUGHT> I need to multiply {num1} by {num2}. Breaking it down: {num1} * {num2} = ({num1//10}0 + {num1%10}) * {num2} = {num1//10*num2}0 + {num1%10*num2} = {result}. </THOUGHT> The answer is {result}. <EOS>"
    }

def create_tool_use_example(id):
    """Creates a synthetic Tool Use example."""
    cities = ["London", "Istanbul", "New York", "Tokyo", "Berlin"]
    city = random.choice(cities)
    temp = random.randint(15, 30)
    
    return {
        "text": f"User: What is the weather like in {city}?\nModel: <THOUGHT> The user is asking for weather information for {city}. I should use the weather tool. </THOUGHT> <TOOL_CALL> get_weather(city='{city}') <TOOL_END>\nSystem: <TOOL_RESULT> {{'temp': {temp}, 'condition': 'Sunny'}} </TOOL_RESULT>\nModel: It is currently {temp} degrees and Sunny in {city}. <EOS>"
    }

def main():
    print("🚀 Preparing Production Data for 100M Model...")
    
    output_file = "data/production_100m.jsonl"
    existing_file = "data/chat_data_real.jsonl"
    
    data = []
    
    # 1. Load Existing Data (if available)
    if os.path.exists(existing_file):
        print(f"   - Loading existing data from {existing_file}...")
        with open(existing_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        print("   ! Warning: chat_data_real.jsonl not found. Starting fresh.")

    # 2. Synthesize Reasoning Data (Boost IQ)
    print("   - Synthesizing Reasoning Examples...")
    for i in range(500): # Add 500 reasoning samples
        data.append(create_reasoning_example(i))
        
    # 3. Synthesize Tool Use Data (Boost Agency)
    print("   - Synthesizing Tool Use Examples...")
    for i in range(500): # Add 500 tool samples
        data.append(create_tool_use_example(i))

    # 4. Shuffle & Save
    random.shuffle(data)
    
    print(f"   - Saving {len(data)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print("✅ Data Preparation Complete.")

if __name__ == "__main__":
    main()
