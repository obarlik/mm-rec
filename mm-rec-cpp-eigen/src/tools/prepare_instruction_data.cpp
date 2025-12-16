/**
 * Prepare Instruction Dataset
 * 
 * Converts JSONL instruction data to binary format with ChatML templates
 * Format: {"instruction": ..., "input": ..., "output": ...}
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

// Simple JSON parser (no dependencies)
struct InstructionSample {
    std::string instruction;
    std::string input;
    std::string output;
};

// ChatML special tokens
const std::string IM_START = "<|im_start|>";
const std::string IM_END = "<|im_end|>";
const std::string NEWLINE = "\n";

std::string extract_json_field(const std::string& json, const std::string& field) {
    std::string search = "\"" + field + "\":";
    size_t start = json.find(search);
    if (start == std::string::npos) return "";
    
    start += search.length();
    // Skip whitespace
    while (start < json.length() && (json[start] == ' ' || json[start] == '\t')) start++;
    
    if (json[start] != '"') return "";
    start++; // Skip opening quote
    
    // Find closing quote (handle escaped quotes)
    std::string result;
    for (size_t i = start; i < json.length(); ++i) {
        if (json[i] == '\\' && i + 1 < json.length()) {
            // Escaped character
            char next = json[i + 1];
            if (next == 'n') result += '\n';
            else if (next == 't') result += '\t';
            else if (next == '"') result += '"';
            else if (next == '\\') result += '\\';
            else result += next;
            i++; // Skip next char
        } else if (json[i] == '"') {
            break;
        } else {
            result += json[i];
        }
    }
    return result;
}

InstructionSample parse_json_line(const std::string& line) {
    InstructionSample sample;
    sample.instruction = extract_json_field(line, "instruction");
    sample.input = extract_json_field(line, "input");
    sample.output = extract_json_field(line, "output");
    return sample;
}

std::string apply_chat_template(const InstructionSample& sample) {
    std::string text;
    
    // System message (implicit)
    text += IM_START + "system" + NEWLINE;
    text += "You are a helpful coding assistant." + NEWLINE;
    text += IM_END + NEWLINE;
    
    // User message
    text += IM_START + "user" + NEWLINE;
    if (!sample.input.empty()) {
        text += sample.instruction + NEWLINE + NEWLINE + sample.input + NEWLINE;
    } else {
        text += sample.instruction + NEWLINE;
    }
    text += IM_END + NEWLINE;
    
    // Assistant message
    text += IM_START + "assistant" + NEWLINE;
    text += sample.output + NEWLINE;
    text += IM_END + NEWLINE;
    
    return text;
}

// Simple char-level tokenizer
std::vector<int32_t> tokenize(const std::string& text, int32_t& assistant_start_idx) {
    std::vector<int32_t> tokens;
    
    // Find where assistant response starts (for loss masking)
    std::string assistant_marker = IM_START + "assistant" + NEWLINE;
    size_t assistant_pos = text.find(assistant_marker);
    assistant_start_idx = -1;
    
    for (size_t i = 0; i < text.length(); ++i) {
        if (assistant_pos != std::string::npos && i == assistant_pos + assistant_marker.length()) {
            assistant_start_idx = tokens.size();
        }
        // Simple: each char = token ID (0-255 for ASCII)
        tokens.push_back(static_cast<int32_t>(static_cast<unsigned char>(text[i])));
    }
    
    return tokens;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: prepare_instruction_data <input.json> <output.bin>" << std::endl;
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open input file: " << input_path << std::endl;
        return 1;
    }
    
    std::cout << "Processing instruction data..." << std::endl;
    
    // Read entire file (it's a JSON array)
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string content = buffer.str();
    infile.close();
    
    std::vector<int32_t> all_tokens;
    std::vector<int32_t> all_masks;
    int num_samples = 0;
    
    // Find all objects in array
    size_t pos = 0;
    while (true) {
        // Find next object start
        pos = content.find('{', pos);
        if (pos == std::string::npos) break;
        
        // Find matching closing brace
        int depth = 1;
        size_t end = pos + 1;
        bool in_string = false;
        char prev = 0;
        
        for (; end < content.length() && depth > 0; ++end) {
            char c = content[end];
            if (c == '"' && prev != '\\') {
                in_string = !in_string;
            } else if (!in_string) {
                if (c == '{') depth++;
                else if (c == '}') depth--;
            }
            prev = c;
        }
        
        if (depth != 0) break;
        
        std::string obj = content.substr(pos, end - pos);
        
        InstructionSample sample = parse_json_line(obj);
        if (!sample.instruction.empty() && !sample.output.empty()) {
            std::string formatted = apply_chat_template(sample);
            
            int32_t assistant_start = -1;
            std::vector<int32_t> tokens = tokenize(formatted, assistant_start);
            
            // Create mask
            std::vector<int32_t> mask(tokens.size(), 0);
            if (assistant_start >= 0) {
                for (size_t i = assistant_start; i < tokens.size(); ++i) {
                    mask[i] = 1;
                }
            }
            
            all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
            all_masks.insert(all_masks.end(), mask.begin(), mask.end());
            
            num_samples++;
            if (num_samples % 1000 == 0) {
                std::cout << "Processed " << num_samples << " samples..." << std::endl;
            }
        }
        
        pos = end;
    }
    
    std::cout << "Total samples: " << num_samples << std::endl;
    std::cout << "Total tokens: " << all_tokens.size() << std::endl;
    
    // Write binary file
    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return 1;
    }
    
    // Header
    int32_t magic = 0x4D4D5245;
    int32_t version = 2;
    int64_t count = all_tokens.size();
    
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    outfile.write(reinterpret_cast<char*>(&version), sizeof(version));
    outfile.write(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Tokens
    for (int32_t token : all_tokens) {
        outfile.write(reinterpret_cast<char*>(&token), sizeof(token));
    }
    
    // Masks
    for (int32_t m : all_masks) {
        outfile.write(reinterpret_cast<char*>(&m), sizeof(m));
    }
    
    outfile.close();
    
    std::cout << "✅ Data written to: " << output_path << std::endl;
    return 0;
}
