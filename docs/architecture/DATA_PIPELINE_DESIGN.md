# High-Performance C++ Streaming Data Pipeline 设计

## 1. Philosophy
> "Disk is the new RAM."

To handle massive datasets (e.g., 1TB+ of text) on finite hardware, we cannot load everything into memory. We must **stream** data directly from disk to the GPU/CPU.

## 2. Architecture Components

### A. The Tokenizer (C++)
*   **Role:** Convert `std::string` -> `std::vector<int>`.
*   **Algorithm:** Byte-Pair Encoding (BPE) or maybe a simpler Character-level for MVP.
*   **Implementation:** Pre-compute vocab map `std::unordered_map<string, int>`.

### B. The Storage Format (`.bin`)
*   **Problem:** Text files (`.txt`, `.jsonl`) are slow to parse (newline detection, string manipulation).
*   **Solution:** **Binary Stream Format**.
    *   Header: `Magic Number`, `Version`, `Total Tokens`.
    *   Body: Contiguous array of `int32_t` (Token IDs).
*   **Benefit:** Zero-copy loading. We can `mmap` this file and treat it as a `int*` array in C++.

### C. The `Dataset` Class
*   **Technique:** `mmap` (Memory Mapped File).
*   **Function:**
    *   Opens the `.bin` file.
    *   OS handles paging. If we access `data[i]`, OS loads that page from SSD.
    *   Supports random access $O(1)$ without RAM overhead.

### D. The `DataLoader` (Streaming Engine)
*   **Role:** Feed the `Trainer`.
*   **Mechanism:** Multi-threaded Prefetching.
    *   **Main Thread:** Asks for `batch`.
    *   **Worker Thread:** Reads from `Dataset`, prepares `Tensor`, pushes to `Queue`.
    *   **Circular Buffer:** A standard `RingBuffer<TrainingBatch>` to smooth out I/O latency.
*   **Streaming Logic:**
    *   Cursor position $P$ moves forward $P \to P + \text{batch\_size} \times \text{seq\_len}$.
    *   When file ends, rewind to 0 (Epoch complete).

## 3. Workflow
1.  **Preprocessing (One-time):** Python script or C++ tool converts `raw.txt` -> `dataset.bin`.
2.  **Training (Loop):**
    ```cpp
    DataLoader loader("dataset.bin", batch_size=64, seq_len=128);
    
    while(true) {
        TrainingBatch batch = loader.next(); // Instant (from buffer)
        trainer.train_step(batch);
    }
    ```

## 4. Performance Goals
*   **Throughput:** > 100 MB/s (Saturate NVMe SSD or PCIe bandwidth).
*   **Latency:** Zero waiting time for the GPU (Hiding I/O behind computation).
