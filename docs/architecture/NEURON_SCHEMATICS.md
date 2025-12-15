# MM-Rec Neuron Schematics (The "Block")

Bu şema, modelin temel yapı taşı olan `MMRecBlock` ("Nöron") mimarisini gösterir.

```mermaid
graph TD
    %% Inputs
    Input(Input Token embedding) --> Split{System}
    Mem_Prev(Memory State t-1) --> GatedMemory
    
    %% 1. Gated Memory (Vertical/Temporal)
    subgraph "1. Vertical Memory (Temporal)"
        Split -->|h_t| GatedMemory[Gated Memory Update]
        GatedMemory -->|Update| Mem_Next(Memory State t)
        GatedMemory -->|New State| Mem_Out(h_mem)
    end
    
    %% 2. Horizontal Specialization (MoE)
    subgraph "2. Horizontal Specialization (MoE)"
        Mem_Out --> Router{Router Gate}
        
        Router -->|Top-K Select| Exp1[Expert 1: Sports]
        Router -->|Top-K Select| Exp2[Expert 2: Finance]
        Router -->|...| ExpN[Expert N: Tech]
        
        Exp1 -->|Weighted Sum| Aggregator((Σ))
        Exp2 --> Aggregator
        ExpN --> Aggregator
    end
    
    %% 3. UBOO (Universal Backward)
    subgraph "3. UBOO Output"
        Aggregator -->|Projection| Logits[Logits: Vocab Prediction]
        Logits --> Loss(Loss Calculation)
    end
    
    %% Connections
    Aggregator --> Output(Output to Next Layer)
    
    %% Styling
    style Input fill:#f9f,stroke:#333
    style Router fill:#ff9,stroke:#333
    style Logits fill:#9ff,stroke:#333
    style Exp1 fill:#fff,stroke:#333,stroke-dasharray: 5 5
    style Exp2 fill:#fff,stroke:#333,stroke-dasharray: 5 5
    style ExpN fill:#fff,stroke:#333,stroke-dasharray: 5 5
```

## Bileşenler

### 1. Gated Memory (Zaman Ekseni)
Standart bir nöronun aksine, bu blok "hatırlama" yeteneğine sahiptir. `GatedMemory` ünitesi, GRU benzeri kapılarla gelen bilgiyi eski hafıza ile birleştirir ($h_t + M_{t-1} \to M_t$).

### 2. MoE (Uzmanlık Ekseni)
Hafızadan çıkan bilgi, **sabit bir işlemden geçmek yerine**, içeriğine göre farklı "Uzmanlara" (Experts) yönlendirilir.
- **Router:** "Bu token sporla ilgili, Uzman 1'e gitsin" der.
- **Dynamic Routing:** Sadece seçilen uzmanlar çalışır (CPU dostu).

### 3. UBOO (Holografik Çıktı)
Her nöron (blok), sadece bir sonraki katmana bilgi göndermekle kalmaz, aynı zamanda **doğrudan tahmin yapmaya çalışır.** Bu, modelin her katmanının "anlamlı" bilgi taşımasını zorunlu kılar.
