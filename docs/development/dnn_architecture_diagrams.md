# DNN Architecture Diagrams

This document contains Mermaid diagrams representing the architecture of the Adaptive Contextual Bandits DNN project. These diagrams cover the data flow, preprocessing, model structure (MLP), and training components.

## 1. System Architecture & Data Flow

This diagram illustrates the complete flow from raw input data through preprocessing, the neural network, and the training loop (loss calculation).

**Components:**
*   **Preprocessing (`src/models/preprocessing.py`)**: Handles `FillNA`, `Normalization` (Numerical), `OneHot` (Categorical), and `Action Encoding`.
*   **Neural Bandit Model (`src/models/neural_bandit.py`)**: The Q-Network (MLP).
*   **Training Logic (`src/train.py` & `train_step`)**: Masking Q-values with Action IDs, calculating Loss (MSE/Focal).
*   **Callbacks**: Validation and Logging.

```mermaid
graph TD
    subgraph Raw_Inputs [Raw Data Inputs]
        I1[(Numerical Features)]
        I2[(Categorical Features)]
        I3[(Action Column)]
        I4[(Label / Reward)]
    end

    subgraph Preprocessing_Model [Preprocessing Submodel]
        direction TB
        P1[FillNA]
        P2[FillNA]
        
        N1[Normalization Layer]
        C1[One-Hot Encoding]
        A1[Action Lookup/Encoding]
        
        I1 --> P1 --> N1
        I2 --> P2 --> C1
        I3 --> A1
        
        Concat[Fully Connected Layers - Concatenate Features]
        N1 --> Concat
        C1 --> Concat
    end

    subgraph Neural_Bandit [Neural Bandit Model - MLP]
        direction TB
        InputLayer[Input Layer]
        HiddenLayers[Hidden Dense Layers<br/>256 - 512 - 512 - 256 - 128 - 64 - 32]
        ReLU{{ReLU Activation}}
        Dropout[Dropout 0.2]
        OutputLayer[Output Layer<br/>Q-Values per Action]
        OutAct{{ReLU or Linear}}
        
        Concat --> InputLayer
        InputLayer --> HiddenLayers
        HiddenLayers --> ReLU
        ReLU --> Dropout
        Dropout --> OutputLayer
        OutputLayer --> OutAct
    end

    subgraph Training_Loop [Training Step and Loss]
        ActionID[Action ID Integer]
        QValues[Predicted Q-Values]
        Masking[Mask Q-Values by Action ID]
        ChosenQ[Chosen Action Q-Value]
        LossCalc{Loss Function<br/>MSE or Focal}
        Backprop[Backpropagation<br/>GradientTape]
        Optimizer[Optimizer<br/>Adam + Exp Decay]
        
        A1 --> ActionID
        OutAct --> QValues
        
        ActionID --> Masking
        QValues --> Masking
        Masking --> ChosenQ
        
        ChosenQ --> LossCalc
        I4 --> LossCalc
        LossCalc --> Backprop
        Backprop --> |∇ Gradients| Optimizer
        Optimizer -.-> |Update Weights| Neural_Bandit
    end

    subgraph Callbacks_Monitor [Callbacks]
        Val[ValidationCallback<br/>Balanced Accuracy]
        TB[TensorBoard<br/>Graph Trace]
        Chk[ModelCheckpoint]
    end
    
    Training_Loop -.-> Val
    Training_Loop -.-> TB
    Training_Loop -.-> Chk

    style Preprocessing_Model fill:#f9f,stroke:#333,stroke-width:2px
    style Neural_Bandit fill:#bbf,stroke:#333,stroke-width:2px
    style Training_Loop fill:#dfd,stroke:#333,stroke-width:2px
```

## 2. MLP "Toy" Representation (Perceptron View)

This diagram provides a simplified "Toy" visualization of the Multi-Layer Perceptron (MLP) defined in `src/models/neural_bandit.py`.

*   **Note**: The actual model has Dense layers with [256, 512, 512, 256, 128, 64, 32] neurons, each followed by ReLU activation. Dropout layers (0.2) are placed after Dense 256 and Dense 64. The output layer uses ReLU (for MSE loss) or Linear (for Focal loss). This diagram uses a simplified node count (2-3 nodes per layer) to represent the *connectivity structure*.

```mermaid
graph LR
    subgraph Input_Layer [Input Layer]
        x1((x1))
        x2((x2))
        x3((x3))
        xn((...))
    end

    subgraph Hidden_Block_1 [Dense 256 + ReLU]
        h1_1((h))
        h1_2((h))
        h1_3((h))
    end

    subgraph Hidden_Block_2 [Dense 512 + ReLU]
        h2_1((h))
        h2_2((h))
        h2_3((h))
    end

    subgraph Hidden_Block_3 [Dense 512 + ReLU]
        h3_1((h))
        h3_2((h))
        h3_3((h))
    end

    subgraph Hidden_Block_4 [Dense 256 + ReLU]
        h4_1((h))
        h4_2((h))
        h4_3((h))
    end
    
    subgraph Dropout_1 [Dropout 0.2]
        d1[x]
    end

    subgraph Hidden_Block_5 [Dense 128 + ReLU]
        h5_1((h))
        h5_2((h))
    end

    subgraph Hidden_Block_6 [Dense 64 + ReLU]
        h6_1((h))
        h6_2((h))
    end
    
    subgraph Dropout_2 [Dropout 0.2]
        d2[x]
    end

    subgraph Hidden_Block_7 [Dense 32 + ReLU]
        h7_1((h))
        h7_2((h))
    end

    subgraph Output_Layer [Output - ReLU or Linear]
        y1((Q1))
        y2((Q2))
        ym((Qm))
    end

    %% Connections - Simplified full connectivity
    x1 & x2 & x3 & xn --- h1_1 & h1_2 & h1_3
    h1_1 & h1_2 & h1_3 --- h2_1 & h2_2 & h2_3
    h2_1 & h2_2 & h2_3 --- h3_1 & h3_2 & h3_3
    h3_1 & h3_2 & h3_3 --- h4_1 & h4_2 & h4_3
    
    h4_1 & h4_2 & h4_3 -.- d1
    d1 -.- h5_1 & h5_2
    h5_1 & h5_2 --- h6_1 & h6_2
    
    h6_1 & h6_2 -.- d2
    d2 -.- h7_1 & h7_2
    
    h7_1 & h7_2 --- y1 & y2 & ym

    %% Styling
    classDef neuron fill:#fff,stroke:#333,stroke-width:2px;
    classDef input fill:#e1f5fe,stroke:#01579b;
    classDef output fill:#e8f5e9,stroke:#1b5e20;
    classDef dropout fill:#ffebee,stroke:#b71c1c,stroke-dasharray: 5 5;

    class x1,x2,x3,xn input;
    class h1_1,h1_2,h1_3,h2_1,h2_2,h2_3,h3_1,h3_2,h3_3,h4_1,h4_2,h4_3 neuron;
    class h5_1,h5_2,h6_1,h6_2,h7_1,h7_2 neuron;
    class y1,y2,ym output;
    class d1,d2 dropout;
```

## How to View
You can view these diagrams directly in tools that support Mermaid (like GitHub, GitLab, or Obsidian) or paste the code blocks into the [Mermaid Live Editor](https://mermaid.live/).
