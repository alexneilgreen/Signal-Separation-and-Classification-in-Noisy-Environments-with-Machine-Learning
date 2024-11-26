````mermaid
graph TD
    A[main.py]

    subgraph Src Folder
        B[analysis.py]
        C[data_processing.py]
        D[model.py]
        E[training.py]
        F[utils.py]
    end

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F

    subgraph Project Structure
        G[Data/]
        H[Figures/]
        I[Results/]
    end

    A --> G
    A --> H
    A --> I

    B --> |Contains| BFA[ForestRecordingAnalyzer]
    B --> |Contains| BFB[check_and_compare_results]

    C --> |Contains| CDA[AudioDataset]
    C --> |Contains| CDB[augment_dataset]

    D --> |Contains| DBM[BirdCallCNN]

    E --> |Contains| EFT[train_model]
    E --> |Contains| EFL[load_dataset]
    E --> |Contains| EFP[plot_training_history]
    E --> |Contains| EFM[print_final_metrics]

    F --> |Contains| FGD[generate_demo]
    ```
````
