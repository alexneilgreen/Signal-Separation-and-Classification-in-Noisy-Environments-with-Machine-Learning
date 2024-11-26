````mermaid
graph TD
    A[main.py]

    subgraph Project Structure
        B[Data/]
        C[Figures/]
        D[Results/]
    end

    A --> B
    A --> C
    A --> D

    A --> E[src/]

    E --> F[analysis.py]
    E --> G[data_processing.py]
    E --> H[model.py]
    E --> I[training.py]
    E --> J[utils.py]

    F --> |Contains| BFA[ForestRecordingAnalyzer]
    F --> |Contains| BFB[check_and_compare_results]

    G --> |Contains| CDA[AudioDataset]
    G --> |Contains| CDB[augment_dataset]

    H --> |Contains| DBM[BirdCallCNN]

    I --> |Contains| EFT[train_model]
    I --> |Contains| EFL[load_dataset]
    I --> |Contains| EFP[plot_training_history]
    I --> |Contains| EFM[print_final_metrics]

    J --> |Contains| FGD[generate_demo]
    ```
````
