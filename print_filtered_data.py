import uproot
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak

# Open the ROOT file
file_path = "./output/filtered_scatters.root"
root_file = uproot.open(file_path)

key = 'Filtered_Scatters;1'
obj = root_file[key]

if hasattr(obj, 'arrays'):
    # Step 1: Get the data (likely as an Awkward Array)
    data = obj.arrays()
    
    # Step 2: Explicitly convert to Pandas
    # For RNTuples, we often need to flatten or simplify the record
    try:
        df = ak.to_dataframe(data)
    except Exception as e:
        print(f"Standard conversion failed: {e}")
        # Fallback: if it's deeply nested, just take the first level
        df = pd.DataFrame(ak.to_list(data))

    print(f"\nStructure of {key}:")
    print(df.info()) 
    print("\nFirst 5 rows:")
    print(df.head())

    # Step 3: Plotting
    cols_to_plot = df.columns[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black')
            axes[i].set_title(f"Distribution of {col}")
    
    plt.tight_layout()
    plt.savefig("./output/filtered_scatters_plot.png")
    plt.show()

    limit = 350
    h = plt.hist2d(df['PostPosition_X'], df['PostPosition_Y'], 
                bins=50, 
                range=[[-limit, limit], [-limit, limit]], 
                cmap='viridis')

    
    plt.axhline(0, color='white', linewidth=0.8, linestyle='--') # Ligne horizontale à 0
    plt.axvline(0, color='white', linewidth=0.8, linestyle='--') # Ligne verticale à 0

    cb = plt.colorbar(h[3])
    cb.set_label('Intensité (Nombre de Scatters)')

    plt.title(f'Distribution des Scatters - {len(df)} points')
    plt.xlabel('PostPosition_X (mm)')
    plt.ylabel('PostPosition_Y (mm)')
    plt.grid(alpha=0.1)

    plt.savefig("./output/scatter_intensity_centered.png")
    plt.show()

else:
    print("Object does not contain array data.")